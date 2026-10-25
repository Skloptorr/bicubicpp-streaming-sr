r"""Train one of the 16 models (4 scales x 4 channel tiers) from the shared cache.

Same recipe as the original train_v2.py (MSE, Adam 5e-4 -> 5e-6 cosine) but on
the new multi-scale cache:

    hr.npy      (N,192,192,3)
    lr_x1.npy   (N,192,192,3)   lr_x15 (N,128,128,3)   lr_x2 (N,96,96,3)   lr_x3 (N,64,64,3)
    src.npy     (N,)            source image index

Two things matter for the 16-model comparison:

  * the split is per SOURCE IMAGE, never per patch. A 4K tile yields 220
    neighbouring patches, so a per-patch split leaks the same scene into both
    sides and inflates the number.
  * the validation patches are FROZEN into `split.json` on the first run and
    reused by every model afterwards, so the differences between c=24 and c=36
    are not sampling noise.

Terminal output:
    滚动进度行  step / % / loss / best dB / it/s / ETA
    每次验证   完整历史表 + 曲线火花线（dB）
    每 1000 步 _last.pth（可续训），每 10000 步 _s{N}.pth，最优存 _best.pth

    python train_multi.py --sr 1.5 --c 24
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

CACHE_DEFAULT = os.environ.get("SR_CACHE", "cache/d15")
OUT_DEFAULT = os.environ.get("SR_WEIGHTS", "weights")
LOG_DEFAULT = os.environ.get("SR_LOGS", "logs")
LR_NAME = {1.0: "lr_x1", 1.5: "lr_x15", 2.0: "lr_x2", 3.0: "lr_x3"}
SPARK = "▁▂▃▄▅▆▇█"


def ps_r(sr):
    r = int(round(2 * sr))
    if abs(r - 2 * sr) > 1e-6:
        raise SystemExit("sr=%g needs 2*sr integral" % sr)
    return r


class Net(nn.Module):
    """Bicubic++ (Bilecen & Ayazoglu, CVPRW 2023) with sr_rate and width exposed."""

    def __init__(self, c, sr):
        super().__init__()
        r = ps_r(sr)
        cout = r * r * 3
        self.conv0 = nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(c, cout, 3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(r)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1)) + x0
        return self.ps(self.conv_out(x2))


def y_psnr(a, b):
    """PSNR on the Y plane. Accepts (H,W,3) or (3,H,W); CHW is transposed first
    -- indexing a[:,0] on CHW silently measures one image row instead of luma."""
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if b.shape[0] == 3 and b.shape[-1] != 3:
        b = b.transpose(1, 2, 0)
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return 10 * np.log10(1.0 / max(float(((ya - yb) ** 2).mean()), 1e-10))


def spark(vals):
    """One-line ASCII curve, so the trend is readable in a plain terminal."""
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / (hi - lo) * (len(SPARK) - 1)))]
                   for v in vals)


def make_split(src, val_frac, seed, n_val_patch, split_path):
    """Freeze the train/val split once so all 16 models see the same val set."""
    if not os.path.exists(split_path):
        raise SystemExit(
            "missing %s\nrun this first:\n  python 03_代码/数据/make_split.py --cache %s --force"
            % (split_path, os.path.dirname(split_path)))
    d = json.load(open(split_path, encoding="utf-8"))
    is_val = np.isin(np.asarray(src), d["val_src"])
    return np.array(d["val_patch"], dtype=np.int64), np.flatnonzero(~is_val), d


def build_runs(idx):
    """Split a sorted index array into contiguous [start, stop) runs.

    Val patches are excluded from the training index, so the train set is a few
    hundred nearly-contiguous segments. Reading a batch from inside one run is a
    single sequential slice of the memmap instead of 32 random seeks -- which
    matters because the cache (145 GB) is 4.6x the machine's RAM (31.6 GB) and
    random access thrashes the page cache.
    """
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(idx)]])
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=float, default=1.5)
    ap.add_argument("--c", type=int, default=28)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--n-val-patch", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--peak-lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--order", default="sequential", choices=["sequential", "random"],
                    help="sequential walks the (already shuffled) cache in order")
    ap.add_argument("--tag", default="")
    ap.add_argument("--loss", default="mse", choices=["mse", "l1", "perc", "grad", "y"],
                    help="perc = L1 + vgg16 features; grad = L1 + gradient distance; "
                         "y = luma-dominant L1")
    ap.add_argument("--y-chroma-weight", type=float, default=0.1,
                    help="chroma weight when --loss y")
    ap.add_argument("--gray-aug", type=float, default=0.0,
                    help="probability of converting a training pair to greyscale, so the "
                         "network learns to leave neutral input neutral")
    ap.add_argument("--grad-weight", type=float, default=0.1,
                    help="weight on the gradient term when --loss grad")
    ap.add_argument("--perc-weight", type=float, default=0.05,
                    help="weight on the perceptual term when --loss perc")
    ap.add_argument("--perc-scale", type=int, default=2,
                    help="downsample factor before the VGG taps (1 = full res)")
    ap.add_argument("--perc-ntaps", type=int, default=4,
                    help="how many VGG taps to use (4 = relu1_2..relu4_3)")
    ap.add_argument("--perc-every", type=int, default=1,
                    help="apply the perceptual term every Nth step")
    a = ap.parse_args()

    lr_name = LR_NAME.get(a.sr)
    if lr_name is None:
        raise SystemExit("sr must be 1, 1.5, 2 or 3")
    lr_mm = np.load(os.path.join(a.cache, lr_name + ".npy"), mmap_mode="r")
    hr_mm = np.load(os.path.join(a.cache, "hr.npy"), mmap_mode="r")
    src = np.load(os.path.join(a.cache, "src.npy"), mmap_mode="r")
    n = lr_mm.shape[0]

    tag = a.tag or "bicubic_%gx_c%d" % (a.sr, a.c)
    va_idx, tr_idx, sinfo = make_split(src, a.val_frac, a.seed, a.n_val_patch,
                                       os.path.join(a.cache, "split.json"))
    print("cache %s | pairs %d | src images %d | val imgs %d | val patches %d | train %d"
          % (os.path.basename(a.cache), n, sinfo["n_images"], sinfo["n_val_images"],
             len(va_idx), len(tr_idx)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
    net = Net(a.c, a.sr).to(dev)
    r = ps_r(a.sr)
    mac = 6.75 * a.c + 4.5 * a.c * a.c + 2.25 * a.c * (r * r * 3)
    nparam = sum(p.numel() for p in net.parameters())
    print("tag %s | sr=%g c=%d | r=%d | params %s | MAC/LR px %.0f | fp32 %.0f KB"
          % (tag, a.sr, a.c, r, format(nparam, ","), mac, nparam * 4 / 1024), flush=True)
    print("steps %d | batch %d | ckpt every %d | amp=%s ema=%s augment=%s"
          % (a.steps, a.batch, a.ckpt_every, a.amp, a.ema or "-", a.augment), flush=True)

    ema = None
    if a.ema > 0:
        ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    start = 1
    if a.resume and os.path.exists(a.resume):
        net.load_state_dict(torch.load(a.resume, map_location=dev))
        base = os.path.basename(a.resume)
        if "_s" in base:
            try:
                start = int(base.rsplit("_s", 1)[-1].split(".")[0]) + 1
            except ValueError:
                start = 1
        print("resumed from %s at step %d" % (base, start), flush=True)

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(LOG_DEFAULT, exist_ok=True)
    csv_path = os.path.join(LOG_DEFAULT, "train_%s.csv" % tag)
    new_csv = not os.path.exists(csv_path)
    cf = open(csv_path, "a", newline="", encoding="utf-8")
    cw = csv.writer(cf)
    if new_csv:
        cw.writerow(["tag", "sr", "c", "step", "loss", "val_psnr_y", "lr", "minutes"])

    opt = torch.optim.Adam(net.parameters(), lr=a.peak_lr)
    from perceptual import build as build_crit
    mse, loss_name = build_crit(a, dev)
    print("  loss: %s" % loss_name, flush=True)
    rng = np.random.default_rng(a.seed + 1)
    best, t0 = 0.0, time.time()
    acc, accn = 0.0, 0
    use_amp = a.amp and dev == "cuda"
    hist = []           # (step, loss, db)

    def val_psnr():
        net.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(va_idx), 32):
                idx = va_idx[i:i + 32]
                l = torch.from_numpy(np.ascontiguousarray(lr_mm[idx])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                h = torch.from_numpy(np.ascontiguousarray(hr_mm[idx])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = net(l)
                out = out.float().clamp(0, 1).cpu().numpy()
                hn = h.cpu().numpy()
                for j in range(len(idx)):
                    ps.append(y_psnr(out[j], hn[j]))
        net.train()
        return float(np.mean(ps))

    def show_curve(step):
        dbs = [h[2] for h in hist]
        print("\n" + "─" * 72, flush=True)
        print(" 验证历史  %s   (step %d)" % (tag, step), flush=True)
        print(" %6s  %10s  %9s  %8s  %6s" % ("step", "loss", "val dB", "Δbest", ""), flush=True)
        bb = -1e9
        for (s, l, v) in hist:
            flag = ""
            if v > bb:
                bb = v
                flag = "★best"
            print(" %6d  %10.6f  %9.3f  %8s  %6s"
                  % (s, l, v, "", flag), flush=True)
        if len(dbs) >= 2:
            print(" 曲线 %s   %.3f → %.3f  (%+.3f dB)"
                  % (spark(dbs), dbs[0], dbs[-1], dbs[-1] - dbs[0]), flush=True)
        print("─" * 72, flush=True)

    if start == 1 and len(va_idx):
        v0 = val_psnr()
        hist.append((0, 0.0, v0))
        print("初始验证 %.3f dB（未训练）" % v0, flush=True)

    # sequential batch source: the cache is globally shuffled, so walking it in
    # order is still a random sample -- with one big page read instead of 32 seeks
    bs = a.batch
    runs = build_runs(tr_idx)
    n_runs = len(runs)
    cache_gb = (n * 192 * 192 * 3 * 2 + n * 4) / 2 ** 30
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (AttributeError, ValueError, OSError):
        ram_gb = 31.6
    print("数据读取 %s | 训练集 %d 段连续区间 | 缓存 %.1f GB vs 内存 %.1f GB"
          % (a.order, n_runs, cache_gb, ram_gb), flush=True)
    state = {"run": int(rng.integers(0, max(n_runs, 1))), "off": 0}

    def next_batch_idx():
        """Return a contiguous (start, stop) from the current run, wrapping epochs."""
        guard = 0
        while guard < n_runs + 2:
            guard += 1
            a0, b0 = runs[state["run"]]
            s = a0 + state["off"]
            if s + bs <= b0:
                state["off"] += bs
                return s, s + bs
            state["run"] = (state["run"] + 1) % n_runs
            state["off"] = 0
        return None

    net.train()
    for step in range(start, a.steps + 1):
        lr_now = a.min_lr + 0.5 * (a.peak_lr - a.min_lr) * (1 + np.cos(np.pi * (step - 1) / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr_now

        if a.order == "sequential":
            sl = next_batch_idx()
            if sl is None:
                continue
            l = torch.from_numpy(np.ascontiguousarray(lr_mm[sl[0]:sl[1]])).permute(0, 3, 1, 2).float().to(dev) / 255.0
            h = torch.from_numpy(np.ascontiguousarray(hr_mm[sl[0]:sl[1]])).permute(0, 3, 1, 2).float().to(dev) / 255.0
        else:
            idx = rng.choice(tr_idx, size=a.batch, replace=False)
            l = torch.from_numpy(np.ascontiguousarray(lr_mm[idx])).permute(0, 3, 1, 2).float().to(dev) / 255.0
            h = torch.from_numpy(np.ascontiguousarray(hr_mm[idx])).permute(0, 3, 1, 2).float().to(dev) / 255.0
        if a.augment:
            if rng.random() < 0.5:
                l = torch.flip(l, [3]); h = torch.flip(h, [3])
            if rng.random() < 0.5:
                l = torch.flip(l, [2]); h = torch.flip(h, [2])
            if rng.random() < 0.5:
                l = l.transpose(2, 3); h = h.transpose(2, 3)

        if a.gray_aug > 0 and rng.random() < a.gray_aug:
            # Teach "neutral in, neutral out". Without this the network barely
            # ever sees colour-free input, so on black-and-white text it still
            # paints fringes. Both sides go greyscale so the target is honest.
            wgt = torch.tensor([0.2126, 0.7152, 0.0722], device=dev).view(1, 3, 1, 1)
            l = (l * wgt).sum(1, keepdim=True).expand(-1, 3, -1, -1)
            h = (h * wgt).sum(1, keepdim=True).expand(-1, 3, -1, -1)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = net(l)
        # the loss runs outside autocast: the VGG taps must stay in fp32
        if a.loss == "perc":
            loss = mse(out.float(), h, step % a.perc_every == 0)
        else:
            loss = mse(out.float(), h)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()

        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                    else:
                        ema[k] = v.clone()

        acc += float(loss.detach()); accn += 1
        if step % 100 == 0:
            el = time.time() - t0
            its = step / el
            eta = (a.steps - step) / max(its, 1e-9)
            print("\r  step %6d/%d %5.1f%% | loss %.6f | best %.3f dB | %.1f it/s | ETA %02d:%02d   "
                  % (step, a.steps, 100 * step / a.steps, acc / accn, best, its,
                     int(eta // 60), int(eta % 60)), end="", flush=True)

        if step % a.val_every == 0 or step == a.steps:
            v = val_psnr()
            hist.append((step, acc / max(accn, 1), v))
            mark = ""
            if v > best:
                best = v
                torch.save(net.state_dict(), os.path.join(a.out, tag + "_best.pth"))
                if ema is not None:
                    torch.save({k: vv.to(net.state_dict()[k].dtype) for k, vv in ema.items()},
                               os.path.join(a.out, tag + "_best_ema.pth"))
                mark = " ★"
            show_curve(step)
            cw.writerow([tag, a.sr, a.c, step, "%.6f" % (acc / accn), "%.4f" % v,
                         "%.2e" % lr_now, "%.1f" % ((time.time() - t0) / 60)])
            cf.flush()
            acc, accn = 0.0, 0

        if a.ckpt_every and step % a.ckpt_every == 0:
            p = os.path.join(a.out, "%s_s%d.pth" % (tag, step))
            torch.save(net.state_dict(), p)
            print("\n  [ckpt] %s  (%.1f KB)" % (os.path.basename(p),
                                                os.path.getsize(p) / 1024), flush=True)
        elif step % 1000 == 0:
            torch.save(net.state_dict(), os.path.join(a.out, tag + "_last.pth"))

    cf.close()
    print("\n" + "=" * 72)
    print("完成 %s | best %.3f dB | 用时 %.0f 分钟" % (tag, best, (time.time() - t0) / 60))
    if len(hist) >= 2:
        print("曲线 %s   %.3f → %.3f  (%+.3f dB)"
              % (spark([h[2] for h in hist]), hist[0][2], hist[-1][2],
                 hist[-1][2] - hist[0][2]))
    print("=" * 72)


if __name__ == "__main__":
    main()
