r"""Python ports of Magpie's CAS and SMAA-Medium for offline comparison renders.

CAS is an exact port of `effects/CAS/CAS.hlsl` (itself a port of ffx-cas):
a 5-tap cross filter whose weight adapts to local contrast, with
    peak = -1 / (8 - 3 * sharpness)

SMAA is the MEDIUM preset from `effects/SMAA/SMAA.hlsli`:
    SMAA_THRESHOLD 0.1, SMAA_MAX_SEARCH_STEPS 8,
    SMAA_DISABLE_DIAG_DETECTION, SMAA_DISABLE_CORNER_DETECTION
With diagonals and corners off, the blending-weight pass is the orthogonal
pattern search only, which is what is ported. The AreaTex/SearchTex LUTs from the
effect are used as-is.

This is not bit-exact with the shader: the GPU version relies on bilinear gather
offsets and a packed search LUT purely for speed. Those are reproduced
behaviourally, not numerically. CAS is exact; SMAA is indicative.
"""
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)   # SMAA's own weights
AREATEX_W, AREATEX_H = 160, 560
AREATEX_MAX_DISTANCE = 16
AREATEX_SUBTEX = 1.0 / 7.0


# ---------------------------------------------------------------- CAS
def cas(rgb, sharpness=0.3):
    """Contrast Adaptive Sharpening. rgb float32 HxWx3 in [0,1]."""
    peak = -1.0 / (8.0 - 3.0 * sharpness)

    def sh(a, dy, dx):
        return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

    b, d, f, h, e = sh(rgb, -1, 0), sh(rgb, 0, -1), sh(rgb, 0, 1), sh(rgb, 1, 0), rgb
    mn = np.minimum(np.minimum(np.minimum(d, e), np.minimum(f, h)), b)
    mx = np.maximum(np.maximum(np.maximum(d, e), np.maximum(f, h)), b)
    rcp = 1.0 / np.maximum(mx, 1e-6)
    amp = np.sqrt(np.clip(np.minimum(mn, 1.0 - mx) * rcp, 0.0, 1.0))
    w = amp * peak
    wG = w[..., 1:2]                       # Magpie weights every channel by green
    rcpWeight = 1.0 / (1.0 + 4.0 * wG)
    out = np.clip((b + d + f + h) * wG + e, 0.0, None) * rcpWeight
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------- SMAA
def _shift(a, dy, dx, fill=0.0):
    h, w = a.shape[:2]
    out = np.full_like(a, fill)
    ys, yd = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
    xs, xd = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
    out[yd, xd] = a[ys, xs]
    return out


def smaa_edges(rgb, threshold=0.1, local_contrast=2.0):
    """SMAALumaEdgeDetectionPS. Returns edges[...,0]=horizontal, [...,1]=vertical."""
    L = (rgb * LUMA).sum(-1)
    Lleft = _shift(L, 0, -1)
    Ltop = _shift(L, -1, 0)
    Lright = _shift(L, 0, 1)
    Lbottom = _shift(L, 1, 0)
    Lll = _shift(L, 0, -2)
    Ltt = _shift(L, -2, 0)

    dl = np.abs(L - Lleft)
    dt = np.abs(L - Ltop)
    e_h = (dl >= threshold).astype(np.float32)
    e_v = (dt >= threshold).astype(np.float32)

    dr = np.abs(L - Lright)
    db = np.abs(L - Lbottom)
    maxDelta = np.maximum(np.maximum(dl, dt), np.maximum(dr, db))
    # left-left / top-top deltas fold into the same max
    dl2 = np.abs(Lleft - Lll)
    dt2 = np.abs(Ltop - Ltt)
    maxDelta = np.maximum(maxDelta, np.maximum(dl2, dt2))

    e_h *= (maxDelta <= local_contrast * dl + 1e-6)
    e_v *= (maxDelta <= local_contrast * dt + 1e-6)
    return np.stack([e_h, e_v], axis=-1)


def _area(area, dist_x, dist_y, e1, e2, offset):
    """Sample AreaTex the way SMAAArea does.

    x = 16*round(4*e1) + sqrt(dist to one end)
    y = 16*round(4*e2) + sqrt(dist to the other end)
    and `offset` picks one of the seven 80-row sub-tables
    (0,1 horizontal left/right; 2,3 vertical up/down; 4..6 diagonals).
    """
    x = (AREATEX_MAX_DISTANCE * np.round(4.0 * e1) + dist_x).astype(np.int32)
    y = (AREATEX_MAX_DISTANCE * np.round(4.0 * e2) + dist_y).astype(np.int32)
    x = np.clip(x, 0, AREATEX_W - 1)
    y = np.clip(y + int(offset * (AREATEX_H // 7)), 0, AREATEX_H - 1)
    return area[y, x, 0], area[y, x, 1]


def _grid(h, w):
    return np.meshgrid(np.arange(h), np.arange(w), indexing="ij")


def _gather(a, ys, xs, fill=0.0):
    """Sample a[h, w] at per-pixel coordinates, with `fill` outside."""
    h, w = a.shape[:2]
    ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    out = np.full((h, w) + a.shape[2:], fill, a.dtype)
    out[ok] = a[ys[ok], xs[ok]]
    return out


def smaa(rgb, area, threshold=0.1, max_steps=8):
    e = smaa_edges(rgb, threshold)
    eh, ev = e[..., 0], e[..., 1]
    h, w = eh.shape
    iy, ix = _grid(h, w)
    weights = np.zeros((h, w, 4), np.float32)

    def run_len(m, axis):
        """Distance to the end of the contiguous run of m through each pixel."""
        neg, pos = np.zeros_like(m, np.int32), np.zeros_like(m, np.int32)
        for k in range(1, max_steps + 1):
            if axis == 1:
                back = _gather(m, iy, ix - k) > 0
                fwd = _gather(m, iy, ix + k) > 0
            else:
                back = _gather(m, iy - k, ix) > 0
                fwd = _gather(m, iy + k, ix) > 0
            neg = np.where((neg == k - 1) & back, k, neg)
            pos = np.where((pos == k - 1) & fwd, k, pos)
        return neg, pos

    # horizontal search is driven by the vertical flag (e.g), along x
    ln, rn = run_len(ev, 1)
    act = ev > 0
    e1 = (_gather(ev, iy, ix - (ln + 1)) > 0).astype(np.float32)
    e2 = (_gather(ev, iy, ix + (rn + 1)) > 0).astype(np.float32)
    sl, sr = np.sqrt(ln.astype(np.float32)), np.sqrt(rn.astype(np.float32))
    wl, _ = _area(area, sl, sr, e1, e2, 0)
    wr, _ = _area(area, sl, sr, e1, e2, 1)
    weights[..., 0] = np.where(act, wl, 0.0)
    weights[..., 1] = np.where(act, wr, 0.0)

    # vertical search is driven by the horizontal flag (e.r), along y
    un, dn = run_len(eh, 0)
    actv = eh > 0
    e1v = (_gather(eh, iy - (un + 1), ix) > 0).astype(np.float32)
    e2v = (_gather(eh, iy + (dn + 1), ix) > 0).astype(np.float32)
    su, sd = np.sqrt(un.astype(np.float32)), np.sqrt(dn.astype(np.float32))
    wt, _ = _area(area, su, sd, e1v, e2v, 2)
    wb, _ = _area(area, su, sd, e1v, e2v, 3)
    weights[..., 2] = np.where(actv, wt, 0.0)
    weights[..., 3] = np.where(actv, wb, 0.0)

    # neighbourhood blending: average the 2x2 block of each weight
    def avg2x2(a):
        return 0.25 * (a + _gather(a, iy, ix - 1) + _gather(a, iy + 1, ix)
                       + _gather(a, iy + 1, ix - 1))

    left, top = avg2x2(weights[..., 0]), avg2x2(weights[..., 2])
    out = rgb.copy()
    hb = (left > 0) & (top <= left) & (left < 1)
    out = np.where(hb[..., None], (1 - left[..., None]) * rgb
                   + left[..., None] * _gather(rgb, iy, ix - 1), out)
    vb = (top > 0) & (left < top) & (top < 1)
    out = np.where(vb[..., None], (1 - top[..., None]) * rgb
                   + top[..., None] * _gather(rgb, iy - 1, ix), out)
    return np.clip(out, 0.0, 1.0)


def load_dds(path):
    """Decode an uncompressed RGBA8/R8 DDS (enough for the SMAA LUTs)."""
    data = open(path, "rb").read()
    if data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    h = int.from_bytes(data[12:16], "little")
    w = int.from_bytes(data[16:20], "little")
    off = 128
    ch = 4 if len(data) - off >= w * h * 4 else 1
    if ch == 4:
        return (np.frombuffer(data[off:off + w * h * 4], np.uint8)
                .reshape(h, w, 4).astype(np.float32) / 255.0)
    return (np.frombuffer(data[off:off + w * h], np.uint8)
            .reshape(h, w, 1).astype(np.float32) / 255.0)
