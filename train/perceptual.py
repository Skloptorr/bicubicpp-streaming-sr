# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Skloptorr

r"""VGG16 perceptual loss without torchvision.

The env has no torchvision and no lpips, but the perceptual loss only needs the
VGG16 conv stack and its ImageNet weights. Both are available without the
package: the architecture is a fixed sequence of conv+ReLU+maxpool (easy to write
by hand) and `vgg16-397923af.pth` holds the matching `features.*` state dict.

Features are taken at relu1_2, relu2_2, relu3_3 and relu4_3, which is the usual
set for SR perceptual losses: shallow layers constrain edges and colour, deeper
layers constrain structure.

MSE training converges to the conditional mean, which is mathematically blurry --
that is why an MSE model scores well on PSNR yet looks soft. Adding this term
pushes the output toward the natural-image manifold instead, trading a little
PSNR for visibly sharper structure.
"""
import os
import torch
import torch.nn as nn

VGG_WEIGHTS = os.environ.get("VGG_WEIGHTS")  # None -> torchvision downloads it


def _make_layers():
    """VGG16 `features` as conv blocks with the ImageNet torchvision layout."""
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"]
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.append(nn.Conv2d(in_ch, v, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v
    return nn.Sequential(*layers)


class VGGPerceptual(nn.Module):
    """L1 distance between VGG16 features of prediction and target.

    Layer indices refer to positions in the sequential built by `_make_layers`:
    ReLUs sit at 1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29.
    relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22 in that list.
    """

    TAPS = (3, 8, 15, 22)

    def __init__(self, weights=VGG_WEIGHTS, device="cuda", scale=2, ntaps=4):
        super().__init__()
        self.TAPS = self.TAPS[:ntaps]
        # VGG at full 192x192 costs ~0.14 s/step on a 4070S; halving the input
        # divides that by four while keeping the structure the loss constrains.
        self.scale = scale
        vgg = _make_layers()
        sd = torch.load(weights, map_location="cpu")
        sd = {k[len("features."):]: v for k, v in sd.items() if k.startswith("features.")}
        missing, unexpected = vgg.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError("unexpected VGG keys: %s" % unexpected[:4])
        # the last maxpool has no weights, so a leftover key is expected and fine
        # channels_last keeps the conv kernels on their fast path on Ampere+;
        # TF32 is fine here because this is a loss term, not a prediction
        self.vgg = vgg[:max(self.TAPS) + 1].to(device, memory_format=torch.channels_last).eval()
        torch.backends.cudnn.allow_tf32 = True
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device)
                             .view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device)
                             .view(1, 3, 1, 1))
        self.device = device

    def forward(self, pred, target):
        if self.scale > 1:
            pred = nn.functional.avg_pool2d(pred, self.scale, self.scale)
            target = nn.functional.avg_pool2d(target, self.scale, self.scale)
        # VGG expects ImageNet-normalised input; both tensors are in [0,1]
        p = ((pred - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        t = ((target - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        loss = 0.0
        xp, xt = p, t
        last = 0
        for tap in self.TAPS:
            xp = self.vgg[last:tap + 1](xp)
            xt = self.vgg[last:tap + 1](xt)
            last = tap + 1
            loss = loss + nn.functional.l1_loss(xp, xt)
        return loss / len(self.TAPS)


class GradLoss(nn.Module):
    """L1 distance between image gradients.

    The VGG route failed here: it is computed on a downsampled input, so fine
    detail barely moves the pooled features while any added high frequency is
    penalised -- the incentive points the wrong way. This term compares first
    differences directly, so it rewards exactly the high-frequency energy that
    the HF/GT metric measures, and it costs almost nothing.
    """

    def forward(self, pred, target):
        loss = 0.0
        for d in (2, 3):
            loss = loss + nn.functional.l1_loss(
                torch.diff(pred, dim=d), torch.diff(target, dim=d))
        return loss / 2


class YLoss(nn.Module):
    """Luma-dominant loss.

    On pure black-and-white text the current model invents up to 25% chroma
    error, and on game captures it raises chroma high-frequency 1.5-1.7x while
    luma only rises 1.14x. The cause is that the loss treats R, G and B equally,
    so the network spends capacity re-synthesising chroma that 4:2:0 never
    carried. This weights luma 1.0 and chroma `cw`: enough to keep the shared
    trunk stable, not enough to ask it to invent colour.
    """

    def __init__(self, cw=0.1):
        super().__init__()
        self.cw = cw
        self.register_buffer("m", torch.tensor(
            [[0.2126, 0.7152, 0.0722],
             [-0.1146, -0.3854, 0.5000],
             [0.5000, -0.4542, -0.0458]], dtype=torch.float32))

    def forward(self, pred, target, perceptual_only=True):
        p = torch.einsum("nchw->nhwc", pred) @ self.m.T
        t = torch.einsum("nchw->nhwc", target) @ self.m.T
        ly = nn.functional.l1_loss(p[..., 0], t[..., 0])
        lc = (nn.functional.l1_loss(p[..., 1], t[..., 1])
              + nn.functional.l1_loss(p[..., 2], t[..., 2]))
        return ly + self.cw * lc


def build(args, device):
    """Return (criterion, name) for the requested --loss setting."""
    if args.loss == "mse":
        return nn.MSELoss(), "mse"
    if args.loss == "l1":
        return nn.L1Loss(), "l1"
    if args.loss == "y":
        cw = getattr(args, "y_chroma_weight", 0.1)
        return YLoss(cw).to(device), "l1_y+%.2g*l1_c" % cw
    if args.loss == "grad":
        gl, gw = GradLoss(), args.grad_weight

        class WithGrad(nn.Module):
            def forward(self, pred, target, perceptual_only=True):
                return nn.functional.l1_loss(pred, target) + gw * gl(pred, target)

        return WithGrad(), "l1+%.3g*grad" % gw
    vgg = VGGPerceptual(device=device, scale=args.perc_scale, ntaps=args.perc_ntaps)
    w = args.perc_weight

    class Combined(nn.Module):
        """L1 + perceptual.

        `perceptual_only` lets the trainer skip the VGG term on most steps. The
        VGG pass is the dominant cost and applying it every other step shifts the
        objective far less than the 2x speedup is worth.
        """

        def forward(self, pred, target, perceptual_only=True):
            pix = nn.functional.l1_loss(pred, target)
            if perceptual_only:
                return pix + w * vgg(pred, target)
            return pix

    return Combined(), "l1+%.3g*vgg(scale=%d,taps=%d,every=%d)" % (
        w, args.perc_scale, args.perc_ntaps, args.perc_every)
