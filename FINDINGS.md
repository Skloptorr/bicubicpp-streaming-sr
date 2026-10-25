# Findings

Four measured results that contradicted the assumptions this project started
with, plus a methodology note. All numbers below were produced on a single
RTX 4070 SUPER; the scripts that produced them are in `eval/`.

The model itself is the least interesting artefact here. These are the part
worth reading.

---

## 1. Perceptual and gradient losses make the output *softer*, not sharper

The usual reason to add a perceptual loss is "MSE is too smooth, VGG features
will restore texture". That did not happen.

Training runs held identical except for the loss. Metric: high-frequency energy
of the output relative to ground truth, as a percentage. Higher = closer to the
sharpness of the real image.

| Loss | HF energy / GT |
|---|---|
| MSE | **85%** |
| L1 only | 76% |
| VGG perceptual (2 taps, every 2 steps) | 78% |
| Gradient, weight 0.10 | 76% |
| Gradient, weight 0.50 | 74% |

MSE was the **sharpest** of the five. Both perceptual and gradient terms pulled
high-frequency energy down, and a stronger gradient weight pulled it down
further — a monotone relationship in the wrong direction.

The likely reason: a perceptual loss treats "plausible to VGG" as the target, and
VGG features are comparatively insensitive to the exact phase of fine detail.
Optimising them rewards a clean, confident mid-frequency structure over
uncertain high-frequency content. Gradient losses do the same thing more
directly by suppressing the high-frequency residual they are penalising.

**Practical conclusion:** if you are chasing sharpness, do not reach for a
perceptual loss by reflex. Measure the frequency response first.

---

## 2. The CNN fabricates chroma that is not in the input

The clearest structural failure found. Feed a **pure greyscale** image — a text
screenshot, where R = G = B exactly everywhere.

| Upscaler | Chroma magnitude (0–255 scale) |
|---|---|
| Bicubic | **0.0 exactly** |
| Bicubic++ style CNN (1.5×, c48) | **64.7** (Cb) / **50.4** (Cr) |

Zero input chroma becomes roughly **25% of full saturation** on output. The
network has no information to justify any colour; it is inventing it from luma
structure, presumably because the training pairs contained coloured content at
similar luminance patterns.

On real game captures the effect is weaker but still systematic. Comparing
high-frequency energy against the source:

| Plane | HF energy ratio (output / bicubic) |
|---|---|
| Y | 1.14× |
| Cb | **1.69×** |
| Cr | **1.49×** |
| worst case observed | **5.43×** |

Luma gains are modest and well-behaved. Chroma gains are 1.5–5× larger, which is
the opposite of what a de-artefacting model should do — compression artefacts in
4:2:0 are *denser* in luma than in chroma.

This is why the recommendation in the main docs is to run the network on the
**luma plane only** and leave chroma to a conventional resampler: it makes the
failure mode structurally impossible rather than something you hope the training
data suppresses.

---

## 3. NNEDI3 is smoother than bicubic, not sharper

NNEDI3 has a reputation for edge-directed sharpness. Measured as high-frequency
energy on the same content:

| Upscaler | HF energy vs bicubic |
|---|---|
| NNEDI3 (znedi3, nns=256, nsize 8×6, qual=2) | **0.891×** |

It is *lower*. NNEDI3 is smooth and edge-directed; it avoids jaggies but does not
add detail. Meanwhile a trained CNN at the same scale produced the
highest-frequency output of the three while also being the least faithful.

So "cleaner" and "sharper" are not the same axis, and the common ranking that
puts NNEDI3 and CNN on the same "better than bicubic" side hides that they are
better in opposite directions:

```
faithfulness / absence of artefacts:  NNEDI3  >  bicubic  >  CNN
high-frequency energy:                CNN     >  bicubic  >  NNEDI3
```

Which one you want depends entirely on whether the viewer is judging detail or
artefacts.

---

## 4. Bicubic captures almost the whole gain; Lanczos adds little

Once both images are filtered to what the viewer can actually resolve (see §5),
the scaling kernel choice is worth much less than the attention it gets.

Upscale quality at 1440p → 4K, acuity-band PSNR, higher is better:

| Upscaler | 24" 4K @ 60 cm | 85" 4K @ 2.5 m |
|---|---|---|
| Bilinear | 49.75 dB | 50.52 dB |
| Bicubic (Catmull-Rom) | **50.71 dB** | **51.55 dB** |
| Lanczos 3 | 50.93 dB | 51.80 dB |
| Trained CNN (this project, 1.5×, c48) | 47.98 dB | 48.51 dB |

- Bilinear → bicubic: **+1.00 dB**
- Bicubic → Lanczos 3: **+0.22 dB**

Bicubic takes the overwhelming majority of the available improvement. Lanczos is
measurably better but by a fifth as much, at 36 texture fetches per plane versus
16 — a significantly worse trade.

The trained model *loses* 2.7–3.0 dB here. That is not a training bug: clean
bicubic-downscaled 4K ground truth is not the distribution the model was tuned
for, and the model was optimising for perceived cleanliness rather than PSNR.

### Where the model does win

Model gain over bicubic versus how much damage the input already carries from
HEVC encoding:

| Input condition | Gain vs bicubic |
|---|---|
| No codec at all (clean downsample) | **−0.69 dB** |
| CRF 34 | +0.34 dB |
| CRF 30 | +0.59 dB |
| CRF 26 | +0.92 dB |
| CRF 22 | **+1.20 dB** |
| CRF 18 | **+1.47 dB** |

The sign flips. On a clean downsample the network is actively harmful; on
realistically encoded input it helps, and it helps *more* as the input gets
cleaner within the encoded range. The minimum it needs is *some* real compression
artefact to work against.

Per plane at CRF 22: **Y +2.09 dB**, Cb +0.56 dB, Cr +0.49 dB. Luma carries
essentially the entire benefit, which is consistent with §2.

---

## 5. Methodology: score at the viewer's acuity limit, not at 1:1 pixels

Most SR comparisons score every pixel equally. On a 4K panel viewed from a sofa,
a large fraction of those pixels cannot be resolved, so the comparison is partly
measuring detail nobody sees.

Pixel density in the fovea, in pixels per degree (PPD). Human acuity is roughly
30 cycles/degree ≈ **60 PPD**.

| Display | PPD | Fraction of acuity |
|---|---|---|
| 24" 1440p @ 60 cm | 53.6 | 89% |
| 24" 4K @ 60 cm (47.76° horizontal FOV) | 108 | 180% |
| 85" 1440p @ 2.5 m | 62.1 | **103%** |
| 85" 4K @ 2.5 m | 93.1 | **155%** |

Two consequences that shaped every number above:

1. **1440p on an 85" screen at 2.5 m already exceeds acuity.** Upscaling it to 4K
   adds detail that is invisible. Only 24% of the 1440p→4K band is inside what the
   eye resolves — the rest is wasted work.
2. **85" 4K at 2.5 m wastes 36% of the panel's resolution.** The correct
   comparison filters both candidate and reference to 93 PPD first.

So every PSNR number in this document was computed after applying a
**low-pass at the acuity limit to both images**. That is why the CNN's apparent
advantage shrinks and the kernel choice dominates: most of what the model adds
lives above the resolvable band.

Script: `eval/hf_analysis.py` and the acuity helpers it imports.

---

## 6. Cost model, for reference

Bicubic++ runs all three heavy convolutions at **half the LR resolution**, so the
cost is naturally priced per LR pixel:

```
MAC/LR-px = 6.75·c + 4.5·c² + 2.25·c·conv_out
```

For a 1.5× upscale, `PixelShuffle(3)` needs `conv_out = 27`; for 1×, 2×, 3× it is
12, 48, 108 respectively. The 16-model matrix in `weights/` sweeps scale × channel
count against this formula.

For comparison, the resampling kernels this project's model competes with cost,
per output pixel, counting both YUV 4:2:0 planes: bilinear **2** fetches, bicubic
**32**, Lanczos 3 **72**. No extra passes, no intermediate render targets.
