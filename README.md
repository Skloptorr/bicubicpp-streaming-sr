# bicubicpp-streaming-sr

Training, evaluation and shader-export tooling for a Bicubic++-style super
resolution model aimed at **video streaming upscaling** (1440p → 4K), plus the
measurements that came out of building it.

**Read [`FINDINGS.md`](FINDINGS.md) first.** The model is the least interesting
part of this repository; the four counter-intuitive results are the point.

---

## What is in here

| Directory | Contents |
|---|---|
| `train/` | `train_multi.py` (multi-scale, 5 loss modes, grey augmentation, per-plane weights), `perceptual.py` (VGG16 perceptual loss with manual weight loading) |
| `export/` | `gen_effect_v2.py` — turns a checkpoint into a multi-pass Magpie HLSL effect for **any** channel count; `export_effect.py`, `sr_pipeline.py`, `run_model.py` |
| `eval/` | `validate_effect.py` (per-pass `fxc` compilation check), `hf_analysis.py` (frequency response), `gain_vs_crf.py` (gain vs input damage), `bench.py`, `eval_vmaf.py`, `eval_hevc.py` |
| `data/` | `build_foliage.py` (100 % procedural Blender scene, no external assets), `make_split.py`, `prep_patches*.py`, `gen_pairs.py`, render helpers |
| `weights/` | `*_best.pth` / `*_last.pth` for the scale × channel matrix |
| `docs/` | Licence analysis and data provenance |

## Quick start

```bash
pip install -r requirements.txt
```

Paths are taken from environment variables, with the repository layout as the
default. To point at your own data:

```bash
export SR_CACHE=./cache/d15           # patch cache with split.json
export SR_WEIGHTS=./weights
export FFMPEG=ffmpeg                  # needs libx265 + libvmaf
export VGG_WEIGHTS=/path/vgg16.pth    # unset -> torchvision downloads it
```

Then:

```bash
python train/train_multi.py --help
python eval/validate_effect.py --help
python eval/gain_vs_crf.py --help
```

`export/gen_effect_v2.py` writes an effect file for Magpie; point
`MAGPIE_EFFECTS` at your `Magpie/effects` directory to have it installed
directly.

## The model in one table

Bicubic++ runs its three heavy convolutions at half the LR resolution, so cost is
expressed per LR pixel:

```
MAC/LR-px = 6.75·c + 4.5·c² + 2.25·c·conv_out
```

`conv_out` is `(2·sr)² · 3`, i.e. 12 / 27 / 48 / 108 for 1× / 1.5× / 2× / 3×.
Channel counts 24–48 are swept in `weights/`. See `FINDINGS.md` §6.

## Using it for streaming

The measured conclusion is that the useful operating point is **not** an
aggressive upscale of clean content:

- On a clean bicubic downsample the model loses ≈0.7 dB against plain bicubic.
- On HEVC-encoded input at CRF 18–26 it gains 0.9–1.5 dB, entirely in luma.
- Chroma should not go through the network at all — see `FINDINGS.md` §2.
- If all you want is a better scaler, bicubic already captures ~1.0 dB of the
  ~1.2 dB available; Lanczos adds only 0.22 dB more for twice the fetch count.

## Licence

**Code: Apache-2.0. Weights: non-commercial.**

Copyright © 2026 Skloptorr. If you redistribute this code or anything derived from
it, keep the copyright notice and `NOTICE` intact (Apache-2.0 §4).

The split is deliberate. The network is an independent implementation of the
Bicubic++ paper — 14 lines of `nn.Conv2d`/`nn.PixelShuffle` restating a figure
from a paper. No upstream source and no upstream weights are included or derived
from, so upstream's CC BY-NC-SA 4.0 does not reach this code. Citing the paper is
academic practice, not a licence term.

The weights are separate because of the training data, not the architecture:
sources include DIV2K and UVG (research-only) and game captures. Whether trained
weights inherit restrictions from training data is unsettled, so
[`LICENSE-WEIGHTS`](LICENSE-WEIGHTS) takes the conservative reading. Rebuild the
dataset from `data/build_foliage.py` plus your own photography and the resulting
weights are yours to license freely.

Details and quotations: [`NOTICE`](NOTICE), [`docs/license_analysis.md`](docs/license_analysis.md).

## What is deliberately **not** here

- **No training data.** The dataset cannot be redistributed (see `NOTICE`).
  `data/` contains the scripts to rebuild it from sources you obtain yourself.
- **No VGG16 weights.** torchvision downloads them on first use; they are
  derived from ImageNet-1K, whose terms are non-commercial research.
- **No Magpie binary.** Magpie is GPL-3.0 and is only a runtime host for the
  exported effects.
- **No znedi3 binary.** GPL-2.0; invoked as an external tool by one evaluation
  script only.

## Reproducing the numbers

`eval/hf_analysis.py` produces §1–§3, `eval/gain_vs_crf.py` produces §4's
per-CRF table, and the acuity-band PSNR uses the low-pass described in
`FINDINGS.md` §5. The split is frozen in `split.json` (seed 1234, 34 validation
images, 238 validation patches) so runs are comparable — do not regenerate it
per run.
