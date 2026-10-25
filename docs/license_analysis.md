# Licence analysis

The short version, and the reasoning behind the split between `LICENSE`
(Apache-2.0, code) and `LICENSE-WEIGHTS` (non-commercial, weights).

## Does the Bicubic++ licence reach this code?

**No.** The upstream repository
`aselsan-research-imaging-team/bicubic-plusplus` is CC BY-NC-SA 4.0, covering
both its code and `pretrained/bicubic_pp_x3.pth`. But copyright protects
*expression*, not ideas or architectures. This repository contains:

- an independent implementation written from the paper, and
- no upstream source file, no copied training loop, no copied helper classes.

The whole model definition is 14 lines in `train/train_multi.py`:

```python
self.conv0 = nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False)
self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
self.conv_out = nn.Conv2d(c, cout, 3, padding=1, bias=False)
self.ps = nn.PixelShuffle(r)
self.act = nn.LeakyReLU(0.1, inplace=True)
```

That is a description of a paper's figure in PyTorch, not a reproduction of
someone's source. There is nothing to license-in.

**Attribution is still expected** — citing the paper is academic practice, and
`NOTICE` does it — but that is a norm, not an obligation inherited from
CC BY-NC-SA.

## Was any upstream weight used as a starting point?

**No.** `Net(c, sr)` is constructed fresh and trained from random initialisation.
`load_state_dict` appears only behind the `--resume` flag.

`bicubic_pp_x3.pth` is referenced by exactly two scripts, both for comparison
rather than training:

- `eval/verify_paper.py` — loads it into our `Net` to check the architecture
  reproduces the paper's numbers
- `extract15.py` (not included here) — same purpose

**It is not in this repository, and must not be redistributed.** If you want to
run those comparisons, download it from upstream yourself and accept their
licence.

## Then why are the weights non-commercial?

Only because of the training data, and only as a conservative reading:

| Source | Term | Consequence |
| --- | --- | --- |
| DIV2K | "academic research purpose only", no redistribution granted | applies to the older checkpoints |
| UVG | research use | applies to the newer ones |
| Game captures | publisher copyright | applies to 35.9% of the newer dataset |

None of these say anything about trained weights. They constrain *use of the
data*. Whether a model trained on research-only images inherits a restriction is
an open question, and the safe answer for a public release is "assume it might".

If you want to remove the ambiguity entirely, retrain from `data/build_foliage.py`
plus your own photography. Nothing in the code changes.

## What this means for reuse

| Asset | Licence | Commercial use |
| --- | --- | --- |
| Code (`train/`, `export/`, `eval/`, `data/`) | Apache-2.0 | Yes |
| The 14-line architecture | Apache-2.0 | Yes |
| Weights in `weights/` | non-commercial (see LICENSE-WEIGHTS) | Only after clearing the data terms |
| Upstream `bicubic_pp_x3.pth` | CC BY-NC-SA 4.0 | No, and not included |

A clean split: the *engineering* is permissively licensed, the *artefacts trained
on research data* are not.
