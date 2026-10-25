# Data provenance

The training dataset is **not distributed with this repository** and cannot be,
for the reasons in `../NOTICE`. This file records what it was made of so that the
published weights are not presented as something they are not.

464,868 patch pairs, 1,717 source images, 192 px patches, HEVC CRF 22.
LR is a Lanczos downscale followed by a HEVC yuv420p round-trip.

| Domain | Patches | Share | Source | Redistributable |
|---|---|---|---|---|
| `games` | 167,055 | 35.9% | Game captures | No — publisher copyright |
| `text` | 77,385 | 16.6% | Text / UI screenshots | Unknown |
| `raw` | 67,298 | 14.5% | Author's own camera RAW files | Author's own |
| `nature_ref` | 56,798 | 12.2% | Public nature video datasets | Dataset-specific |
| `uvg` | 52,424 | 11.3% | UVG test sequences | Research use only |
| `pnc` | 38,954 | 8.4% | Unrecorded | Unknown |
| `blender` | 4,954 | 1.1% | `build_foliage.py`, generated | Yes — author's own |

Only about **15.6 %** of the corpus (`raw` + `blender`) is unambiguously the
author's own work.

## Consequences

- Every checkpoint in `weights/` was trained on a mixture that includes
  third-party copyrighted material. That is disclosed here rather than hidden.
- Weights are not a copy of the training images, and the legal status of trained
  weights is unsettled in most jurisdictions. This repository does not claim the
  question is resolved, and its non-commercial licence is the conservative
  reading.
- If you need a model you can license freely, rebuild the dataset from
  `build_foliage.py` (100 % procedural, no external assets) plus your own
  photography, and retrain. That is the only path that does not inherit the
  problem.

## Two open questions, recorded honestly

1. The `pnc` domain (8.4%) has no recorded provenance. It should not be in a
   dataset you redistribute.
2. The `text` domain (16.6%) may contain third-party UI content. It was treated
   as self-captured but that was never verified image by image.
