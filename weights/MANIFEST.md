# Weights

`*_best.pth` and `*_last.pth` for the scale x channel sweep.

Naming: `bicubic_<scale>x_c<channels>[_seed<N>]_{best,last}.pth`

Bicubic++ runs its three heavy convolutions at half the LR resolution, so cost is
priced per LR pixel:

    MAC/LR-px = 6.75*c + 4.5*c^2 + 2.25*c*conv_out

`conv_out` is `(2*sr)^2 * 3`, i.e. 12 / 27 / 48 / 108 for 1x / 1.5x / 2x / 3x.

Training data provenance is in `../docs/data_provenance.md`.
These checkpoints are covered by the repository licence (CC BY-NC-SA 4.0).
