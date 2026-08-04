# Two-Stage Deformable-Convolutional Inverse Design of Nanophotonic Absorbers from Optical Spectra

Reference implementation for the proposed method in *"Two-Stage Deformable-Convolutional
Inverse Design of Nanophotonic Absorbers from Optical Spectra"* (Waseer, Jabbar, Ibrahim,
Khan; manuscript draft). Given an 80-point absorption spectrum, the model directly
generates the 64×64 top-layer geometry mask of a metal-insulator-metal (MIM) resonator
that would produce it — no iterative solver, no parameter-space search.

## Problem

A MIM absorber unit cell (100 nm gold back reflector, 200 nm Al₂O₃ spacer, 100 nm patterned
gold resonator, 3.2 × 3.2 μm periodic cell) has a resonant absorption spectrum that depends
nonlinearly on its resonator outline — arm length, gap width, symmetry, connectivity. The
forward map (geometry → spectrum) is cheap to query once simulated; the inverse map
(spectrum → geometry) is what this codebase learns directly, as image generation rather
than a low-dimensional parameter regression, so it can represent free-form resonator shapes.

Dataset: the open 10,000-sample FDTD-simulated MIM dataset from Yeung et al. (2020, *ACS
Photonics*). Each pair is an 800-sample absorption curve (reduced to 80 points, spanning
4–12 μm) and a 40×40 top-layer mask (resized to 64×64, normalized to [0, 1]).

## Method

**Generator** (`InverseModel` in `build_model.py`): the spectrum is projected by a linear
layer (80 → 2400) and reshaped to 150×4×4, then decoded by four spatial layers
(150→96→64→32→1 channels, 5×5 kernels, same padding) interleaved with nearest-neighbor
upsampling (×2, ×2, ×4) to reach 64×64, BatchNorm after the first two upsampling stages,
ReLU throughout. In the proposed configuration every spatial layer is a **deformable
convolution** (`DeformableConv2d`) rather than a plain `nn.Conv2d` — a learned per-location
offset lets the sampling grid adapt to resonator arms, gaps, and corners instead of being
fixed to a Cartesian grid.

**Discriminator** (`Discriminator` in `build_model.py`): a 5-layer patch discriminator
(1→32→64→32→16→32 channels, 5×5 same-padded, LeakyReLU(0.2), BatchNorm on layers 2–4,
sigmoid), also built from the same deformable-conv primitive. No spatial downsampling, so
every output location assesses a local receptive field with dense feedback over the image.

**Two-stage training** (`train_gan_model_abl` in `train_fn.py`):
1. **Supervised reconstruction** — the generator is trained alone against pixelwise MSE.
2. **LSGAN refinement** — initialized from the best stage-1 checkpoint, the discriminator
   is introduced and the generator is fine-tuned on reconstruction + least-squares
   adversarial loss. This staged schedule is what the paper's training-strategy ablation
   found necessary: an LSGAN trained from scratch converges to a substantially worse
   optimum than supervised pretraining followed by adversarial refinement.

The codebase also implements the paper's controlled operator ablation (only the conv
primitive changes: plain / deformable / involution / dynamic / ODConv, via `--conv_type`
in `config.py`) and training-strategy ablation, but this README focuses on reproducing the
**proposed** (deformable, two-stage) configuration only.

## Reproducing the proposed result

```bash
pip install -r requirements.txt
```

`Deform_convII.py` (providing `DeformableConv2d`) is included in this directory.

```bash
python train.py --conv_type deform --use_gan --gan_mode ls --two_stage \
    --recon_loss mse --use_pretrained --epochs 500 \
    --pretrained_generator_ckpt logs/log_dir_here/checkpoints/best.pth \
    --seed 42   # repeat with --seed 43, --seed 44 for the paper's 3-run mean±std
```

`--pretrained_generator_ckpt` should point at a checkpoint from stage 1 (a prior
`--conv_type deform` run *without* `--use_gan`) — this is the supervised checkpoint that
stage 2 initializes from. Each run creates `logs/log_<NUM>_<TIMESTAMP>/`
(`config.json`, `train.log`, checkpoints), and automatically invokes `evaluate.py` on the
fixed held-out test split at the end (see `python evaluate.py --log_dir ...` to re-run
standalone, and `--num_samples`/`--save_samples` for qualitative GT-vs-predicted figures).

## Results

Mean ± sample standard deviation over 3 independent runs (seeds 42/43/44), from the
manuscript, final two-stage deformable-convolution model vs. the plain-convolution
baseline:

| Metric | Proposed (DeformConv) | Plain-conv baseline |
|---|---|---|
| PSNR (dB) ↑ | **20.79 ± 0.31** | 18.63 ± 0.11 |
| SSIM ↑ | **0.8501 ± 0.0082** | 0.7670 ± 0.0071 |
| Image MSE ↓ | **0.01378 ± 0.00073** | 0.01982 ± 0.00081 |
| Dice ↑ | **0.9623 ± 0.0027** | 0.9489 ± 0.0013 |
| HD95 (px) ↓ | **1.883 ± 0.109** | 2.224 ± 0.215 |
| ASD (px) ↓ | **0.353 ± 0.024** | 0.553 ± 0.033 |

i.e. +2.16 dB PSNR, −30.5% MSE, −36.1% ASD relative to plain convolution, and the best
score among all five operators tested (plain / deformable / involution / dynamic / ODConv).
`evaluate.py` reports these plus IoU, boundary F-score, connected-component error, and
topology-validity rate (see `evaluate_on_test_set` in `train_fn.py`) — a superset of the
manuscript's boundary-aware metrics.

## Spectral (forward-surrogate) validation

The manuscript's stated highest-priority remaining validation (§8, "Limitations and
Remaining Validation") is passing each predicted geometry through an independently
validated forward model and checking spectral consistency — PSNR/RMSE/R², peak-wavelength
error, peak-amplitude error against the input spectrum. This codebase implements that:

- `forward_model/train.py` trains the image→spectrum forward surrogate (MobileNetV2
  backbone + MCSR refinement, optional Savitzky-Golay post-processing at inference) on the
  same dataset/split as the inverse model, so its held-out test set matches exactly.
- `evaluate_spectrum_roundtrip.py --log_dir <inverse_run> --forward_log_dir <forward_run>`
  runs the full cycle (spectrum → generated geometry → re-simulated spectrum), saves the
  spectrum arrays plus qualitative GT/Forward-Predicted/Roundtrip-Predicted figures, and
  reports the metrics above for both the raw and binarized predicted geometry.

## Repository layout

| File | Contents |
|---|---|
| `train.py` / `evaluate.py` | Train the inverse model; standalone/auto-invoked evaluation on the fixed test split. |
| `build_data.py` | `ImageSpectrumDataset`, forward (XPIE-augmentation) simulator, loader builders. |
| `build_model.py` | `InverseModel` (generator), `Discriminator`. |
| `conv_layers.py` / `Deform_convII.py` | The spatial operators (`DeformableConv2d` is the proposed one). |
| `train_fn.py` | Two-stage training loop, test-set evaluation (image + boundary metrics), qualitative sample dumps. |
| `config.py` | Argparse definition and CLI/preset/default merge logic. |
| `evaluate_spectrum_roundtrip.py` | Forward-surrogate spectral validation (see above). |
| `forward_model/` | Image→spectrum forward surrogate: its own `train.py`/`evaluate.py`/`model.py`. |
| `utils.py` | Seeding, PSNR/SSIM/segmentation metrics, run-dir management, file loggers, config save/load. |

## Citation

Waseer, W., Jabbar, M.S., Ibrahim, M.S., Khan, S. *Two-Stage Deformable-Convolutional
Inverse Design of Nanophotonic Absorbers from Optical Spectra.* Manuscript in preparation.
