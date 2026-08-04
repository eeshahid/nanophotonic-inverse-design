"""
evaluate_spectrum_roundtrip.py
================================
Spectrum -> image -> spectrum cycle-consistency check for a trained run: for each test-set
item, predict an image from its GT spectrum with the trained generator, then re-simulate a
spectrum from that predicted image with a forward (image->spectrum) model -- once from the
raw [0,1] prediction and once from a binarized version. Saves the GT spectrum + both
predicted spectrum arrays to <log_dir>/spectrum_roundtrip/, and reports, for both the raw
and binarized variants: spectral MSE / RMSE / R^2 / PSNR (pooled over all test samples and
spectral points), plus peak-wavelength error (um) and peak-amplitude error (per-sample,
then averaged -- see _peak_metrics).

Forward model: by default this is the legacy PretrainedModelToSpectrum used elsewhere in
the pipeline for XPIE augmentation (--forward_model_path). Pass --forward_log_dir to use a
run trained by forward_model/train.py (MobileNetV2+MCSR) instead -- its intended deployment
mode applies Savitzky-Golay smoothing to its predictions as post-processing (optional here
via --apply_savgol/--no-apply_savgol, on by default when --forward_log_dir is set; ignored
for the legacy forward model).

The saved .npy arrays are meant to be reused for any additional metrics later without
re-running the model, e.g.:
    gt = np.load("<log_dir>/spectrum_roundtrip/spectrum_gt.npy")
    pred_raw = np.load("<log_dir>/spectrum_roundtrip/spectrum_pred_raw.npy")

Usage
-----
python evaluate_spectrum_roundtrip.py --log_dir logs/log_003_20260712_153045

# Using a forward_model/train.py run, with Savitzky-Golay post-processing (window=11, polyorder=2):
python evaluate_spectrum_roundtrip.py --log_dir logs/log_003_20260712_153045 \\
    --forward_log_dir forward_model/logs/log_000_20260803_155339
"""

# Terminal Runs:
# python evaluate_spectrum_roundtrip.py --log_dir /home/coe/shahid/mxene/inverse/src/logs/log_002_20260721_164839 --forward_log_dir /home/coe/shahid/mxene/inverse/src/forward_model/logs/log_003_20260803_162542/
# python evaluate_spectrum_roundtrip.py --log_dir /home/coe/shahid/mxene/inverse/src/logs/log_040_20260730_004725/ --forward_log_dir /home/coe/shahid/mxene/inverse/src/forward_model/logs/log_004_20260803_162553/

import argparse
import os

import numpy as np
import torch

from build_data import build_test_loader, load_forward_model
from build_model import build_models
from config import PRESETS, add_shared_args
from forward_model.model import SpectrumForwardModel
from forward_model.train_fn import savgol_smooth
from train_fn import SPECTRUM_WAVELENGTHS_UM, save_spectrum_roundtrip_samples
from utils import build_logger, load_config, save_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spectrum -> image -> spectrum cycle-consistency check for a trained run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_dir", type=str, required=True,
                         help="Run directory produced by train.py (contains config.json + checkpoints/).")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best",
                         help="Which checkpoint in <log_dir>/checkpoints/ to evaluate.")
    parser.add_argument("--binarize_threshold", type=float, default=0.5,
                         help="Threshold used to binarize the predicted image before its "
                              "second (binarized) pass through the forward model.")
    parser.add_argument("--output_dir_name", type=str, default="spectrum_roundtrip",
                         help="Subdirectory of --log_dir to save spectra/metrics into.")

    samples = parser.add_argument_group("qualitative sample dump")
    samples.add_argument("--save_samples", action=argparse.BooleanOptionalAction, default=True,
                          help="Save --num_samples GT-vs-roundtrip-predicted spectrum figures "
                               "(individual + combined + grid) to <output_dir_name>/samples/.")
    samples.add_argument("--num_samples", type=int, default=8,
                          help="Number of test-set samples to save when --save_samples is set.")

    fwd = parser.add_argument_group("forward model (image -> spectrum)")
    fwd.add_argument("--forward_log_dir", type=str, default=None,
                      help="Use a forward_model/train.py run (MobileNetV2+MCSR) at this "
                           "log_dir instead of the legacy --forward_model_path checkpoint.")
    fwd.add_argument("--forward_checkpoint", choices=["best", "latest"], default="best",
                      help="Which checkpoint under --forward_log_dir/checkpoints/ to use.")
    fwd.add_argument("--apply_savgol", action=argparse.BooleanOptionalAction, default=True,
                      help="Apply Savitzky-Golay smoothing to --forward_log_dir's predictions "
                           "(its intended post-processing-at-inference deployment mode). "
                           "Ignored when using the legacy --forward_model_path.")
    fwd.add_argument("--savgol_window", type=int, default=11)
    fwd.add_argument("--savgol_polyorder", type=int, default=2)

    # Re-expose the same shared flags as evaluate.py so users CAN override e.g. a moved
    # forward_model_path/data path; anything left as None falls back to config.json.
    parser = add_shared_args(parser)
    return parser


def _pooled_regression_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Spectral MSE / RMSE / R^2 / PSNR pooled over every (sample, spectral point) --
    gt, pred: (N, L)."""
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    mse = float(np.mean((gt - pred) ** 2))
    rmse = float(np.sqrt(mse))
    data_range = float(gt.max() - gt.min())
    psnr = float(10 * np.log10((data_range ** 2) / mse)) if mse > 0 else float("inf")
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "r2": r2, "psnr": psnr, "data_range": data_range}


def _peak_metrics(gt: np.ndarray, pred: np.ndarray, wavelengths: np.ndarray) -> dict:
    """Per-sample peak-wavelength error (um, |argmax wavelength difference|) and
    peak-amplitude error (|peak value difference|), averaged (MAE) over all N samples.
    gt, pred: (N, L); wavelengths: (L,)."""
    gt_peak_idx = np.argmax(gt, axis=1)
    pred_peak_idx = np.argmax(pred, axis=1)
    peak_wavelength_error = np.abs(wavelengths[pred_peak_idx] - wavelengths[gt_peak_idx])

    rows = np.arange(gt.shape[0])
    gt_peak_amp = gt[rows, gt_peak_idx]
    pred_peak_amp = pred[rows, pred_peak_idx]
    peak_amplitude_error = np.abs(pred_peak_amp - gt_peak_amp)

    return {
        "peak_wavelength_error_um": float(peak_wavelength_error.mean()),
        "peak_amplitude_error": float(peak_amplitude_error.mean()),
    }


def _spectrum_metrics(gt: np.ndarray, pred: np.ndarray, wavelengths: np.ndarray) -> dict:
    metrics = _pooled_regression_metrics(gt, pred)
    metrics.update(_peak_metrics(gt, pred, wavelengths))
    return metrics


def _load_forward_model(cfg: dict, args, device, logger) -> tuple:
    """Returns (forward_model, forward_info) where forward_info records what was used (for
    the saved result json). Legacy mode (--forward_log_dir unset): the PretrainedModelToSpectrum
    checkpoint used elsewhere in the pipeline for XPIE augmentation, used verbatim. New mode:
    a forward_model/train.py run (MobileNetV2+MCSR), with Savitzky-Golay applied to its
    predictions as post-processing if --apply_savgol (its intended deployment mode)."""
    if args.forward_log_dir is None:
        forward_model = load_forward_model(cfg["forward_model_path"], device)
        forward_model.eval()
        logger.info(f"Forward model: legacy PretrainedModelToSpectrum ({cfg['forward_model_path']})")
        return forward_model, {
            "mode": "legacy_forward_model_path", "forward_model_path": cfg["forward_model_path"],
            "apply_savgol": False, "savgol_window": None, "savgol_polyorder": None,
        }

    forward_cfg = load_config(os.path.join(args.forward_log_dir, "config.json"))
    ckpt_path = os.path.join(args.forward_log_dir, "checkpoints", f"{args.forward_checkpoint}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Forward model checkpoint not found: {ckpt_path}")

    forward_model = SpectrumForwardModel(
        output_dim=forward_cfg["output_dim"], pretrained=False, hidden_dim=forward_cfg["hidden_dim"],
        dropout=forward_cfg["dropout"], use_mcsr=forward_cfg["use_mcsr"], use_savgol=False,
    ).to(device)
    forward_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    forward_model.eval()
    logger.info(f"Forward model: forward_model/train.py run at {args.forward_log_dir} "
                f"(checkpoint={args.forward_checkpoint}, apply_savgol={args.apply_savgol})")
    return forward_model, {
        "mode": "forward_log_dir", "forward_log_dir": args.forward_log_dir,
        "forward_checkpoint": args.forward_checkpoint, "apply_savgol": bool(args.apply_savgol),
        "savgol_window": args.savgol_window if args.apply_savgol else None,
        "savgol_polyorder": args.savgol_polyorder if args.apply_savgol else None,
    }


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    saved_cfg = load_config(os.path.join(args.log_dir, "config.json"))
    cfg = dict(saved_cfg)
    # Back-compat with pre-conv_type run dirs (see config.py / evaluate.py).
    if "conv_type" not in cfg:
        cfg["conv_type"] = "deform" if cfg.get("use_deform") else "plain"
    if args.preset is not None:
        cfg.update(PRESETS[args.preset])

    for field in ("conv_type", "batch_size", "device", "forward_model_path",
                  "data_imag_path", "spectrum_file"):
        val = getattr(args, field, None)
        if val is not None:
            cfg[field] = val

    out_dir = os.path.join(args.log_dir, args.output_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = build_logger("spectrum_roundtrip", os.path.join(out_dir, "spectrum_roundtrip.log"))
    logger.info(f"Evaluating spectrum round-trip for run: {args.log_dir}")

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    ckpt_path = os.path.join(args.log_dir, "checkpoints", f"{args.checkpoint}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- generator ----
    model, _ = build_models(cfg["conv_type"], use_gan=False, device=device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["generator_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch={state.get('epoch')}, loss={state.get('loss')})")

    # ---- forward (image -> spectrum) model, frozen ----
    forward_model, forward_info = _load_forward_model(cfg, args, device, logger)

    # ---- data (fixed test split) ----
    test_loader = build_test_loader(
        batch_size=cfg["batch_size"], data_imag_path=cfg["data_imag_path"],
        spectrum_file=cfg["spectrum_file"],
    )

    # ---- spectrum -> image -> spectrum round trip (+ forward-model-only baseline) ----
    threshold = args.binarize_threshold
    gt_spectra, gt_images, inverse_pred_images = [], [], []
    pred_spectra_raw, pred_spectra_bin, pred_spectra_forward = [], [], []
    with torch.no_grad():
        for spectra, images in test_loader:
            spectra = spectra.to(device)
            images = images.to(device)
            pred_image_raw = model(spectra).clamp(0, 1)
            pred_image_bin = (pred_image_raw > threshold).float()

            pred_spectrum_raw = forward_model(pred_image_raw)
            pred_spectrum_bin = forward_model(pred_image_bin)
            pred_spectrum_forward = forward_model(images)  # forward-model-only: GT image -> spectrum

            gt_spectra.append(spectra.cpu().numpy())
            gt_images.append(images.cpu().numpy())
            inverse_pred_images.append(pred_image_raw.cpu().numpy())
            pred_spectra_raw.append(pred_spectrum_raw.cpu().numpy())
            pred_spectra_bin.append(pred_spectrum_bin.cpu().numpy())
            pred_spectra_forward.append(pred_spectrum_forward.cpu().numpy())

    gt_spectra = np.concatenate(gt_spectra, axis=0)
    gt_images = np.concatenate(gt_images, axis=0)
    inverse_pred_images = np.concatenate(inverse_pred_images, axis=0)
    pred_spectra_raw = np.concatenate(pred_spectra_raw, axis=0)
    pred_spectra_bin = np.concatenate(pred_spectra_bin, axis=0)
    pred_spectra_forward = np.concatenate(pred_spectra_forward, axis=0)

    # ---- optional Savitzky-Golay post-processing of the forward model's predictions ----
    if forward_info["apply_savgol"]:
        savgol_kwargs = dict(window_length=forward_info["savgol_window"], polyorder=forward_info["savgol_polyorder"])
        pred_spectra_raw = savgol_smooth(pred_spectra_raw, **savgol_kwargs)
        pred_spectra_bin = savgol_smooth(pred_spectra_bin, **savgol_kwargs)
        pred_spectra_forward = savgol_smooth(pred_spectra_forward, **savgol_kwargs)
        logger.info(f"Applied Savitzky-Golay (window={forward_info['savgol_window']}, "
                    f"polyorder={forward_info['savgol_polyorder']}) to forward model predictions")

    np.save(os.path.join(out_dir, "spectrum_gt.npy"), gt_spectra)
    np.save(os.path.join(out_dir, "spectrum_pred_raw.npy"), pred_spectra_raw)
    np.save(os.path.join(out_dir, "spectrum_pred_bin.npy"), pred_spectra_bin)
    np.save(os.path.join(out_dir, "spectrum_pred_forward.npy"), pred_spectra_forward)
    logger.info(f"Saved {gt_spectra.shape[0]} GT/predicted spectra "
                f"({gt_spectra.shape[1]} points each) to {out_dir}")

    # ---- qualitative samples (GT/inverse-pred images + Ground-Truth/Forward Predicted/Roundtrip Predicted spectra) ----
    if args.save_samples:
        save_spectrum_roundtrip_samples(
            gt_spectra, pred_spectra_raw, pred_spectra_bin, pred_spectra_forward,
            gt_images, inverse_pred_images, os.path.join(out_dir, "samples"),
            num_samples=args.num_samples, wavelengths=SPECTRUM_WAVELENGTHS_UM, logger=logger,
        )

    # ---- metrics ----
    metrics = {
        "forward_model": forward_info,
        "raw": _spectrum_metrics(gt_spectra, pred_spectra_raw, SPECTRUM_WAVELENGTHS_UM),
        "binarized": _spectrum_metrics(gt_spectra, pred_spectra_bin, SPECTRUM_WAVELENGTHS_UM),
        "forward_only": _spectrum_metrics(gt_spectra, pred_spectra_forward, SPECTRUM_WAVELENGTHS_UM),
    }
    for tag in ("raw", "binarized", "forward_only"):
        m = metrics[tag]
        logger.info(
            f"[{tag}] MSE={m['mse']:.6f}  RMSE={m['rmse']:.6f}  R^2={m['r2']:.4f}  "
            f"PSNR={m['psnr']:.4f} dB  peak_wavelength_err={m['peak_wavelength_error_um']:.4f} um  "
            f"peak_amplitude_err={m['peak_amplitude_error']:.4f}"
        )

    result_path = os.path.join(out_dir, "spectrum_roundtrip_result.json")
    save_config(metrics, result_path)
    logger.info(f"Saved metrics to {result_path}")

    return metrics


if __name__ == "__main__":
    main()
