"""
evaluate.py
============
Reproduces every notebook's final cell (section 7): load the best checkpoint,
score it on the fixed FDTD test split (PSNR / SSIM / MSE), print + log the
result, and append one row to the shared master CSV
(<results_dir>/<master_csv_name>, default results_abl_inv/ablation_summary_inv.csv).

Usage
-----
# Run standalone against a finished run -- only --log_dir is required; every other
# key arg (model flags, data paths, batch size, ...) is loaded from that run's
# saved config.json unless you explicitly override it on the CLI:
python evaluate.py --log_dir logs/log_003_20260712_153045

# Override e.g. which checkpoint to score, or point at a moved dataset:
python evaluate.py --log_dir logs/log_003_20260712_153045 --checkpoint latest \\
    --data_imag_path /new/path/Images --spectrum_file /new/path/Spectra.csv

This script is also invoked automatically by train.py at the end of a training run.
"""
import argparse
import csv
import math
import os

import torch

from build_data import build_test_loader
from build_model import build_models
from config import PRESETS, add_shared_args
from train_fn import evaluate_on_test_set, save_gt_vs_pred_samples
from utils import build_logger, load_config, save_config

# Columns for the shared master CSV. Rewritten in full on every append (see
# _append_master_csv) so adding a column here never desyncs older rows/headers.
CSV_COLUMNS = [
    "variant", "seed", "psnr", "ssim", "mse",
    "dsc", "hd95", "asd", "iou", "boundary_f", "cc_error", "topology_validity_rate",
    "params_M", "conv_type", "use_xpie", "use_gan", "gan_mode", "two_stage", "recon_loss",
    "log_dir", "checkpoint",
]


def _append_master_csv(master_csv: str, row: dict) -> None:
    """Appends `row` (keyed by CSV_COLUMNS) to the shared master CSV. Rewrites the whole
    file with the current CSV_COLUMNS schema each time, so rows written before a column
    was added (e.g. dsc/hd95/asd) just get a blank cell for it instead of desyncing the
    header from later rows."""
    rows = []
    if os.path.exists(master_csv):
        with open(master_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the fixed FDTD test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_dir", type=str, required=True,
                         help="Run directory produced by train.py (contains config.json + checkpoints/).")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best",
                         help="Which checkpoint in <log_dir>/checkpoints/ to evaluate.")
    parser.add_argument("--binarize_threshold", type=float, default=0.5,
                         help="Threshold used to binarize GT/predicted images for the "
                              "segmentation metrics (DSC/HD95/ASD/IoU/boundary-F/topology).")
    parser.add_argument("--boundary_tolerance_px", type=int, default=2,
                         help="Pixel tolerance used for the boundary F-score match.")
    samples = parser.add_argument_group("qualitative sample dump")
    samples.add_argument("--save_samples", action=argparse.BooleanOptionalAction, default=True,
                          help="Save --num_samples (spectrum, GT image, predicted image) figures "
                               "to <log_dir>/samples/.")
    samples.add_argument("--num_samples", type=int, default=8,
                          help="Number of test-set samples to save when --save_samples is set.")
    # Re-expose the same shared flags as train.py so users CAN override any key arg;
    # anything left as None falls back to the run's saved config.json.
    parser = add_shared_args(parser)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    saved_cfg = load_config(os.path.join(args.log_dir, "config.json"))

    # Start from the saved training config, then apply any explicit CLI overrides
    # using the exact same "explicit-CLI > preset > default" merge, but seeded
    # from saved_cfg instead of HARD_DEFAULTS so unset fields keep their trained values.
    # We deliberately do NOT call resolve_config(args) here: that would fill in
    # HARD_DEFAULTS for every field, clobbering the run's saved config. Instead we
    # start from the run's saved config, optionally layer --preset on top (if the
    # user wants to evaluate this checkpoint "as if" it were a different preset's
    # flags), and finally apply anything *explicitly passed* on this evaluate.py
    # CLI call (all such fields default to None -- see config.py). Precedence:
    # explicit CLI > --preset > saved_cfg.
    cfg = dict(saved_cfg)
    # Back-compat: older run dirs were trained before the use_deform bool -> conv_type
    # string rename (see config.py). Derive conv_type from the saved use_deform flag
    # if this run's config.json predates the rename.
    if "conv_type" not in cfg:
        cfg["conv_type"] = "deform" if cfg.get("use_deform") else "plain"

    if args.preset is not None:
        cfg.update(PRESETS[args.preset])

    explicit_fields = [
        "variant", "table1_row", "conv_type", "use_xpie", "use_gan", "gan_mode",
        "two_stage", "recon_loss", "use_pretrained", "epochs", "seed", "warmup",
        "alpha_max", "patience", "lr_g", "lr_d", "batch_size",
        "pretrained_generator_ckpt", "pretrained_discriminator_ckpt", "device",
        "forward_model_path", "xpie_data_path", "data_imag_path", "spectrum_file",
        "results_save_dir", "model_path", "inverse_model_path", "gan_model_path",
        "log_root", "results_dir", "master_csv_name",
    ]
    for field in explicit_fields:
        val = getattr(args, field, None)
        if val is not None:
            cfg[field] = val

    eval_logger = build_logger("eval", os.path.join(args.log_dir, "eval.log"))
    eval_logger.info(f"Evaluating run: {args.log_dir}")
    eval_logger.info(f"Effective eval config: {cfg}")

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    eval_logger.info(f"Device: {device}")

    ckpt_path = os.path.join(args.log_dir, "checkpoints", f"{args.checkpoint}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- model ----
    model, _ = build_models(cfg["conv_type"], use_gan=False, device=device)  # discriminator not needed for eval
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["generator_state_dict"])
    model.eval()
    eval_logger.info(f"Loaded checkpoint: {ckpt_path} (epoch={state.get('epoch')}, loss={state.get('loss')})")

    # ---- data (fixed test split; independent of use_xpie) ----
    test_loader = build_test_loader(
        batch_size=cfg["batch_size"], data_imag_path=cfg["data_imag_path"],
        spectrum_file=cfg["spectrum_file"],
    )

    # ---- evaluate ----
    metrics = evaluate_on_test_set(model, test_loader, device, eval_logger,
                                    binarize_threshold=args.binarize_threshold,
                                    boundary_tolerance_px=args.boundary_tolerance_px)
    variant = cfg.get("variant", "custom")
    eval_logger.info(f"VARIANT: {variant}")

    # ---- qualitative samples (spectrum + GT + predicted) ----
    if args.save_samples:
        save_gt_vs_pred_samples(
            model, test_loader, device, os.path.join(args.log_dir, "samples"),
            num_samples=args.num_samples, logger=eval_logger,
        )

    # ---- persist per-run result ----
    eval_result = {"variant": variant, "checkpoint": args.checkpoint, **metrics}
    save_config(eval_result, os.path.join(args.log_dir, "eval_result.json"))

    # ---- append to shared master CSV (same behaviour as every notebook's final cell) ----
    os.makedirs(cfg["results_dir"], exist_ok=True)
    master_csv = os.path.join(cfg["results_dir"], cfg["master_csv_name"])
    _append_master_csv(master_csv, {
        "variant": variant, "seed": cfg.get("seed"),
        "psnr": round(metrics["psnr"], 4), "ssim": round(metrics["ssim"], 4),
        "mse": round(metrics["mse"], 6),
        "dsc": round(metrics["dsc"], 4),
        "hd95": "" if math.isnan(metrics["hd95"]) else round(metrics["hd95"], 4),
        "asd": "" if math.isnan(metrics["asd"]) else round(metrics["asd"], 4),
        "iou": round(metrics["iou"], 4),
        "boundary_f": round(metrics["boundary_f"], 4),
        "cc_error": round(metrics["cc_error"], 4),
        "topology_validity_rate": round(metrics["topology_validity_rate"], 4),
        "params_M": round(metrics["params_M"], 3),
        "conv_type": cfg.get("conv_type"), "use_xpie": cfg.get("use_xpie"),
        "use_gan": cfg.get("use_gan"), "gan_mode": cfg.get("gan_mode"),
        "two_stage": cfg.get("two_stage"), "recon_loss": cfg.get("recon_loss"),
        "log_dir": args.log_dir, "checkpoint": args.checkpoint,
    })
    eval_logger.info(f"Appended to {master_csv}")

    return eval_result


if __name__ == "__main__":
    main()
