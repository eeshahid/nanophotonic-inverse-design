"""
train.py
=========
Drop-in replacement for "run one of the 11 ablation notebooks end-to-end".

Usage examples
--------------
# Reproduce a specific notebook exactly:
python train.py --preset 11_proposed \\
    --forward_model_path weights/sh_best_model.pth \\
    --xpie_data_path data/XPIE_mask \\
    --data_imag_path data/Training_Data/Images \\
    --spectrum_file data/Training_Data/Spectra.csv \\
    --pretrained_generator_ckpt logs/log_003_.../checkpoints/best.pth \\
    --pretrained_discriminator_ckpt logs/log_007_.../checkpoints/best.pth

# Fully custom ablation:
python train.py --conv_type deform --use_xpie --use_gan --gan_mode ls --two_stage \\
    --recon_loss mse --epochs 200 ...

Every run creates logs/log_<NUM>_<TIMESTAMP>/ containing:
    config.json             -- the fully-resolved config (argparser args) for this run
    train.log                -- training log
    eval.log                 -- evaluation log (written by evaluate.py, run automatically below)
    eval_result.json         -- evaluation metrics (written by evaluate.py)
    checkpoints/best.pth      -- best (lowest val loss) checkpoint
    checkpoints/latest.pth    -- most recent epoch's checkpoint

After training finishes, evaluate.py is automatically invoked on this run's log_dir
(pass --skip_eval to disable this).
"""
import argparse
import os
import subprocess
import sys

import torch

from build_data import build_loaders_abl, load_forward_model
from build_model import build_models
from config import add_shared_args, resolve_config
from train_fn import evaluate_on_test_set, train_gan_model_abl
from utils import build_logger, make_run_dir, save_config, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one ablation configuration of the inverse-design model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_shared_args(parser)
    parser.add_argument("--skip_eval", action="store_true",
                         help="Do not automatically run evaluate.py after training.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = resolve_config(args)

    # ---- experiment dir + loggers -------------------------------------------------
    run_dir = make_run_dir(cfg["log_root"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    latest_ckpt_path = os.path.join(ckpt_dir, "latest.pth")

    train_logger = build_logger("train", os.path.join(run_dir, "train.log"))
    train_logger.info(f"Run dir: {run_dir}")
    train_logger.info(f"Resolved config: {cfg}")

    cfg["run_dir"] = run_dir
    save_config(cfg, os.path.join(run_dir, "config.json"))

    # ---- device / seed --------------------------------------------------------------
    device = torch.device(cfg["device"] if cfg["device"] else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_logger.info(f"Device: {device}")
    set_seed(cfg["seed"])

    # ---- forward model + data --------------------------------------------------------
    forward_model = load_forward_model(cfg["forward_model_path"], device)
    train_loader, val_loader, test_loader, len_tr, len_va = build_loaders_abl(
        use_xpie=cfg["use_xpie"], seed=cfg["seed"], batch_size=cfg["batch_size"],
        xpie_data_path=cfg["xpie_data_path"], data_imag_path=cfg["data_imag_path"],
        spectrum_file=cfg["spectrum_file"], forward_model=forward_model, device=device,
    )
    train_logger.info(f"train/val/test sizes: {len_tr}/{len_va}/{len(test_loader.dataset)}")

    # ---- model ------------------------------------------------------------------------
    model, discriminator = build_models(cfg["conv_type"], cfg["use_gan"], device)

    # ---- train ------------------------------------------------------------------------
    train_gan_model_abl(
        model, discriminator, train_loader, val_loader, len_tr, len_va,
        best_ckpt_path=best_ckpt_path, latest_ckpt_path=latest_ckpt_path,
        device=device, logger=train_logger,
        use_gan=cfg["use_gan"], gan_mode=cfg["gan_mode"], two_stage=cfg["two_stage"],
        recon_loss=cfg["recon_loss"], num_epochs=cfg["epochs"], warmup=cfg["warmup"],
        alpha_max=cfg["alpha_max"], lr=cfg["lr_g"], d_lr=cfg["lr_d"], patience=cfg["patience"],
        use_pretrained=cfg["use_pretrained"],
        pretrained_generator_ckpt=cfg["pretrained_generator_ckpt"],
        pretrained_discriminator_ckpt=cfg["pretrained_discriminator_ckpt"],
    )
    train_logger.info("Training complete.")

    # ---- auto-evaluate ------------------------------------------------------------------
    if not args.skip_eval:
        train_logger.info("Launching evaluate.py on this run...")
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"),
             "--log_dir", run_dir],
            check=False,
        )
        if result.returncode != 0:
            train_logger.error(f"evaluate.py exited with code {result.returncode}")
        else:
            train_logger.info("Evaluation complete.")

    return run_dir


if __name__ == "__main__":
    main()
