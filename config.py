"""
config.py
==========
Single source of truth for:
  - the 11 original notebooks' exact FLAGS/epoch combinations, exposed as `--preset`
  - the full argparse definition shared by train.py and evaluate.py
  - the merge logic: explicit CLI flag > --preset value > hard-coded default

IMPORTANT / KNOWN QUIRK (see also README):
Notebooks 09 (`bce_gan_pretrained`) and 10 (`lsgan_pretrained_bce`) are *named* and
*described* as "(Pretrained)" but their actual code never sets `use_pretrained=True`
and never loads a warm-start checkpoint -- only notebook 11 (`proposed`) does. This
was verified by diffing all 11 notebooks byte-for-byte: only `11_proposed.ipynb` has
the warm-start block. Per instruction, this is preserved EXACTLY as authored by the
collaborator (Waleed) rather than "fixed" -- see `PRESETS["09_bce_gan_pretrained"]`
and `PRESETS["10_lsgan_pretrained_bce"]` below, both of which keep `use_pretrained=False`.
If you want a run that actually performs the Waleed-style pretrained warm-start with
those flag combinations, pass `--use_pretrained --pretrained_generator_ckpt ... `
explicitly (flag name kept as `use_pretrained_waleed_quirk` is NOT a separate knob --
it is documented here so nobody "fixes" this silently and changes reported numbers).
"""
import argparse

from conv_layers import CONV_LAYER_REGISTRY

# Terminal Runs
# python train.py --conv_type deform --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 200
# python train.py --conv_type deform --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 # with None as --pretrained_discriminator_ckpt

# nohup python train.py --conv_type plain --gan_mode ls --recon_loss mse --epochs 100 &> nh_C3Gen_plain &

# 2Stage-Part1: Pretrained Generator
# nohup python train.py --conv_type plain --gan_mode ls --recon_loss mse --epochs 100 --seed 43 &> nh_C3Gen_plain_run4 &
# nohup python train.py --conv_type deform --gan_mode ls --recon_loss mse --epochs 100 --seed 43 &> nh_C3Gen_deform_run4 &
# nohup python train.py --conv_type involution --gan_mode ls --recon_loss mse --epochs 100 --seed 43 &> nh_C3Gen_involution_run4 &
# nohup python train.py --conv_type dynamic --gan_mode ls --recon_loss mse --epochs 100 --seed 43 &> nh_C3Gen_dynamic_run4 &
# nohup python train.py --conv_type odconv --gan_mode ls --recon_loss mse --epochs 100 --seed 43 &> nh_C3Gen_odconv_run4 &

# 2Stage-Part2: 
# nohup python train.py --conv_type plain --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt /home/coe/shahid/mxene/inverse/src/logs/log_003_20260722_102600/checkpoints/best.pth &> nh_C11_4_plain &
# nohup python train.py --conv_type involution --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt /home/coe/shahid/mxene/inverse/src/logs/log_004_20260722_102925/checkpoints/best.pth &> nh_C11_4_involution &
# nohup python train.py --conv_type dynamic --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt /home/coe/shahid/mxene/inverse/src/logs/log_005_20260722_102931/checkpoints/best.pth &> nh_C11_4_dynamic &
# nohup python train.py --conv_type odconv --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt /home/coe/shahid/mxene/inverse/src/logs/log_006_20260722_102938/checkpoints/best.pth &> nh_C11_4_odconv &

# nohup python train.py --conv_type plain --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt logs/log_0 --seed 43 &> nh_C11_4_plain_run4 &
# nohup python train.py --conv_type deform --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt logs/log_0 --seed 43 &> nh_C11_4_deform_run4 &
# nohup python train.py --conv_type involution --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt logs/log_0 --seed 43 &> nh_C11_4_involution_run4 &
# nohup python train.py --conv_type dynamic --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt logs/log_0 --seed 43 &> nh_C11_4_dynamic_run4 &
# nohup python train.py --conv_type odconv --use_gan --gan_mode ls --two_stage --recon_loss mse --use_pretrained --epochs 500 --pretrained_generator_ckpt logs/log_0 --seed 43 &> nh_C11_4_odconv_run4 &


# Single Stage
# nohup python train.py --conv_type plain --use_gan --gan_mode ls --recon_loss mse --epochs 500 --variant lsgan_1stage --seed 43 &> nh_lsgan_1stage_plain_run4 &
# nohup python train.py --conv_type deform --use_gan --gan_mode ls --recon_loss mse --epochs 500 --variant lsgan_1stage --seed 43 &> nh_lsgan_1stage_deform_run4 &
# nohup python train.py --conv_type involution --use_gan --gan_mode ls --recon_loss mse --epochs 500 --variant lsgan_1stage --seed 43 &> nh_lsgan_1stage_involution_run4 &
# nohup python train.py --conv_type dynamic --use_gan --gan_mode ls --recon_loss mse --epochs 500 --variant lsgan_1stage --seed 43 &> nh_lsgan_1stage_dynamic_run4 &
# nohup python train.py --conv_type odconv --use_gan --gan_mode ls --recon_loss mse --epochs 500 --variant lsgan_1stage --seed 43 &> nh_lsgan_1stage_odconv_run4 &


# --------------------------------------------------------------------------- #
# Exact FLAGS / epochs / variant name / table1_row from each of the 11 notebooks.
# --------------------------------------------------------------------------- #
PRESETS = {
    "01_simple_cnn_noaug": dict(
        variant="simple_cnn_noaug",
        table1_row="(new control) plain CNN, MSE, no aug -- to run",
        conv_type="plain", use_xpie=False, use_gan=False, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "02_simple_cnn_xpie": dict(
        variant="simple_cnn_xpie",
        table1_row="(new control) plain CNN, MSE, +XPIE -- to run",
        conv_type="plain", use_xpie=True, use_gan=False, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "03_deform_mse_noaug": dict(
        variant="deform_mse_noaug",
        table1_row="Deform INVERSE no AUG  -> 17.05 / 0.6835",
        conv_type="deform", use_xpie=False, use_gan=False, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "04_deform_mse_xpie": dict(
        variant="deform_mse_xpie",
        table1_row="Deform INVERSE+XPIE  -> 18.87 / 0.7911",
        conv_type="deform", use_xpie=True, use_gan=False, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "05_bce_gan_noaug": dict(
        variant="bce_gan_noaug",
        table1_row="SIMPLE DEFORM GAN (NO AUG)  -> 16.54 / 0.7683",
        conv_type="deform", use_xpie=False, use_gan=True, gan_mode="bce",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "06_bce_gan_xpie": dict(
        variant="bce_gan_xpie",
        table1_row="SIMPLE DEFORM GAN+xpie  -> 15.52 / 0.7332",
        conv_type="deform", use_xpie=True, use_gan=True, gan_mode="bce",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "07_lsgan_noaug": dict(
        variant="lsgan_noaug",
        table1_row="DEFORM LS GAN no AUG  -> 18.33 / 0.7683",
        conv_type="deform", use_xpie=False, use_gan=True, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "08_lsgan_xpie": dict(
        variant="lsgan_xpie",
        table1_row="DEFORM LS GAN+XPIE  -> 18.53 / 0.7693",
        conv_type="deform", use_xpie=True, use_gan=True, gan_mode="ls",
        two_stage=False, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    # -- "(Pretrained)" in name/description only; see module docstring above. --
    "09_bce_gan_pretrained": dict(
        variant="bce_gan_pretrained",
        table1_row="Deform GAN +xpie (Pretrained)  -> 19.75 / 0.8128",
        conv_type="deform", use_xpie=True, use_gan=True, gan_mode="bce",
        two_stage=True, recon_loss="mse", use_pretrained=False, epochs=100,
    ),
    "10_lsgan_pretrained_bce": dict(
        variant="lsgan_pretrained_bce",
        table1_row="Deform LS GAN +xpie (Pretrained) BCE LOSS  -> 19.93 / 0.8228",
        conv_type="deform", use_xpie=True, use_gan=True, gan_mode="ls",
        two_stage=True, recon_loss="bce", use_pretrained=False, epochs=100,
    ),
    "11_proposed": dict(
        variant="proposed",
        table1_row="Deform LS GAN +xpie (Pretrained) MSE LOSS (PROPOSED)  -> 20.00 / 0.8238",
        conv_type="deform", use_xpie=True, use_gan=True, gan_mode="ls",
        two_stage=True, recon_loss="mse", use_pretrained=True, epochs=500,
    ),
}

# Hard-coded defaults shared by every notebook (section 1), used when a field is
# neither explicitly passed on the CLI nor supplied by --preset.
HARD_DEFAULTS = dict(
    variant="custom",
    table1_row="",
    conv_type="plain",
    use_xpie=False,
    use_gan=False,
    gan_mode="ls",
    two_stage=False,
    recon_loss="mse",
    use_pretrained=False,
    epochs=100,
    warmup=50,
    alpha_max=0.1,
    patience=50,
    lr_g=1e-4,
    lr_d=1e-6,
    seed=42,
    batch_size=32,
)

PRESET_OVERRIDABLE_FIELDS = (
    "variant", "table1_row", "conv_type", "use_xpie", "use_gan", "gan_mode",
    "two_stage", "recon_loss", "use_pretrained", "epochs",
)


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Args shared by train.py and evaluate.py. Boolean/choice/epoch fields default
    to None so we can tell "not explicitly passed" apart from "explicitly passed the
    same value as the default" -- see `resolve_config()`."""
    preset = parser.add_argument_group("preset (optional convenience)")
    preset.add_argument(
        "--preset", choices=sorted(PRESETS.keys()), default=None,
        help="Reproduce one of the 11 original notebooks exactly (sets variant, "
             "table1_row, conv_type, use_xpie, use_gan, gan_mode, two_stage, "
             "recon_loss, use_pretrained, epochs). Any of those flags passed "
             "explicitly on the CLI still takes precedence over the preset.",
    )

    exp = parser.add_argument_group("experiment identity")
    exp.add_argument("--variant", type=str, default=None, help="Short name for this run (used in filenames/CSV rows).")
    exp.add_argument("--table1_row", type=str, default=None, help="Free-text description of this run.")

    flags = parser.add_argument_group("ablation flags (the notebooks' FLAGS dict)")
    flags.add_argument("--conv_type", choices=sorted(CONV_LAYER_REGISTRY.keys()), default=None,
                        help="Conv layer used by generator + discriminator. 'plain' = nn.Conv2d, "
                             "'deform' = DeformableConv2d (the original notebooks' use_deform=True/False "
                             "knob). Optional stronger replacements (see conv_layers.py): 'dcnv4' "
                             "(official DCNv4 CUDA op, CVPR'24 -- needs the optional DCNv4 package), "
                             "'involution' (content-adaptive spatial kernels, CVPR'21), 'dynamic' "
                             "(CondConv-style mixture-of-kernels attention, CVPR'20), 'odconv' "
                             "(omni-dimensional 4D kernel attention, ICLR'22).")
    flags.add_argument("--use_xpie", action=argparse.BooleanOptionalAction, default=None,
                        help="Add the XPIE forward-augmented dataset to training/validation.")
    flags.add_argument("--use_gan", action=argparse.BooleanOptionalAction, default=None,
                        help="Train with an adversarial discriminator (GAN) in addition to reconstruction loss.")
    flags.add_argument("--gan_mode", choices=["ls", "bce"], default=None,
                        help="'ls' = LSGAN (MSELoss adversarial term), 'bce' = vanilla GAN (BCELoss).")
    flags.add_argument("--two_stage", action=argparse.BooleanOptionalAction, default=None,
                        help="Reconstruction-only warmup for `warmup` epochs before turning on the adversarial term.")
    flags.add_argument("--recon_loss", choices=["mse", "bce"], default=None,
                        help="Pixel reconstruction loss.")
    flags.add_argument("--use_pretrained", action=argparse.BooleanOptionalAction, default=None,
                        help="Warm-start generator (+discriminator, if --use_gan) from checkpoints "
                             "given by --pretrained_generator_ckpt / --pretrained_discriminator_ckpt. "
                             "Only notebook 11 ('proposed') actually used this; see config.py docstring.")

    pretrain = parser.add_argument_group("pretrained warm-start (only used if --use_pretrained)")
    # pretrain.add_argument("--pretrained_generator_ckpt", type=str, default="/home/coe/shahid/mxene/inverse/src/logs_presets/log_004_20260714_124037/checkpoints/best.pth", # log_004_20260714_124037
    pretrain.add_argument("--pretrained_generator_ckpt", type=str, default="/home/coe/shahid/mxene/inverse/src/logs_presets/log_003_20260714_124025/checkpoints/best.pth", # 
                           help="Path to a checkpoint (.pth with a 'generator_state_dict' key) to warm-start the generator from -- "
                                "e.g. another run's <log_dir>/checkpoints/best.pth.")
    # pretrain.add_argument("--pretrained_discriminator_ckpt", type=str, default="/home/coe/shahid/mxene/inverse/src/logs_presets/log_006_20260714_124102/checkpoints/best.pth", # log_006_20260714_124102
    # pretrain.add_argument("--pretrained_discriminator_ckpt", type=str, default="/home/coe/shahid/mxene/inverse/src/logs_presets/log_005_20260714_124048/checkpoints/best.pth",
    pretrain.add_argument("--pretrained_discriminator_ckpt", type=str, default=None,
                           help="Path to a checkpoint (.pth with a 'discriminator_state_dict' key) to warm-start the discriminator from.")

    train = parser.add_argument_group("training hyperparameters")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None, help="Max epochs (ABL_EPOCHS in the notebooks).")
    train.add_argument("--warmup", type=int, default=None, help="Reconstruction-only warmup epochs (ABL_WARMUP).")
    train.add_argument("--alpha_max", type=float, default=None, help="Adversarial loss weight after warmup (ALPHA_MAX).")
    train.add_argument("--patience", type=int, default=None, help="Early-stop patience on val pixel-MSE (ABL_PATIENCE).")
    train.add_argument("--lr_g", type=float, default=None, help="Generator learning rate (LR_G).")
    train.add_argument("--lr_d", type=float, default=None, help="Discriminator learning rate (LR_D).")
    train.add_argument("--batch_size", type=int, default=None)
    train.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'. Default: cuda if available else cpu.")

    data = parser.add_argument_group("data / model paths (all hardcoded Windows paths in the "
                                      "original notebooks; override for your machine)")
    data.add_argument("--forward_model_path", type=str, default="/home/coe/shahid/mxene/inverse/Result_Waleed/Weights/sh_best_model.pth",
                       help="Frozen forward (image->spectrum) simulator checkpoint.")
    data.add_argument("--xpie_data_path", type=str, default="/home/coe/shahid/mxene/mxene_forward/data_inverse/data_inverse/XPIE_mask",
                       help="Directory of XPIE mask images (used only if --use_xpie).")
    data.add_argument("--data_imag_path", type=str, default="/home/coe/shahid/mxene/mxene_forward/data_inverse/data_inverse/Data/Images",
                       help="Directory of the main FDTD training images.")
    data.add_argument("--spectrum_file", type=str, default="/home/coe/shahid/mxene/mxene_forward/data_inverse/data_inverse/Data/Spectra.csv",
                       help="CSV of spectra paired with data_imag_path images.")
    # The following three paths are defined in every notebook's section 1 but are never
    # actually read anywhere in the notebooks (dead/vestigial config). Kept here only for
    # config fidelity; not used by any code path.
    data.add_argument("--results_save_dir", type=str, default="Result/run", help="[unused, kept for fidelity]")
    data.add_argument("--model_path", type=str, default="Weights/model.pth", help="[unused, kept for fidelity]")
    data.add_argument("--inverse_model_path", type=str, default="Weights/simple_inverse.pth", help="[unused, kept for fidelity]")
    data.add_argument("--gan_model_path", type=str, default="Weights/gan_generator.pth", help="[unused, kept for fidelity]")

    logdir = parser.add_argument_group("logging / output")
    logdir.add_argument("--log_root", type=str, default="logs",
                         help="Parent directory for per-run log_<NUM>_<TIMESTAMP> dirs.")
    logdir.add_argument("--results_dir", type=str, default="results_abl_inv",
                         help="Directory for the shared cross-run master CSV (ablation_summary_inv.csv).")
    logdir.add_argument("--master_csv_name", type=str, default="ablation_summary_inv.csv")

    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge priority: explicit CLI value > --preset value > HARD_DEFAULTS.
    Any arg the user did not pass on the CLI is None at this point (because every
    preset-overridable arg above uses default=None), so `is not None` reliably
    means "the user explicitly passed this flag"."""
    cfg = dict(HARD_DEFAULTS)

    if args.preset is not None:
        cfg.update(PRESETS[args.preset])

    for field in PRESET_OVERRIDABLE_FIELDS:
        explicit = getattr(args, field)
        if explicit is not None:
            cfg[field] = explicit

    for field in ("seed", "warmup", "alpha_max", "patience", "lr_g", "lr_d", "batch_size"):
        explicit = getattr(args, field)
        if explicit is not None:
            cfg[field] = explicit

    # Straight passthrough fields (no preset involvement).
    for field in (
        "pretrained_generator_ckpt", "pretrained_discriminator_ckpt", "device",
        "forward_model_path", "xpie_data_path", "data_imag_path", "spectrum_file",
        "results_save_dir", "model_path", "inverse_model_path", "gan_model_path",
        "log_root", "results_dir", "master_csv_name",
    ):
        cfg[field] = getattr(args, field)

    return cfg
