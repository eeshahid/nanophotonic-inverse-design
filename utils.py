"""
utils.py
========
Generic helpers shared by train.py / evaluate.py:
  - reproducibility (set_seed)
  - metrics (calculate_psnr, calculate_ssim)  -- verbatim from the notebooks
  - experiment-log-dir management (log_NUM_TIMESTAMP)
  - file logger setup (separate train.log / eval.log per run)
  - config save/load (json)

Nothing here changes numerical behaviour vs. the original notebooks; this file only
factors out code that was duplicated verbatim across all 11 notebooks.
"""
import json
import logging
import os
import random
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

try:
    from medpy.metric import binary as _medpy_binary
except ImportError:
    _medpy_binary = None


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Verbatim from every notebook's section 4."""
    random.seed(seed)                          # Python random
    np.random.seed(seed)                       # Numpy
    torch.manual_seed(seed)                    # CPU
    torch.cuda.manual_seed(seed)               # Current GPU
    torch.cuda.manual_seed_all(seed)           # All GPUs (if using multi-GPU)

    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable autotuner that can change results


# --------------------------------------------------------------------------- #
# Metrics  (verbatim from every notebook's section 5)
# --------------------------------------------------------------------------- #
def calculate_psnr(true_image: torch.Tensor, predicted_image: torch.Tensor) -> float:
    mse = F.mse_loss(true_image, predicted_image)
    max_pixel_value = 1.0  # Since the image is normalized between 0 and 1
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value.item()


def calculate_ssim(true_image: torch.Tensor, predicted_image: torch.Tensor) -> float:
    true_image_np = true_image.squeeze().detach().cpu().numpy()
    predicted_image_np = predicted_image.squeeze().detach().cpu().numpy()
    ssim_value = ssim(
        true_image_np,
        predicted_image_np,
        data_range=true_image_np.max() - true_image_np.min(),
    )
    return ssim_value


def _boundary_f_score(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 2) -> float:
    """Boundary F-score: precision/recall of boundary pixels matched within `tolerance_px`
    (Euclidean, in pixels), via a distance transform to the opposite mask's boundary --
    the standard "BF score" / boundary-matching definition used e.g. for Cityscapes/COCO
    boundary evaluation. 1.0 if both masks have no boundary (both empty or both full);
    0.0 if only one does."""
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import find_boundaries

    pred_b = find_boundaries(pred, mode="inner")
    gt_b = find_boundaries(gt, mode="inner")
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    gt_dist = distance_transform_edt(~gt_b)
    pred_dist = distance_transform_edt(~pred_b)
    precision = float((gt_dist[pred_b] <= tolerance_px).mean())
    recall = float((pred_dist[gt_b] <= tolerance_px).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _topology_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """cc_error: |#connected components(pred) - #connected components(gt)| (8-connectivity).
    topology_valid: 1.0 iff pred and gt have the same Euler number (#components - #holes) --
    i.e. genuinely the same topology, not just the same component count. Both are well-defined
    for empty masks (0 components / Euler number 0), no special-casing needed."""
    from skimage.measure import euler_number, label

    n_cc_pred = int(label(pred, connectivity=2).max())
    n_cc_gt = int(label(gt, connectivity=2).max())
    euler_pred = euler_number(pred, connectivity=2)
    euler_gt = euler_number(gt, connectivity=2)
    return {
        "cc_error": abs(n_cc_pred - n_cc_gt),
        "topology_valid": 1.0 if euler_pred == euler_gt else 0.0,
    }


# --------------------------------------------------------------------------- #
# Segmentation metrics -- both images are binarized at `threshold` (GT is already
# ~binary; predictions are continuous [0,1] model output). DSC/HD95/ASD/IoU via
# medpy.metric.binary (the same library nnU-Net-style pipelines use); boundary F-score
# and topology stats via scipy/skimage (see helpers above).
# --------------------------------------------------------------------------- #
def calculate_segmentation_metrics(true_image: torch.Tensor, predicted_image: torch.Tensor,
                                    threshold: float = 0.5, boundary_tolerance_px: int = 2) -> dict:
    """Returns {"dsc", "hd95", "asd", "iou", "boundary_f", "cc_error", "topology_valid"}
    (distances in pixels -- this dataset has no physical spacing). hd95/asd are undefined
    when either mask has no foreground pixels: NaN if only one is empty (dsc=iou=0.0), or a
    trivial perfect match (dsc=iou=1.0, hd95=asd=0.0) if both are."""
    if _medpy_binary is None:
        raise ImportError(
            "calculate_segmentation_metrics requires the 'medpy' package. Install it with:\n"
            "  pip install medpy"
        )
    gt = true_image.squeeze().detach().cpu().numpy() > threshold
    pred = predicted_image.squeeze().detach().cpu().numpy() > threshold

    gt_empty = not gt.any()
    pred_empty = not pred.any()
    if gt_empty and pred_empty:
        dsc, hd95, asd, iou = 1.0, 0.0, 0.0, 1.0
    elif gt_empty or pred_empty:
        dsc, hd95, asd, iou = 0.0, float("nan"), float("nan"), 0.0
    else:
        dsc = _medpy_binary.dc(pred, gt)
        hd95 = _medpy_binary.hd95(pred, gt)
        asd = _medpy_binary.asd(pred, gt)
        iou = _medpy_binary.jc(pred, gt)

    result = {"dsc": dsc, "hd95": hd95, "asd": asd, "iou": iou}
    result["boundary_f"] = _boundary_f_score(pred, gt, tolerance_px=boundary_tolerance_px)
    result.update(_topology_stats(pred, gt))
    return result


# --------------------------------------------------------------------------- #
# Experiment log-dir management: logs/log_000_20260712_153045
# --------------------------------------------------------------------------- #
_LOG_DIR_RE = re.compile(r"^log_(\d{3})_\d{8}_\d{6}$")


def make_run_dir(log_root: str) -> str:
    """Create and return a new experiment directory:
        <log_root>/log_<NUM>_<YYYYMMDD_HHMMSS>
    NUM is zero-padded (000, 001, ...) and is one greater than the highest NUM
    already present under log_root (so runs are never overwritten / reused).
    """
    os.makedirs(log_root, exist_ok=True)
    existing = []
    for name in os.listdir(log_root):
        m = _LOG_DIR_RE.match(name)
        if m:
            existing.append(int(m.group(1)))
    next_num = (max(existing) + 1) if existing else 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"log_{next_num:03d}_{timestamp}"
    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def find_latest_run_dir(log_root: str) -> str:
    """Convenience: return the most-recently-created log_NUM_TIMESTAMP dir under log_root."""
    candidates = []
    if os.path.isdir(log_root):
        for name in os.listdir(log_root):
            if _LOG_DIR_RE.match(name):
                candidates.append(name)
    if not candidates:
        raise FileNotFoundError(f"No log_NUM_TIMESTAMP directories found under {log_root!r}")
    candidates.sort()
    return os.path.join(log_root, candidates[-1])


# --------------------------------------------------------------------------- #
# Logging: independent file loggers for training vs. evaluation
# --------------------------------------------------------------------------- #
def build_logger(name: str, log_file: str, also_console: bool = True) -> logging.Logger:
    """Creates a logger that writes to `log_file` (and optionally stdout).
    Uses a unique logger name per (name, log_file) so multiple loggers in the
    same process (train + eval) don't share/duplicate handlers.
    """
    logger = logging.getLogger(f"{name}:{log_file}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # already configured (e.g. re-entrant call)
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# --------------------------------------------------------------------------- #
# Config save / load
# --------------------------------------------------------------------------- #
def save_config(config: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
