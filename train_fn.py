"""
train_fn.py
=============
Training loop, verbatim in numerical behaviour from every notebook's section 5/6
(`train_gan_model_abl`), plus `evaluate_inverse_model` (defined in every notebook
but never actually called there -- kept here for fidelity / reuse).

Only additions vs. the notebooks:
  - writes to a `logging.Logger` instead of only `print`
  - saves a `latest.pth` checkpoint every epoch IN ADDITION to `best.pth` on
    improvement (the notebooks only ever saved `best.pth`)
  - `use_pretrained` warm-start now takes explicit checkpoint paths instead of
    the notebook's hardcoded `weights_abl_inv/abl_..._best.pth` filenames
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import calculate_psnr, calculate_segmentation_metrics, calculate_ssim


# --------------------------------------------------------------------------- #
# Defined-but-unused-in-the-original-notebooks evaluation helper.
# Kept verbatim (parametrized with `device`) for fidelity; NOT used by the
# actual end-of-notebook evaluation cell, which is reproduced in evaluate.py.
# --------------------------------------------------------------------------- #
def evaluate_inverse_model(model, dataloader, device):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for spectra, true_images in dataloader:
            spectra, true_images = spectra.to(device), true_images.to(device)

            # Forward pass: Predict images from spectra
            predicted_images = model(spectra)

            # Calculate MSE for the batch
            mse_loss = F.mse_loss(predicted_images, true_images.unsqueeze(1))  # Unsqueeze to add the channel dimension
            total_mse += mse_loss.item() * spectra.size(0)

            # Calculate PSNR and SSIM for each sample in the batch
            for i in range(spectra.size(0)):
                psnr_value = calculate_psnr(true_images[i].unsqueeze(0), predicted_images[i])
                ssim_value = calculate_ssim(true_images[i], predicted_images[i])

                total_psnr += psnr_value
                total_ssim += ssim_value

            num_samples += spectra.size(0)

    # Calculate average metrics
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print("Performance Metrics:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return avg_mse, avg_psnr, avg_ssim


# --------------------------------------------------------------------------- #
# Main training loop -- verbatim numerical behaviour from every notebook's
# `train_gan_model_abl`. `logger` replaces bare `print`; checkpointing extended
# to also keep a rolling `latest.pth`.
# --------------------------------------------------------------------------- #
def train_gan_model_abl(model, discriminator, train_loader, val_loader, len_train, len_val,
                         best_ckpt_path, latest_ckpt_path, device, logger,
                         use_gan=True, gan_mode='ls', two_stage=True,
                         recon_loss='mse', num_epochs=100, warmup=50,
                         alpha_max=0.1, lr=1e-4, d_lr=1e-6, patience=50,
                         use_pretrained=False, pretrained_generator_ckpt=None,
                         pretrained_discriminator_ckpt=None):
    """Parametrized version of Proposed.ipynb's train_gan_model. adversarial_loss=MSELoss
    is the symmetric least-squares (LSGAN) term; BCELoss is the vanilla GAN term. The
    two-stage reconstruction warmup IS the 'Pretrained' stage (self-contained).
    The discriminator step and the adversarial term share ONE condition (`adversarial`),
    so the loop has a single if-branch; the generator is trained on every iteration."""
    logger.info(f"Using device: {device}")
    optimizer_G = optim.AdamW(model.parameters(), lr=lr)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=d_lr) if use_gan else None

    # -- Pretrained warm start (only the 'proposed' variant sets use_pretrained=True
    #    in the original notebooks; see README for the 09/10 "pretrained"-in-name-only quirk) --
    if use_pretrained:
        if not pretrained_generator_ckpt or not os.path.exists(pretrained_generator_ckpt):
            raise FileNotFoundError(
                f"--use_pretrained was set but --pretrained_generator_ckpt "
                f"({pretrained_generator_ckpt!r}) does not exist."
            )
        checkpoint = torch.load(pretrained_generator_ckpt, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded pretrained generator from {pretrained_generator_ckpt}")

        if use_gan and discriminator is not None:
            if not pretrained_discriminator_ckpt:
                logger.info("No --pretrained_discriminator_ckpt given; training discriminator from scratch.")
            elif not os.path.exists(pretrained_discriminator_ckpt):
                raise FileNotFoundError(
                    f"--pretrained_discriminator_ckpt ({pretrained_discriminator_ckpt!r}) does not exist."
                )
            else:
                checkpointg = torch.load(pretrained_discriminator_ckpt, map_location=device)
                discriminator.load_state_dict(checkpointg['discriminator_state_dict'])
                if optimizer_D is not None and 'optimizer_D_state_dict' in checkpointg:
                    optimizer_D.load_state_dict(checkpointg['optimizer_D_state_dict'])
                logger.info(f"Loaded pretrained discriminator from {pretrained_discriminator_ckpt}")

    adversarial_loss = nn.MSELoss() if gan_mode == 'ls' else nn.BCELoss()
    pixelwise_loss = nn.MSELoss()
    criterion = nn.MSELoss()

    def _recon(pred, target):
        if recon_loss == 'bce':
            return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target.clamp(0, 1))
        return pixelwise_loss(pred, target)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        if use_gan:
            discriminator.train()

        # -- DynamicConv (conv_type='dynamic') anneals its attention softmax temperature
        #    34 -> 1 once per epoch; other conv layers don't define step_temperature() so
        #    this is a no-op for them --
        for m in model.modules():
            if hasattr(m, "step_temperature"):
                m.step_temperature()
        if use_gan:
            for m in discriminator.modules():
                if hasattr(m, "step_temperature"):
                    m.step_temperature()

        # -- adversarial weight schedule (two-stage reconstruction warmup -> adversarial) --
        if not use_gan:
            alpha = 0.0
        elif two_stage:
            alpha = 0.0 if epoch <= warmup else alpha_max
        else:
            alpha = alpha_max
        adversarial = use_gan and alpha > 0

        # give the adversarial phase a fresh patience window when it switches on
        if two_stage and epoch == warmup + 1:
            epochs_no_improve = 0

        g_loss = None
        for spectra, real_images in train_loader:
            spectra = spectra.to(device)
            real_images = real_images.to(device)
            fake_images = model(spectra)

            optimizer_G.zero_grad()
            g_loss = _recon(fake_images, real_images)  # pixel term (== full loss when no adversary)
            if adversarial:
                # --- Train Discriminator ---
                optimizer_D.zero_grad()
                real_scores = discriminator(real_images)
                fake_scores = discriminator(fake_images.detach())
                d_loss = alpha * (adversarial_loss(real_scores, torch.ones_like(real_scores))
                                   + adversarial_loss(fake_scores, torch.zeros_like(fake_scores))) / 2
                d_loss.backward()
                optimizer_D.step()
                # --- Adversarial term for the Generator ---
                fake_scores = discriminator(fake_images)
                g_gan_loss = adversarial_loss(fake_scores, torch.ones_like(fake_scores))
                g_loss = alpha * g_gan_loss + (1 - alpha) * g_loss
            g_loss.backward()
            optimizer_G.step()

        # -- validation (pixel MSE, as Proposed.ipynb) --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for spectra, images in val_loader:
                outputs = model(spectra.to(device))
                val_loss += criterion(outputs, images.to(device)).item() * spectra.size(0)
        val_loss = val_loss / len_val
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], alpha={alpha:.2f}, "
            f"Train Loss: {g_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # -- checkpointing: latest every epoch, best on improvement --
        ckpt = {'generator_state_dict': model.state_dict(), 'epoch': epoch, 'loss': val_loss}
        if use_gan:
            ckpt['discriminator_state_dict'] = discriminator.state_dict()
        torch.save(ckpt, latest_ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, best_ckpt_path)
            logger.info(f"  Validation loss improved. Saving best model to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"  Validation loss did not improve for {epochs_no_improve} epochs (patience={patience}).")
            if epochs_no_improve >= patience:
                logger.info(f"  Early stopping triggered after {epoch + 1} epochs.")
                break

    return best_val_loss


# --------------------------------------------------------------------------- #
# Final test-set evaluation -- verbatim from every notebook's final cell
# (section 7), plus segmentation metrics (DSC/HD95/ASD/IoU/boundary-F/connected-component
# error/topology-validity rate -- see calculate_segmentation_metrics in utils.py for exact
# definitions -- both images binarized at `binarize_threshold`). Shared by evaluate.py and,
# at the end of a run, by train.py.
# --------------------------------------------------------------------------- #
def evaluate_on_test_set(model, test_loader, device, logger, binarize_threshold=0.5,
                          boundary_tolerance_px=2):
    model.eval()
    total_mse = 0.0
    psnr_list = []
    ssim_list = []
    dsc_list = []
    hd95_list = []
    asd_list = []
    iou_list = []
    boundary_f_list = []
    cc_error_list = []
    topology_valid_list = []
    n = 0
    n_undefined_distance = 0
    with torch.no_grad():
        for spectra, true_images in test_loader:
            spectra = spectra.to(device)
            true_images = true_images.to(device)
            pred = model(spectra).clamp(0, 1)
            total_mse += F.mse_loss(pred, true_images).item() * spectra.size(0)
            for i in range(spectra.size(0)):
                psnr_list.append(calculate_psnr(true_images[i].unsqueeze(0), pred[i].unsqueeze(0)))
                ssim_list.append(calculate_ssim(true_images[i], pred[i]))
                seg = calculate_segmentation_metrics(true_images[i], pred[i], threshold=binarize_threshold,
                                                      boundary_tolerance_px=boundary_tolerance_px)
                dsc_list.append(seg["dsc"])
                iou_list.append(seg["iou"])
                boundary_f_list.append(seg["boundary_f"])
                cc_error_list.append(seg["cc_error"])
                topology_valid_list.append(seg["topology_valid"])
                if np.isnan(seg["hd95"]):
                    n_undefined_distance += 1
                else:
                    hd95_list.append(seg["hd95"])
                    asd_list.append(seg["asd"])
            n += spectra.size(0)
    avg_mse = total_mse / n
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_dsc = float(np.mean(dsc_list))
    avg_hd95 = float(np.mean(hd95_list)) if hd95_list else float("nan")
    avg_asd = float(np.mean(asd_list)) if asd_list else float("nan")
    avg_iou = float(np.mean(iou_list))
    avg_boundary_f = float(np.mean(boundary_f_list))
    avg_cc_error = float(np.mean(cc_error_list))
    topology_validity_rate = float(np.mean(topology_valid_list))
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info("=" * 60)
    logger.info(f"  PSNR = {avg_psnr:.4f} dB")
    logger.info(f"  SSIM = {avg_ssim:.4f}")
    logger.info(f"  MSE  = {avg_mse:.6f}")
    logger.info(f"  DSC  = {avg_dsc:.4f}")
    logger.info(f"  HD95 = {avg_hd95:.4f} px")
    logger.info(f"  ASD  = {avg_asd:.4f} px")
    logger.info(f"  IoU  = {avg_iou:.4f}")
    logger.info(f"  Boundary-F ({boundary_tolerance_px}px) = {avg_boundary_f:.4f}")
    logger.info(f"  Connected-component error = {avg_cc_error:.4f}")
    logger.info(f"  Topology-validity rate = {topology_validity_rate:.4f}")
    logger.info(f"  params = {params_M:.2f} M")
    if n_undefined_distance:
        logger.info(f"  (HD95/ASD undefined for {n_undefined_distance}/{n} samples -- "
                     f"one of GT/predicted mask was empty; excluded from their averages)")
    logger.info("=" * 60)

    return {
        "mse": avg_mse, "psnr": avg_psnr, "ssim": avg_ssim,
        "dsc": avg_dsc, "hd95": avg_hd95, "asd": avg_asd, "iou": avg_iou,
        "boundary_f": avg_boundary_f, "cc_error": avg_cc_error,
        "topology_validity_rate": topology_validity_rate,
        "params_M": params_M,
    }


# --------------------------------------------------------------------------- #
# Qualitative GT-vs-predicted sample dump, called by evaluate.py after scoring
# (both standalone `python evaluate.py` and train.py's auto-eval at the end of a run).
# Saved at 300dpi so the PNGs are usable directly in a manuscript.
# --------------------------------------------------------------------------- #
SAMPLE_FIG_DPI = 300

# The dataset's 80 spectral points span 4-12 um, linearly spaced (per dataset collaborator).
SPECTRUM_WAVELENGTHS_UM = np.linspace(4, 12, 80)


def _plot_spectrum(ax, spectrum_np, title=None):
    wavelengths = SPECTRUM_WAVELENGTHS_UM if len(spectrum_np) == len(SPECTRUM_WAVELENGTHS_UM) else np.arange(len(spectrum_np))
    ax.plot(wavelengths, spectrum_np)
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Absorption")
    if title:
        ax.set_title(title)


def _plot_image(ax, image_np, title=None):
    ax.imshow(image_np, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def save_gt_vs_pred_samples(model, test_loader, device, out_dir, num_samples=8, logger=None):
    """Saves the first `num_samples` test-set items as:
      - <out_dir>/sample_00_spectrum.png, sample_00_GT.png, sample_00_Pred.png (individual
        single-panel figures per sample, one per postfix)
      - <out_dir>/sample_00.png, sample_01.png, ... (one combined 3-panel figure per sample)
      - <out_dir>/samples_grid.png (all samples combined into one grid figure)
    Uses dataset[i] directly (not the dataloader's shuffled batches) so sample order is
    stable across runs/checkpoints."""
    dataset = test_loader.dataset
    num_samples = min(num_samples, len(dataset))
    if num_samples <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            spectrum, gt_image = dataset[i]
            pred_image = model(spectrum.unsqueeze(0).to(device)).clamp(0, 1).squeeze(0).cpu()
            samples.append((
                spectrum.numpy(),
                gt_image.squeeze().numpy(),
                pred_image.squeeze().numpy(),
                calculate_psnr(gt_image.unsqueeze(0), pred_image.unsqueeze(0)),
                calculate_ssim(gt_image, pred_image),
            ))

    def _plot_row(axes_row, spectrum_np, gt_np, pred_np, psnr_value, ssim_value, header):
        _plot_spectrum(axes_row[0], spectrum_np, title="Input spectrum" if header else None)
        _plot_image(axes_row[1], gt_np, title="Ground truth" if header else None)
        _plot_image(axes_row[2], pred_np, title="Predicted" if header else None)
        axes_row[2].set_xlabel(f"PSNR={psnr_value:.2f}  SSIM={ssim_value:.3f}")

    for i, (spectrum_np, gt_np, pred_np, psnr_value, ssim_value) in enumerate(samples):
        # individual single-panel figures
        fig, ax = plt.subplots(figsize=(4, 3))
        _plot_spectrum(ax, spectrum_np, title=f"sample_{i:02d} spectrum")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_spectrum.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3))
        _plot_image(ax, gt_np, title=f"sample_{i:02d} GT")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_GT.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3.3))
        _plot_image(ax, pred_np, title=f"sample_{i:02d} Pred")
        ax.set_xlabel(f"PSNR={psnr_value:.2f}  SSIM={ssim_value:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_Pred.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        # combined 3-panel figure for this sample
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        _plot_row(axes, spectrum_np, gt_np, pred_np, psnr_value, ssim_value, header=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

    fig, axes_grid = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples), squeeze=False)
    for i, sample in enumerate(samples):
        _plot_row(axes_grid[i], *sample, header=(i == 0))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "samples_grid.png"), dpi=SAMPLE_FIG_DPI)
    plt.close(fig)

    if logger:
        logger.info(f"Saved {num_samples} GT-vs-predicted samples (individual + combined, {SAMPLE_FIG_DPI}dpi) to {out_dir}")


# --------------------------------------------------------------------------- #
# Qualitative GT-vs-roundtrip-predicted spectrum dump, called by
# evaluate_spectrum_roundtrip.py after computing the full test-set spectrum arrays.
# Unlike save_gt_vs_pred_samples above (which re-runs the model on dataset[i]), this just
# slices the first `num_samples` rows out of the already-computed (N, L) arrays -- no
# re-running needed, and sample order matches the saved .npy files exactly.
# --------------------------------------------------------------------------- #
def save_spectrum_roundtrip_samples(gt_spectra, pred_spectra_raw, pred_spectra_bin, pred_spectra_forward,
                                     gt_images, inverse_pred_images, out_dir, num_samples=8,
                                     wavelengths=None, logger=None):
    """Saves the first `num_samples` samples as:
      - sample_00_GT.png / _ForwardPred.png / _PredRaw.png / _PredBin.png (individual
        single-curve spectrum figures)
      - sample_00_GTImage.png / _InversePredImage.png (individual images: the dataset's true
        image, and the inverse generator's image predicted from the GT spectrum)
      - sample_00.png (combined spectrum overlay: Ground-Truth / Forward Predicted /
        Roundtrip Predicted, each labeled with its RMSE vs. GT -- "Forward Predicted" is the
        forward model applied directly to the GT image; "Roundtrip Predicted" is the forward
        model applied to the *binarized* inverse-predicted image, i.e. the full round trip)
      - sample_00_panels.png (3-panel: GT image | inverse-predicted image | the same spectrum overlay)
      - samples_grid.png (all samples' spectrum overlay combined into one grid figure)

    gt_spectra, pred_spectra_raw, pred_spectra_bin, pred_spectra_forward: (N, L) arrays (as
    saved to spectrum_gt.npy / spectrum_pred_raw.npy / spectrum_pred_bin.npy /
    spectrum_pred_forward.npy). gt_images, inverse_pred_images: (N, 1, H, W) arrays."""
    if wavelengths is None:
        wavelengths = SPECTRUM_WAVELENGTHS_UM
    num_samples = min(num_samples, gt_spectra.shape[0])
    if num_samples <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _plot_curve(ax, y, title, color, xlabel="Wavelength (μm)"):
        ax.plot(wavelengths, y, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absorption")
        if title:
            ax.set_title(title)

    def _plot_overlay(ax, gt, fwd_pred, roundtrip_pred, rmse_fwd, rmse_rt, title=None):
        ax.plot(wavelengths, gt, label="Ground-Truth", color="tab:blue")
        ax.plot(wavelengths, fwd_pred, label=f"Forward Predicted (RMSE={rmse_fwd:.4f})", color="tab:orange")
        ax.plot(wavelengths, roundtrip_pred, label=f"Roundtrip Predicted (RMSE={rmse_rt:.4f})", color="tab:green")
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Absorption")
        ax.legend(fontsize=7)
        if title:
            ax.set_title(title)

    for i in range(num_samples):
        gt, pred_raw, pred_bin, pred_fwd = (gt_spectra[i], pred_spectra_raw[i],
                                             pred_spectra_bin[i], pred_spectra_forward[i])
        rmse_raw, rmse_bin, rmse_fwd = _rmse(gt, pred_raw), _rmse(gt, pred_bin), _rmse(gt, pred_fwd)

        # individual spectrum curves
        fig, ax = plt.subplots(figsize=(4, 3))
        _plot_curve(ax, gt, f"sample_{i:02d} Ground-Truth", "tab:blue")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_GT.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 3.3))
        _plot_curve(ax, pred_fwd, f"sample_{i:02d} Forward Predicted", "tab:orange",
                    xlabel=f"Wavelength (μm)\nRMSE={rmse_fwd:.4f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_ForwardPred.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 3.3))
        _plot_curve(ax, pred_raw, f"sample_{i:02d} Pred (raw)", "tab:red",
                    xlabel=f"Wavelength (μm)\nRMSE={rmse_raw:.4f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_PredRaw.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 3.3))
        _plot_curve(ax, pred_bin, f"sample_{i:02d} Roundtrip Predicted", "tab:green",
                    xlabel=f"Wavelength (μm)\nRMSE={rmse_bin:.4f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_PredBin.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        # individual images
        fig, ax = plt.subplots(figsize=(3, 3))
        _plot_image(ax, gt_images[i].squeeze(), title=f"sample_{i:02d} GT image")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_GTImage.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3, 3))
        _plot_image(ax, inverse_pred_images[i].squeeze(), title=f"sample_{i:02d} Inverse Pred image")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_InversePredImage.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        # combined spectrum overlay: Ground-Truth / Forward Predicted / Roundtrip Predicted
        fig, ax = plt.subplots(figsize=(5, 3.5))
        _plot_overlay(ax, gt, pred_fwd, pred_bin, rmse_fwd, rmse_bin, title=f"sample_{i:02d}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

        # 3-panel: GT image | inverse-predicted image | spectrum overlay
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        _plot_image(axes[0], gt_images[i].squeeze(), title="GT image")
        _plot_image(axes[1], inverse_pred_images[i].squeeze(), title="Inverse Pred image")
        _plot_overlay(axes[2], gt, pred_fwd, pred_bin, rmse_fwd, rmse_bin, title="Spectrum")
        fig.suptitle(f"sample_{i:02d}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i:02d}_panels.png"), dpi=SAMPLE_FIG_DPI)
        plt.close(fig)

    fig, axes = plt.subplots(num_samples, 1, figsize=(5, 3 * num_samples), squeeze=False)
    for i in range(num_samples):
        gt, pred_bin, pred_fwd = gt_spectra[i], pred_spectra_bin[i], pred_spectra_forward[i]
        rmse_fwd, rmse_bin = _rmse(gt, pred_fwd), _rmse(gt, pred_bin)
        _plot_overlay(axes[i, 0], gt, pred_fwd, pred_bin, rmse_fwd, rmse_bin, title=f"sample_{i:02d}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "samples_grid.png"), dpi=SAMPLE_FIG_DPI)
    plt.close(fig)

    if logger:
        logger.info(f"Saved {num_samples} GT/Forward-Predicted/Roundtrip-Predicted spectrum+image "
                     f"samples (individual + combined + panels, {SAMPLE_FIG_DPI}dpi) to {out_dir}")
