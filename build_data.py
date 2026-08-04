"""
build_data.py
==============
Forward model (spectrum predictor, used only to synthesize spectra for the XPIE
mask images) + the two Dataset classes + the loader-builder function.

This is verbatim from every notebook's sections 2 and 4 (all 11 notebooks are
byte-identical here) -- only refactored into functions/args instead of module-level
globals and hardcoded Windows paths.
"""
import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, UnidentifiedImageError
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
from torchvision import transforms


# --------------------------------------------------------------------------- #
# Forward model (spectrum simulator) -- verbatim from section 2
# --------------------------------------------------------------------------- #
class PretrainedModelToSpectrum(nn.Module):
    def __init__(self, pretrained=True):
        super(PretrainedModelToSpectrum, self).__init__()

        # Load the pretrained MobileNetV2 model
        self.base_model = models.mobilenet_v2(pretrained=pretrained)

        # Modify the first convolutional layer to accept 1 channel (grayscale)
        self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Replace the classifier with custom layers
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.base_model.last_channel, 512),
        )

        # Final fully connected layer for spectrum prediction
        self.fc = nn.Linear(512, 80)

        # 1D convolution layer to smooth the spectrum
        self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model(x)
        x = self.fc(x)

        # Add 1D convolution for spectrum refinement
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1d1(x)
        x = self.conv1d2(x)
        x = x.squeeze(1)  # Remove the channel dimension

        return x


def load_forward_model(forward_model_path: str, device: torch.device) -> PretrainedModelToSpectrum:
    """Loads + freezes the forward (image -> spectrum) simulator used to synthesize
    spectra for the augmentation-only XPIE dataset. Verbatim from section 2."""
    forward_model = PretrainedModelToSpectrum(pretrained=True).to(device)
    forward_model.load_state_dict(torch.load(forward_model_path, map_location=device))
    forward_model.eval()
    print("Loading: Forward_Simulator_SKv3_SmoothArch_best_model")
    return forward_model


# --------------------------------------------------------------------------- #
# Datasets -- verbatim from section 4
# --------------------------------------------------------------------------- #
class InverseModelDataset(Dataset):
    """XPIE-mask dataset: images only; spectra are synthesized on-the-fly via the
    (frozen) forward model. Optional random rotate / zoom / noise augmentation."""

    def __init__(self, image_dir, transform=None, augment=False, forward_model=None, device=None):
        self.image_dir = image_dir
        # Filter out non-image files (e.g., 'desktop.ini')
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform
        self.augment = augment
        self.forward_model = forward_model
        self.device = device

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load and process image
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        try:
            image = Image.open(img_name).convert('L')  # Convert to grayscale
        except UnidentifiedImageError:
            raise RuntimeError(f"Cannot open the image file {img_name}")

        if self.transform:
            image = self.transform(image)

        # Add optional augmentation
        if self.augment:
            image = self.apply_augmentation(image)

        # Forward pass through the forward model to get the spectrum
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        spectrum = self.forward_model(image).squeeze(0).cpu()  # Get spectrum and remove batch dimension

        # Return spectrum and the original image
        return spectrum, image.squeeze(0).cpu()

    # Define augmentation operations (random rotation, zoom, noise)
    def apply_augmentation(self, image):
        if random.random() > 0.5:
            angle = random.uniform(-180, 180)  # Random rotation between -180 to 180 degrees
            image = transforms.functional.rotate(image, angle)

        if random.random() > 0.5:
            scale = random.uniform(0.25, 5)  # Random zoom between 90% to 110%
            image = transforms.functional.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)

        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.05  # Add random noise
            image = image + noise
            image = torch.clamp(image, 0, 1)  # Ensure pixel values remain between 0 and 1

        return image


class ImageSpectrumDataset(Dataset):
    """Main FDTD dataset: (image, spectrum) pairs read from a CSV spectrum file."""

    def __init__(self, image_dir, spectrum_file, transform=None):
        # Load the spectrum data from the .xlsx file
        # self.spectrum_data = pd.read_excel(spectrum_file)
        self.spectrum_data = pd.read_csv(spectrum_file)

        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.spectrum_data.iloc[:, 0].str.replace('results', 'index').str.replace('.mat', '.png').str.replace('-Excel', '')

        # self.image_names = self.spectrum_data.iloc[:, 0].str.replace('.mat', '.png')  # Base image names without extension
        self.spectra = self.spectrum_data.iloc[:, 1:].values  # Spectrum data (502, 102)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the base name of the image
        base_image_name = self.image_names[idx]

        # Check if the image exists in .png or .bmp format
        png_path = os.path.join(self.image_dir, base_image_name)
        bmp_path = os.path.join(self.image_dir, base_image_name + '.bmp')

        if os.path.exists(png_path):
            img_name = png_path
        elif os.path.exists(bmp_path):
            img_name = bmp_path
        else:
            raise FileNotFoundError(f"No image file found for {base_image_name} in .png or .bmp format.")

        # Load the image and convert it to grayscale
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        # Get the corresponding spectrum
        spectrum = torch.tensor(self.spectra[idx], dtype=torch.float32)

        return spectrum, image


# Define image transformations -- verbatim from section 4
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def build_loaders_abl(use_xpie, seed, batch_size, xpie_data_path, data_imag_path,
                       spectrum_file, forward_model, device):
    """Reproduces the Proposed.ipynb data pipeline with the XPIE forward-augmented
    set toggled on/off. use_xpie=True -> XPIE(aug)+FDTD; False -> FDTD only, no aug.
    The FDTD test split is fixed (seed 42) so every variant is scored on the SAME
    held-out test set. Verbatim from section 4."""
    dataset2 = ImageSpectrumDataset(spectrum_file=spectrum_file, image_dir=data_imag_path, transform=IMAGE_TRANSFORM)
    train_size2 = int(0.8 * len(dataset2))
    val_size2 = int(0.1 * len(dataset2))
    test_size = len(dataset2) - train_size2 - val_size2
    train_dataset2, val_dataset2, test_dataset = random_split(
        dataset2, [train_size2, val_size2, test_size], generator=torch.Generator().manual_seed(42))

    if use_xpie:
        dataset1 = InverseModelDataset(
            image_dir=xpie_data_path, transform=IMAGE_TRANSFORM, augment=True,
            forward_model=forward_model, device=device,
        )
        train_size1 = int(0.9 * len(dataset1))
        val_size1 = len(dataset1) - train_size1
        train_dataset1, val_dataset1 = random_split(
            dataset1, [train_size1, val_size1], generator=torch.Generator().manual_seed(seed))
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])
    else:
        train_dataset = train_dataset2
        val_dataset = val_dataset2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset)


def build_test_loader(batch_size, data_imag_path, spectrum_file):
    """The held-out FDTD test split depends only on `dataset2` + the fixed seed=42
    3-way random_split in `build_loaders_abl` -- it does NOT depend on use_xpie.
    This helper reproduces just that split (test_size = len - 0.8*len - 0.1*len,
    generator seed 42) without touching the XPIE dataset/forward model, so
    evaluate.py can score a checkpoint without paying for the training-only data."""
    dataset2 = ImageSpectrumDataset(spectrum_file=spectrum_file, image_dir=data_imag_path, transform=IMAGE_TRANSFORM)
    train_size2 = int(0.8 * len(dataset2))
    val_size2 = int(0.1 * len(dataset2))
    test_size = len(dataset2) - train_size2 - val_size2
    _, _, test_dataset = random_split(
        dataset2, [train_size2, val_size2, test_size], generator=torch.Generator().manual_seed(42))
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
