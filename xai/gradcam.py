"""
xai/gradcam.py — Core GradCAM / GradCAM++ Engine
=================================================
Generates class-activation heatmaps for the DRISHYAMTLModel.

The model returns a tuple (logits, masks). The pytorch-grad-cam library
expects a single tensor output, so we wrap the model with ClassificationWrapper
that runs only the encoder → pool → classifier path (skipping the decoder).

Usage:
    from xai.gradcam import generate_heatmap, get_target_layer, ClassificationWrapper
    
    wrapper = ClassificationWrapper(model)
    target_layer = get_target_layer(model, backbone_name)
    heatmap = generate_heatmap(wrapper, target_layer, input_tensor, target_class=2)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ClassificationWrapper(nn.Module):
    """
    Wraps DRISHYAMTLModel so that forward() runs only the classification
    path: encoder → global_pool → classifier.

    The segmentation decoder is completely bypassed. This is important for
    two reasons:
      1. pytorch-grad-cam expects a single tensor output.
      2. Skipping the decoder avoids ~40% unnecessary computation and
         keeps the computation graph minimal for cleaner gradients.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Run only the encoder → pool → classifier path (no decoder)
        features = self.model.unet.encoder(x)
        pooled = self.model.global_pool(features[-1]).flatten(1)
        logits = self.model.classifier(pooled)
        return logits


# ─── Target Layer Auto-Detection ──────────────────────────────────────────────

# Mapping from backbone name prefix to the attribute path for the
# deepest convolutional block (the best layer for GradCAM).
_BACKBONE_TARGET_LAYERS = {
    "tf_efficientnet": "model.blocks[-1]",    # EfficientNet family (MBConv)
    "efficientnet":    "model.blocks[-1]",    # EfficientNet via timm
    "convnext":        "model.stages[-1]",    # ConvNeXt family
    "regnety":         "model.s4",            # RegNetY Stage 4
    "regnetx":         "model.s4",            # RegNetX Stage 4
}


def get_target_layer(model, backbone_name="tu-tf_efficientnet_b4.ns_jft_in1k"):
    """
    Automatically detects and returns the correct target layer for GradCAM
    based on the backbone architecture name.

    The target layer is the deepest convolutional block in the encoder —
    the layer just before global average pooling. This layer has the richest
    spatial-semantic features for heatmap generation.
    
    Args:
        model: DRISHYAMTLModel instance (with model.unet.encoder)
        backbone_name: timm backbone name string (e.g., 'tu-tf_efficientnet_b4.ns_jft_in1k')

    Returns:
        nn.Module: The target layer for GradCAM hook registration
        
    Raises:
        ValueError: If the backbone architecture is not recognized
    """
    # The SMP encoder wraps the timm model inside model.unet.encoder.model
    encoder_model = model.unet.encoder.model

    # Strip the 'tu-' prefix that SMP adds
    clean_name = backbone_name.replace("tu-", "")

    # Find matching backbone
    for prefix, attr_path in _BACKBONE_TARGET_LAYERS.items():
        if prefix in clean_name:
            # Navigate the attribute path (e.g., "model.blocks[-1]")
            # We use the encoder_model as the root since we already extracted it
            parts = attr_path.replace("model.", "").split(".")
            layer = encoder_model
            for part in parts:
                if "[" in part:
                    # Handle indexing like blocks[-1]
                    attr_name = part.split("[")[0]
                    index = int(part.split("[")[1].rstrip("]"))
                    layer = getattr(layer, attr_name)[index]
                else:
                    layer = getattr(layer, part)
            return layer

    raise ValueError(
        f"Unrecognized backbone '{backbone_name}'. "
        f"Supported prefixes: {list(_BACKBONE_TARGET_LAYERS.keys())}"
    )


# ─── Fundus FOV Masking ──────────────────────────────────────────────────────

def create_fundus_fov_mask(rgb_image_01, threshold=10):
    """
    Create a binary mask of the circular fundus field-of-view.
    Pixels inside the retinal disc → 1, black background → 0.

    Args:
        rgb_image_01: np.ndarray (H, W, 3), float32 in [0, 1]
        threshold: int — grayscale intensity threshold (0–255) to separate
                   the retina from the black background

    Returns:
        mask: np.ndarray (H, W), uint8 with values 0 or 255
    """
    gray = cv2.cvtColor((rgb_image_01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological close to fill small holes inside the retinal disc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component (the retinal disc)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Label 0 is background; find the largest among labels 1..N
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Optional slight erosion to avoid edge artifacts
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    return mask


def _apply_fov_mask(heatmap, fov_mask):
    """
    Zero out heatmap values outside the fundus field-of-view and
    re-normalize the remaining values to [0, 1].

    This prevents black-background corners from dominating the color scale.

    Args:
        heatmap: np.ndarray (H, W), float32 in [0, 1] — raw GradCAM output
        fov_mask: np.ndarray (H_img, W_img), uint8 — 255 inside retina, 0 outside

    Returns:
        masked: np.ndarray (H, W), float32 in [0, 1] — masked and re-normalized
    """
    h, w = heatmap.shape[:2]

    # Resize mask to match heatmap dimensions
    mask_resized = cv2.resize(fov_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 127).astype(np.float32)

    # Zero out background
    masked = heatmap * binary_mask

    # Re-normalize to [0, 1] using only the retinal region
    max_val = masked.max()
    if max_val > 1e-7:
        masked = masked / max_val

    return masked


# ─── Heatmap Generation ──────────────────────────────────────────────────────

def generate_heatmap(
    wrapper,
    target_layers,
    input_tensor,
    target_class=None,
    method="gradcam++",
    fov_mask=None,
):
    """
    Generate a GradCAM or GradCAM++ heatmap for a single input image.

    Args:
        wrapper: ClassificationWrapper instance (returns logits only)
        target_layers: list of nn.Module — the layer(s) to hook for GradCAM
        input_tensor: torch.Tensor of shape (1, 3, H, W), already normalized
        target_class: int or None. If None, uses the model's predicted class (argmax)
        method: 'gradcam' or 'gradcam++' (default: 'gradcam++')
        fov_mask: np.ndarray (H, W) uint8 or None. If provided, the heatmap is
                  masked to the fundus field-of-view before normalization so that
                  black-background corners do not dominate the color scale.

    Returns:
        heatmap: np.ndarray of shape (H, W), float32 in [0, 1] range
    """
    # Select GradCAM method
    if method.lower() in ("gradcam++", "gradcampp", "gradcam_plus_plus"):
        cam_class = GradCAMPlusPlus
    elif method.lower() == "gradcam":
        cam_class = GradCAM
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'gradcam' or 'gradcam++'.")

    # If no target class specified, use the model's own prediction
    if target_class is None:
        with torch.no_grad():
            logits = wrapper(input_tensor)
            target_class = int(torch.argmax(logits, dim=1).item())

    targets = [ClassifierOutputTarget(target_class)]

    # Construct the CAM object and generate the heatmap
    with cam_class(model=wrapper, target_layers=target_layers) as cam:
        # cam() returns shape (batch, H, W) — we take the first (and only) image
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0, :]  # (H, W) float32 in [0, 1]

    # Mask out the black background so corners don't dominate the heatmap
    if fov_mask is not None:
        heatmap = _apply_fov_mask(heatmap, fov_mask)

    return heatmap


def generate_heatmap_all_classes(
    wrapper,
    target_layers,
    input_tensor,
    num_classes=5,
    method="gradcam++",
    fov_mask=None,
):
    """
    Generate GradCAM heatmaps for ALL 5 ICDR classes (0–4) on a single image.
    Useful for understanding what the model focuses on for each severity level.

    Args:
        wrapper: ClassificationWrapper instance
        target_layers: list of nn.Module
        input_tensor: torch.Tensor of shape (1, 3, H, W)
        num_classes: number of classes (default: 5 for ICDR grades 0–4)
        method: 'gradcam' or 'gradcam++'
        fov_mask: np.ndarray or None — fundus field-of-view mask

    Returns:
        dict: {class_idx: heatmap_ndarray} for each class 0 through num_classes-1
    """
    heatmaps = {}
    for cls_idx in range(num_classes):
        heatmaps[cls_idx] = generate_heatmap(
            wrapper, target_layers, input_tensor,
            target_class=cls_idx, method=method,
            fov_mask=fov_mask,
        )
    return heatmaps
