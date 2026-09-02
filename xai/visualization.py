"""
xai/visualization.py — Heatmap Overlay & Comparison Panel Rendering
===================================================================
All rendering and image-saving logic for the XAI module.

Usage:
    from xai.visualization import overlay_heatmap, create_comparison_panel, save_outputs
    
    overlay = overlay_heatmap(rgb_01, heatmap, alpha=0.4)
    panel = create_comparison_panel(rgb_01, heatmap, overlay)
    save_outputs(rgb_01, heatmap, overlay, panel, "outputs/xai", "IDRiD_001")
"""

import cv2
import numpy as np

# ─── ICDR Grade Labels ───────────────────────────────────────────────────────

GRADE_LABELS = {
    0: "Grade 0: No DR",
    1: "Grade 1: Mild NPDR",
    2: "Grade 2: Moderate NPDR",
    3: "Grade 3: Severe NPDR",
    4: "Grade 4: Proliferative DR",
}

# Supported OpenCV colormap constants
_COLORMAPS = {
    "jet":     cv2.COLORMAP_JET,
    "turbo":   cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma":   cv2.COLORMAP_MAGMA,
    "hot":     cv2.COLORMAP_HOT,
}


# ─── Overlay ──────────────────────────────────────────────────────────────────

def overlay_heatmap(rgb_image_01, heatmap, alpha=0.4, colormap="jet", fov_mask=None):
    """
    Blend a GradCAM heatmap onto the original fundus image.

    Args:
        rgb_image_01: np.ndarray (H, W, 3), float32 in [0, 1] — the original RGB image
        heatmap: np.ndarray (H, W), float32 in [0, 1] — the GradCAM heatmap
        alpha: float — blend factor for the heatmap (0 = invisible, 1 = fully opaque)
        colormap: str — OpenCV colormap name: 'jet', 'turbo', 'inferno', 'magma', 'hot'
        fov_mask: np.ndarray (H, W), uint8 or None — mask where retina is present

    Returns:
        overlay_rgb: np.ndarray (H, W, 3), uint8 RGB — the blended overlay image
        heatmap_colored: np.ndarray (H, W, 3), uint8 RGB — the heatmap alone (colorized)
    """
    if colormap not in _COLORMAPS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {list(_COLORMAPS.keys())}")

    # Resize heatmap to match image dimensions if needed
    h, w = rgb_image_01.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert heatmap to uint8 and apply colormap (OpenCV produces BGR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, _COLORMAPS[colormap])
    heatmap_colored = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Convert original image to uint8
    img_uint8 = np.uint8(255 * rgb_image_01)

    # Blend
    overlay_rgb = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_colored, alpha, 0)

    # If FOV mask is provided, keep background black outside the retina
    if fov_mask is not None:
        mask_resized = cv2.resize(fov_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bg_pixels = (mask_resized <= 127)
        heatmap_colored[bg_pixels] = 0
        overlay_rgb[bg_pixels] = img_uint8[bg_pixels]

    return overlay_rgb, heatmap_colored


# ─── Comparison Panel ─────────────────────────────────────────────────────────

def _add_label(image, text, font_scale=0.55, thickness=1):
    """Add a text label at the bottom of an image with a dark background strip."""
    h, w = image.shape[:2]
    labeled = image.copy()

    # Dark strip at the bottom
    strip_h = 30
    cv2.rectangle(labeled, (0, h - strip_h), (w, h), (0, 0, 0), -1)

    # White text centered
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - strip_h // 2 + text_size[1] // 2
    cv2.putText(labeled, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return labeled


def create_comparison_panel(rgb_image_01, heatmap, overlay_rgb, heatmap_colored=None, fov_mask=None, gt_class=None):
    """
    Create a 3-panel horizontal comparison image with clinical explainability footer:
    [ Original Fundus | GradCAM++ Heatmap | Saliency Overlay ]
    + Clinical Explanation Legend Strip at bottom.

    Args:
        rgb_image_01: np.ndarray (H, W, 3), float32 [0,1] — original image
        heatmap: np.ndarray (H, W), float32 [0,1] — raw grayscale heatmap
        overlay_rgb: np.ndarray (H, W, 3), uint8 — blended overlay
        heatmap_colored: np.ndarray (H, W, 3), uint8 or None — if None, generated with JET
        fov_mask: np.ndarray (H, W), uint8 or None — fundus field of view mask
        gt_class: int or None — ground truth ICDR class (0–4)

    Returns:
        panel: np.ndarray, uint8 RGB — the annotated comparison panel
    """
    img_uint8 = np.uint8(255 * rgb_image_01)

    if heatmap_colored is None:
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        if fov_mask is not None:
            mask_resized = cv2.resize(fov_mask, (heatmap_colored.shape[1], heatmap_colored.shape[0]), interpolation=cv2.INTER_NEAREST)
            heatmap_colored[mask_resized <= 127] = 0

    # Labels for individual panels
    orig_label = "Original Fundus" if gt_class is None else f"Original (GT: Grade {gt_class})"
    panel_original = _add_label(img_uint8, orig_label)
    panel_heatmap = _add_label(heatmap_colored, "GradCAM++ Saliency")
    panel_overlay = _add_label(overlay_rgb, "Lesion Attention Overlay")

    # Join 3 panels with a 4-pixel dark gap
    gap = np.zeros((panel_original.shape[0], 4, 3), dtype=np.uint8)
    core_panel = np.hstack([panel_original, gap, panel_heatmap, gap, panel_overlay])

    # Add clinical explainability footer banner
    pw = core_panel.shape[1]
    footer_h = 28
    footer = np.zeros((footer_h, pw, 3), dtype=np.uint8)

    legend_text = "[EXPLAINABILITY] RED: High DR Lesion Detection (Exudates/Hemorrhages/MA) | YELLOW: Moderate Saliency | BLUE: Normal Retina"
    cv2.putText(footer, legend_text, (15, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 255), 1, cv2.LINE_AA)

    # Combine core panel and explainability footer
    divider = np.zeros((2, pw, 3), dtype=np.uint8)
    divider[:, :] = (50, 50, 50)
    full_panel = np.vstack([core_panel, divider, footer])

    return full_panel


def add_prediction_banner(overlay_rgb, pred_class, confidence_pct, gt_class=None):
    """
    Draw a semi-transparent prediction banner at the top of the overlay image.
    Shows: "Grade 2: Moderate NPDR | Conf: 94.3% [GT: Grade 2 (MATCH)]"

    Args:
        overlay_rgb: np.ndarray (H, W, 3), uint8 RGB
        pred_class: int (0–4)
        confidence_pct: float (e.g. 94.3)
        gt_class: int or None (ground truth class)

    Returns:
        labeled: np.ndarray (H, W, 3), uint8 RGB — overlay with banner
    """
    labeled = overlay_rgb.copy()
    h, w = labeled.shape[:2]

    # Semi-transparent banner
    banner_h = 36
    banner_overlay = labeled.copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(banner_overlay, 0.65, labeled, 0.35, 0, labeled)

    # Text
    grade_text = GRADE_LABELS.get(pred_class, f"Grade {pred_class}")
    if gt_class is not None:
        status_tag = "MATCH" if pred_class == gt_class else f"DIFF (GT:{gt_class})"
        full_text = f"{grade_text} | Conf: {confidence_pct:.1f}% [{status_tag}]"
    else:
        full_text = f"{grade_text} | Confidence: {confidence_pct:.1f}%"

    cv2.putText(labeled, full_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return labeled


# ─── Output Saving ────────────────────────────────────────────────────────────

def save_outputs(rgb_image_01, heatmap, overlay_rgb, comparison_panel,
                 output_dir, image_name, heatmap_colored=None):
    """
    Save all XAI output images to the specified directory.

    Saves:
        - {image_name}_heatmap.png    — raw colorized heatmap
        - {image_name}_overlay.png    — heatmap blended on fundus
        - {image_name}_comparison.png — 3-panel side-by-side

    Args:
        rgb_image_01: np.ndarray (H, W, 3), float32 [0,1]
        heatmap: np.ndarray (H, W), float32 [0,1]
        overlay_rgb: np.ndarray (H, W, 3), uint8 RGB
        comparison_panel: np.ndarray, uint8 RGB
        output_dir: str — directory to save into
        image_name: str — base name (without extension)
        heatmap_colored: np.ndarray or None — if None, generated with JET
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if heatmap_colored is None:
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Convert RGB → BGR for cv2.imwrite
    cv2.imwrite(
        os.path.join(output_dir, f"{image_name}_heatmap.png"),
        cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{image_name}_overlay.png"),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{image_name}_comparison.png"),
        cv2.cvtColor(comparison_panel, cv2.COLOR_RGB2BGR)
    )
