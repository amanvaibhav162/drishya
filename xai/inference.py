"""
xai/inference.py — Model Loading, Preprocessing & Heatmap Pipeline Runner
==========================================================================
Handles the full pipeline: load model → preprocess image → run inference →
generate heatmap → call visualization → save outputs.

Usage (CLI):
    # Single image
    python -m xai.inference \\
        --image path/to/fundus.jpg \\
        --checkpoint models/teacher_tu-tf_efficientnet_b4.ns_jft_in1k_fold0_best.pth \\
        --output-dir outputs/xai/

    # Batch mode
    python -m xai.inference \\
        --image-dir data/processed/dr_images/ \\
        --checkpoint models/teacher_tu-tf_efficientnet_b4.ns_jft_in1k_fold0_best.pth \\
        --output-dir outputs/xai_batch/

Usage (Python):
    from xai.inference import load_teacher_model, preprocess_image, run_single_image
    
    model, device = load_teacher_model("models/teacher_...pth")
    result = run_single_image("fundus.jpg", model, device)
"""

import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from xai.model_mtl import DRISHYAMTLModel
from xai.gradcam import ClassificationWrapper, get_target_layer, generate_heatmap, create_fundus_fov_mask
from xai.visualization import (
    overlay_heatmap,
    create_comparison_panel,
    add_prediction_banner,
    save_outputs,
    GRADE_LABELS,
)

# ImageNet normalization constants (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_BACKBONE = "tu-tf_efficientnet_b4.ns_jft_in1k"
DEFAULT_IMG_SIZE = 384


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_teacher_model(
    checkpoint_path,
    backbone_name=DEFAULT_BACKBONE,
    device=None,
):
    """
    Load a trained DRISHYAMTLModel teacher checkpoint.

    Args:
        checkpoint_path: str — path to the .pth checkpoint file
        backbone_name: str — timm backbone name (must match what was used during training)
        device: torch.device or None (auto-detects GPU/CPU)

    Returns:
        model: DRISHYAMTLModel in eval mode
        device: torch.device that the model is on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the architecture (pretrained=False since we load our own weights)
    model = DRISHYAMTLModel(backbone_name=backbone_name, pretrained=False)

    # Load trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"✔ Loaded checkpoint: {checkpoint_path}")

    model.to(device)
    model.eval()
    print(f"✔ Model ready on {device}")

    return model, device


# ─── Image Preprocessing ─────────────────────────────────────────────────────

def preprocess_image(image_path, target_size=DEFAULT_IMG_SIZE):
    """
    Load and preprocess a fundus image for model inference.

    Steps:
        1. Read image (BGR) via OpenCV
        2. Resize to target_size × target_size
        3. Convert BGR → RGB, normalize to [0, 1]
        4. Apply ImageNet normalization
        5. Convert to tensor (1, 3, H, W)

    Args:
        image_path: str — path to the fundus image
        target_size: int — resize dimension (default: 384)

    Returns:
        input_tensor: torch.Tensor (1, 3, H, W), float32, ImageNet-normalized
        rgb_image_01: np.ndarray (H, W, 3), float32 in [0, 1] — for overlay rendering
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Resize
    img_bgr = cv2.resize(img_bgr, (target_size, target_size))

    # BGR → RGB, normalize to [0, 1]
    rgb_image_01 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # To tensor and apply ImageNet normalization
    input_tensor = torch.from_numpy(rgb_image_01).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std

    return input_tensor, rgb_image_01


# ─── Single Image Pipeline ───────────────────────────────────────────────────

def run_single_image(
    image_path,
    model,
    device,
    backbone_name=DEFAULT_BACKBONE,
    method="gradcam++",
    alpha=0.4,
    colormap="jet",
    output_dir=None,
    target_class=None,
    gt_class=None,
):
    """
    Full end-to-end pipeline for a single fundus image:
    preprocess → inference → GradCAM → overlay → save.

    Args:
        image_path: str — path to fundus image
        model: DRISHYAMTLModel — loaded and in eval mode
        device: torch.device
        backbone_name: str — for target layer detection
        method: str — 'gradcam' or 'gradcam++'
        alpha: float — overlay blend factor
        colormap: str — 'jet', 'turbo', 'inferno', 'magma', 'hot'
        output_dir: str or None — if provided, saves output images here
        target_class: int or None — if None, uses model's predicted class
        gt_class: int or None — ground truth class from labels.csv

    Returns:
        dict with keys:
            'predicted_class': int
            'confidence_pct': float
            'grade_label': str
            'gt_class': int or None
            'heatmap': np.ndarray (H, W)
            'overlay': np.ndarray (H, W, 3) uint8
            'comparison_panel': np.ndarray uint8
    """
    # 1. Preprocess
    input_tensor, rgb_image_01 = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    fov_mask = create_fundus_fov_mask(rgb_image_01)

    # 2. Forward pass — get prediction
    with torch.no_grad():
        logits, masks = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence_pct = float(probs[pred_class]) * 100.0

    # Use the model's prediction if no target class specified
    if target_class is None:
        target_class = pred_class

    grade_label = GRADE_LABELS.get(pred_class, f"Grade {pred_class}")
    gt_str = f" | GT: Grade {gt_class} {'(MATCH)' if pred_class == gt_class else '(DIFF)'}" if gt_class is not None else ""
    print(f"  Prediction: {grade_label} | Confidence: {confidence_pct:.1f}%{gt_str}")

    # 3. GradCAM heatmap
    wrapper = ClassificationWrapper(model)
    target_layer = get_target_layer(model, backbone_name)
    heatmap = generate_heatmap(
        wrapper, [target_layer], input_tensor,
        target_class=target_class, method=method,
        fov_mask=fov_mask,
    )

    # 4. Visualization
    overlay_rgb, heatmap_colored = overlay_heatmap(
        rgb_image_01, heatmap, alpha=alpha, colormap=colormap, fov_mask=fov_mask
    )
    overlay_with_banner = add_prediction_banner(overlay_rgb, pred_class, confidence_pct, gt_class=gt_class)
    comparison = create_comparison_panel(
        rgb_image_01, heatmap, overlay_with_banner, heatmap_colored, fov_mask=fov_mask, gt_class=gt_class
    )

    # 5. Save outputs
    if output_dir is not None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_outputs(
            rgb_image_01, heatmap, overlay_with_banner, comparison,
            output_dir, image_name, heatmap_colored,
        )
        print(f"  ✔ Saved to {output_dir}/{image_name}_*.png")

    return {
        "predicted_class": pred_class,
        "confidence_pct": confidence_pct,
        "grade_label": grade_label,
        "gt_class": gt_class,
        "heatmap": heatmap,
        "overlay": overlay_with_banner,
        "comparison_panel": comparison,
    }


# ─── Batch Pipeline ──────────────────────────────────────────────────────────

def run_batch(
    image_dir,
    model,
    device,
    backbone_name=DEFAULT_BACKBONE,
    method="gradcam++",
    alpha=0.4,
    colormap="jet",
    output_dir="outputs/xai_batch",
    labels_csv=None,
):
    """
    Run GradCAM pipeline on all images in a directory.
    Cross-checks with labels.csv if available.

    Args:
        image_dir: str — directory containing .png / .jpg fundus images
        model, device, backbone_name, method, alpha, colormap: same as run_single_image
        output_dir: str — where to save results
        labels_csv: str or None — path to labels.csv for ground truth crosscheck

    Returns:
        list of result dicts from run_single_image
    """
    # Auto-detect labels.csv if not provided
    if labels_csv is None:
        for candidate in [
            os.path.join(image_dir, "labels.csv"),
            os.path.join(os.path.dirname(image_dir), "labels.csv"),
            "images_test/labels.csv",
        ]:
            if os.path.exists(candidate):
                labels_csv = candidate
                break

    gt_labels = {}
    if labels_csv and os.path.exists(labels_csv):
        import pandas as pd
        try:
            df_labels = pd.read_csv(labels_csv)
            for _, row in df_labels.iterrows():
                gt_labels[str(row["id_code"])] = int(row["label"])
            print(f"✔ Loaded ground truth labels for {len(gt_labels)} samples from {labels_csv}")
        except Exception as e:
            print(f"Notice: Could not parse labels.csv: {e}")

    # Collect image files (supports both flat directory and dataset structure with subfolders)
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    mask_suffixes = ("_ex", "_hemorrhage", "_ma", "_vessel")
    
    image_paths = []
    entries = sorted(os.listdir(image_dir))
    for entry in entries:
        full_entry = os.path.join(image_dir, entry)
        if os.path.isdir(full_entry):
            for sub_file in sorted(os.listdir(full_entry)):
                if sub_file.lower().endswith(valid_exts):
                    base_name = os.path.splitext(sub_file)[0]
                    if not any(base_name.endswith(sfx) for sfx in mask_suffixes):
                        image_paths.append(os.path.join(full_entry, sub_file))
        elif entry.lower().endswith(valid_exts):
            base_name = os.path.splitext(entry)[0]
            if not any(base_name.endswith(sfx) for sfx in mask_suffixes):
                image_paths.append(full_entry)

    if not image_paths:
        print(f"⚠ No fundus images found in {image_dir}")
        return []

    print(f"\n{'='*70}")
    print(f"  DRISHYA XAI: Batch GradCAM Processing & Clinical Verification")
    print(f"  Images: {len(image_paths)} | Method: {method}")
    print(f"{'='*70}\n")

    results = []
    matches = 0
    for image_path in tqdm(image_paths, desc="Processing"):
        filename = os.path.basename(image_path)
        id_code = os.path.splitext(filename)[0]
        gt_class = gt_labels.get(id_code)

        try:
            result = run_single_image(
                image_path, model, device,
                backbone_name=backbone_name,
                method=method, alpha=alpha, colormap=colormap,
                output_dir=output_dir,
                gt_class=gt_class,
            )
            result["filename"] = filename
            result["id_code"] = id_code
            if gt_class is not None and result["predicted_class"] == gt_class:
                matches += 1
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    total_with_gt = sum(1 for r in results if r.get("gt_class") is not None)
    if total_with_gt > 0:
        acc = (matches / total_with_gt) * 100.0
        print(f"\n✔ Processed {len(results)}/{len(image_paths)} images.")
        print(f"🎯 Ground Truth Match Rate: {matches}/{total_with_gt} ({acc:.1f}%)")
    else:
        print(f"\n✔ Processed {len(results)}/{len(image_paths)} images successfully.")

    return results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DRISHYA XAI — GradCAM Heatmap Generation for Fundus Images"
    )

    # Input (single image or batch directory — mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single fundus image")
    input_group.add_argument("--image-dir", type=str, help="Directory of fundus images for batch processing")

    # Model
    parser.add_argument(
        "--checkpoint", type=str,
        default="models/teacher_tu-tf_efficientnet_b4.ns_jft_in1k_fold0_best.pth",
        help="Path to the teacher model checkpoint (.pth)"
    )
    parser.add_argument(
        "--backbone", type=str,
        default=DEFAULT_BACKBONE,
        help=f"Backbone architecture name (default: {DEFAULT_BACKBONE})"
    )

    # GradCAM settings
    parser.add_argument("--method", type=str, default="gradcam++", choices=["gradcam", "gradcam++"],
                        help="CAM method (default: gradcam++)")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Target class for GradCAM (default: auto = model's prediction)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Overlay blend factor (default: 0.4)")
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "turbo", "inferno", "magma", "hot"],
                        help="Heatmap colormap (default: jet)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/xai",
                        help="Output directory for saved images")

    args = parser.parse_args()

    # Load model
    model, device = load_teacher_model(args.checkpoint, args.backbone)

    if args.image:
        # Single image mode
        print(f"\n  Processing: {args.image}")
        run_single_image(
            args.image, model, device,
            backbone_name=args.backbone,
            method=args.method,
            alpha=args.alpha,
            colormap=args.colormap,
            output_dir=args.output_dir,
            target_class=args.target_class,
        )
    else:
        # Batch mode
        run_batch(
            args.image_dir, model, device,
            backbone_name=args.backbone,
            method=args.method,
            alpha=args.alpha,
            colormap=args.colormap,
            output_dir=args.output_dir,
        )

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
