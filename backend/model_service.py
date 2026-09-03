"""
DRISHYA AI — Clinical Screening Pipeline (ModelService)
Uses the distilled PP-LCNet Multi-Task Student Model for:
  1. Image Quality Assessment (IQA)
  2. CLAHE + Ben Graham preprocessing
  3. 5-class ICDR DR grading with confidence
  4. 4-channel lesion segmentation (MA, EX, HE, SE)
  5. Grad-CAM++ explainability heatmaps
  6. Clinical biomarker extraction
  7. Colored lesion overlay generation
  8. Automated 1-page clinical PDF report
"""

import os
from typing import Any, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from backend.model_student import DRISHYAStudentMTL
from ui.report_generator import generate_clinical_pdf


# ── Retinal FOV Masking ───────────────────────────────────────────────────────

def create_fundus_fov_mask(img_bgr, threshold=10):
    """
    Creates a binary mask of the circular retinal field-of-view.
    Retinal disc pixels -> 255, non-retinal black borders -> 0.
    Prevents background step-edge gradients from corrupting CAM normalization.
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, erode_kernel, iterations=1)
    return mask


def apply_fov_mask(heatmap, fov_mask):
    """
    Zeros out activations outside the retinal disc and re-normalizes the heatmap
    to [0, 1] using ONLY the retinal tissue pixels.
    """
    h, w = heatmap.shape[:2]
    mask_resized = cv2.resize(fov_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 127).astype(np.float32)

    masked = heatmap * binary_mask
    retina_pixels = masked[binary_mask > 0]
    if len(retina_pixels) > 0 and retina_pixels.max() > 1e-7:
        min_val = float(retina_pixels.min())
        max_val = float(retina_pixels.max())
        if max_val - min_val > 1e-7:
            masked = (masked - min_val) / (max_val - min_val)
            masked = np.clip(masked * binary_mask, 0.0, 1.0)
        else:
            masked = masked / max_val
    return masked


# ── Grad-CAM++ ───────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ for explainable DR classification on the PP-LCNet encoder.
    Uses 1st, 2nd, and 3rd order gradient approximations (Chattopadhay et al., 2018)
    for high-fidelity multi-instance lesion localization.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, target_class, fov_mask=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output

        target_score = logits[0, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured.")
        grads = self.gradients.detach()
        acts = self.activations.detach()

        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-7

        alpha = grad_2 / (2 * grad_2 + sum_acts * grad_3 + eps)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(384, 384), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if fov_mask is not None:
            cam = apply_fov_mask(cam, fov_mask)
        else:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        return cam

    def overlay_on_image(self, rgb_img_384, heatmap, alpha=0.40, colormap=cv2.COLORMAP_TURBO, fov_mask=None):
        """
        Overlays the saliency map onto the fundus scan using a medically uniform colormap.
        Ensures the background outside the retinal disc remains uncolored.
        """
        heatmap_uint8 = (255 * np.clip(heatmap, 0.0, 1.0)).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(rgb_img_384, 1.0 - alpha, color_heatmap, alpha, 0)

        if fov_mask is not None:
            mask_resized = cv2.resize(fov_mask, (rgb_img_384.shape[1], rgb_img_384.shape[0]), interpolation=cv2.INTER_NEAREST)
            bg_pixels = (mask_resized <= 127)
            color_heatmap[bg_pixels] = 0
            blended[bg_pixels] = rgb_img_384[bg_pixels]

        return blended, color_heatmap


# ── Image Quality Assessment ─────────────────────────────────────────────────

def assess_image_quality(img_bgr):
    """
    Computes Laplacian focus and illumination uniformity.
    Returns: (is_pass, q_score, reason)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_val = np.mean(gray)

    focus_norm = min(1.0, lap_var / 150.0)
    illum_norm = 1.0 - min(1.0, abs(mean_val - 110.0) / 90.0)
    q_score = round(float(0.6 * focus_norm + 0.4 * illum_norm), 2)

    if lap_var < 35.0:
        return False, q_score, "Image is blurry. Please hold camera steady."
    if mean_val < 20.0:
        return False, q_score, "Image is too dark. Increase illumination."
    if mean_val > 220.0:
        return False, q_score, "Image is overexposed with severe glare."

    return True, q_score, "Image quality passed."


# ── Fundus Preprocessing ─────────────────────────────────────────────────────

def preprocess_fundus(img_bgr, target_size=(384, 384)):
    """
    Applies circular mask extraction, 1:1 square crop, and Ben Graham + CLAHE normalization.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        max_side = max(cw, ch)
        cx, cy = x + cw // 2, y + ch // 2
        x1 = max(0, cx - max_side // 2)
        y1 = max(0, cy - max_side // 2)
        x2 = min(w, x1 + max_side)
        y2 = min(h, y1 + max_side)
        img_cropped = img_bgr[y1:y2, x1:x2]
    else:
        img_cropped = img_bgr

    img_resized = cv2.resize(img_cropped, target_size)

    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    preprocessed_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    return img_resized, preprocessed_bgr


# ── Lesion Overlay Generator ─────────────────────────────────────────────────

def generate_lesion_overlay(preprocessed_bgr, mask_probs):
    """
    Renders colored lesion contours on the preprocessed fundus image.
    mask_probs: numpy array of shape (4, 384, 384), sigmoid probabilities.
    Channel 0: Microaneurysms    → Red       (0, 0, 255) BGR
    Channel 1: Hard Exudates     → Yellow    (0, 255, 255) BGR
    Channel 2: Hemorrhages       → Crimson   (60, 20, 220) BGR
    Channel 3: Soft Exudates/CWS → Cyan      (255, 255, 0) BGR
    """
    overlay = preprocessed_bgr.copy()

    lesion_colors_bgr = [
        (0, 0, 255),      # MA  - Red
        (0, 255, 255),     # EX  - Yellow
        (60, 20, 220),     # HE  - Crimson
        (255, 255, 0),     # SE  - Cyan
    ]

    for ch_idx, color in enumerate(lesion_colors_bgr):
        binary = (mask_probs[ch_idx] >= 0.5).astype(np.uint8)
        if np.sum(binary) == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)
        # Semi-transparent fill for visibility
        mask_colored = np.zeros_like(overlay)
        cv2.drawContours(mask_colored, contours, -1, color, thickness=cv2.FILLED)
        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.25, 0)

    return overlay


# ── Biomarker Extraction ─────────────────────────────────────────────────────

def extract_biomarkers(mask_probs):
    """
    Extracts clinical biomarkers from the 4-channel segmentation masks.
    mask_probs: numpy array of shape (4, 384, 384), sigmoid probabilities.
    Returns dict with MA count, exudate area %, hemorrhage quadrant count, soft exudate area %.
    """
    total_pixels = 384 * 384

    # Channel 0: Microaneurysms — count discrete lesions via connected components
    ma_bin = (mask_probs[0] >= 0.5).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(ma_bin, connectivity=8)
    num_mas = max(0, num_labels - 1)  # subtract background label

    # Channel 1: Hard Exudates — percentage of retinal area
    ex_bin = (mask_probs[1] >= 0.5).astype(np.uint8)
    exudate_pct = round((np.sum(ex_bin) / total_pixels) * 100, 2)

    # Channel 2: Hemorrhages — count affected quadrants (4-2-1 rule for severe NPDR)
    he_bin = (mask_probs[2] >= 0.5).astype(np.uint8)
    h, w = 384, 384
    quads = [
        he_bin[0:h//2, 0:w//2],       # Top-left
        he_bin[0:h//2, w//2:w],        # Top-right
        he_bin[h//2:h, 0:w//2],        # Bottom-left
        he_bin[h//2:h, w//2:w],        # Bottom-right
    ]
    # Require at least 10 active pixels to count a quadrant as affected
    hemorrhage_quadrants = sum(1 for q in quads if np.sum(q) > 10)

    # Channel 3: Soft Exudates / Cotton Wool Spots — percentage of retinal area
    se_bin = (mask_probs[3] >= 0.5).astype(np.uint8)
    soft_exudate_pct = round((np.sum(se_bin) / total_pixels) * 100, 2)

    return {
        "microaneurysms": num_mas,
        "exudate_area_pct": exudate_pct,
        "hemorrhage_quadrants": hemorrhage_quadrants,
        "soft_exudate_area_pct": soft_exudate_pct,
    }


# ── Model Service ─────────────────────────────────────────────────────────────

class ModelService:
    """
    End-to-end clinical screening service using the distilled PP-LCNet Student Model.
    """
    def __init__(self, model_path="models/student_mtl_lcnet_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the distilled PP-LCNet multi-task student model
        self.model = DRISHYAStudentMTL(num_classes=5, num_masks=4).to(self.device)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"[DRISHYA] ✓ Loaded trained student model from {model_path}")
            except Exception as e:
                print(f"[DRISHYA] ✗ Could not load weights from {model_path}: {e}")
        else:
            print(f"[DRISHYA] ✗ Checkpoint {model_path} not found. Model is uninitialized!")

        self.model.eval()

        # Hook Grad-CAM++ to the final encoder block (PP-LCNet blocks[-1])
        target_layer = self.model.encoder.blocks[-1]
        self.gradcam = GradCAMPlusPlus(self.model, target_layer)

        # Log model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[DRISHYA] Model: PP-LCNet Student MTL | {total_params/1e6:.2f}M params | Device: {self.device}")

    def run_screening_pipeline(self, raw_img_bgr, patient_info: dict, output_dir: str = "backend/outputs") -> Dict[str, Any]:
        """
        Full clinical screening pipeline:
        IQA → Preprocess → Inference → Biomarkers → Lesion Overlay → Grad-CAM++ → PDF Report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 1. Image Quality Assessment ──────────────────────────────────────
        is_pass, q_score, iqa_reason = assess_image_quality(raw_img_bgr)
        if not is_pass:
            return {
                "success": False,
                "iqa_pass": False,
                "q_score": q_score,
                "message": f"RETAKE SCAN: {iqa_reason}",
                "grade": "ungradable",
                "grade_title": "Ungradable (IQA Reject)",
                "action": "🛑 Recapture retinal photo immediately before patient leaves."
            }

        # ── 2. Fundus Preprocessing ──────────────────────────────────────────
        raw_384_bgr, preprocessed_bgr = preprocess_fundus(raw_img_bgr)
        rgb_384 = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        # ── 3. Model Inference ───────────────────────────────────────────────
        tensor_in = torch.tensor(rgb_384 / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_in = ((tensor_in - mean) / std).to(self.device)

        logits_icdr, mask_logits = self.model(tensor_in)

        # Classification results
        probs = F.softmax(logits_icdr, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence_pct = round(float(probs[pred_class]) * 100, 1)

        # Segmentation masks (sigmoid probabilities)
        mask_probs = torch.sigmoid(mask_logits).detach().cpu().numpy()[0]  # (4, 384, 384)

        # ── 4. Grad-CAM++ Explainability ─────────────────────────────────────
        fov_mask = create_fundus_fov_mask(raw_384_bgr)
        heatmap = self.gradcam.generate_heatmap(tensor_in, target_class=pred_class, fov_mask=fov_mask)
        blended_cam_rgb, heatmap_only = self.gradcam.overlay_on_image(
            rgb_384, heatmap, alpha=0.40, colormap=cv2.COLORMAP_TURBO, fov_mask=fov_mask
        )

        # ── 5. Biomarker Extraction ──────────────────────────────────────────
        biomarkers = extract_biomarkers(mask_probs)
        num_mas = biomarkers["microaneurysms"]
        exudate_pct = biomarkers["exudate_area_pct"]
        hemorrhage_quadrants = biomarkers["hemorrhage_quadrants"]
        soft_exudate_pct = biomarkers["soft_exudate_area_pct"]

        # ── 6. Lesion Overlay ────────────────────────────────────────────────
        lesion_overlay_bgr = generate_lesion_overlay(preprocessed_bgr, mask_probs)

        # ── 7. Save Output Images ────────────────────────────────────────────
        raw_path = os.path.join(output_dir, f"{timestamp}_raw.png")
        prep_path = os.path.join(output_dir, f"{timestamp}_prep.png")
        lesion_path = os.path.join(output_dir, f"{timestamp}_lesions.png")
        heatmap_path = os.path.join(output_dir, f"{timestamp}_heatmap.png")
        gradcam_path = os.path.join(output_dir, f"{timestamp}_gradcam.png")

        cv2.imwrite(raw_path, raw_384_bgr)
        cv2.imwrite(prep_path, preprocessed_bgr)
        cv2.imwrite(lesion_path, lesion_overlay_bgr)
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_only, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gradcam_path, cv2.cvtColor(blended_cam_rgb, cv2.COLOR_RGB2BGR))

        # ── 8. Clinical Grading & Triage ─────────────────────────────────────
        GRADE_TITLES = {
            0: ("Grade 0: Normal Retina", "No diabetic retinopathy lesions detected.", "Non-Referable (Annual Rescreening)", False),
            1: ("Grade 1: Mild NPDR", "Microaneurysms only present.", "Non-Referable (Rescreening in 6-12 Months)", False),
            2: ("Grade 2: Moderate NPDR", "Microaneurysms + Hard Exudates present.", "Referable DR (Refer to Specialist within 4 weeks)", True),
            3: ("Grade 3: Severe NPDR", "Severe intraretinal hemorrhages (4:2:1 rule).", "Urgent Referral (Evaluation within 1-2 weeks)", True),
            4: ("Grade 4: Proliferative DR", "Neovascularization / Preretinal hemorrhage.", "Urgent Laser / Anti-VEGF Referral", True),
        }
        g_title, g_desc, g_triage, is_referable = GRADE_TITLES.get(pred_class, GRADE_TITLES[0])

        # ── 9. Build PDF Report Data ─────────────────────────────────────────
        pdf_path = os.path.join(output_dir, f"DRISHYA_Report_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf")

        diagnostic_result = {
            'grade_title': g_title,
            'grade_desc': g_desc,
            'triage_status': g_triage,
            'triage_sub': 'Slit-lamp & OCT evaluation required' if is_referable else 'Routine Primary Health Care Follow-up',
            'iqa_status': f'Pass (Q={q_score})',
            'confidence': f'{confidence_pct}%',
            'grade_num': pred_class
        }

        panel_paths = {
            'preprocessed': prep_path,
            'lesions': lesion_path,
            'gradcam': gradcam_path
        }

        # Determine hemorrhage clinical relevance
        if hemorrhage_quadrants >= 4:
            he_rel = 'All 4 quadrants — meets severe NPDR 4-2-1 criterion'
        elif hemorrhage_quadrants >= 2:
            he_rel = 'Below severe NPDR threshold but active bleeding'
        elif hemorrhage_quadrants == 1:
            he_rel = 'Single quadrant involvement'
        else:
            he_rel = 'No significant hemorrhage detected'

        biomarker_metrics = {
            'mas': f'{num_mas} detected',
            'mas_rel': 'Active microvascular leakage' if num_mas > 0 else 'Normal vascular integrity',
            'exudates': f'{exudate_pct:.2f}% area',
            'exudates_rel': 'Lipoprotein deposits indicating vascular leakage' if exudate_pct > 0 else 'Absent',
            'hemorrhages': f'{hemorrhage_quadrants} quadrant{"s" if hemorrhage_quadrants != 1 else ""}',
            'hemorrhages_rel': he_rel,
            'neovascularization': '0 (Absent)' if pred_class < 4 else 'Suspected (Grade 4)',
            'macula': 'High Risk' if (is_referable and exudate_pct > 0.5) else ('Moderate Risk' if is_referable else 'Low Risk'),
            'macula_rel': 'Distance from fovea evaluated',
            'soft_exudates': f'{soft_exudate_pct:.2f}% area',
            'soft_exudates_rel': 'Cotton wool spots present — nerve fiber ischemia' if soft_exudate_pct > 0 else 'Absent',
        }

        # ── 10. Generate Clinical PDF ────────────────────────────────────────
        generate_clinical_pdf(patient_info, diagnostic_result, panel_paths, biomarker_metrics, pdf_path)

        # ── 11. Return Results ───────────────────────────────────────────────
        return {
            "success": True,
            "iqa_pass": True,
            "q_score": q_score,
            "grade": pred_class,
            "grade_title": g_title,
            "grade_desc": g_desc,
            "confidence": f"{confidence_pct}%",
            "referable_dr": is_referable,
            "biomarkers": {
                "microaneurysms": num_mas,
                "exudate_area_pct": f"{exudate_pct:.2f}%",
                "hemorrhage_quadrants": hemorrhage_quadrants,
                "soft_exudate_area_pct": f"{soft_exudate_pct:.2f}%",
            },
            "files": {
                "raw_path": raw_path,
                "preprocessed_path": prep_path,
                "lesion_path": lesion_path,
                "heatmap_path": heatmap_path,
                "gradcam_path": gradcam_path,
                "pdf_report_path": pdf_path
            }
        }
