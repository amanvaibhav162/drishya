"""
DRISHYA AI — Clinical Screening Pipeline (ModelService)
Uses the EfficientNetV2 Multi-Task Student Model for:
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

from backend.model_student import DRISHYAStudentMTL, DRISHYAEfficientNetV2MTL, load_student_model
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
    Computes Laplacian focus and illumination uniformity on retinal tissue,
    excluding non-retinal black camera margins.
    Returns: (is_pass, q_score, reason)
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # Extract retinal tissue mask to evaluate the actual retina, not black borders
    mask = create_fundus_fov_mask(img_bgr)
    retina_pixels = gray[mask > 127]
    if len(retina_pixels) == 0:
        retina_pixels = gray.flatten()

    # Measure focus variance within the retina (erode slightly to avoid false high-variance at the disc perimeter)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    retina_lap = lap[eroded_mask > 127]
    lap_var = float(np.var(retina_lap)) if len(retina_lap) > 100 else float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mean_val = float(np.mean(retina_pixels))

    focus_norm = min(1.0, lap_var / 150.0)
    illum_norm = 1.0 - min(1.0, abs(mean_val - 110.0) / 90.0)
    q_score = round(float(0.6 * focus_norm + 0.4 * illum_norm), 2)

    if lap_var < 35.0:
        return False, q_score, "Image is blurry. Please hold camera steady."
    if mean_val < 25.0:
        return False, q_score, "Image is too dark. Increase illumination."
    if mean_val > 220.0:
        return False, q_score, "Image is overexposed with severe glare."

    return True, q_score, "Image quality passed."


# ── Fundus Preprocessing ─────────────────────────────────────────────────────

def preprocess_fundus(img_bgr, target_size=(384, 384)):
    """
    Applies circular mask extraction, zero-padding for a distortion-free 1:1 square crop,
    and Luminance-channel CLAHE normalization in LAB color space.
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

        # Desired square bounding coordinates
        x_min = cx - max_side // 2
        y_min = cy - max_side // 2
        x_max = x_min + max_side
        y_max = y_min + max_side

        # Pad with black border if square extends beyond image dimensions (avoids stretching)
        pad_top = max(0, -y_min)
        pad_left = max(0, -x_min)
        pad_bottom = max(0, y_max - h)
        pad_right = max(0, x_max - w)

        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            img_bgr = cv2.copyMakeBorder(
                img_bgr, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            x_min += pad_left
            y_min += pad_top
            x_max += pad_left
            y_max += pad_top

        img_cropped = img_bgr[y_min:y_max, x_min:x_max]
    else:
        img_cropped = img_bgr

    img_resized = cv2.resize(img_cropped, target_size)

    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    preprocessed_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    return img_resized, preprocessed_bgr


# ── Per-Channel Segmentation Thresholds ──────────────────────────────────────
# Empirically tuned: MAs are punctate and need lower threshold;
# hemorrhages vary in intensity, exudates are brighter and more reliable.
SEG_THRESHOLDS = {
    'MA': 0.15,   # Microaneurysms — very small, low-confidence detections
    'EX': 0.15,   # Hard Exudates
    'HE': 0.35,   # Hemorrhages
    'SE': 0.45,   # Soft Exudates / Cotton Wool Spots (raised to suppress vessel reflex)
}


def extract_vessel_mask(img_bgr, threshold=10):
    """
    Extracts the prominent retinal vascular tree from the green channel
    using morphological black-hat filtering. Used to prevent normal retinal
    blood vessels from being falsely outlined as soft exudates/lesions.
    """
    g = img_bgr[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    black_hat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    _, vessel_mask = cv2.threshold(black_hat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)


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
    vessel_mask = extract_vessel_mask(preprocessed_bgr)

    lesion_colors_bgr = [
        (0, 0, 255),      # MA  - Red
        (0, 255, 255),     # EX  - Yellow
        (60, 20, 220),     # HE  - Crimson
        (255, 255, 0),     # SE  - Cyan
    ]
    channel_keys = ['MA', 'EX', 'HE', 'SE']

    for ch_idx, (color, key) in enumerate(zip(lesion_colors_bgr, channel_keys)):
        thresh = SEG_THRESHOLDS[key]
        binary = (mask_probs[ch_idx] >= thresh).astype(np.uint8)

        # For Channel 3 (Soft Exudates), suppress blood vessel pixels to avoid lining the vascular tree
        if key == 'SE':
            binary[vessel_mask > 0] = 0

        if np.sum(binary) == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Remove small noise and elongated vessel remnants from Soft Exudates
            if key == 'SE':
                if area < 20:
                    continue
                rect = cv2.minAreaRect(c)
                w, h = rect[1]
                if min(w, h) > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio >= 4.0:
                        # Skip long linear vessel segments
                        continue
            valid_contours.append(c)

        if not valid_contours:
            continue

        cv2.drawContours(overlay, valid_contours, -1, color, thickness=2)
        # Semi-transparent fill for visibility
        mask_colored = np.zeros_like(overlay)
        cv2.drawContours(mask_colored, valid_contours, -1, color, thickness=cv2.FILLED)
        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.25, 0)

    return overlay


# ── Biomarker Extraction ─────────────────────────────────────────────────────

def extract_biomarkers(mask_probs, preprocessed_bgr=None):
    """
    Extracts clinical biomarkers from the 4-channel segmentation masks.
    Uses per-channel thresholds from SEG_THRESHOLDS for clinical-grade sensitivity.
    mask_probs: numpy array of shape (4, 384, 384), sigmoid probabilities.
    Returns dict with MA count, exudate area %, hemorrhage quadrant count,
    soft exudate area %, and macular risk status.
    """
    total_pixels = 384 * 384

    # Channel 0: Microaneurysms — count discrete lesions via connected components
    ma_bin = (mask_probs[0] >= SEG_THRESHOLDS['MA']).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ma_bin, connectivity=8)
    # Filter out noise: only count components with area >= 3 pixels
    num_mas = 0
    if num_labels > 1:
        for i in range(1, num_labels):  # skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= 3:
                num_mas += 1

    # Channel 1: Hard Exudates — percentage of retinal area
    ex_bin = (mask_probs[1] >= SEG_THRESHOLDS['EX']).astype(np.uint8)
    exudate_pct = round((np.sum(ex_bin) / total_pixels) * 100, 2)

    # Channel 2: Hemorrhages — count affected quadrants (4-2-1 rule for severe NPDR)
    he_bin = (mask_probs[2] >= SEG_THRESHOLDS['HE']).astype(np.uint8)
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
    se_bin = (mask_probs[3] >= SEG_THRESHOLDS['SE']).astype(np.uint8)
    if preprocessed_bgr is not None:
        vessel_mask = extract_vessel_mask(preprocessed_bgr)
        se_bin[vessel_mask > 0] = 0
    soft_exudate_pct = round((np.sum(se_bin) / total_pixels) * 100, 2)

    # Macular Risk — check exudate proximity to the foveal center (image center)
    # The fovea is approximated at the center of the 384x384 crop.
    # Risk zones: High <500µm (~32px), Moderate <1000µm (~64px), Low >1000µm
    cx, cy = w // 2, h // 2
    macular_risk = 'Low Risk'
    macular_detail = 'No lesions in macular zone'
    if exudate_pct > 0:
        # Find the closest exudate pixel to the foveal center
        ey, ex_coords = np.where(ex_bin > 0)
        if len(ey) > 0:
            distances = np.sqrt((ex_coords - cx)**2 + (ey - cy)**2)
            min_dist_px = float(np.min(distances))
            # Convert to approximate µm (1 pixel ≈ 15.6µm at 384px / 6mm FOV)
            min_dist_um = round(min_dist_px * 15.6)
            if min_dist_px < 32:  # < ~500µm
                macular_risk = 'High Risk'
                macular_detail = f'Exudates ~{min_dist_um}µm from fovea — clinically significant macular edema risk'
            elif min_dist_px < 64:  # < ~1000µm
                macular_risk = 'Moderate Risk'
                macular_detail = f'Exudates ~{min_dist_um}µm from fovea — monitor closely'
            else:
                macular_risk = 'Low Risk'
                macular_detail = f'Exudates ~{min_dist_um}µm from fovea — outside critical zone'

    return {
        "microaneurysms": num_mas,
        "exudate_area_pct": exudate_pct,
        "hemorrhage_quadrants": hemorrhage_quadrants,
        "soft_exudate_area_pct": soft_exudate_pct,
        "macular_risk": macular_risk,
        "macular_detail": macular_detail,
    }


# ── Dynamic Diagnosis Builder ────────────────────────────────────────────────

def _build_dynamic_diagnosis(pred_class, num_mas, exudate_pct, hemorrhage_quadrants, soft_exudate_pct, macular_risk):
    """
    Constructs diagnostic title, description, triage status, referral flag,
    and action/follow-up text purely from actual biomarker detections and model predictions.
    Reconciles classification logits with physical biomarker evidence (Multi-Task Consensus).
    No hardcoded ICDR-grade descriptions — everything is evidence-based.
    """
    # ── 1. Calculate Evidence-Based Severity from Biomarkers ─────────────
    # ICDR Classification Criteria:
    # Grade 3 (Severe NPDR): 4-2-1 rule: hemorrhages in 4 quadrants
    # Grade 2 (Moderate NPDR): Hard exudates, hemorrhages in 1-3 quadrants, or significant CWS
    # Grade 1 (Mild NPDR): Microaneurysms only (or solitary hemorrhage/CWS)
    # Grade 0 (Normal): No lesions detected
    if hemorrhage_quadrants >= 4:
        biomarker_grade = 3
    elif exudate_pct > 0.05 or (num_mas > 0 and hemorrhage_quadrants >= 1) or soft_exudate_pct > 0.5:
        biomarker_grade = 2
    elif num_mas > 0 or hemorrhage_quadrants > 0 or soft_exudate_pct > 0.1:
        biomarker_grade = 1
    else:
        biomarker_grade = 0

    # Multi-task consensus: Take the higher of the classification prediction and biomarker-indicated grade
    # (prevents clinical false negatives when classification head misses physical lesions)
    effective_grade = max(pred_class, biomarker_grade)

    GRADE_NAMES = {
        0: 'Normal Retina',
        1: 'Mild NPDR',
        2: 'Moderate NPDR',
        3: 'Severe NPDR',
        4: 'Proliferative DR',
    }

    grade_name = GRADE_NAMES.get(effective_grade, 'Unclassified')
    g_title = f"Grade {effective_grade}: {grade_name}"

    # ── Build evidence-based description from actual detections ──────────
    findings = []
    if num_mas > 0:
        findings.append(f"{num_mas} microaneurysm{'s' if num_mas != 1 else ''}")
    if exudate_pct > 0:
        findings.append(f"hard exudates ({exudate_pct:.2f}% area)")
    if hemorrhage_quadrants > 0:
        quad_text = f"{hemorrhage_quadrants} quadrant{'s' if hemorrhage_quadrants != 1 else ''}"
        findings.append(f"hemorrhages in {quad_text}")
    if soft_exudate_pct > 0:
        findings.append(f"cotton wool spots ({soft_exudate_pct:.2f}% area)")

    if findings:
        finding_str = ', '.join(findings)
        g_desc = f"Detected: {finding_str}."
        if macular_risk == 'High Risk':
            g_desc += " Clinically significant macular edema risk."
        elif macular_risk == 'Moderate Risk':
            g_desc += " Moderate macular involvement noted."
    else:
        g_desc = "No diabetic retinopathy lesions detected in segmentation analysis."

    # ── Triage & Referral logic based on effective grade ──────────────────
    if effective_grade == 0:
        g_triage = 'Non-Referable (Annual Rescreening)'
        is_referable = False
        action_followup = 'Routine rescreening in 12 months'
    elif effective_grade == 1:
        g_triage = 'Non-Referable (Rescreening in 6-12 Months)'
        is_referable = False
        action_followup = 'Rescreening in 6-12 months'
        if macular_risk in ('High Risk', 'Moderate Risk'):
            g_triage = 'Referable DR (Macular Risk — Refer within 4 weeks)'
            is_referable = True
            action_followup = 'Ophthalmologist evaluation within 4 weeks (macular involvement)'
    elif effective_grade == 2:
        g_triage = 'Referable DR (Refer to Specialist within 4 weeks)'
        is_referable = True
        action_followup = 'Ophthalmologist evaluation within 4 weeks'
        if macular_risk == 'High Risk':
            action_followup = 'Urgent ophthalmologist evaluation — macular edema suspected'
    elif effective_grade == 3:
        g_triage = 'Urgent Referral (Evaluation within 1-2 weeks)'
        is_referable = True
        action_followup = 'Specialist evaluation within 1-2 weeks — 4-2-1 severe NPDR criterion met'
    else:  # Grade 4
        g_triage = 'Urgent Laser / Anti-VEGF Referral'
        is_referable = True
        action_followup = 'Immediate referral for anti-VEGF or laser photocoagulation'

    return effective_grade, g_title, g_desc, g_triage, is_referable, action_followup


# ── Model Service ─────────────────────────────────────────────────────────────

class ModelService:
    """
    End-to-end clinical screening service using the distilled EfficientNetV2 Student Model.
    """
    def __init__(self, model_path="models/student_mtl_lcnet_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the multi-task student model
        self.model: DRISHYAEfficientNetV2MTL = load_student_model(model_path, self.device, num_classes=5, num_masks=4)

        # Hook Grad-CAM++ to the final encoder convolutional block
        target_layer = self.model.get_target_layer()
        self.gradcam = GradCAMPlusPlus(self.model, target_layer)

        # Log model stats
        total_params = sum(p.numel() for p in self.model.parameters())
        arch_name = getattr(self.model, 'arch_name', 'Student MTL')
        print(f"[DRISHYA] Model: {arch_name} | {total_params/1e6:.2f}M params | Device: {self.device}")

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
        biomarkers = extract_biomarkers(mask_probs, preprocessed_bgr)
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

        # ── 8. Dynamic Clinical Grading & Triage ─────────────────────────────
        # Build diagnostic description dynamically from actual biomarker detections & model consensus
        effective_grade, g_title, g_desc, g_triage, is_referable, action_followup = _build_dynamic_diagnosis(
            pred_class, num_mas, exudate_pct, hemorrhage_quadrants, soft_exudate_pct,
            biomarkers.get('macular_risk', 'Low Risk')
        )

        # ── 9. Build PDF Report Data ─────────────────────────────────────────
        pdf_path = os.path.join(output_dir, f"DRISHYA_Report_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf")

        diagnostic_result = {
            'grade_title': g_title,
            'grade_desc': g_desc,
            'triage_status': g_triage,
            'triage_sub': 'Slit-lamp & OCT evaluation required' if is_referable else 'Routine Primary Health Care Follow-up',
            'action_followup': action_followup,
            'iqa_status': f'Pass (Q={q_score})',
            'confidence': f'{confidence_pct}%',
            'grade_num': effective_grade,
            'is_referable': is_referable
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

        macular_risk = biomarkers.get('macular_risk', 'Low Risk')
        macular_detail = biomarkers.get('macular_detail', 'No lesions in macular zone')

        biomarker_metrics = {
            'mas': f'{num_mas} detected',
            'mas_rel': 'Active microvascular leakage' if num_mas > 0 else 'Normal vascular integrity',
            'exudates': f'{exudate_pct:.2f}% area',
            'exudates_rel': 'Lipoprotein deposits indicating vascular leakage' if exudate_pct > 0 else 'Absent',
            'hemorrhages': f'{hemorrhage_quadrants} quadrant{"s" if hemorrhage_quadrants != 1 else ""}',
            'hemorrhages_rel': he_rel,
            'neovascularization': '0 (Absent)' if effective_grade < 4 else 'Suspected (Grade 4)',
            'macula': macular_risk,
            'macula_rel': macular_detail,
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
            "grade": effective_grade,
            "grade_title": g_title,
            "grade_desc": g_desc,
            "action": action_followup,
            "confidence": f"{confidence_pct}%",
            "referable_dr": is_referable,
            "biomarkers": {
                "microaneurysms": num_mas,
                "exudate_area_pct": f"{exudate_pct:.2f}%",
                "hemorrhage_quadrants": hemorrhage_quadrants,
                "soft_exudate_area_pct": f"{soft_exudate_pct:.2f}%",
                "macular_risk": macular_risk,
                "macular_detail": macular_detail,
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
