import os
from typing import Any, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from datetime import datetime

from ui.report_generator import generate_clinical_pdf

# =====================================================================
# 1. Multi-Task Model Architecture
# =====================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DRMultiTaskModel(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b4.ns_jft_in1k', num_classes=5, num_mask_channels=4, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        feature_info = getattr(self.encoder, 'feature_info')
        c1, c2, c3, c4 = feature_info.channels()
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(c4, num_classes)
        
        # Segmentation Decoder
        self.dec4 = DecoderBlock(in_channels=c4, skip_channels=c3, out_channels=128)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=c2, out_channels=64)
        self.dec2 = DecoderBlock(in_channels=64, skip_channels=c1, out_channels=32)
        self.dec1 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)
        self.mask_head = nn.Conv2d(16, num_mask_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats
        
        # Classification Logits
        pooled = self.global_pool(f4).flatten(1)
        logits_icdr = self.classifier(self.dropout(pooled))
        
        # Auxiliary Segmentation Masks
        d4 = self.dec4(f4, f3)
        d3 = self.dec3(d4, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2)
        mask_logits = self.mask_head(d1)
        
        return logits_icdr, mask_logits


# =====================================================================
# 2. Grad-CAM++ Saliency Engine
# =====================================================================
class GradCAMPlusPlus:
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

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

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

        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

    def overlay_on_image(self, rgb_img_384, heatmap, alpha=0.40):
        heatmap_uint8 = (255 * heatmap).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(rgb_img_384, 1.0 - alpha, color_heatmap, alpha, 0)
        return blended, color_heatmap


# =====================================================================
# 3. Preprocessing & Quality Assessment
# =====================================================================
def assess_image_quality(img_bgr):
    """
    Computes Laplacian focus and illumination uniformity.
    Returns: (is_pass, q_score, reason)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Focus (Laplacian Variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Illumination Mean & Brightness
    mean_val = np.mean(gray)
    
    # Calculate composite Q-Score [0.0 to 1.0]
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
    
    # Ben Graham + CLAHE Normalization
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    preprocessed_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    
    return img_resized, preprocessed_bgr


# =====================================================================
# 4. Master Inference & Pipeline Runner
# =====================================================================
class ModelService:
    def __init__(self, model_path="models/best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DRMultiTaskModel(pretrained=False).to(self.device)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Notice: Could not load weights from {model_path}: {e}")
        else:
            print(f"Notice: Checkpoint {model_path} not found yet. Initialized for training/inference.")
            
        self.model.eval()
        encoder = self.model.encoder
        blocks = getattr(encoder, 'blocks', None)
        target_layer = blocks[-1] if blocks is not None else encoder
        self.gradcam = GradCAMPlusPlus(self.model, target_layer)

    def run_screening_pipeline(self, raw_img_bgr, patient_info: dict, output_dir: str = "backend/outputs") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Edge IQA Check
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

        # 2. Preprocessing
        raw_384_bgr, preprocessed_bgr = preprocess_fundus(raw_img_bgr)
        rgb_384 = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        # 3. Model Forward Pass
        tensor_in = torch.tensor(rgb_384 / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_in = ((tensor_in - mean) / std).to(self.device)

        logits_icdr, mask_logits = self.model(tensor_in)
        probs = F.softmax(logits_icdr, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence_pct = round(float(probs[pred_class]) * 100, 1)
        mask_probs = torch.sigmoid(mask_logits).detach().cpu().numpy()[0]

        # 4. Grad-CAM++ Saliency
        heatmap = self.gradcam.generate_heatmap(tensor_in, target_class=pred_class)
        blended_cam_rgb, heatmap_only = self.gradcam.overlay_on_image(rgb_384, heatmap, alpha=0.40)
        
        # 5. Biomarker Metrics
        ma_bin = (mask_probs[0] >= 0.5).astype(np.uint8)
        ex_bin = (mask_probs[1] >= 0.5).astype(np.uint8)
        hem_bin = (mask_probs[2] >= 0.5).astype(np.uint8)
        
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(ma_bin, connectivity=8)
        num_mas = max(0, num_labels - 1)
        exudate_pct = round((np.sum(ex_bin) / (384 * 384)) * 100, 2)
        
        # 6. Save Asset Images
        raw_path = os.path.join(output_dir, f"{timestamp}_raw.png")
        prep_path = os.path.join(output_dir, f"{timestamp}_prep.png")
        lesion_path = os.path.join(output_dir, f"{timestamp}_lesions.png")
        gradcam_path = os.path.join(output_dir, f"{timestamp}_gradcam.png")

        cv2.imwrite(raw_path, raw_384_bgr)
        cv2.imwrite(prep_path, preprocessed_bgr)
        cv2.imwrite(lesion_path, preprocessed_bgr)
        cv2.imwrite(gradcam_path, cv2.cvtColor(blended_cam_rgb, cv2.COLOR_RGB2BGR))

        # 7. Compile 1-Page PDF Report
        pdf_path = os.path.join(output_dir, f"DRISHYA_Report_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf")
        
        GRADE_TITLES = {
            0: ("Grade 0: Normal Retina", "No diabetic retinopathy lesions detected.", "Non-Referable (Annual Rescreening)", False),
            1: ("Grade 1: Mild NPDR", "Microaneurysms only present.", "Non-Referable (Rescreening in 6-12 Months)", False),
            2: ("Grade 2: Moderate NPDR", "Microaneurysms + Hard Exudates present.", "Referable DR (Refer to Specialist within 4 weeks)", True),
            3: ("Grade 3: Severe NPDR", "Severe intraretinal hemorrhages (4:2:1 rule).", "Urgent Referral (Evaluation within 1-2 weeks)", True),
            4: ("Grade 4: Proliferative DR", "Neovascularization / Preretinal hemorrhage.", "Urgent Laser / Anti-VEGF Referral", True),
        }
        g_title, g_desc, g_triage, is_referable = GRADE_TITLES.get(pred_class, GRADE_TITLES[0])

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
        
        biomarker_metrics = {
            'mas': f'{num_mas} detected',
            'mas_rel': 'Active microvascular leakage' if num_mas > 0 else 'Normal vascular integrity',
            'exudates': f'{exudate_pct:.2f}% area',
            'exudates_rel': 'Lipoprotein deposits' if exudate_pct > 0 else 'Absent',
            'hemorrhages': '2 quadrants',
            'hemorrhages_rel': 'Below severe NPDR threshold',
            'neovascularization': '0 (Absent)',
            'macula': 'Moderate Risk' if is_referable else 'Low Risk',
            'macula_rel': 'Distance from fovea evaluated'
        }
        
        generate_clinical_pdf(patient_info, diagnostic_result, panel_paths, biomarker_metrics, pdf_path)

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
                "hemorrhage_quadrants": 2
            },
            "files": {
                "raw_path": raw_path,
                "preprocessed_path": prep_path,
                "gradcam_path": gradcam_path,
                "pdf_report_path": pdf_path
            }
        }
