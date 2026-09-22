import os
import sys
import pytest
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from ui.report_generator import generate_clinical_pdf

def test_generate_clinical_pdf(tmp_path):
    output_pdf = str(tmp_path / "test_report.pdf")
    
    # Create 4 dummy panel images
    panel_paths = {}
    for name in ["raw", "preprocessed", "lesion", "heatmap"]:
        img_path = str(tmp_path / f"{name}.png")
        img = Image.new("RGB", (384, 384), color=(180, 100, 50))
        img.save(img_path)
        panel_paths[name] = img_path
        
    patient_info = {
        "name": "Drishya Verification Patient",
        "abha_id": "91-1234-5678-9999",
        "age_gender": "55 Yrs / Female",
        "laterality": "OD (Right Eye)",
        "exam_id": "EX-2026-VAL-001"
    }
    
    diagnostic_result = {
        "grade": 3,
        "grade_title": "Severe Non-Proliferative Retinopathy",
        "grade_desc": "Significant intraretinal hemorrhages in 4 quadrants satisfying ETDRS 4-2-1 rule.",
        "confidence": "94.6%",
        "referable_dr": True,
        "action": "Urgent vitreo-retinal evaluation within 2-4 weeks. Anti-VEGF / Pan-retinal photocoagulation assessment."
    }
    
    biomarker_metrics = {
        "microaneurysms": 38,
        "exudate_area_pct": "1.42%",
        "hemorrhage_quadrants": "4 / 4 Quadrants",
        "soft_exudate_area_pct": "0.35%",
        "macular_risk": "Moderate Risk",
        "macular_detail": "Hard exudates 0.85 DD from fovea"
    }
    
    generate_clinical_pdf(patient_info, diagnostic_result, panel_paths, biomarker_metrics, output_pdf)
    
    assert os.path.exists(output_pdf), f"PDF was not generated at {output_pdf}"
    file_size = os.path.getsize(output_pdf)
    assert file_size > 10000, f"Expected PDF > 10KB, got {file_size} bytes"
    
    with open(output_pdf, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-", f"Invalid PDF header: {header}"
