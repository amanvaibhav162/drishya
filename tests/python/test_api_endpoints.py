import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from backend.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "DRISHYA" in data["service"]

def test_screen_patient_invalid_upload():
    # Attempting upload with empty or non-image content
    response = client.post(
        "/api/screen-patient",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"name": "Test Patient", "abha_id": "91-1234-5678-9012"}
    )
    # Backend should handle gracefully or return error
    assert response.status_code in [400, 422, 500]

def test_screen_patient_valid_image():
    sample_img_path = os.path.join(BASE_DIR, "data", "aptos", "images", "s00r00000", "s00r00000.jpg")
    if not os.path.exists(sample_img_path):
        pytest.skip(f"Sample image {sample_img_path} not found")
        
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()
        
    response = client.post(
        "/api/screen-patient",
        files={"file": ("s00r00000.jpg", img_bytes, "image/jpeg")},
        data={
            "name": "Sita Devi",
            "age": "52",
            "gender": "Female",
            "abha_id": "91-8899-2233-4455"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    assert "grade" in data
    assert "grade_title" in data
    assert "referable_dr" in data
    assert "biomarkers" in data
    assert "files" in data
