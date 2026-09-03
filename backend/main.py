from typing import Any, Dict
import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import PORT
from backend.model_service import ModelService
from backend.database import save_screening_record, get_all_screenings, upload_file_to_supabase, STORAGE_BUCKET_REPORTS, STORAGE_BUCKET_IMAGES

app = FastAPI(
    title="DRISHYA AI Screening Backend",
    description="FastAPI backend connecting PyTorch Multi-Task Model, Grad-CAM++, Supabase, and Clinical Reporting.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService(model_path="models/student_mtl_lcnet_best.pth")

os.makedirs("backend/outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="backend/outputs"), name="outputs")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DRISHYA AI Engine",
        "device": str(model_service.device)
    }


@app.post("/api/screen-patient")
async def screen_patient(
    file: UploadFile = File(...),
    name: str = Form("Anonymous Patient"),
    age: str = Form("N/A"),
    gender: str = Form("N/A"),
    phone: str = Form("N/A"),
    abha_id: str = Form("N/A")
):
    """
    Receives patient fundus scan, runs IQA, AI Grading, Grad-CAM++, generates PDF, and syncs to Supabase.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        patient_name = name.strip() if name and name.strip() else "Anonymous Patient"
        patient_age = age.strip() if age and age.strip() else "N/A"
        patient_gender = gender.strip() if gender and gender.strip() else "N/A"
        patient_phone = phone.strip() if phone and phone.strip() else "N/A"
        patient_abha = abha_id.strip() if abha_id and abha_id.strip() else "N/A"

        # Format age / sex string for clinical report
        if patient_age != "N/A" and patient_gender != "N/A":
            age_sex_str = f"{patient_age} Yrs / {patient_gender}"
        elif patient_age != "N/A":
            age_sex_str = f"{patient_age} Yrs"
        elif patient_gender != "N/A":
            age_sex_str = patient_gender
        else:
            age_sex_str = "Adult Screening"

        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "age_sex": age_sex_str,
            "phone": patient_phone,
            "abha_id": patient_abha,
            "eye": "Left Eye (OS)",
            "center": "PHC Rampur | Rural Eye Care Hub"
        }

        result: Dict[str, Any] = model_service.run_screening_pipeline(img_bgr, patient_info, output_dir="backend/outputs")

        if not result.get("success"):
            return JSONResponse(status_code=200, content=result)

        files = result.get("files", {})
        pdf_path = files.get("pdf_report_path", "")
        gradcam_path = files.get("gradcam_path", "")
        
        pdf_filename = os.path.basename(pdf_path)
        gradcam_filename = os.path.basename(gradcam_path)

        supabase_pdf_url = upload_file_to_supabase(pdf_path, STORAGE_BUCKET_REPORTS, pdf_filename)
        supabase_img_url = upload_file_to_supabase(gradcam_path, STORAGE_BUCKET_IMAGES, gradcam_filename)
        
        heatmap_path = files.get("heatmap_path", "")
        if heatmap_path:
            heatmap_filename = os.path.basename(heatmap_path)
            upload_file_to_supabase(heatmap_path, STORAGE_BUCKET_IMAGES, heatmap_filename)

        biomarkers = result.get("biomarkers", {})
        db_record = {
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "patient_phone": patient_phone,
            "abha_id": patient_abha,
            "icdr_grade": result.get("grade"),
            "grade_title": result.get("grade_title"),
            "confidence": result.get("confidence"),
            "referable_dr": result.get("referable_dr"),
            "iqa_score": result.get("q_score"),
            "num_microaneurysms": biomarkers.get("microaneurysms", 0),
            "exudate_area_pct": biomarkers.get("exudate_area_pct", "0.00%"),
            "pdf_report_url": supabase_pdf_url or f"/outputs/{pdf_filename}",
            "gradcam_image_url": supabase_img_url or f"/outputs/{gradcam_filename}"
        }
        
        save_screening_record(db_record)

        result["supabase_synced"] = bool(supabase_pdf_url)
        result["pdf_download_url"] = f"/api/download-report/{pdf_filename}"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/screenings")
def list_screenings(limit: int = 50):
    """
    Returns historical screening records from Supabase.
    """
    return get_all_screenings(limit=limit)


@app.get("/api/download-report/{filename}")
def download_report(filename: str):
    """
    Downloads the compiled 1-page clinical PDF report.
    """
    file_path = os.path.join("backend/outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report file not found.")
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )


# ── Production UI Serving ─────────────────────────────────────────────────────
# After `npm run build`, the React SPA lives in ui/dist/.
# We serve it directly from FastAPI so the entire app runs on a single port.
# The catch-all route below handles both Vite bundles and public assets.

UI_DIST = Path("ui/dist")
UI_PUBLIC_ASSETS = Path("ui/public/assets")



@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """
    SPA catch-all: serves the React index.html for any non-API, non-static route.
    This enables client-side routing (React Router, hash routing, etc.).
    """
    # Try to serve static files from ui/dist first
    static_file = UI_DIST / full_path
    if static_file.is_file():
        return FileResponse(str(static_file))

    # Try ui/public/assets
    public_file = UI_PUBLIC_ASSETS / full_path.removeprefix("assets/")
    if public_file.is_file():
        return FileResponse(str(public_file))

    # Fallback: serve index.html (SPA client-side routing)
    index_path = UI_DIST / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())

    # No built UI yet — return a helpful message
    return JSONResponse(
        status_code=200,
        content={
            "message": "DRISHYA API is running. Build the UI with: cd ui && npm run build",
            "api_docs": "/docs",
            "health": "/api/health"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=True)

