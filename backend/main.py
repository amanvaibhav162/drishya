from typing import Any, Dict
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    name: str = Form("Ramesh Kumar"),
    phone: str = Form("+91 98765 43210"),
    abha_id: str = Form("91-4820-1940-52")
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

        patient_info = {
            "name": name,
            "phone": phone,
            "abha_id": abha_id,
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

        biomarkers = result.get("biomarkers", {})
        db_record = {
            "patient_name": name,
            "patient_phone": phone,
            "abha_id": abha_id,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=True)
