"""
Pydantic Schemas for DRISHYA AI Screening Backend.

Defines typed request, response, and clinical biomarker models
used across FastAPI routes, OpenAPI documentation, and Supabase integration.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """System health check and hardware compute device status."""

    status: str = Field(
        ...,
        description="Current operational status of the service",
        examples=["healthy"],
    )
    service: str = Field(
        ...,
        description="Name of the backend service",
        examples=["DRISHYA AI Engine"],
    )
    version: str = Field(
        default="1.0.0",
        description="API semantic version",
        examples=["1.0.0"],
    )
    device: str = Field(
        ...,
        description="Active PyTorch execution compute device (e.g., 'cuda:0' or 'cpu')",
        examples=["cuda:0"],
    )
    model_loaded: bool = Field(
        default=True,
        description="Whether the multi-task deep learning model is loaded in memory",
        examples=[True],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "service": "DRISHYA AI Engine",
                "version": "1.0.0",
                "device": "cuda:0",
                "model_loaded": True,
            }
        }
    )


class BiomarkersResponse(BaseModel):
    """Quantitative retinal biomarkers extracted from segmentation masks and vessel geometry."""

    microaneurysms: int = Field(
        default=0,
        ge=0,
        description="Count of detected microaneurysms across all four retinal quadrants",
        examples=[4],
    )
    exudate_area_pct: str = Field(
        default="0.00%",
        description="Percentage of total retinal area occupied by hard lipid exudates",
        examples=["1.25%"],
    )
    hemorrhage_quadrants: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Number of retinal quadrants (0-4) containing intraretinal hemorrhages (for 4:2:1 rule evaluation)",
        examples=[2],
    )
    soft_exudate_area_pct: Optional[str] = Field(
        default="0.00%",
        description="Percentage of retinal area occupied by soft exudates (cotton wool spots)",
        examples=["0.00%"],
    )
    macular_risk: Optional[str] = Field(
        default="Low Risk",
        description="Clinical risk level for diabetic macular edema (DME)",
        examples=["Moderate Risk"],
    )
    macular_detail: Optional[str] = Field(
        default="No lesions in macular zone",
        description="Spatial distribution of lesions relative to the foveal avascular zone (FAZ)",
        examples=["Hard exudates within 1 disc diameter of fovea center"],
    )

    model_config = ConfigDict(extra="ignore")


class ScreeningFilesResponse(BaseModel):
    """Local filesystem paths to diagnostic visual assets and reports."""

    raw_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to raw resized 384x384 fundus image",
        examples=["backend/outputs/20260904_120000_raw.png"],
    )
    preprocessed_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to contrast-enhanced fundus image",
        examples=["backend/outputs/20260904_120000_prep.png"],
    )
    lesion_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to multi-lesion color-coded segmentation map",
        examples=["backend/outputs/20260904_120000_lesions.png"],
    )
    heatmap_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to raw Grad-CAM++ activation heatmap",
        examples=["backend/outputs/20260904_120000_heatmap.png"],
    )
    gradcam_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to Grad-CAM++ blended overlay image",
        examples=["backend/outputs/20260904_120000_gradcam.png"],
    )
    pdf_report_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to compiled 1-page clinical PDF report",
        examples=["backend/outputs/DRISHYA_Report_Patient_20260904_120000.pdf"],
    )

    model_config = ConfigDict(extra="ignore")


class ScreeningResultResponse(BaseModel):
    """Comprehensive diagnostic result returned by the retinal screening pipeline."""

    success: bool = Field(
        ...,
        description="True if the screening pipeline ran without fatal error or IQA rejection",
        examples=[True],
    )
    iqa_pass: bool = Field(
        ...,
        description="True if image quality assessment passed illumination, blur, and contrast checks",
        examples=[True],
    )
    q_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Automated Image Quality Assessment score (0-100 scale)",
        examples=[85.5],
    )
    grade: Union[int, str] = Field(
        ...,
        description="ICDR diabetic retinopathy grade (0 to 4) or 'ungradable'",
        examples=[2],
    )
    grade_title: str = Field(
        ...,
        description="Clinical diagnostic category title",
        examples=["Moderate Non-Proliferative Diabetic Retinopathy (NPDR)"],
    )
    grade_desc: Optional[str] = Field(
        default=None,
        description="Synthesized clinical diagnostic rationale based on detected biomarkers",
        examples=["More than microaneurysms; cotton wool spots, venous beading, or extensive intraretinal hemorrhages."],
    )
    action: Optional[str] = Field(
        default=None,
        description="Actionable clinical referral recommendation or rescreening timeframe",
        examples=["Refer to Ophthalmologist within 2-4 weeks for slit-lamp biomicroscopy."],
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Deep learning model prediction confidence percentage",
        examples=["94.2%"],
    )
    referable_dr: Optional[bool] = Field(
        default=None,
        description="Clinical referral trigger (True if ICDR grade >= 2 or high macular risk)",
        examples=[True],
    )
    biomarkers: Optional[BiomarkersResponse] = Field(
        default=None,
        description="Quantitative retinal biomarker metrics extracted from lesion masks",
    )
    files: Optional[ScreeningFilesResponse] = Field(
        default=None,
        description="Generated diagnostic asset paths on server storage",
    )
    supabase_synced: Optional[bool] = Field(
        default=False,
        description="Indicates if records and report assets were successfully synced to Supabase",
        examples=[True],
    )
    pdf_download_url: Optional[str] = Field(
        default=None,
        description="Relative API URL to download the compiled 1-page clinical PDF report",
        examples=["/api/download-report/DRISHYA_Report_Patient_20260904_120000.pdf"],
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional diagnostic notes or quality retake explanation when IQA fails",
        examples=["RETAKE SCAN: Image is too dark (mean intensity 18.2 < 30.0)."],
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "success": True,
                "iqa_pass": True,
                "q_score": 85.5,
                "grade": 2,
                "grade_title": "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
                "grade_desc": "More than microaneurysms; cotton wool spots or extensive intraretinal hemorrhages.",
                "action": "Refer to Ophthalmologist within 2-4 weeks.",
                "confidence": "94.2%",
                "referable_dr": True,
                "biomarkers": {
                    "microaneurysms": 4,
                    "exudate_area_pct": "1.25%",
                    "hemorrhage_quadrants": 2,
                    "soft_exudate_area_pct": "0.00%",
                    "macular_risk": "Moderate Risk",
                    "macular_detail": "Lesions within 1 disc diameter of fovea",
                },
                "files": {
                    "raw_path": "backend/outputs/20260904_120000_raw.png",
                    "preprocessed_path": "backend/outputs/20260904_120000_prep.png",
                    "lesion_path": "backend/outputs/20260904_120000_lesions.png",
                    "heatmap_path": "backend/outputs/20260904_120000_heatmap.png",
                    "gradcam_path": "backend/outputs/20260904_120000_gradcam.png",
                    "pdf_report_path": "backend/outputs/DRISHYA_Report_Patient_20260904_120000.pdf",
                },
                "supabase_synced": True,
                "pdf_download_url": "/api/download-report/DRISHYA_Report_Patient_20260904_120000.pdf",
                "message": None,
            }
        },
    )


class ScreeningRecordResponse(BaseModel):
    """Historical patient screening record stored in the Supabase 'screenings' table."""

    id: Optional[str] = Field(
        default=None,
        description="Unique UUID primary key generated by Supabase",
        examples=["c9b5d38f-51d2-4e89-8d14-38ff31b8162e"],
    )
    created_at: Optional[str] = Field(
        default=None,
        description="UTC creation timestamp ISO-8601 string",
        examples=["2026-09-04T12:00:00.000Z"],
    )
    patient_name: str = Field(
        ...,
        description="Full name of screened patient",
        examples=["Ramesh Kumar"],
    )
    patient_age: Optional[str] = Field(
        default="N/A",
        description="Patient age in years",
        examples=["54"],
    )
    patient_gender: Optional[str] = Field(
        default="N/A",
        description="Patient biological sex or gender",
        examples=["Male"],
    )
    patient_phone: Optional[str] = Field(
        default="N/A",
        description="Patient contact phone number",
        examples=["+91 9876543210"],
    )
    abha_id: Optional[str] = Field(
        default="N/A",
        description="Ayushman Bharat Health Account ID",
        examples=["14-1234-5678-9012"],
    )
    icdr_grade: int = Field(
        ...,
        ge=0,
        le=4,
        description="ICDR Diabetic Retinopathy numerical grade (0 to 4)",
        examples=[2],
    )
    grade_title: str = Field(
        ...,
        description="Clinical diagnostic category title",
        examples=["Moderate Non-Proliferative DR"],
    )
    confidence: str = Field(
        ...,
        description="Model confidence rating string",
        examples=["94.2%"],
    )
    referable_dr: bool = Field(
        ...,
        description="True if patient requires tertiary referral",
        examples=[True],
    )
    iqa_score: float = Field(
        ...,
        description="Image quality assessment score (0-100)",
        examples=[85.5],
    )
    num_microaneurysms: Optional[int] = Field(
        default=0,
        description="Count of detected microaneurysms",
        examples=[4],
    )
    exudate_area_pct: Optional[str] = Field(
        default="0.00%",
        description="Hard exudate area percentage",
        examples=["1.25%"],
    )
    pdf_report_url: Optional[str] = Field(
        default=None,
        description="URL for PDF clinical report (Supabase Storage public URL or local URL)",
        examples=["https://xyz.supabase.co/storage/v1/object/public/drishya-reports/DRISHYA_Report_Patient_20260904_120000.pdf"],
    )
    gradcam_image_url: Optional[str] = Field(
        default=None,
        description="URL for Grad-CAM++ visualization asset (Supabase Storage or local path)",
        examples=["https://xyz.supabase.co/storage/v1/object/public/drishya-scans/20260904_120000_gradcam.png"],
    )

    model_config = ConfigDict(extra="allow", from_attributes=True)


class ErrorResponse(BaseModel):
    """Standardized error response payload across all API endpoints."""

    detail: str = Field(
        ...,
        description="Human-readable error explanation",
        examples=["Invalid image file format. Please upload a valid retinal fundus image."],
    )
    status_code: int = Field(
        ...,
        description="HTTP status code associated with the error",
        examples=[400],
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Machine-readable error category or identifier",
        examples=["bad_request"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid image file format. Please upload a valid retinal fundus image.",
                "status_code": 400,
                "error_type": "bad_request",
            }
        }
    )


class SPAFallbackResponse(BaseModel):
    """Fallback response served when the production React frontend bundle is not yet built."""

    message: str = Field(
        ...,
        description="Informational status message",
        examples=["DRISHYA API is running. Build the UI with: cd ui && npm run build"],
    )
    api_docs: str = Field(
        ...,
        description="Relative URL path to interactive OpenAPI documentation",
        examples=["/docs"],
    )
    health: str = Field(
        ...,
        description="Relative URL path to health check endpoint",
        examples=["/api/health"],
    )
