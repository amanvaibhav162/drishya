# 🩺 DRISHYA — AI-Powered Diabetic Retinopathy Screening

> **दृष्टि** *(Drishya)* — Sanskrit for "Vision"
>
> A full-stack clinical AI system for rural diabetic retinopathy screening. Built for deployment in Primary Health Centres (PHCs) and low-resource settings where specialist ophthalmologists are unavailable.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Model](#ai-model)
- [Pipeline Stages](#pipeline-stages)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [UI Modes](#ui-modes)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [Training Datasets](#training-datasets)

---

## Overview

DRISHYA is an end-to-end clinical screening system that takes a **retinal fundus photograph** as input and outputs:

- **ICDR DR Grade** (0–4, 5-class classification)
- **Referable DR flag** (Grade ≥ 2 → refer to ophthalmologist)
- **Image Quality Assessment (IQA)** score with reject/enhance/accept decision
- **Lesion segmentation masks** for Microaneurysms, Exudates, Hemorrhages, and Sub-retinal Exudates
- **Grad-CAM++ explainability heatmap** overlaid on the fundus image
- **Automated 1-page clinical PDF report** uploaded to Supabase cloud

The system is designed for **health workers** with no medical training — they register a patient, upload a photo, and receive a clinical-grade report in seconds.

---

## Key Features

| Feature | Detail |
|---|---|
| 🧠 **Multi-Task AI** | Simultaneous DR grading + 4-channel lesion segmentation in a single forward pass |
| 🔍 **Image Quality Gate** | MATLAB IQA pipeline rejects/enhances low-quality scans before inference |
| 🌡️ **Grad-CAM++ XAI** | Retinal-FOV-masked explainability heatmap for clinical trust |
| 📄 **Clinical PDF Report** | Auto-generated 1-page report with biomarkers, heatmap, and referral recommendation |
| ☁️ **Supabase Sync** | All reports and scans auto-uploaded to Supabase Storage; records saved to DB |
| 🖥️ **Single-Port Deployment** | FastAPI serves both the REST API and the React SPA from port 8000 |
| 👩‍⚕️ **Dual UI Modes** | Health Worker Portal (screening) + Judge Inspector (explainability & pipeline debug) |
| 📱 **Offline Fallback** | Runs fully locally without Supabase; falls back to local file serving |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DRISHYA System                           │
│                                                                 │
│  ┌──────────────┐    HTTP POST     ┌─────────────────────────┐  │
│  │  React SPA   │ ──────────────► │   FastAPI Backend        │  │
│  │  (Vite/JSX)  │                 │   backend/main.py        │  │
│  │              │ ◄────────────── │   port 8000              │  │
│  └──────────────┘    JSON Result  └──────────┬──────────────┘  │
│                                              │                  │
│                                   ┌──────────▼──────────────┐  │
│                                   │    ModelService          │  │
│                                   │  model_service.py        │  │
│                                   │                          │  │
│                                   │  1. IQA (Python)         │  │
│                                   │  2. CLAHE Preprocessing  │  │
│                                   │  3. EfficientNetV2 MTL   │  │
│                                   │  4. Lesion Segmentation  │  │
│                                   │  5. Grad-CAM++           │  │
│                                   │  6. Biomarker Extraction │  │
│                                   │  7. PDF Report (ReportLab│  │
│                                   └──────────┬──────────────┘  │
│                                              │                  │
│                              ┌───────────────▼────────────┐    │
│                              │        Supabase             │    │
│                              │  Storage: PDF + Heatmaps    │    │
│                              │  DB Table: screenings       │    │
│                              └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## AI Model

### EfficientNetV2-B0 Multi-Task Student Model (`DRISHYAEfficientNetV2MTL`)

The core AI model is a **lightweight multi-task student model** distilled for edge/PHC deployment.

| Property | Value |
|---|---|
| **Architecture** | EfficientNetV2-B0 + scSE UNet Decoder + MSAG |
| **Parameters** | ~7.67M |
| **Classification Head** | 5-class ICDR DR Grading (Grades 0–4) |
| **Segmentation Head** | 4-channel lesion masks (MA, EX, HE, SE) |
| **Attention** | Spatial & Channel Squeeze-Excitation (scSE) |
| **Localization** | Multi-Scale Attention Gate (MSAG) |
| **Explainability** | Grad-CAM++ with retinal FOV masking |
| **Checkpoint** | `models/student_mtl_lcnet_best.pth` |
| **Export** | ONNX-compatible |

#### ICDR Grade Mapping

| Grade | Label | Referral |
|---|---|---|
| 0 | No DR | Routine rescreening in 12 months |
| 1 | Mild NPDR | Monitor, rescreening in 6 months |
| 2 | Moderate NPDR | Refer to ophthalmologist within 4 weeks |
| 3 | Severe NPDR | Urgent referral within 1 week |
| 4 | PDR (Proliferative) | Emergency referral — same day |

#### Lesion Segmentation Channels

| Channel | Lesion | Clinical Significance |
|---|---|---|
| 0 | Microaneurysms (MA) | Earliest sign of DR |
| 1 | Hard Exudates (EX) | Lipid leakage from damaged vessels |
| 2 | Hemorrhages (HE) | Intraretinal bleeding |
| 3 | Soft Exudates (SE) | Cotton-wool spots, nerve fiber infarcts |

---

## Pipeline Stages

DRISHYA operates as a **multi-stage clinical pipeline**:

### Stage 1 — Pre-Processing Pipeline (`pre_processing_pipeline/`)

> Implemented in **MATLAB** for signal-processing-grade quality control.

1. **Retinal Mask Extraction** (`extract_retinal_mask.m`) — Crops fundus image to the circular retinal FOV, removes black borders.
2. **Image Quality Assessment** (`assess_quality.m`, `evaluate_iqa.m`) — Computes focus sharpness (F), contrast (C), and composite quality score (Q).
3. **IQA Decision Engine** — Three-tier decision:
   - `Q < 0.76` → **UNGRADABLE** — Reject, request retake
   - `0.76 ≤ Q < 0.78` → **BORDERLINE** — Apply adaptive CLAHE enhancement
   - `Q ≥ 0.78` → **ACCEPTABLE** — Pass directly to Stage 2
4. **Adaptive Enhancement** (`adaptive_enhance.m`) — CLAHE + Ben Graham preprocessing for borderline images.

Entry point: `pre_processing_pipeline/run_iqa_enhancement_pipeline.m`

---

### Stage 2 — Lesion Feature Pipeline (`pipeline_stage2/`)

> Offline training-time pipeline combining **Python + MATLAB** for dataset preparation.

Processes five major DR datasets: **APTOS, DRIVE, EyePACS, IDRiD, MESSIDOR-2**

Key steps:
- **Optic Disc & Macula landmark extraction** (`extract_landmarks.m`)
- **Vessel segmentation** via Frangi filter (`extract_vessels.m`)
- **Hemorrhage classification** (`classify_hemorrhages.m`)
- **Lesion detection & mask generation** (`detect_lesions.m`)
- **Multicore batch processing** (`run_stage2_multicore.py`) — parallel processing with Python `multiprocessing`
- **Mask integrity verification** (`verify_masks.py`)

Entry point: `pipeline_stage2/run_stage2_master.py`

---

### Stage 3 — Clinical Inference Pipeline (`backend/model_service.py`)

> Real-time inference on patient upload. Runs on **CPU or CUDA GPU**.

Sequence:
1. **Retinal FOV mask** computation (`create_fundus_fov_mask`)
2. **CLAHE + Ben Graham** preprocessing
3. **EfficientNetV2 MTL forward pass** → (logits, lesion masks)
4. **Grad-CAM++ heatmap** generation with FOV masking
5. **Colorized lesion overlay** (4-channel → RGBA composite)
6. **Biomarker extraction** — MA count, exudate area %, hemorrhage quadrants, macular risk zone
7. **Clinical PDF report** generation via ReportLab
8. **Supabase upload** — PDF to `drishya-reports`, heatmap/overlay to `drishya-scans`

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| API Framework | FastAPI 0.115+ |
| Model Runtime | PyTorch 2.0+ |
| Backbone | timm (EfficientNetV2-B0) |
| Segmentation | segmentation-models-pytorch |
| Explainability | Grad-CAM++ (custom implementation) |
| Image Processing | OpenCV, Albumentations |
| PDF Generation | ReportLab 4.0+ |
| Cloud Database | Supabase (PostgreSQL) |
| Cloud Storage | Supabase Storage |
| ONNX Export | onnx, onnxruntime-gpu |
| Server | Uvicorn |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 19 + Vite 8 |
| Language | JSX (JavaScript) |
| Icons | lucide-react |
| Linting | oxlint |
| Build | Vite (outputs to `ui/dist/`) |

### Pre-Processing
| Component | Technology |
|---|---|
| IQA Pipeline | MATLAB |
| Vessel Segmentation | Frangi filter (scikit-image) |
| Batch Processing | Python multiprocessing |

---

## Project Structure

```
drishya/
│
├── backend/                        # FastAPI backend server
│   ├── main.py                     # API entrypoint, routes, SPA serving
│   ├── model_service.py            # Clinical inference pipeline (ModelService)
│   ├── model_student.py            # EfficientNetV2 MTL model architecture
│   ├── database.py                 # Supabase DB + Storage integration
│   ├── config.py                   # Environment config loader
│   ├── schema.sql                  # Supabase PostgreSQL schema
│   ├── .env                        # Local secrets (git-ignored)
│   └── .env.example                # Environment variable template
│
├── ui/                             # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx                 # Root app, state management, API calls
│   │   ├── components/
│   │   │   ├── HealthWorkerMode.jsx    # Patient registration & screening UI
│   │   │   ├── JudgeInspectorMode.jsx # Pipeline explainability & debug view
│   │   │   ├── PdfPreviewModal.jsx    # Clinical PDF preview modal
│   │   │   └── Sidebar.jsx            # Mode navigation sidebar
│   │   ├── index.css               # Global styles
│   │   └── main.jsx                # React entrypoint
│   ├── public/                     # Static assets
│   ├── report_generator.py         # ReportLab PDF generation (called by backend)
│   ├── package.json                # Node dependencies
│   └── vite.config.js              # Vite build config
│
├── pre_processing_pipeline/        # MATLAB + Python IQA pipeline
│   ├── run_iqa_enhancement_pipeline.m  # Master IQA pipeline
│   ├── extract_retinal_mask.m          # Circular FOV crop
│   ├── assess_quality.m                # Focus + contrast metrics
│   ├── evaluate_iqa.m                  # Q-score decision engine
│   ├── adaptive_enhance.m              # CLAHE enhancement
│   ├── run_test.m / run_test.py        # Test runners
│   ├── batch_process_usb.py            # USB-mount batch processor
│   ├── count_status.py                 # IQA status statistics
│   └── sample_q_scores.py              # Q-score sampling utility
│
├── pipeline_stage2/                # Lesion feature extraction pipeline
│   ├── run_stage2_master.py        # Python master pipeline (single-core)
│   ├── run_stage2_multicore.py     # Parallel multiprocessing pipeline
│   ├── run_stage2_python.py        # Pure Python pipeline variant
│   ├── run_stage2_test.m           # MATLAB test runner
│   ├── extract_landmarks.m         # Optic disc & macula detection
│   ├── extract_vessels.m           # Frangi vessel segmentation
│   ├── detect_lesions.m            # Lesion detection & masking
│   ├── classify_hemorrhages.m      # Hemorrhage quadrant classification
│   ├── verify_masks.py             # Mask integrity verification
│   └── debug_ma.py                 # Microaneurysm debug utility
│
├── models/                         # Model checkpoints (placeholder)
│   └── .gitkeep                    # Add .pth files here (see Setup)
│
├── requirements.txt                # Python dependencies
├── start.sh                        # One-command startup script
├── pyrightconfig.json              # Python type checker config
└── README.md                       # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- MATLAB R2021b+ *(only required for pre-processing pipeline)*
- CUDA-capable GPU *(optional — CPU inference is supported)*

### 1. Clone the repository

```bash
git clone https://github.com/amanvaibhav162/drishya.git
cd drishya
```

### 2. Create Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Linux with a CUDA GPU, `onnxruntime-gpu` will be installed automatically. On macOS/Windows, only CPU inference is used.

### 4. Install frontend dependencies

```bash
cd ui
npm install
cd ..
```

### 5. Configure environment variables

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
MODEL_PATH=models/student_mtl_lcnet_best.pth
PORT=8000
STORAGE_BUCKET_REPORTS=drishya-reports
STORAGE_BUCKET_IMAGES=drishya-scans
```

> **No Supabase?** The system runs in **local fallback mode** — all files are served locally and DB inserts are skipped. Just leave the Supabase fields empty.

### 6. Place the model checkpoint

Download the trained model checkpoint and place it at:

```
models/student_mtl_lcnet_best.pth
```

### 7. Set up Supabase (optional)

Run the SQL schema to create the `screenings` table and storage buckets:

```bash
# Apply the schema to your Supabase project via the SQL editor
# File: backend/schema.sql
```

---

## Running the Application

### Option A — One-command start (recommended)

```bash
chmod +x start.sh
./start.sh
```

This script will:
1. Activate the `.venv` virtual environment
2. Verify the model checkpoint exists
3. Build the React UI (`npm run build`)
4. Start the FastAPI server on port 8000

Then open: **http://localhost:8000**

---

### Option B — Development mode (hot-reload)

Run the backend and frontend in separate terminals:

**Terminal 1 — FastAPI backend:**
```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Vite dev server:**
```bash
cd ui
npm run dev
```

Then open: **http://localhost:5173** (Vite dev server with HMR)

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### `GET /api/health`

Check service health and inference device.

**Response:**
```json
{
  "status": "healthy",
  "service": "DRISHYA AI Engine",
  "device": "cuda:0"
}
```

---

### `POST /api/screen-patient`

Run the full clinical screening pipeline on a retinal fundus image.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | Yes | Retinal fundus photograph (JPG/PNG) |
| `name` | string | No | Patient name (default: "Anonymous Patient") |
| `age` | string | No | Patient age |
| `gender` | string | No | Patient gender |
| `phone` | string | No | Patient phone number |
| `abha_id` | string | No | ABHA health ID |

**Response (success):**
```json
{
  "success": true,
  "iqa_pass": true,
  "q_score": 0.84,
  "grade": 2,
  "grade_title": "Moderate NPDR",
  "confidence": 0.91,
  "referable_dr": true,
  "action": "Refer to Ophthalmologist within 2-4 weeks",
  "biomarkers": {
    "microaneurysms": 12,
    "exudate_area_pct": "2.34%",
    "hemorrhage_quadrants": 1,
    "macular_risk": "Moderate Risk"
  },
  "files": {
    "raw_path": "backend/outputs/raw_...",
    "preprocessed_path": "backend/outputs/preprocessed_...",
    "lesion_path": "backend/outputs/lesion_...",
    "heatmap_path": "backend/outputs/heatmap_...",
    "gradcam_path": "backend/outputs/gradcam_...",
    "pdf_report_path": "backend/outputs/report_..."
  },
  "supabase_synced": true,
  "pdf_download_url": "/api/download-report/report_....pdf"
}
```

**Response (IQA rejection):**
```json
{
  "success": false,
  "iqa_pass": false,
  "q_score": 0.71,
  "grade_title": "Ungradable (Quality Rejected)",
  "action": "RETAKE SCAN REQUIRED"
}
```

---

### `GET /api/screenings`

Retrieve recent screening records from Supabase.

| Param | Default | Description |
|---|---|---|
| `limit` | `50` | Max records to return |

---

### `GET /api/download-report/{filename}`

Download a compiled clinical PDF report.

---

## UI Modes

### Health Worker Portal

The primary clinical interface for health workers at PHCs:

- **Patient Registration** — Name, age, gender, phone, ABHA ID
- **Image Upload** — Drag & drop or file picker for fundus photo
- **Live Pipeline Visualization** — Step-by-step progress (IQA → Grading → XAI → PDF)
- **Results Dashboard** — DR grade, confidence, referral recommendation
- **Clinical PDF** — One-tap download/preview of the full report

### Judge Inspector Mode

A technical/research mode for pipeline transparency:

- **Side-by-side image comparison** — Raw → Preprocessed → Lesion overlay → Heatmap → Grad-CAM
- **Biomarker Panel** — Detailed lesion statistics
- **Pipeline Step Debugger** — Step-by-step execution trace
- **Explainability review** — Grad-CAM++ heatmap analysis

---

## Database Schema

Screening records are stored in the Supabase `screenings` table. See `backend/schema.sql` for the full schema.

Key columns:

| Column | Type | Description |
|---|---|---|
| `id` | UUID | Auto-generated primary key |
| `patient_name` | TEXT | Patient name |
| `patient_age` | TEXT | Patient age |
| `patient_gender` | TEXT | Patient gender |
| `patient_phone` | TEXT | Contact number |
| `abha_id` | TEXT | ABHA health ID |
| `icdr_grade` | INTEGER | DR grade (0–4) |
| `grade_title` | TEXT | Grade label |
| `confidence` | FLOAT | Model confidence (0–1) |
| `referable_dr` | BOOLEAN | Referral flag |
| `iqa_score` | FLOAT | Image quality score |
| `num_microaneurysms` | INTEGER | MA count |
| `exudate_area_pct` | TEXT | Exudate area percentage |
| `pdf_report_url` | TEXT | Supabase Storage URL |
| `gradcam_image_url` | TEXT | Grad-CAM image URL |
| `created_at` | TIMESTAMP | Record creation time |

---

## Configuration

All configuration is managed via environment variables in `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `SUPABASE_URL` | — | Your Supabase project URL |
| `SUPABASE_KEY` | — | Supabase anon/service key |
| `MODEL_PATH` | `models/student_mtl_lcnet_best.pth` | Path to model checkpoint |
| `PORT` | `8000` | FastAPI server port |
| `STORAGE_BUCKET_REPORTS` | `drishya-reports` | Supabase bucket for PDFs |
| `STORAGE_BUCKET_IMAGES` | `drishya-scans` | Supabase bucket for images |

---

## Training Datasets

The Stage 2 lesion pipeline was designed to process the following public DR datasets:

| Dataset | Description |
|---|---|
| **APTOS 2019** | Kaggle Blindness Detection challenge |
| **DRIVE** | Digital Retinal Images for Vessel Extraction |
| **EyePACS** | Large-scale DR grading dataset |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset |
| **MESSIDOR-2** | French multi-centre DR grading dataset |

---

## License

This project is intended for academic and research use. Please ensure compliance with the licenses of all constituent datasets and pretrained model weights before clinical deployment.

---

<div align="center">
  <strong>Built for rural India. Powered by AI. Designed for lives.</strong><br/>
  <em>दृष्य — दृष्टि बचाओ, जीवन बचाओ</em>
</div>
