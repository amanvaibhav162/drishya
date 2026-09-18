# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
- **Primary:** Frontline ASHA workers (Accredited Social Health Activists), Primary Health Centre (PHC) nurses, and rural vision technicians with zero formal ophthalmology training.
- **Secondary:** Remote tele-ophthalmologists and district medical officers reviewing escalated referable cases.
- **Patients:** Rural screening attendees, primarily diabetic adults in underserved areas with limited access to retinal specialists.

## Product Purpose
DRISHYA (दृष्टि — Sanskrit for "Vision") is an edge-ready, autonomous clinical AI screening and triage platform for Diabetic Retinopathy (DR). It exists to eliminate preventable blindness in rural India by providing sub-3-second point-of-care clinical triage, generating immediate 1-page clinical reports, and accurately filtering out normal eyes so scarce ophthalmologists only review patients requiring intervention.

## Positioning
Unlike black-box cloud AI diagnostic APIs that require constant high-speed internet and external per-scan fees, DRISHYA is a 100% offline-first edge screening appliance that simultaneously computes 5-class ICDR severity, 4-channel lesion segmentation, and FOV-masked Grad-CAM++ explainability in a single forward pass, packaged in a tri-lingual interface tailored for Indian frontline health workers.

## Operating Context
- **Deployment Setting:** Rural PHCs, sub-centres, and temporary mobile screening camps with variable power, dust, and non-mydriatic handheld/tabletop fundus cameras.
- **Connectivity:** 100% offline edge deployment on budget laptops/tablets (Intel Core i3/i5, 8GB RAM) with zero internet dependence.
- **Operational Ritual:** Patient registration -> Fundus photo capture/upload -> Automated Quality Check & CLAHE -> Instant AI Inference (< 3s) -> Print/Display 1-Page Clinical Diagnostic Slip -> Clear patient or trigger urgent referral.

## Capabilities and Constraints
- **Core Capabilities:**
  - Automated Image Quality Assessment (IQA: focus variance & illumination gatekeeper).
  - 5-Class ICDR DR Severity grading (Grade 0: Normal to Grade 4: Proliferative DR).
  - Simultaneous 4-channel lesion segmentation (Microaneurysms, Hard Exudates, Hemorrhages, Sub-retinal Exudates).
  - Retinal-FOV-masked Grad-CAM++ neural activation maps.
  - Quantitative biomarker extraction (MA counts, exudate area %, hemorrhage distribution).
  - Real-time zero-latency tri-lingual switching: English, Hindi (हिन्दी), and Marathi (मराठी).
  - Dual-purpose 1-page clinical diagnostic PDF report generation (instant triage banner + detailed clinical biomarkers and heatmaps).
  - Dual portals: Frontline Health Worker Portal (prioritized) + Judge/Specialist Inspector Mode.
- **Technical Constraints:**
  - Must run inference entirely on edge CPU in < 1 second (and sub-100 ms for district simulation compliance).
  - Single-port serving: FastAPI serves both the production REST API and compiled React SPA from port 8000.
  - Zero external API dependency at screening runtime.

## Brand Commitments
- **Name:** DRISHYA (दृष्टि) — Sanskrit for "Vision".
- **Visual Identity:** Clinical medical teal/cyan accents, clean dark slate surfaces for fundus inspection, high-contrast readable typography for rural health workers.
- **Linguistic Parity:** Equal diagnostic authority across English, Hindi, and Marathi terminology.

## Evidence on Hand
- Full-stack React 19 + Vite 8 frontend with tri-lingual i18n dictionaries (`ui/src/i18n/translations.js`).
- FastAPI backend (`backend/main.py`, `backend/model_service.py`) with integrated student models (`models/student_mtl_lcnet_best.pth`, `models/drishya_student_effnetv2.onnx`).
- ReportLab-based 1-page clinical diagnostic PDF generator (`ui/report_generator.py`).
- MATLAB / Python pre-processing and lesion verification pipelines (`pre_processing_pipeline/`, `pipeline_stage2/`).
- Documented simulation and architecture specifications (`README.md`, `README_APPROACH2.md`).

## Product Principles
1. **ASHA-First Usability:** Every primary workflow must be operable by a health worker with zero ophthalmology training in under 3 clicks.
2. **Deterministic Triage Clarity:** Critical clinical status (Normal vs. Refer to Specialist) must be glaringly obvious at a glance through unmistakable color tokens and vernacular copy.
3. **No Unexplained Black Boxes:** Every referral decision must be visually corroborated by Grad-CAM neural attention and segmented lesion masks.
4. **Resilient Offline Autonomy:** The tool must function completely when disconnected from internet, cloud databases, or power stability.

## Accessibility & Inclusion
- Multilingual accessibility with native script rendering (Devanagari for Hindi and Marathi).
- High visual contrast against dark fundus backgrounds (WCAG AA/AAA compliant text).
- Responsive adaptation down to mobile viewports (`< 640px`) and tablet drawer navigation (`< 1000px`) for handheld PHC devices.
