# 👁️ DRISHYA (दृष्य) — Explainable AI for Diabetic Retinopathy Screening in Rural India

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026-blue?style=flat-square)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/PS-SIH26038-orange?style=flat-square)](https://www.sih.gov.in/)
[![Organization](https://img.shields.io/badge/Org-MathWorks-red?style=flat-square)](https://www.mathworks.com/)
[![Category](https://img.shields.io/badge/Category-Software-green?style=flat-square)](#)
[![Theme](https://img.shields.io/badge/Theme-Clean%20%26%20Green%20Technology-teal?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

An end-to-end, clinically validated, and explainable retinal image analysis pipeline developed for automated **Diabetic Retinopathy (DR)** screening in primary healthcare centers (PHCs) and rural camps across India. Built on a high-efficiency **Hybrid Python $\longleftrightarrow$ MATLAB $\longleftrightarrow$ Simulink** architecture.

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Key Features & Win Criteria](#-key-features--win-criteria)
- [Hybrid Architecture](#-hybrid-architecture)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Preprocessing & Enhancement](#-preprocessing--enhancement)
- [System Requirements](#-system-requirements)
- [Quickstart Guide](#-quickstart-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Preprocessing & Training (Python)](#2-preprocessing--training-python)
  - [3. Export to ONNX](#3-export-to-onnx)
  - [4. MATLAB Clinical Inference & Reporting](#4-matlab-clinical-inference--reporting)
  - [5. Simulink Telemedicine Simulation](#5-simulink-telemedicine-simulation)
- [Clinical Benchmark Targets](#-clinical-benchmark-targets)
- [Contributing & Team](#-contributing--team)

---

## 🏥 Executive Summary

India is home to **77+ million diabetic adults** (the 2nd highest globally), with ~18% affected by Diabetic Retinopathy (DR)—a leading cause of preventable blindness. Early screening can prevent **90% of vision loss**, yet rural India has only **~1 ophthalmologist per 100,000 population**.

Standard black-box deep learning models fail under variable field conditions (uneven illumination, blur, non-mydriatic portable cameras) and lack clinical transparency. **DRISHYA** addresses these challenges through:
1. **Automated Fundus Image Quality Assessment (IQA)** with field-technician recapture feedback.
2. **Adaptive Contrast & Illumination Enhancement** (Ben Graham's local illumination subtraction + CLAHE).
3. **ICDR Severity Grading (Levels 0–4)** and **Referable DR Detection** (>90% Sensitivity, >85% Specificity).
4. **Explainable AI (XAI)** combining Grad-CAM / Grad-CAM++ with sub-pixel lesion segmentations (Microaneurysms, Hemorrhages, Hard/Soft Exudates).
5. **Automated 1-Page Clinical PDF Reports** enabling ophthalmologist review in **<30 seconds**.
6. **Simulink Discrete-Event Simulation** optimizing bandwidth, compute queues, and reviewer allocation for district-level programs serving **100,000+ patients annually**.

---

## 🎯 Key Features & Win Criteria

| Feature | Target Metric / Deliverable | Status |
|---|---|---|
| **Referable DR Sensitivity** | **> 90%** (Clinically non-negotiable) | Target Set |
| **Referable DR Specificity** | **> 85%** | Target Set |
| **Image QA & Recapture** | 3-way triage: `PASS`, `ENHANCE`, `REJECT` with actionable feedback | In Development |
| **Explainability (XAI)** | Grad-CAM heatmaps validated against IDRiD lesion masks | In Development |
| **Clinical Report** | 1-page automated PDF with QA, Grade, Cam overlay & recommendations | In Development |
| **Simulink Simulation** | Telemedicine queueing & capacity model for 100k+ patients/year | In Development |
| **Hardware Compatibility** | Full training & inference optimized for **$\le$ 8 GB VRAM** GPUs | Supported |

---

## 🏗️ Hybrid Architecture

To combine rapid experimentation in PyTorch with MathWorks compliance and workflow simulation, **DRISHYA** utilizes a **Hybrid Workflow**:

```
 ┌────────────────────────────────────────────────────────┐
 │            1. Python / PyTorch GPU Engine              │
 │  • Fast mixed-precision (FP16) training on 8 GB GPU    │
 │  • EfficientNet-B4 / Swin-T backbones                  │
 │  • 5-Fold Stratified Cross-Validation + Augmentation   │
 └──────────────────────────┬─────────────────────────────┘
                            │ Exports to ONNX (.onnx)
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │            2. MATLAB Clinical Pipeline (.m)            │
 │  • importNetworkFromONNX into dlnetwork                │
 │  • Automated Image QA (Laplacian blur, illumination)   │
 │  • Adaptive CLAHE & vessel/lesion segmentation         │
 │  • MATLAB gradCAM generation & overlay                 │
 │  • 1-Page PDF Clinical Screening Report generation     │
 └──────────────────────────┬─────────────────────────────┘
                            │ Workflow parameters
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │       3. Simulink Telemedicine Simulation Model        │
 │  • SimEvents / Stateflow discrete-event simulation     │
 │  • Rural PHCs ──> 4G/2G Bandwidth ──> AI Batch Queue   │
 │  • Recapture loops & Doctor validation (<30s/case)     │
 │  • Resource & cost optimization for 100,000+ patients  │
 └────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```plaintext
drishya/
├── README.md                           # Project documentation & execution guide
├── requirements.txt                     # Python dependencies
├── flake.nix / .envrc                   # Reproducible NixOS dev shell (optional)
│
├── data/                               # Dataset storage (gitignored)
│   ├── raw/                            # APTOS 2019, IDRiD, Messidor-2
│   ├── processed/                      # Preprocessed, cropped, illumination-corrected images
│   └── splits/                         # 5-fold stratified train/val/test splits
│
├── src/
│   ├── python/                         # PyTorch Deep Learning Pipeline
│   │   ├── dataset.py                  # PyTorch Dataset loaders with Ben Graham transforms
│   │   ├── preprocess.py               # Auto-crop, circular mask, illumination normalization
│   │   ├── models.py                   # EfficientNet-B4, Swin-T, U-Net lesion segmenter
│   │   ├── train.py                    # FP16 + Gradient Accumulation training loop (8GB VRAM)
│   │   ├── evaluate.py                 # Sensitivity, specificity, Quadratic Weighted Kappa
│   │   ├── explainability.py           # Grad-CAM / Grad-CAM++ generation
│   │   └── export_onnx.py              # Export trained weights to standard ONNX
│   │
│   ├── matlab/                         # MATLAB Clinical Pipeline
│   │   ├── import_onnx_model.m         # Imports .onnx into MATLAB dlnetwork
│   │   ├── image_qa.m                  # Focus/blur, illumination, FOV quality assessment
│   │   ├── enhance_fundus.m            # CLAHE, illumination correction & green channel filter
│   │   ├── predict_dr.m                # Runs MATLAB inference & ICDR classification
│   │   ├── explain_gradcam.m           # MATLAB gradCAM feature visualization
│   │   ├── generate_report.m           # Generates 1-page clinical PDF report
│   │   └── run_pipeline.m              # End-to-end batch processing runner
│   │
│   └── simulink/                       # Simulink Telemedicine Simulation
│       ├── telemedicine_network.slx    # SimEvents model for rural screening logistics
│       ├── run_simulation.m            # Scripts to vary bandwidth, doctor count, & arrival rates
│       └── parameters.m                # Telemedicine system parameters (100k patients/year)
│
├── models/                             # Saved checkpoints (.pth) and exported ONNX models (.onnx)
├── reports/                            # Generated screening PDF reports & simulation charts
└── tests/                              # Unit and integration tests
```

---

## 📊 Datasets

| Dataset | Images | Annotations | Role | Source |
|---|---|---|---|---|
| **APTOS 2019** | 3,662 | ICDR Grades 0–4 (CSV) | Primary Classification & 5-Fold CV | [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **IDRiD** | 516 | Grades 0–4 + Pixel Masks for MA, HE, EX, SE | Lesion Segmentation & Grad-CAM Validation | [IEEE DataPort IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **Messidor-2** | 1,748 | Referable DR (adjudicated) | External Holdout Validation | [ADCIS Messidor-2](https://www.adcis.net/en/third-party/messidor2/) |
| **EyePACS** | ~35,000 | Grades 0–4 | Optional Domain Pre-training | [Kaggle EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) |

---

## ⚙️ Preprocessing & Enhancement

Fundus images exhibit significant variation across camera sensors. The pipeline normalizes all inputs:

1. **Auto-Crop & Centering**: Thresholding (`gray > 10`) removes useless black borders and centers the retina.
2. **Ben Graham's Illumination Normalization**: Subtracts local average color to eliminate flash variations:
   $$I_{\text{normalized}} = 4 \cdot I - 4 \cdot \text{GaussianBlur}(I, \sigma=10) + 128$$
3. **Circular FOV Masking**: Cleans artifacts outside the $45^\circ$ optical cone.
4. **Adaptive Enhancement (CLAHE)**: Enhances local contrast of microaneurysms and micro-hemorrhages.
5. **Standard Resizing**: Formatted to $384 \times 384$ pixels (preserving fine lesion structures while optimizing VRAM).

---

## 💻 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with **$\ge$ 8 GB VRAM** (e.g., RTX 3060/3070/4060, T4, V100).
- **RAM**: 16 GB minimum (32 GB recommended for dataset caching).
- **Disk Space**: ~50 GB SSD storage.

### Software Requirements
- **Python**: Python 3.10+ with PyTorch 2.x, CUDA 12.x, `timm`, `albumentations`, `opencv-python`, `onnx`.
- **MATLAB**: R2022b or later with:
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Medical Imaging Toolbox
  - Simulink & SimEvents

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/drishya.git
cd drishya

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocessing & Training (Python)
Optimized for **8 GB VRAM** using Mixed Precision (FP16), Batch Size 8, and Gradient Accumulation:
```bash
# Preprocess raw dataset
python src/python/preprocess.py --data_dir data/raw/aptos2019 --out_dir data/processed/aptos2019 --img_size 384

# Train EfficientNet-B4 classifier (Referable DR + ICDR 5-class)
python src/python/train.py \
    --model_name tf_efficientnet_b4.ns_jft_in1k \
    --img_size 384 \
    --batch_size 8 \
    --accum_steps 4 \
    --use_amp \
    --epochs 25 \
    --lr 3e-4
```

### 3. Export to ONNX
```bash
python src/python/export_onnx.py \
    --checkpoint models/best_efficientnet_b4.pth \
    --output models/dr_classifier_b4.onnx \
    --img_size 384
```

### 4. MATLAB Clinical Inference & Reporting
Run the complete clinical pipeline directly in MATLAB or via batch mode:
```bash
matlab -batch "addpath('src/matlab'); run_pipeline('data/samples/sample_01.png', 'models/dr_classifier_b4.onnx');"
```

### 5. Simulink Telemedicine Simulation
Open and simulate the district-level screening workflow:
```matlab
% In MATLAB command prompt:
cd src/simulink
open_system('telemedicine_network.slx')
sim('telemedicine_network.slx')
```

---

## 📈 Clinical Benchmark Targets

| Model / Approach | Dataset | Referable DR Sens. | Referable DR Spec. |
|---|---|---|---|
| Published Baseline (EfficientNet-B4) | APTOS 2019 | 93.2% | 91.5% |
| Published Attention ResNet50 | IDRiD | 94.8% | 90.7% |
| **DRISHYA Target (Our Floor)** | **Holdout / Cross-Dataset** | **> 90.0%** | **> 85.0%** |

---

## 👥 Contributing & Team

- **Project**: DRISHYA (दृष्य)
- **Team**: SIH 2026 MathWorks Problem Statement Team
- **Problem Statement**: SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India
- **Repository**: Maintained under active development for SIH 2026 submissions.
