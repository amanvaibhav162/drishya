# DRISHYA Backend Service

FastAPI backend service connecting the **PyTorch Multi-Task Deep Learning Model**, **Grad-CAM++ Explainability Engine**, **Supabase Cloud Database & Storage**, and the **1-Page Clinical PDF Generator**.

---

## 🛠️ Tech Stack & Services

* **Framework**: FastAPI + Uvicorn
* **Deep Learning Engine**: PyTorch + `timm` (EfficientNet-B4 Multi-Task)
* **Explainability**: Grad-CAM++ (2nd & 3rd order partial derivatives)
* **Cloud Database & Storage**: Supabase (`supabase-py`)
* **Clinical Reporting**: ReportLab PDF Engine
* **Image Processing**: OpenCV + NumPy

---

## 📁 Directory Structure

```
backend/
├── main.py              # FastAPI application and API routes
├── model_service.py     # IQA, Preprocessing, PyTorch Model & Grad-CAM++ engine
├── database.py          # Supabase client, storage upload & database sync
├── config.py            # Environment configuration loader
├── schema.sql           # SQL script to set up Supabase database & storage
└── .env.example         # Template for Supabase URL and Keys
```

---

## 🚀 Quick Setup & Execution

### 1. Configure Supabase Credentials
Copy `.env.example` to `.env`:
```bash
cp backend/.env.example backend/.env
```
Open `backend/.env` and insert your Supabase project credentials:
```ini
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-key
MODEL_PATH=models/best_model.pth
PORT=8000
```

### 2. Set Up Database Schema in Supabase
1. Log into your [Supabase Dashboard](https://app.supabase.com).
2. Go to **SQL Editor** and paste the contents of `backend/schema.sql`. Click **Run**.
3. Go to **Storage** and create two public buckets:
   * `drishya-reports`
   * `drishya-scans`

### 3. Start the Backend Server
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & device status (`cuda` / `cpu`). |
| `POST` | `/api/screen-patient` | Multipart upload (image + patient info). Runs IQA $\rightarrow$ AI $\rightarrow$ Grad-CAM++ $\rightarrow$ PDF report $\rightarrow$ Supabase sync. |
| `GET` | `/api/screenings` | Retrieves past patient screening records from Supabase. |
| `GET` | `/api/download-report/{filename}` | Downloads the generated 1-page clinical PDF report. |
| `GET` | `/docs` | Interactive Swagger API documentation UI. |
