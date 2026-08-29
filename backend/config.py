import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
PORT = int(os.getenv("PORT", 8000))
STORAGE_BUCKET_REPORTS = os.getenv("STORAGE_BUCKET_REPORTS", "drishya-reports")
STORAGE_BUCKET_IMAGES = os.getenv("STORAGE_BUCKET_IMAGES", "drishya-scans")
