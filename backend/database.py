import os
from supabase import create_client, Client
from backend.config import SUPABASE_URL, SUPABASE_KEY, STORAGE_BUCKET_REPORTS, STORAGE_BUCKET_IMAGES

supabase: Client | None = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase successfully!")
    except Exception as e:
        print(f"Warning: Failed to connect to Supabase: {e}")
else:
    print("Supabase credentials not configured in .env. Running in local fallback mode.")


import json
from datetime import datetime

LOCAL_DB_PATH = "backend/outputs/screenings_local.json"


def _read_local_screenings(limit: int = 50):
    if not os.path.exists(LOCAL_DB_PATH):
        return []
    try:
        with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        # Sort descending by created_at
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records[:limit]
    except Exception as e:
        print(f"Error reading local screenings storage: {e}")
        return []


def _save_local_screening(record_data: dict):
    os.makedirs(os.path.dirname(LOCAL_DB_PATH), exist_ok=True)
    records = []
    if os.path.exists(LOCAL_DB_PATH):
        try:
            with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Check if record with same id or created_at already exists, otherwise prepend
    new_record = dict(record_data)
    if "created_at" not in new_record:
        new_record["created_at"] = datetime.now().isoformat()
    if "id" not in new_record:
        new_record["id"] = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Prepend to records
    records.insert(0, new_record)

    try:
        with open(LOCAL_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"Error saving to local screenings storage: {e}")

    return new_record


def save_screening_record(record_data: dict):
    """
    Saves a patient screening record to both local persistent storage and Supabase 'screenings' table.
    """
    # Always save to local storage first for resilient offline-first support
    saved_local = _save_local_screening(record_data)

    if not supabase:
        print("Local Mode: Saved screening to local storage. Supabase client offline.")
        return saved_local

    try:
        response = supabase.table("screenings").insert(record_data).execute()
        print("✓ Successfully saved record to Supabase DB.")
        return response.data
    except Exception as e:
        print(f"Notice: Supabase DB insert encountered ({e}). Saved to local screening storage.")
        # If columns patient_age / patient_gender are not yet created in remote table, retry without them
        if "patient_age" in record_data or "patient_gender" in record_data:
            try:
                fallback_record = {k: v for k, v in record_data.items() if k not in ("patient_age", "patient_gender")}
                fallback_resp = supabase.table("screenings").insert(fallback_record).execute()
                print("✓ Successfully saved record using schema fallback.")
                return fallback_resp.data
            except Exception as e2:
                print(f"Fallback insert to Supabase encountered: {e2}")
        return saved_local


def get_all_screenings(limit: int = 50):
    """
    Retrieves recent patient screening records from Supabase (or local persistent storage when offline).
    """
    if supabase:
        try:
            response = supabase.table("screenings").select("*").order("created_at", desc=True).limit(limit).execute()
            if response and response.data and len(response.data) > 0:
                return response.data
        except Exception as e:
            print(f"Notice: Supabase fetch error ({e}). Returning records from local storage.")

    return _read_local_screenings(limit=limit)


def upload_file_to_supabase(file_path: str, bucket_name: str, remote_filename: str):
    """
    Uploads a local image or PDF file to a Supabase Storage bucket.
    """
    if not supabase or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        content_type = "application/pdf" if file_path.endswith(".pdf") else "image/png"
        
        supabase.storage.from_(bucket_name).upload(
            path=remote_filename,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        public_url = supabase.storage.from_(bucket_name).get_public_url(remote_filename)
        return public_url
    except Exception as e:
        print(f"Error uploading file to Supabase storage ({bucket_name}): {e}")
        return None
