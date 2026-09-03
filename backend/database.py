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


def save_screening_record(record_data: dict):
    """
    Saves a patient screening record to the Supabase 'screenings' table.
    """
    if not supabase:
        print("Local Mode: Skipping Supabase DB insert.")
        return record_data

    try:
        response = supabase.table("screenings").insert(record_data).execute()
        return response.data
    except Exception as e:
        print(f"Error saving record to Supabase DB: {e}")
        # If columns patient_age / patient_gender are not yet created in remote table, retry without them
        if "patient_age" in record_data or "patient_gender" in record_data:
            try:
                fallback_record = {k: v for k, v in record_data.items() if k not in ("patient_age", "patient_gender")}
                fallback_resp = supabase.table("screenings").insert(fallback_record).execute()
                print("✓ Successfully saved record using schema fallback.")
                return fallback_resp.data
            except Exception as e2:
                print(f"Error saving fallback record to Supabase DB: {e2}")
        return None


def get_all_screenings(limit: int = 50):
    """
    Retrieves recent patient screening records from Supabase.
    """
    if not supabase:
        return []

    try:
        response = supabase.table("screenings").select("*").order("created_at", desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching screenings from Supabase: {e}")
        return []


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
