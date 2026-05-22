"""
Upload trained model files to Supabase Storage ('models' bucket).
Run this once after training, and again whenever you retrain.

Usage:
    python upload_models_to_storage.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from supabase import create_client

SUPABASE_URL      = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY", "")
MODELS_DIR = Path(__file__).parent.parent / "models"

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Ensure bucket exists (private)
try:
    sb.storage.create_bucket("models", options={"public": False})
    print("Created 'models' storage bucket")
except Exception:
    print("'models' bucket already exists")

for filename in ["spread_model.joblib", "total_model.joblib"]:
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found — run train_models.py first")
        continue

    data = path.read_bytes()
    size_mb = len(data) / 1_048_576

    # Remove old version first (storage upsert isn't always available)
    try:
        sb.storage.from_("models").remove([filename])
    except Exception:
        pass

    sb.storage.from_("models").upload(
        filename, data,
        file_options={"content-type": "application/octet-stream"},
    )
    print(f"  Uploaded {filename}  ({size_mb:.1f} MB)")

print("\nModels are now in Supabase Storage → GitHub Actions can download them.")
