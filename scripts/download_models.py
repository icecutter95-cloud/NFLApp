"""
Download trained models from Supabase Storage to the local models/ directory.
Used by GitHub Actions before running score_week.py.

Usage:
    python download_models.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from supabase import create_client

SUPABASE_URL         = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY", "")
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

for filename in ["spread_model.joblib", "total_model.joblib"]:
    print(f"Downloading {filename}...")
    data = sb.storage.from_("models").download(filename)
    dest = MODELS_DIR / filename
    dest.write_bytes(data)
    print(f"  Saved to {dest}  ({len(data)/1_048_576:.1f} MB)")

print("Models ready.")
