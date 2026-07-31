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

REQUIRED_MODEL_FILES = ["spread_model.joblib", "total_model.joblib"]
# Calibration files are optional — score_week.py falls back to the old guessed
# linear EV formula if they're missing, so a missing calibration file should
# warn, not crash the whole run the way a missing prediction model must.
OPTIONAL_MODEL_FILES = ["spread_calibration.joblib", "total_calibration.joblib",
                        "movement_model.joblib", "movement_features.joblib",
                        "margin_model.joblib", "margin_features.joblib",
                        "total_movement_model.joblib", "total_movement_features.joblib"]

for filename in REQUIRED_MODEL_FILES:
    print(f"Downloading {filename}...")
    data = sb.storage.from_("models").download(filename)
    dest = MODELS_DIR / filename
    dest.write_bytes(data)
    print(f"  Saved to {dest}  ({len(data)/1_048_576:.1f} MB)")

for filename in OPTIONAL_MODEL_FILES:
    try:
        print(f"Downloading {filename}...")
        data = sb.storage.from_("models").download(filename)
        dest = MODELS_DIR / filename
        dest.write_bytes(data)
        print(f"  Saved to {dest}  ({len(data)/1_048_576:.1f} MB)")
    except Exception as exc:
        print(f"  WARNING: could not download {filename}: {exc}")

print("Models ready.")
