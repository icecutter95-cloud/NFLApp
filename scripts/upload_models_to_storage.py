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

MODEL_FILES = [
    "spread_model.joblib", "total_model.joblib",
    "spread_calibration.joblib", "total_calibration.joblib",
    # Line-movement model + its exact feature order (CLV tracking).
    "movement_model.joblib", "movement_features.joblib",
    "margin_model.joblib", "margin_features.joblib",
    "total_movement_model.joblib", "total_movement_features.joblib",
    # Added 2026-08-04. NFL spreads now run the residual model; the old
    # rule's margin model ships too so it can run as a shadow arm. The cfb_*
    # artifacts power the college tab, which flags nothing as a bet.
    "nfl_residual_model.joblib",
    "nfl_residual_features.joblib",
    "nfl_margin_shadow_model.joblib",
    "nfl_margin_shadow_features.joblib",
    "nfl_total_movement_model.joblib",
    "nfl_total_movement_features.joblib",
    "cfb_movement_model.joblib",
    "cfb_movement_features.joblib",
    "cfb_margin_model.joblib",
    "cfb_margin_features.joblib",
    "cfb_total_residual_model.joblib",
    "cfb_total_residual_features.joblib",

]

for filename in MODEL_FILES:
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
