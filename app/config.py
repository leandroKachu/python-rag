from pathlib import Path

BASE_DIR = Path("/app/data")

STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = BASE_DIR / "uploads"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
