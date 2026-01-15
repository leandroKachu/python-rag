import shutil
from app.config import STORAGE_DIR

def wipe_all_data():
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        STORAGE_DIR.mkdir()
    print("Todos os dados locais foram removidos.")
