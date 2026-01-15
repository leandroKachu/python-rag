from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_document
from app.ingest import load_vector_store
from app.llm import ask_llm
from app.delete_faiss import wipe_all_data
from contextlib import asynccontextmanager
from app.config import UPLOADS_DIR
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 1. INGESTÃO (PDF/TXT)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    load_vector_store(
        model.get_sentence_embedding_dimension()
    )
    print("🚀 API iniciada")

    yield

    # SHUTDOWN
    print("🛑 API finalizada")
app = FastAPI(title="Local RAG + Gemini", lifespan=lifespan)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = UPLOADS_DIR / file.filename
    
    print(file_path)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_document(str(file_path))

    return {"status": "ok", "file": file.filename}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: QueryRequest):
    answer = ask_llm(payload.question)
    return {"answer": answer}

@app.delete("/wipe")
def wite():
    wipe_all_data()
    return {"status": "all local data removed"}

