import faiss
import pdfplumber
import numpy as np
import json
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from app.config import STORAGE_DIR

# from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 100
OVERLAP = 25
MIN_CHARS = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.75
chunks = []
index = None
dimension = None

model = SentenceTransformer(MODEL_NAME)
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_PATH = STORAGE_DIR / "index.faiss"
def read_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else: 
        raise ValueError(f"Formated not supported: {ext}")


def chunk_text(text, source_file="unknown"):
    """
    Divide texto em chunks respeitando limites de sentenças.
    Não quebra no meio de uma frase.
    """
    # Divide em sentenças
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    local_chunks = []
    chunk_id = len(chunks)
    current_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # Verifica se adicionar essa sentença ultrapassaria o limite
        would_exceed = current_word_count + sentence_word_count > CHUNK_SIZE
        
        if current_sentences and would_exceed:
            # Finaliza o chunk atual
            chunk_text = " ".join(current_sentences)
            local_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_file,
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
            
            # Cria overlap pegando últimas sentenças
            overlap_sentences = []
            overlap_words = 0
            
            for sent in reversed(current_sentences):
                sent_words = len(sent.split())
                if overlap_words + sent_words <= OVERLAP:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                else:
                    break
            
            # Reinicia com o overlap
            current_sentences = overlap_sentences
            current_word_count = overlap_words
        
        # Adiciona a sentença atual
        current_sentences.append(sentence)
        current_word_count += sentence_word_count
    
    # Processa o último chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        
        # Se for muito pequeno e já existir chunk anterior, mescla
        if len(chunk_text) < MIN_CHARS and local_chunks:
            local_chunks[-1]["text"] += " " + chunk_text
        else:
            local_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_file,
                    "chunk_id": chunk_id
                }
            })
    return local_chunks

def ingest_document(path: str):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    text = read_file(path)
    new_chunks = chunk_text(text, source_file=path)

    embeddings = model.encode(
        [c["text"] for c in new_chunks],
        normalize_embeddings=True
    ).astype("float32")

    if Path(INDEX_PATH).exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(str(CHUNKS_PATH), "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        all_chunks = []

    index.add(embeddings)
    all_chunks.extend(new_chunks)

    faiss.write_index(index, str(INDEX_PATH))
    with open(str(CHUNKS_PATH), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return len(new_chunks)

def load_vector_store(dimension: int):
    global index, all_chunks
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if Path(INDEX_PATH).exists() and Path(CHUNKS_PATH).exists():
        index = faiss.read_index(str(INDEX_PATH))

        with open(str(CHUNKS_PATH), "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        if index.ntotal != len(all_chunks):
            raise RuntimeError(
                f"Inconsistência FAISS ({index.ntotal}) vs chunks ({len(all_chunks)})"
            )
        print("Vector store carregado do disco")
    else:
        index = faiss.IndexFlatIP(dimension)
        all_chunks = []
        print("Vector store vazio (primeira execução)")

def retrieve_context(query: str, top_k=3) -> str:
    query_embedding = model.encode([query], normalize_embeddings=True)

    scores, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    contexts = []
    
    for idx, score in zip(indices[0], scores[0]):
        chunk = all_chunks[idx]
        print(f"- {chunk}")
        contexts.append(
            f"- {chunk['text']} (fonte: {chunk['metadata']['source']}, score: {score:.3f})"
        )

    return "\n".join(contexts)