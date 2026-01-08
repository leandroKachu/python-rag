import faiss
import numpy as np
import time
import sys
from sentence_transformers import SentenceTransformer


document= """
O Pix é um sistema de pagamentos instantâneos criado pelo Banco Central do Brasil.
Ele permite transferências 24 horas por dia, 7 dias por semana.
As transações são concluídas em poucos segundos.
O Pix pode ser usado por pessoas físicas e empresas.
"""
chunks =[]
model = SentenceTransformer("all-MiniLM-L6-v2")

def animated_print(msg, width=60, delay=0.01):
    for i  in range(1, width + 1):
        sys.stdout.write("\r" + "="* i)
        sys.stdout.flush()
        time.sleep(delay)
    
    print()
    print(msg.center(width))
    print("=" * width + "\n")

# =========================
# 2. CHUNKING
# =========================

def read_text(path):
    with open(path) as f:
        return f.read()
    

def chunk_text(text, chunk_size=120):
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

def create_embeddings(chunks):
    animated_print("CHUNKS")

    for i, chunk in enumerate(chunks):
        animated_print(f"[{i}] {chunk}")

    # =========================
    # 3. EMBEDDINGS
    # =========================
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    animated_print(f"\n Dimensions of embeddings: {embeddings.shape}")
    return embeddings

# =========================
# 4. INDEXAÇÃO VETORIAL
# =========================

def index_vetor(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    animated_print(f"Total of vetors index: { index.ntotal}")
    return index

# =========================
# 5. BUSCA SEMÂNTICA
# =========================

def question_request(query):
    animated_print(f"Question: {query}")
    
    embeddings =  create_embeddings(chunks)
    index = index_vetor(embeddings)
    query_embeddings = model.encode([query])

    k = 3 ## número de resultados

    distances, indices = index.search(np.array(query_embeddings), k)

    animated_print("Results of search")
    for i, d in zip(indices[0], distances[0]):
        animated_print(f" Index: {i}, Distances: {d}, Result: {chunks[i]}")
    

chunk_text(document)
question_request("Pix funciona á noite?")
