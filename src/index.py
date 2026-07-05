"""
index.py — Embed chunks and build a FAISS vector index.

Embedding model choice (defend this in the report):
  Model: sentence-transformers/all-MiniLM-L6-v2
  - Free & local: no API cost, runs on CPU on a normal laptop.
  - Fast: small model, encodes 1000 short records in seconds.
  - 384 dimensions: compact, plenty for short structured book records.
  - Good general-purpose semantic quality (trained on 1B+ sentence pairs).
  - English dataset -> an English model is the right fit.
  Rejected: OpenAI/Gemini embedding APIs (cost + network per call for no
  real quality gain on this small, simple corpus).

Vector store choice:
  FAISS (IndexFlatIP with normalized vectors = cosine similarity).
  - Exact search, trivial for 1000 vectors, no external service.
  - Chroma was considered; FAISS is lighter for a single-file local index.

Outputs:
  data/faiss.index   — the FAISS index
  data/store.pkl      — parallel list of chunk dicts (id, text, metadata)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data"
CHUNKS = DATA / "chunks.json"
INDEX_FILE = DATA / "faiss.index"
STORE_FILE = DATA / "store.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode and L2-normalize so inner product == cosine similarity."""
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(vecs)
    return vecs.astype("float32")


def build_index():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    print(f"Loading model '{MODEL_NAME}'...")
    model = load_model()

    print(f"Embedding {len(texts)} chunks...")
    vecs = embed_texts(model, texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(vecs)
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(INDEX_FILE))
    with open(STORE_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved -> {INDEX_FILE.name}, {STORE_FILE.name}")


if __name__ == "__main__":
    build_index()
