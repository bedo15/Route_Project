import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data"
INDEX_FILE = DATA / "faiss.index"
STORE_FILE = DATA / "store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(STORE_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        return vec.astype("float32")

    def search(self, query: str, k: int = 5, metadata_filter: dict | None = None):
        """
        Return top-k chunks as a list of dicts:
          {rank, score, id, text, metadata}

        metadata_filter (optional, used in Level 2): a dict of constraints,
        e.g. {"category": "Travel", "max_price": 30, "min_rating": 4}.

        Price/rating/stock are applied as HARD filters. Category is applied
        as a SOFT boost: matching books get a small score bump so they rank
        higher, but non-matching books are still eligible (the dataset's 50
        fine-grained genres make exact category matching too brittle).
        We over-fetch so we still return k results after filtering.
        """
        fetch_k = k if not metadata_filter else max(k * 10, 50)
        qvec = self.embed_query(query)
        scores, ids = self.index.search(qvec, fetch_k)

        wanted_cat = (metadata_filter or {}).get("category")
        CATEGORY_BOOST = 0.05

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if metadata_filter and not self._passes(chunk["metadata"], metadata_filter):
                continue
            final_score = float(score)
            if wanted_cat and self._category_soft_match(
                chunk["metadata"].get("category", ""), wanted_cat
            ):
                final_score += CATEGORY_BOOST
            results.append(
                {
                    "score": final_score,
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:k]
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        return results

    @staticmethod
    def _passes(meta: dict, f: dict) -> bool:
        
        if "max_price" in f and f["max_price"] is not None:
            if meta.get("price", 1e9) > f["max_price"]:
                return False
        if "min_price" in f and f["min_price"] is not None:
            if meta.get("price", 0) < f["min_price"]:
                return False
        if "min_rating" in f and f["min_rating"] is not None:
            if meta.get("rating", 0) < f["min_rating"]:
                return False
        if "in_stock" in f and f["in_stock"] is not None:
            if meta.get("in_stock") != f["in_stock"]:
                return False
        
        return True

    @staticmethod
    def _category_soft_match(meta_cat: str, wanted_cat: str) -> bool:
        """Loose category match: substring either direction, case-insensitive."""
        a, b = meta_cat.lower(), wanted_cat.lower()
        return a in b or b in a


def _demo():
    r = DenseRetriever()
    for hit in r.search("cheap travel book about mountains", k=3):
        print(f"[{hit['rank']}] score={hit['score']:.3f}  {hit['metadata']['title']}")


if __name__ == "__main__":
    _demo()
