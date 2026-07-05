import pickle
import re
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data"
INDEX_FILE = DATA / "faiss.index"
STORE_FILE = DATA / "store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

K_RRF = 60


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(STORE_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(MODEL_NAME)
        self.tokenized_corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def _dense_ranking(self, query: str, top_n: int) -> list[int]:
        vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        vec = vec.astype("float32")
        _, ids = self.index.search(vec, top_n)
        return [int(i) for i in ids[0] if i != -1]

    def _bm25_ranking(self, query: str, top_n: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(scores)[::-1][:top_n])


    @staticmethod
    def _rrf(rankings: list[list[int]], k_rrf: int = K_RRF) -> dict[int, float]:
        """Reciprocal Rank Fusion over several ranked id-lists."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        return fused


    def search(self, query: str, k: int = 5, metadata_filter: dict | None = None,
               pool: int = 50):
        """Hybrid search. Returns the same result shape as DenseRetriever."""
        dense_ids = self._dense_ranking(query, pool)
        bm25_ids = self._bm25_ranking(query, pool)
        fused = self._rrf([dense_ids, bm25_ids])

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        wanted_cat = (metadata_filter or {}).get("category")
        CATEGORY_BOOST = 0.01  
        results = []
        for doc_id, rrf_score in ordered:
            chunk = self.chunks[doc_id]
            if metadata_filter and not self._passes(chunk["metadata"], metadata_filter):
                continue
            score = rrf_score
            if wanted_cat and self._category_soft_match(
                chunk["metadata"].get("category", ""), wanted_cat
            ):
                score += CATEGORY_BOOST
            results.append(
                {
                    "score": float(score),
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
        a, b = meta_cat.lower(), wanted_cat.lower()
        return a in b or b in a


def _demo():
    r = HybridRetriever()
    for hit in r.search("history books", k=3):
        print(f"[{hit['rank']}] {hit['score']:.4f}  {hit['metadata']['title']} "
              f"({hit['metadata']['category']})")


if __name__ == "__main__":
    _demo()
