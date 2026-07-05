from retrieve import DenseRetriever
from generate import generate_answer
from query import process_query

_retriever = None
_hybrid_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever()
    return _retriever


def get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from retrieve_hybrid import HybridRetriever
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def answer_question(query: str, k: int = 5, use_query_intelligence: bool = True,
                    use_hybrid: bool = True) -> dict:
    retriever = get_hybrid_retriever() if use_hybrid else get_retriever()

    if use_query_intelligence:
        qi = process_query(query)
        search_query = qi["rewritten_query"]
        filters = qi["filters"] or None
        query_class = qi["query_class"]
    else:
        search_query, filters, query_class = query, None, None

    # Out-of-scope queries: don't bother retrieving/answering from the catalogue
    if query_class == "out_of_scope":
        return {
            "query": query,
            "rewritten_query": search_query,
            "query_class": query_class,
            "filters": filters or {},
            "answer": "That question isn't about the book catalogue, so I can't "
                      "answer it from the available data.",
            "sources": [],
            "chunks": [],
        }

    chunks = retriever.search(search_query, k=k, metadata_filter=filters)

    if not chunks and filters:
        chunks = retriever.search(search_query, k=k, metadata_filter=None)

    answer = generate_answer(query, chunks)

    return {
        "query": query,
        "rewritten_query": search_query,
        "query_class": query_class,
        "filters": filters or {},
        "retrieval_mode": "hybrid" if use_hybrid else "dense",
        "answer": answer,
        "sources": [
            {
                "rank": c["rank"],
                "title": c["metadata"]["title"],
                "score": round(c["score"], 3),
                "category": c["metadata"]["category"],
                "price": c["metadata"]["price"],
                "rating": c["metadata"]["rating"],
                "url": c["metadata"].get("source_url", ""),
            }
            for c in chunks
        ],
        "chunks": chunks,
    }


def _demo():
    for q in ["a cheap travel book about mountains",
              "highly rated history books under £40"]:
        r = answer_question(q, k=3)
        print("=" * 60)
        print("Original: ", r["query"])
        print("Rewritten:", r["rewritten_query"])
        print("Class:    ", r["query_class"])
        print("Filters:  ", r["filters"])
        print("\nAnswer:\n", r["answer"])
        print("\nSources:")
        for s in r["sources"]:
            print(f"  [{s['rank']}] {s['title']} (score={s['score']}, £{s['price']}, {s['rating']}*)")


if __name__ == "__main__":
    _demo()
