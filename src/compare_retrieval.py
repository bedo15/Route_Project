from retrieve import DenseRetriever
from retrieve_hybrid import HybridRetriever

QUERIES = [
    "history books",
    "cheap travel book about mountains",
    "vampire romance",
    "books about running and marathons",
]


def _show(title, hits):
    print(f"  {title}")
    if not hits:
        print("    (no results)")
    for h in hits:
        m = h["metadata"]
        print(f"    [{h['rank']}] {m['title'][:55]:55s} "
              f"({m['category']}, £{m['price']}, {m['rating']}*)")


def main():
    dense = DenseRetriever()
    hybrid = HybridRetriever()

    for q in QUERIES:
        print("=" * 70)
        print(f"QUERY: {q}\n")
        _show("DENSE-ONLY:", dense.search(q, k=3))
        print()
        _show("HYBRID (dense+BM25, RRF):", hybrid.search(q, k=3))
        print()


if __name__ == "__main__":
    main()
