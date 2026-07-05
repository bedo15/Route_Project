import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "books_raw.json"
OUT = DATA / "chunks.json"

MAX_TEXT_CHARS = 2000

def make_chunk_text(book: dict) -> str:
    """Compose the searchable text blob for one book."""
    stars = "star" if book["rating"] == 1 else "stars"
    parts = [
        f"Title: {book['title']}",
        f"Category: {book['category']}",
        f"Rating: {book['rating']} {stars} out of 5",
        f"Price: £{book['price']:.2f}",
        f"Availability: {'in stock' if book['in_stock'] else 'out of stock'}"
        f" ({book['stock_count']} available)",
    ]
    if book.get("description"):
        parts.append(f"Description: {book['description']}")
    text = "\n".join(parts)
    return text[:MAX_TEXT_CHARS]


def build_chunks() -> list[dict]:
    books = json.loads(RAW.read_text(encoding="utf-8"))
    chunks = []
    for i, book in enumerate(books):
        chunks.append(
            {
                "id": book.get("upc") or f"book-{i}",
                "text": make_chunk_text(book),
                "metadata": {
                    "title": book["title"],
                    "category": book["category"],
                    "price": book["price"],
                    "rating": book["rating"],
                    "in_stock": book["in_stock"],
                    "stock_count": book["stock_count"],
                    "num_reviews": book["num_reviews"],
                    "upc": book["upc"],
                    "source_url": book["source_url"],
                },
            }
        )
    return chunks


def main():
    chunks = build_chunks()
    OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(chunks)} chunks -> {OUT}")
    if chunks:
        print("\nExample chunk:\n")
        print(chunks[0]["text"])


if __name__ == "__main__":
    main()
