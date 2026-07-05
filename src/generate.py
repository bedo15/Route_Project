import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"

def _get_client():
    
    if not API_KEY:
        return None
    return genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for a book catalogue. "
    "You are given a numbered list of context books retrieved for the user's "
    "question. Base your answer on these books: you may recommend, compare, "
    "rank, or summarize them, and reason about their category, price, and "
    "rating to address the question. "
    "Only discuss books that appear in the context; never invent titles, "
    "prices, or details that are not shown. "
    "If NONE of the context books are relevant to the question, say you don't "
    "have anything suitable in the catalogue. "
    "Refer to books by their title. Be concise and helpful."
)




def _format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a numbered context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        blocks.append(
            f"[{i}] {meta['title']} "
            f"(Category: {meta['category']}, Price: £{meta['price']:.2f}, "
            f"Rating: {meta['rating']}/5)\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    client = _get_client()
    if client is None:
        return ("ERROR: No GEMINI_API_KEY found. Copy .env.example to .env "
                "and paste your key.")
    if not chunks:
        return "I couldn't find anything relevant in the catalogue for that question."

    context = _format_context(chunks)
    prompt = (
        f"Context books:\n{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer using only the context above."
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return resp.text.strip()


def _demo():
    fake = [
        {"text": "Title: A Walk in the Woods\nCategory: Travel\nRating: 4 stars\n"
                 "Price: £24.30\nDescription: A hike along the Appalachian Trail.",
         "metadata": {"title": "A Walk in the Woods", "category": "Travel",
                      "price": 24.30, "rating": 4}},
    ]
    print(generate_answer("Recommend a cheap travel book about hiking", fake))


if __name__ == "__main__":
    _demo()
