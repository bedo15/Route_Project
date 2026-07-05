import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"
def _get_client():
    """Fresh client per call — survives Streamlit script re-runs."""
    if not API_KEY:
        return None
    return genai.Client(api_key=API_KEY)


QUERY_CLASSES = ["factual", "recommendation", "comparison", "out_of_scope"]


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_query": {"type": "string"},
        "query_class": {"type": "string", "enum": QUERY_CLASSES},
        "filters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "min_rating": {"type": "integer"},
                "in_stock": {"type": "boolean"},
            },
        },
    },
    "required": ["rewritten_query", "query_class", "filters"],
}

SYSTEM_INSTRUCTION = (
    "You process user queries for a book-catalogue search engine. "
    "For each query you return a JSON object with three things:\n"
    "1. rewritten_query: a clean, explicit English version of the query, "
    "good for semantic + keyword search. Expand vague wording; keep the "
    "user's intent.\n"
    "2. query_class: one of 'factual' (a specific fact lookup, e.g. the "
    "price or rating of a named book), 'recommendation' (asking for "
    "suggestions), 'comparison' (comparing two or more books/options), or "
    "'out_of_scope' (not about books in the catalogue at all).\n"
    "3. filters: structured constraints IMPLIED by the query. Include only "
    "the fields the user actually implies. Fields: category (a book genre "
    "like Travel, Mystery, History), min_price, max_price (numbers, in "
    "pounds), min_rating (1-5 integer), in_stock (boolean). If the user says "
    "'cheap' you may set a sensible max_price (books here range roughly "
    "£10-£60, so 'cheap' means about max_price 25, not lower); if they name "
    "a genre set category; if they say 'highly rated' set min_rating to 4. "
    "If no filter is implied, return an empty filters object.\n"
    "Return ONLY the JSON object."
)


def _fallback(query: str) -> dict:
    """If the API isn't available or fails, degrade gracefully to a no-op
    (use the raw query, treat as recommendation, no filters)."""
    return {
        "rewritten_query": query,
        "query_class": "recommendation",
        "filters": {},
    }


def process_query(query: str) -> dict:
    """Return {rewritten_query, query_class, filters} for a user query."""
    client = _get_client()
    if client is None:
        return _fallback(query)

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"User query: {query}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        data = json.loads(resp.text)
        data.setdefault("rewritten_query", query)
        data.setdefault("query_class", "recommendation")
        data.setdefault("filters", {})
        data["filters"] = {
            k: v for k, v in (data.get("filters") or {}).items() if v not in (None, "")
        }
        return data
    except Exception as e:
        print(f"[query.py] falling back (reason: {e})")
        return _fallback(query)


def _demo():
    examples = [
        "cheap travel book about mountains",
        "what's the price of Sapiens",
        "compare the two most expensive mystery books",
        "how do I bake sourdough bread",
        "highly rated history books under £40",
    ]
    for q in examples:
        out = process_query(q)
        print(f"\nQ: {q}")
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()
