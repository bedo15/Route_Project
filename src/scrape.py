import json
import time
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/"
CATALOGUE = urljoin(BASE, "catalogue/")
LISTING_URL = urljoin(CATALOGUE, "page-{}.html")

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = OUT_DIR / "books_raw.json"

HEADERS = {"User-Agent": "Route-RAG-Project/1.0 (educational graduation project)"}


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return BeautifulSoup(resp.text, "html.parser")


def parse_price(text: str) -> float:
    """'£51.77' -> 51.77 . Strip any stray non-ASCII currency bytes."""
    cleaned = re.sub(r"[^0-9.]", "", text)
    return float(cleaned) if cleaned else 0.0


def get_book_links(page_soup: BeautifulSoup) -> list[str]:
    links = []
    for h3 in page_soup.select("article.product_pod h3 a"):
        href = h3.get("href")
        links.append(urljoin(CATALOGUE, href))
    return links


def parse_book_page(url: str) -> dict:
    soup = get_soup(url)

    title = soup.select_one("div.product_main h1").get_text(strip=True)

    price_incl = soup.select_one("p.price_color").get_text(strip=True)
    price = parse_price(price_incl)

    rating_word = "Zero"
    star_p = soup.select_one("p.star-rating")
    if star_p:
        classes = star_p.get("class", [])
        for c in classes:
            if c in RATING_WORDS:
                rating_word = c
    rating = RATING_WORDS.get(rating_word, 0)

    avail_text = soup.select_one("p.availability").get_text(strip=True)
    stock_match = re.search(r"\((\d+) available\)", avail_text)
    stock_count = int(stock_match.group(1)) if stock_match else 0
    in_stock = "In stock" in avail_text

    crumbs = soup.select("ul.breadcrumb li a")
    category = crumbs[-1].get_text(strip=True) if len(crumbs) >= 3 else "Unknown"

    desc_el = soup.select_one("#product_description ~ p")
    description = desc_el.get_text(strip=True) if desc_el else ""

    info = {}
    for row in soup.select("table.table.table-striped tr"):
        key = row.select_one("th").get_text(strip=True)
        val = row.select_one("td").get_text(strip=True)
        info[key] = val

    return {
        "title": title,
        "category": category,
        "price": price,
        "rating": rating,
        "in_stock": in_stock,
        "stock_count": stock_count,
        "description": description,
        "upc": info.get("UPC", ""),
        "price_excl_tax": parse_price(info.get("Price (excl. tax)", "0")),
        "price_incl_tax": parse_price(info.get("Price (incl. tax)", "0")),
        "tax": parse_price(info.get("Tax", "0")),
        "num_reviews": int(info.get("Number of reviews", "0") or 0),
        "source_url": url,
    }


def scrape_all(max_pages: int = 50, delay: float = 0.3) -> list[dict]:
    all_books = []
    for page in range(1, max_pages + 1):
        listing_url = LISTING_URL.format(page)
        try:
            page_soup = get_soup(listing_url)
        except requests.HTTPError:
            print(f"Stopping — page {page} not found.")
            break

        links = get_book_links(page_soup)
        if not links:
            break

        print(f"Page {page:2d}: {len(links)} books")
        for link in links:
            try:
                all_books.append(parse_book_page(link))
            except Exception as e:  
                print(f"  ! failed {link}: {e}")
            time.sleep(delay)
        time.sleep(delay)

    return all_books


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    books = scrape_all()
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(books)} books -> {OUT_FILE}")


if __name__ == "__main__":
    main()
