"""
Web Scraper Template – Zurich Apartment Listings
=================================================
Provides a structural template for scraping apartment listings from Swiss
real-estate portals such as Homegate or ImmoScout24.

IMPORTANT NOTES
---------------
* Check the target site's robots.txt and Terms of Service before crawling.
* Add polite request delays (REQUEST_DELAY) to avoid overloading servers.
* The CSS selectors below are ILLUSTRATIVE; update them after inspecting the
  actual HTML of the portal you intend to scrape.
* If scraping is not immediately possible, run
      python generate_sample_data.py
  to create a realistic synthetic dataset at data/apartments_zurich_raw.csv
  that the rest of the pipeline can use without modification.

Usage
-----
    pip install requests beautifulsoup4 lxml
    python scraper.py
"""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Generator

# Optional imports – only needed for actual scraping
try:
    import requests
    from bs4 import BeautifulSoup
    _SCRAPING_LIBS_AVAILABLE = True
except ImportError:
    _SCRAPING_LIBS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path("data/apartments_zurich_raw.csv")
BASE_URL      = "https://www.homegate.ch/rent/real-estate/city-zurich/matching-list"
REQUEST_DELAY = 2.5   # seconds between requests – be a polite crawler
MAX_PAGES     = 50    # maximum number of search-result pages to scrape


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ApartmentListing:
    """Represents a single apartment listing scraped from a portal."""
    title:               str
    description:         str
    price:               float | None
    number_of_rooms:     float | None
    apartment_size_sqm:  float | None
    address:             str
    zip_code:            str
    city:                str
    latitude:            float | None
    longitude:           float | None
    publisher:           str


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _get_headers() -> dict[str, str]:
    """Returns browser-like HTTP request headers."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _parse_price(text: str) -> float | None:
    """Extracts a numeric price from strings like "CHF 2'400.—"."""
    cleaned = re.sub(r"[^\d.]", "", text.replace("'", "").replace(",", "."))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_rooms(text: str) -> float | None:
    """Extracts room count from strings like "3.5 Zimmer" or "3.5-room"."""
    match = re.search(r"(\d+(?:[.,]\d)?)", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def _parse_size(text: str) -> float | None:
    """Extracts floor area from strings like "78 m²"."""
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


# ── HTTP layer ────────────────────────────────────────────────────────────────

def fetch_page(session: "requests.Session", url: str) -> "BeautifulSoup | None":
    """
    Fetches a single URL and returns a parsed BeautifulSoup document.
    Returns None if the request fails.
    """
    if not _SCRAPING_LIBS_AVAILABLE:
        logger.error(
            "requests and beautifulsoup4 are required for scraping. "
            "Install with: pip install requests beautifulsoup4 lxml"
        )
        return None
    try:
        response = session.get(url, headers=_get_headers(), timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ── HTML parsing ──────────────────────────────────────────────────────────────

def parse_listings(soup: "BeautifulSoup") -> list[ApartmentListing]:
    """
    Parses listing cards from a search-results page.

    NOTE: The CSS selectors below are illustrative placeholders.
    Inspect the portal's HTML with your browser DevTools and update the
    selectors accordingly before running a real scrape.

    Typical structure on Homegate (subject to change):
        <div class="ResultList_listItem__...">
            <span class="ListItem_title__...">…</span>
            <span class="ListItem_price__...">CHF 2'400.—</span>
            <span class="ListItem_rooms__...">3.5 Zi.</span>
            <span class="ListItem_surface__...">78 m²</span>
            <address class="ListItem_address__...">Seestrasse 12, 8002 Zürich</address>
        </div>
    """
    listings: list[ApartmentListing] = []

    for card in soup.select("div[class*='ResultList_listItem']"):
        try:
            title_el   = card.select_one("[class*='ListItem_title']")
            price_el   = card.select_one("[class*='ListItem_price']")
            rooms_el   = card.select_one("[class*='ListItem_rooms']")
            size_el    = card.select_one("[class*='ListItem_surface']")
            address_el = card.select_one("address")
            desc_el    = card.select_one("[class*='ListItem_description']")

            raw_address = address_el.get_text(strip=True) if address_el else ""
            # Try to extract zip code and city from "Musterstrasse 1, 8001 Zürich"
            addr_match = re.search(r"(\d{4})\s+(.+)$", raw_address)
            zip_code = addr_match.group(1) if addr_match else ""
            city     = addr_match.group(2).strip() if addr_match else ""

            listings.append(ApartmentListing(
                title=              title_el.get_text(strip=True)   if title_el   else "",
                description=        desc_el.get_text(strip=True)    if desc_el    else "",
                price=              _parse_price(price_el.get_text(strip=True)) if price_el else None,
                number_of_rooms=    _parse_rooms(rooms_el.get_text(strip=True)) if rooms_el else None,
                apartment_size_sqm= _parse_size(size_el.get_text(strip=True))  if size_el  else None,
                address=            raw_address,
                zip_code=           zip_code,
                city=               city,
                latitude=           None,   # enrich via geocoding if needed
                longitude=          None,
                publisher=          "Homegate",
            ))
        except Exception as exc:
            logger.debug("Skipping malformed listing card: %s", exc)

    return listings


# ── Pagination ────────────────────────────────────────────────────────────────

def scrape_all_pages(
    max_pages: int = MAX_PAGES,
) -> Generator[ApartmentListing, None, None]:
    """
    Generator that iterates over search-result pages and yields ApartmentListing
    objects.  Stops early if a page returns no listings.
    """
    if not _SCRAPING_LIBS_AVAILABLE:
        logger.error(
            "Scraping libraries not installed. "
            "Run: pip install requests beautifulsoup4 lxml"
        )
        return

    import requests as req

    session = req.Session()
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}?ep={page}"
        logger.info("Scraping page %d/%d: %s", page, max_pages, url)

        soup = fetch_page(session, url)
        if soup is None:
            break

        listings = parse_listings(soup)
        if not listings:
            logger.info("No listings on page %d – stopping.", page)
            break

        yield from listings
        time.sleep(REQUEST_DELAY)


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_listings_to_csv(
    listings: list[ApartmentListing],
    output_path: Path = RAW_DATA_PATH,
) -> None:
    """Writes a list of ApartmentListing objects to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields(ApartmentListing)]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for listing in listings:
            writer.writerow(
                {f.name: getattr(listing, f.name) for f in fields(ApartmentListing)}
            )
    logger.info("Saved %d listings to '%s'", len(listings), output_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def run_scraper(max_pages: int = MAX_PAGES) -> None:
    """Scrapes all available pages and saves results to the raw data CSV."""
    logger.info("Starting Zurich apartment scraper…")
    all_listings: list[ApartmentListing] = []

    for listing in scrape_all_pages(max_pages=max_pages):
        all_listings.append(listing)
        if len(all_listings) % 50 == 0:
            logger.info("Collected %d listings so far.", len(all_listings))

    if all_listings:
        save_listings_to_csv(all_listings)
        logger.info("Scraping complete. Total: %d listings.", len(all_listings))
    else:
        logger.warning(
            "No listings were scraped.  "
            "Check CSS selectors or generate synthetic data with: "
            "python generate_sample_data.py"
        )


if __name__ == "__main__":
    run_scraper()
