"""
Robust RSE Web Scraper
----------------------

Usage:
  - Save this file as `robust_scraper.py`.
  - Create a virtual environment and install dependencies:
      pip install requests beautifulsoup4 html5lib tqdm
  - Run:
      python robust_scraper.py

What it does:
  - Uses a configurable dict of company -> start_url
  - Respects a max_depth parameter (BFS)
  - Filters candidate URLs by keywords (english only)
  - Verifies Content-Type before parsing (only text/html)
  - Uses html5lib parser fallback for difficult HTML
  - Optionally checks page text for keywords (not just URL)
  - Saves one JSON file per company with structured fields

Notes:
  - This is a research/educational scraper. Check each site's robots.txt and terms of use before large-scale scraping.
  - The script uses polite delays and a simple retry mechanism.

"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import random
import logging
import sys
from tqdm import tqdm
import os


print("--- Strating the scraper ---") 

# ----------------------
# Configuration
# ----------------------
HEADERS = {
    "User-Agent" :  "Your usuer agent string here"
}
#strict kyewords 
KEYWORDS = [#strict_keywords
    "esg",
    "csr",
    "esg report",
    "sustainability report",
    "impact report",
    "non-financial report",
    "integrated report",
    "corporate social responsibility",
    "net zero",
    "carbon footprint",
    "scope 1",
    "scope 2",
    "scope 3"
    # Environment
    "sustainability",
    "sustainable",
    "environmental",
    "climate",
    "emissions",
    "renewable",
    "biodiversity",
    "recycling",
    "waste management",
    "energy efficiency",
    "circular economy",
    "water management",
    "deforestation",

    # Social
    "human rights",
    "supply chain",
    "diversity",
    "inclusion",
    "health and safety",
    "decent work",

    # Governance
    "governance",
    "transparency",
    "anticorruption",
    "compliance",
    "stakeholder"]
#not used but kept for reference
REVISED_KEYWORDS = [
    # Mots-clés généraux / Rapports
    "esg", "csr", "sustainability", "sustainable", "responsibility", "corporate responsibility",
    "impact", "ethics", "ethical", "esg report", "sustainability report", "impact report",

    # Pilier Environnement (E)
    "environment", "environmental", "climate", "carbon", "carbon footprint", "net zero", 
    "emissions", "renewable", "biodiversity", "recycling", "waste management", "energy efficiency",
    "circular economy", "water management", "deforestation",

    # Pilier Social (S)
    "social", "human rights", "supply chain", "diversity", "inclusion", "health and safety", 
    "community", "decent work",

    # Pilier Gouvernance (G)
    "governance", "governance policy", "transparency", "anticorruption", "compliance", "stakeholder"
]
# improved english-only keywords (compact) used in the scraper
SOFT_KEYWORDS = [
    "esg", "csr", "sustainability", "sustainable", "responsibility", "corporate responsibility",
    "environment", "environmental", "social", "governance", "impact", "ethics", "ethical",
    "climate", "carbon", "carbon footprint", "net zero", "emissions",
    "renewable", "biodiversity", "recycling", "waste management", "energy efficiency",
    "governance policy", "transparency", "esg report", "sustainability report","impact report" 
    ]
# normalize keywords lowercased
KEYWORDS = [k.lower() for k in KEYWORDS]

# polite settings
REQUEST_TIMEOUT = 12
MIN_DELAY = 0.4
MAX_DELAY = 1.0
MAX_RETRIES = 2

# ----------------------
# Utility helpers
# ----------------------

def same_domain(base_url, new_url):
    try:
        return urlparse(base_url).netloc == urlparse(new_url).netloc
    except Exception:
        return False


def is_relevant_url(url):
    u = url.lower()
    return any(k in u for k in KEYWORDS)


def contains_keyword_text(text):
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in KEYWORDS)


# safe get with retries
def safe_get(session, url):
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            return r
        except requests.RequestException as e:
            logging.debug(f"Request error {e} for url {url} (attempt {attempt})")
            time.sleep(0.5 + attempt * 0.5)
    return None


# parse page to structured dict
def extract_page(soup, url, company):
    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
    subtitles = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    text = "\n".join(paragraphs)
    # get internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        full = urljoin(url, href)
        links.append(full)
    return {
        "company": company,
        "url": url,
        "title": title,
        "subtitles": subtitles,
        "text": text,
        "links": links
    }


# ----------------------
# Main scraping function
# ----------------------

def scrape_company(company, start_url, max_depth=1, check_text_for_keywords=True, out_dir="output"):
    session = requests.Session()

    visited = set()
    results = []

    # BFS queue of (url, depth)
    to_visit = [(start_url, 0)]

    # prepare output dir
    os.makedirs(out_dir, exist_ok=True)

    pbar = tqdm(total=0, desc=f"{company}", unit="page", leave=False)

    while to_visit:
        url, depth = to_visit.pop(0)

        if url in visited:
            continue
        if depth > max_depth:
            continue

        visited.add(url)

        # polite delay
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        r = safe_get(session, url)
        if r is None:
            logging.debug(f"Failed to fetch {url}")
            continue

        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            # skip non-HTML content
            logging.debug(f"Skipping non-HTML content: {url} ({content_type})")
            continue

        # try parse with default parser, fallback to html5lib on failure
        soup = None
        try:
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            try:
                soup = BeautifulSoup(r.text, "html5lib")
            except Exception as e:
                logging.debug(f"Parser failed for {url}: {e}")
                continue

        page_struct = extract_page(soup, url, company)

        # decide whether to keep this page
        keep = False
        # 1) url contains keywords
        if is_relevant_url(url):
            keep = True
        # 2) OR page text contains keywords (if enabled)
        if not keep and check_text_for_keywords and contains_keyword_text(page_struct.get("text", "") + " " + page_struct.get("title", "")):
            keep = True

        if keep:
            results.append(page_struct)
            pbar.total += 1
            pbar.refresh()

        # find candidate links to continue crawling
        for link in page_struct["links"]:
            if not same_domain(start_url, link):
                continue
            # normalize fragment
            link = link.split('#')[0]
            if link in visited:
                continue
            # add link if either url looks relevant or depth < max_depth
            # we use url-relevance to reduce queue size
            if is_relevant_url(link) or depth < max_depth:
                to_visit.append((link, depth + 1))

    pbar.close()

    # save results
    filename = os.path.join(out_dir, f"{company.replace(' ', '_')}_rse.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Scraping finished for {company} — {len(results)} pages saved to {filename}")
    return filename


# ----------------------
# Example companies (replace / extend as needed)
# ----------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

#name the companies to scrape here "company_name": "start_url",
    companies = {}

    # change max_depth here if needed (recommended 1 or 2)
    MAX_DEPTH = 1

    for company, url in companies.items():
        try:
            scrape_company(company, url, max_depth=MAX_DEPTH, check_text_for_keywords=True, out_dir="scraped_output")
        except KeyboardInterrupt:
            print("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logging.exception(f"Error scraping {company}: {e}")
            continue