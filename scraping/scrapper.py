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
from companies_to_scrape import companies_to_scrape


print("--- Strating the scraper ---") 

# ----------------------
# Configuration
# ----------------------

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

KEYWORDS = [ 
    #strict_keywords
    "esg", "csr","esg report",
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
    "stakeholder"
]

KEYWORDS_URL = [
    # --- Piliers Généraux ---
    "sustainability", "sustainable", "csr", "esg", "responsibility", 
    "impact", "corporate-responsibility", "social-responsibility",
    
    # --- Rapports et Données (Crucial pour ton analyse) ---
    "report", "disclosure", "data", "metrics", "index", "performance",
    "gri", "sasb", "tcfd", "non-financial", "integrated-report",
    
    # --- Environnement & Climat ---
    "climate", "carbon", "emissions", "net-zero", "environmental", 
    "planet", "energy", "nature", "biodiversity", "water", "waste", 
    "circular-economy", "green",
    
    # --- Social & Humain ---
    "social", "human-rights", "diversity", "inclusion", "equity", 
    "employees", "people", "community", "labor", "supply-chain",
    
    # --- Gouvernance & Éthique ---
    "governance", "ethics", "compliance", "policy", "integrity", 
    "transparency", "stakeholder"
]
EXCLUDE_URLS = ["facebook", "twitter", "linkedin", "instagram", "youtube", "login", "register"]

KEYWORDS = [k.lower() for k in KEYWORDS]
KEYWORDS_URL = [k.lower() for k in KEYWORDS_URL]
EXCLUDE_URLS = [u.lower() for u in EXCLUDE_URLS]

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

    if any(x in u for x in EXCLUDE_URLS):
        return False 
    
    for k in KEYWORDS_URL:
        if k in u or k.replace('-', '_') in u:
            return True 
            
    return False 


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



def extract_page(soup, url, company):

    # 1. Links (avant nettoyage)
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a.get("href"))
        links.append(full)

    # 2. Remove noise
    for noise in soup(["script", "style", "nav", "footer", "header", "aside"]):
        noise.decompose()

    # 3. Metadata
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    subtitles = [
        h.get_text(strip=True)
        for h in soup.find_all(["h1","h2","h3","h4"])
        if h.get_text(strip=True)
    ]

    # 4. Main content
    main = soup.find("main") or soup.find("article")
    if main:
        text = main.get_text(separator=" ", strip=True)
    else:
        text = " ".join(
            p.get_text(strip=True)
            for p in soup.find_all("p")
        )

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
            if is_relevant_url(link) and depth < max_depth:
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


    
    # change max_depth here if needed (recommended 1 or 2)
    MAX_DEPTH = 1

    for company, url in companies_to_scrape.items():
        try:
            scrape_company(company, url, max_depth=MAX_DEPTH, check_text_for_keywords=True, out_dir="Scraped_output")
        except KeyboardInterrupt:
            print("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logging.exception(f"Error scraping {company}: {e}")
            continue