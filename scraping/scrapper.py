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

import requests                         # HTTP client for web requests
from bs4 import BeautifulSoup           # HTML parsing library
from urllib.parse import urljoin, urlparse  # URL normalization and domain checks
import json                             # JSON serialization
import time                             # Delays between requests
import random                           # Randomized polite delays
import logging                          # Logging of errors and debug information
import sys                              # System exit handling
from tqdm import tqdm                   # Progress bar for scraping status
import os                               # File system operations

from companies_to_scrape import companies_to_scrape  # Input: company -> start URL


# ======================
# Configuration
# ======================

# HTTP headers used to mimic a real browser and reduce basic bot blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# List of ESG-related keywords used for text-based relevance filtering
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
    "scope 3",
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

# Keywords used specifically for URL-based filtering
# These keywords help reduce the crawl space early in the process
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

EXCLUDE_URLS = [
    "facebook", "twitter", "linkedin", "instagram", "youtube", "login", "register"]
# URLs containing these terms are excluded to avoid irrelevant or external pages
KEYWORDS = [k.lower() for k in KEYWORDS]
KEYWORDS_URL = [k.lower() for k in KEYWORDS_URL]
EXCLUDE_URLS = [u.lower() for u in EXCLUDE_URLS]

# Polite crawling parameters
REQUEST_TIMEOUT = 12    # Maximum time waiting for a server response
MIN_DELAY = 0.4         # Minimum delay between requests
MAX_DELAY = 1.0         # Maximum delay between requests
MAX_RETRIES = 2         # Number of retry attempts per request

# ======================
# Utility helper functions
# ======================

def same_domain(base_url, new_url):
    """
    Checks whether two URLs belong to the same domain.
    This prevents the crawler from leaving the target website.
    """
    try:
        return urlparse(base_url).netloc == urlparse(new_url).netloc
    except Exception:
        return False

def is_relevant_url(url):
    """
    Determines whether a URL is potentially relevant based on ESG-related keywords.
    URL-based filtering helps reduce the number of pages to crawl.
    """
    u = url.lower()
     # Exclude known irrelevant or external platforms
    if any(x in u for x in EXCLUDE_URLS):
        return False 
    
    # Check for the presence of ESG-related keywords in the URL
    for k in KEYWORDS_URL:
        if k in u or k.replace('-', '_') in u:
            return True 
            
    return False 

def contains_keyword_text(text):
    """
    Checks whether a text contains any ESG-related keyword.
    This function is used for content-based relevance filtering.
    """
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in KEYWORDS)

def safe_get(session, url):
    """
    Performs an HTTP GET request with retry logic.
    This helps handle transient network errors gracefully.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            return r
        except requests.RequestException as e:
            logging.debug(f"Request error {e} for url {url} (attempt {attempt})")
            time.sleep(0.5 + attempt * 0.5)
    return None

def extract_page(soup, url, company):
    """
    Extracts structured information from a parsed HTML page.
    """

    # 1. Extract all hyperlinks before cleaning the DOM
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a.get("href"))
        links.append(full)

    # 2. Remove non-informative or navigational elements
    for noise in soup(["script", "style", "nav", "footer", "header", "aside"]):
        noise.decompose()

    # 3. Extract metadata
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    subtitles = [
        h.get_text(strip=True)
        for h in soup.find_all(["h1","h2","h3","h4"])
        if h.get_text(strip=True)
    ]

    # 4. Extract main textual content
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

# ======================
# Main scraping function
# ======================

def scrape_company(company, start_url, max_depth=1, check_text_for_keywords=True, out_dir="output"):
    """
    Scrapes ESG-related pages for a single company using BFS traversal.
    """
        
    session = requests.Session()

    visited = set()      # Keeps track of already visited URLs
    results = []         # Stores extracted ESG-relevant pages

    # BFS queue of (url, depth)
    to_visit = [(start_url, 0)]

    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Progress bar indicating number of relevant pages collected
    pbar = tqdm(total=0, desc=f"{company}", unit="page", leave=False)

    while to_visit:
        url, depth = to_visit.pop(0)

        # Skip already visited URLs
        if url in visited:
            continue

        # Stop traversal when maximum depth is reached
        if depth > max_depth:
            continue

        visited.add(url)

        # Apply a polite randomized delay
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        #Fetch the page safely
        r = safe_get(session, url)
        if r is None:
            logging.debug(f"Failed to fetch {url}")
            continue

        #Only process HTML pages
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

        # Determine wether the page shoulld be kept
        keep = False

        # Criterion 1: URL-based relevance
        if is_relevant_url(url):
            keep = True
       # Criterion 2: content-based relevance
        if not keep and check_text_for_keywords and contains_keyword_text(page_struct.get("text", "") + " " + page_struct.get("title", "")):
            keep = True

        # Store relevant pages
        if keep:
            results.append(page_struct)
            pbar.total += 1
            pbar.refresh()

        # Discover new URLs to visit
        for link in page_struct["links"]:
            if not same_domain(start_url, link):
                continue
            # normalize fragment
            link = link.split('#')[0]
            if link in visited:
                continue
            # Add link if either url looks relevant or depth < max_depth
            # We use url-relevance to reduce queue size
            if is_relevant_url(link) and depth < max_depth:
                to_visit.append((link, depth + 1))

    pbar.close()

    # Save extracted data as a JSON file
    filename = os.path.join(out_dir, f"{company.replace(' ', '_')}_rse.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Scraping finished for {company} — {len(results)} pages saved to {filename}")
    return filename

# ======================
# Script entry point
# ======================

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