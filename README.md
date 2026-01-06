# CSR Benchmark Analysis

## Introduction

The objective of this project is to analyze and compare the CSR (Corporate Social Responsibility) discourses of major corporations, utilizing their institutional web pages.

Acting as consultants, our mission is to collect, analyze, and compare the CSR commitments published on the websites of large companies from various sectors. This strategic benchmark aims to identify **dominant trends and themes**, extract key **insights**, and analyze the relationships between the data in order to provide informed strategic recommendations.

## Table of Contents

- [**Project Plan**](#project-plan)
- [**Getting started**](#getting-started)
- [**Scraping**](#scraping)
- [**Text Mining**](#text-mining)
- [**Link Analysis**](#link-analysis)

## Project Plan

This project is divided into several technical and analytical objectives:

### 1. Data Collection (Scraping)

* Implement **web scraping** techniques to extract the textual content from the pages dedicated to Sustainable Development or Social Responsibility across multiple corporate websites.

### 2. Textual Analysis (Text Mining)

* Utilize **Text Mining (NLP)** methods to analyze the collected data, including:
    * Identification of **key themes and promises**.
    * Extraction of **dominant keywords and concepts**.
    * Comparison of the **lexicons** used by the companies.

### 3. Network Analysis (Web Mining)

* Create complex visualizations to represent the relationships between the concepts:
    * Construction of a **concept graph** shared among the companies.
    * Visualization of a **semantic network** of sub-themes addressed in the CSR discourses.

## Getting Started

To begin working on the project, please follow the steps below to set up the development environment.

### Prerequisites

Ensure that the following tools are installed on your system:

* **Python 3.9 or higher** ([download](https://www.python.org/downloads/))
* **pip** (Python Package Manager) ([installation guide](https://pip.pypa.io/en/stable/installation/))
* **git** (To clone the repository) ([download](https://git-scm.com/install/))

### Step 1: Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/Yusuf-Bulut-1907/CSR-Benchmark.git
cd CSR-Benchmark
```

### Step 2: Install Python Libraries

Install all necessary project libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
After installing the libraries, download the SpaCy English model required for NLP tasks:

```bash
pip python -m spacy download en_core_web_md
```
## Scraping

The objective of this step is to automatically collect the CSR-related pages from the websites of selected companies. This allows us to build a structured corpus of textual data for further analysis.

The list of companies and their corresponding URLs is stored in the Python dictionary [`companies_to_scrape.py`](./scraping/companies_to_scrape.py). Companies were chosen to represent a variety of sectors and to ensure that relevant CSR content is available on their official websites.

Before running the full scraper, a [`smoke test`](./scraping/scrap_smoke_testing.py) can be performed to verify that each company's URL is accessible and likely to return meaningful content.
Run:

```bash
python scraping/scrap_smoke_testing.py
```
This script will generate a `smoke_test_results.txt` file in the root of the repository containing three sections:<br>
`SUCCESS`: URLs that responded correctly and can be scraped<br>
`BLOCKED`: URLs that returned access errors (e.g., 401 or 403)<br>
`ERRORS`: URLs that could not be reached for other reasons

Performing this test helps identify sites that might need manual adjustment or that cannot be scraped.

To reproduce the data collection:

1. Ensure that all dependencies from `requirements.txt` are installed.
2. Run the scraper script:

```bash
python scraping/scrapper.py
```
The scraper will navigate each company's website listed in companies_to_scrape.py, following links within the same domain that match relevant keywords. It respects a polite delay between requests and retries failed connections.

For each company, the scraper produces a JSON file in the `Scraped_output/` folder. Each file contains:<br>
`company`: company name<br>
`url`: URL of the page<br>
`title`: page title<br>
`subtitles`: all headings (h1–h4)<br>
`text`: main text content<br>
`links`: list of extracted links

## Text Mining

Now that we have obtained the raw text data from the company websites through the scraping process, the goal of this step is to transform it into a clean and structured corpus suitable for textual analysis. This stage also generates descriptive statistics to help understand the quality, distribution, and coverage of the collected data.

The script [`load_corpus.py`](./text_mining/load_corpus.py) is responsible for loading the scraped JSON files from the `Scraped_output/` folder. It performs the following tasks:

1. Reads all JSON files produced by the scraper.  
2. Extracts the main textual content from each page while removing boilerplate elements (navigation menus, ads, etc.) using the `justext` library.  
3. Collects metadata for each document, including:
   - `company`: the company name  
   - `url`: the original page URL  
   - `title`: the page title 

```bash
python text_mining/load_corpus.py
```
By default, you need to specify the path to the JSON folder in the `JSON_FOLDER_PATH` variable.

Once the corpus is loaded, `stats_and_cleaning.py` processes the documents to create a clean and structured dataset suitable for analysis. The script performs several key steps, including removing very short or irrelevant documents, eliminating duplicates, and filtering out non-English content. After cleaning, it generates descriptive statistics about the corpus, such as the total number of documents, vocabulary size, and distribution of documents across companies.

Run the following command:
```bash
python text_mining/stats_and_cleaning.py
```
Running this script will generate `corpus_statistics.txt` in the `results/`  folder.

The next step consists in transforming the cleaned textual corpus into numerical representations that can be used for quantitative analysis. This is handled by the script [`text_mining_tf-idf.py`](./text_mining/text_mining_tf-idf.py).

This script applies a linguistic preprocessing pipeline based on `spaCy` to normalize the text through lemmatization and to focus on linguistically meaningful terms. It extracts noun-based unigrams as well as compound expressions (bigrams and trigrams) that are particularly relevant in CSR and ESG discourse. To reduce noise, both generic stopwords and domain-specific stopwords related to corporate, legal, and GDPR vocabulary are removed.

Once the texts are processed, documents are aggregated at the company level so that each company is represented by a single consolidated document. From this aggregated corpus, the script builds a Term-Document Matrix (TDM) and computes TF-IDF representations, which quantify the importance of each term relative to companies.

These representations are exported as CSV files and are intended to serve as inputs for downstream analyses such as clustering, similarity analysis, or topic modeling.

To generate the term-document matrices and TF-IDF representations, run the following command:
```bash
python text_mining/text_mining_tf-idf.py
```
### Text Analytics & Clustering

This stage performs company-level text analytics and unsupervised learning on the TF-IDF representations generated in the previous step. The pipeline is organized into modular components, each responsible for a specific analytical task. These components are imported and orchestrated by a central execution script [`main_text_analytics.py`](./text_mining_analytics.py/main_text_analytics.py), which ensures reproducibility and clarity of the workflow.
   
   - [`company_metadata`](text_mining_analytics.py/company_metadata.py)
        Provides structured company-level metadata (sector, industry, headquarters country, listing status).
        The file is auto-generated from a private csv source.
   - [`loaders.py`](text_mining_analytics.py/loaders.py)
        Contains utility functions to load TF-IDF matrices and company metadata, and to merge analytical results (e.g. clustering outputs) with descriptive company information for interpretation.
   - [`processing.py`](./text_mining_analytics.py/processing.py)
        Implements similarity and term-level analyses, including cosine similarity between companies, term co-occurrence analysis, and identification of globally important terms based on TF-IDF weights.
   - [`models.py`](./text_mining_analytics.py/models.py)
        Centralizes unsupervised learning methods, including K-Means clustering, angular clustering, Non-negative Matrix Factorization (NMF) for topic modeling, cluster interpretation via keywords, and internal validation using silhouette scores.
   - [`visualizers.py`](./text_mining_analytics.py/visualizers.py)
        Provides visualization tools to support model validation and interpretation, including heatmaps for cluster profiling, elbow curves for cluster selection, and silhouette plots for cluster quality assessment.


## Link Analysis

## Authors
- **Bastian Minet** – [Bastian-Mnt](https://github.com/Bastian-Mnt)
- **Matteo Galizia** – [Matteo-glz](https://github.com/Matteo-glz)
- **Yusuf Bulut** – [Yusuf-Bulut-1907](https://github.com/Yusuf-Bulut-1907)