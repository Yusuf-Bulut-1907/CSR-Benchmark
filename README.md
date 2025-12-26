# CSR Benchmark Analysis

## Introduction

The objective of this project is to analyze and compare the CSR (Corporate Social Responsibility) discourses of major corporations, utilizing their institutional web pages.

Acting as consultants, our mission is to collect, analyze, and compare the CSR commitments published on the websites of large companies from various sectors. This strategic benchmark aims to identify **dominant trends and themes**, extract key **insights**, and analyze the relationships between the data in order to provide informed strategic recommendations.

## Table of Contents

- [**Project Plan**](#project-plan)
 - [1. Data Collection (Scraping)](#1-data-collection-scraping)
 - [2. Textual Analysis (Text Mining)](#2-textual-analysis-text-mining)
 - [3. Network Analysis (Web Mining)](#3-network-analysis-web-mining)
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

In this part,

## Link Analysis

## Authors
- **Bastian Minet** – [Bastian-Mnt](https://github.com/Bastian-Mnt)
- **Matteo Galizia** – [Matteo-glz](https://github.com/Matteo-glz)
- **Yusuf Bulut** – [Yusuf-Bulut-1907](https://github.com/Yusuf-Bulut-1907)