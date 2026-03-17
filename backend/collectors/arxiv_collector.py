"""
arXiv paper collector v4.
Fetches by keyword, then filters by CCEL-relevant arXiv categories in Python.
This avoids the URL-too-long issue from v2 while keeping results relevant.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# CCEL-relevant arXiv categories (post-fetch filter)
# Catalysis/battery papers live under cond-mat.mtrl-sci and physics.chem-ph
ALLOWED_CATEGORIES = {
    # Condensed Matter
    "cond-mat.mtrl-sci",   # Materials Science (battery, catalysis materials)
    "cond-mat.str-el",     # Strongly Correlated Electrons
    "cond-mat.other",      # Other condensed matter
    "cond-mat.stat-mech",  # Statistical Mechanics (thermodynamics)
    "cond-mat.soft",       # Soft Condensed Matter (polymers, electrolytes)
    # Physics
    "physics.chem-ph",     # Chemical Physics (DFT, reaction mechanisms)
    "physics.comp-ph",     # Computational Physics (simulations)
    "physics.atm-clus",    # Atomic and Molecular Clusters
    "physics.app-ph",      # Applied Physics
    # Chemistry (if present)
    "chem-ph",             # Chemical Physics (older tag)
    # Machine Learning (for MLIP, GNN potentials)
    "cs.LG",               # Machine Learning
    "cs.AI",               # Artificial Intelligence
    "stat.ML",             # Statistical ML
    "cs.CE",               # Computational Engineering
}


def _has_relevant_category(categories: List[str]) -> bool:
    """Check if any of the paper's categories match CCEL-relevant ones."""
    for cat in categories:
        # Exact match
        if cat in ALLOWED_CATEGORIES:
            return True
        # Parent match (e.g., "cond-mat.mtrl-sci" matches if "cond-mat" is checked)
        parent = cat.split(".")[0]
        if parent == "cond-mat" or parent == "physics":
            # Allow all cond-mat.* and physics.* subcategories
            # since catalysis/battery papers appear across many
            return True
    return False


def _parse_entry(entry) -> Optional[dict]:
    """Parse a single arXiv Atom entry into a paper dict."""
    try:
        arxiv_id = entry.find("atom:id", NS).text.strip().split("/abs/")[-1]
        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", NS).text[:10]
        updated = entry.find("atom:updated", NS).text[:10]

        authors = []
        for author_el in entry.findall("atom:author", NS):
            name = author_el.find("atom:name", NS).text.strip()
            authors.append(name)

        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        categories = []
        for cat in entry.findall("arxiv:primary_category", NS):
            categories.append(cat.get("term"))
        for cat in entry.findall("atom:category", NS):
            t = cat.get("term")
            if t not in categories:
                categories.append(t)

        doi_el = entry.find("arxiv:doi", NS)
        doi = doi_el.text.strip() if doi_el is not None else None

        return {
            "source": "arXiv",
            "arxiv_id": arxiv_id,
            "doi": doi,
            "title": title,
            "authors": authors,
            "authors_str": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
            "abstract": summary,
            "date": published,
            "updated": updated,
            "categories": categories,
            "pdf_url": pdf_url,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "open_access": True,
        }
    except Exception as e:
        logger.warning(f"Failed to parse arXiv entry: {e}")
        return None


def collect(keywords: List[str], days_back: int = 3, max_results_per_query: int = 30) -> List[dict]:
    """
    Search arXiv for recent papers matching keywords.
    Post-fetch filtering by CCEL-relevant categories to remove irrelevant results.
    """
    all_papers = {}
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    for keyword in keywords:
        search_query = f'ti:"{keyword}" OR abs:"{keyword}"'

        logger.info(f"arXiv search: '{keyword}'")
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results_per_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CCEL-DailyNews/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            root = ET.fromstring(data)
            entries = root.findall("atom:entry", NS)

            date_ok = 0
            cat_ok = 0
            for entry in entries:
                paper = _parse_entry(entry)
                if paper is None:
                    continue
                # 1) Date filter
                if paper["date"] < cutoff:
                    continue
                date_ok += 1
                # 2) Category filter (post-fetch, in Python)
                if not _has_relevant_category(paper.get("categories", [])):
                    continue
                cat_ok += 1
                # 3) Dedup
                key = paper["arxiv_id"]
                if key not in all_papers:
                    all_papers[key] = paper

            logger.info(f"  Fetched {len(entries)}, date-ok {date_ok}, category-ok {cat_ok}")

        except Exception as e:
            logger.error(f"arXiv API error for '{keyword}': {e}")

        # Respect arXiv rate limit
        time.sleep(3)

    papers = list(all_papers.values())
    logger.info(f"arXiv total: {len(papers)} unique relevant papers")
    return papers