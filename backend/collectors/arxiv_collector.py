"""
arXiv paper collector v3.
No category pre-filtering - Claude AI will classify relevance later.
Only filters by date and deduplicates.
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
    No category filtering - keywords are specific enough for CCEL topics.
    Claude AI will handle relevance scoring later.
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

            matched = 0
            for entry in entries:
                paper = _parse_entry(entry)
                if paper is None:
                    continue
                # Date filter only
                if paper["date"] < cutoff:
                    continue
                # Dedup
                key = paper["arxiv_id"]
                if key not in all_papers:
                    all_papers[key] = paper
                    matched += 1

            logger.info(f"  Fetched {len(entries)}, accepted {matched} (date-filtered)")

        except Exception as e:
            logger.error(f"arXiv API error for '{keyword}': {e}")

        # Respect arXiv rate limit
        time.sleep(3)

    papers = list(all_papers.values())
    logger.info(f"arXiv total: {len(papers)} unique papers")
    return papers