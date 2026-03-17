"""
Semantic Scholar collector.
Tracks specific authors (research group PIs) and fetches their recent papers.
"""

import urllib.request
import urllib.parse
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,abstract,year,publicationDate,authors,citationCount,journal,openAccessPdf,url"


def _api_get(endpoint: str, params: dict = None, api_key: str = None) -> Optional[dict]:
    """Make a GET request to the Semantic Scholar API."""
    url = f"{S2_API}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    headers = {"User-Agent": "CCEL-DailyNews/1.0"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("S2 rate limit hit, waiting 60s...")
            time.sleep(60)
            return _api_get(endpoint, params, api_key)
        logger.error(f"S2 API error {e.code}: {e.reason}")
        return None
    except Exception as e:
        logger.error(f"S2 API request failed: {e}")
        return None


def _parse_paper(paper: dict, group_id: str = None) -> Optional[dict]:
    """Convert an S2 paper object to our standard format."""
    if not paper or not paper.get("title"):
        return None

    doi = None
    ext = paper.get("externalIds", {})
    if ext:
        doi = ext.get("DOI")

    authors = []
    for a in paper.get("authors", []):
        if a.get("name"):
            authors.append(a["name"])

    pub_date = paper.get("publicationDate")
    if not pub_date:
        year = paper.get("year")
        pub_date = f"{year}-01-01" if year else None

    journal = paper.get("journal", {})
    journal_name = journal.get("name", "Unknown") if journal else "Unknown"

    oa = paper.get("openAccessPdf")
    pdf_url = oa.get("url") if oa else None

    return {
        "source": journal_name,
        "s2_id": paper.get("paperId"),
        "doi": doi,
        "title": paper["title"],
        "authors": authors,
        "authors_str": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
        "abstract": paper.get("abstract", ""),
        "date": pub_date,
        "citations": paper.get("citationCount", 0),
        "pdf_url": pdf_url,
        "url": paper.get("url", ""),
        "open_access": pdf_url is not None,
        "group": group_id,
    }


def collect_by_author(author_id: str, group_id: str, days_back: int = 30,
                      api_key: str = None) -> List[dict]:
    """
    Fetch recent papers by a specific author.
    Default 30 days back (was 7 - too short for most groups).
    """
    logger.info(f"S2 author search: {group_id} (ID: {author_id}, {days_back}d window)")

    data = _api_get(
        f"author/{author_id}/papers",
        params={"fields": S2_FIELDS, "limit": 100},
        api_key=api_key,
    )
    if not data or "data" not in data:
        logger.warning(f"No data for author {author_id}")
        return []

    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    papers = []

    for item in data["data"]:
        paper = _parse_paper(item, group_id)
        if paper is None:
            continue
        # Include papers with no date (newly indexed, date not yet set)
        if paper["date"] is None or paper["date"] >= cutoff:
            papers.append(paper)

    logger.info(f"  {group_id}: {len(papers)} papers in {days_back}d window")
    time.sleep(1)
    return papers


def collect_groups(groups_config: dict, days_back: int = 30,
                   api_key: str = None) -> List[dict]:
    """
    Collect recent papers from all tracked research groups.
    Default window is 30 days for better coverage.
    """
    all_papers = {}

    for gid, gcfg in groups_config.items():
        s2_id = gcfg.get("semantic_scholar_id")
        if not s2_id:
            logger.warning(f"No S2 ID for group {gid}, skipping")
            continue

        papers = collect_by_author(s2_id, gid, days_back, api_key)
        for p in papers:
            key = p.get("doi") or p.get("s2_id") or p["title"]
            if key not in all_papers:
                all_papers[key] = p

    result = list(all_papers.values())
    logger.info(f"S2 groups total: {len(result)} papers")
    return result