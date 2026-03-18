"""
OpenAlex collector.
Tracks specific authors (research group PIs) and fetches their recent papers.
Replaces s2_collector for better coverage of recent publications.
"""

import urllib.request
import urllib.parse
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

OPENALEX_API = "https://api.openalex.org"
WORKS_SELECT = "id,doi,title,display_name,publication_date,authorships,cited_by_count,open_access,primary_location,abstract_inverted_index"


def _api_get(url: str, api_key: str = None, retries: int = 2) -> Optional[dict]:
    """Make a GET request to the OpenAlex API."""
    if api_key:
        sep = "&" if "?" in url else "?"
        url += f"{sep}api_key={api_key}"

    headers = {"User-Agent": "CCEL-DailyNews/1.0 (mailto:ccel@snu.ac.kr)"}

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"OpenAlex rate limit hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"OpenAlex API error {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return None
    return None


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct plain text abstract from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    positions = {}
    for word, indices in inverted_index.items():
        for idx in indices:
            positions[idx] = word
    if not positions:
        return ""
    max_pos = max(positions.keys())
    return " ".join(positions.get(i, "") for i in range(max_pos + 1))


def _parse_work(work: dict, group_id: str = None) -> Optional[dict]:
    """Convert an OpenAlex work object to our standard paper dict format."""
    if not work or not work.get("title"):
        return None

    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)

    pub_date = work.get("publication_date")

    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    journal_name = source.get("display_name", "Unknown")

    oa = work.get("open_access") or {}
    pdf_url = location.get("pdf_url") or oa.get("oa_url")

    landing_url = location.get("landing_page_url") or ""
    if not landing_url and doi:
        landing_url = f"https://doi.org/{doi}"

    openalex_id = work.get("id", "")
    if openalex_id.startswith("https://openalex.org/"):
        openalex_id = openalex_id[len("https://openalex.org/"):]

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    return {
        "source": journal_name,
        "openalex_id": openalex_id,
        "doi": doi,
        "title": work["title"],
        "authors": authors,
        "authors_str": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
        "abstract": abstract,
        "date": pub_date,
        "citations": work.get("cited_by_count", 0),
        "pdf_url": pdf_url,
        "url": landing_url,
        "open_access": oa.get("is_oa", False),
        "group": group_id,
    }


def collect_by_author(author_id: str, group_id: str, days_back: int = 30,
                      api_key: str = None) -> List[dict]:
    """
    Fetch recent papers by a specific author using OpenAlex.
    Uses server-side date filtering and cursor pagination for complete coverage.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    logger.info(f"OpenAlex author search: {group_id} (ID: {author_id}, from {cutoff})")

    papers = []
    cursor = "*"

    while cursor:
        params = {
            "filter": f"author.id:{author_id},from_publication_date:{cutoff}",
            "select": WORKS_SELECT,
            "per_page": 100,
            "cursor": cursor,
            "sort": "publication_date:desc",
        }
        url = f"{OPENALEX_API}/works?" + urllib.parse.urlencode(params)

        data = _api_get(url, api_key=api_key)
        if not data or "results" not in data:
            logger.warning(f"No data for author {author_id}")
            break

        for item in data["results"]:
            paper = _parse_work(item, group_id)
            if paper:
                papers.append(paper)

        next_cursor = data.get("meta", {}).get("next_cursor")
        if next_cursor and next_cursor != cursor and data["results"]:
            cursor = next_cursor
            time.sleep(0.2)
        else:
            break

    logger.info(f"  {group_id}: {len(papers)} papers since {cutoff}")
    time.sleep(0.5)
    return papers


def collect_groups(groups_config: dict, days_back: int = 30,
                   api_key: str = None) -> List[dict]:
    """
    Collect recent papers from all tracked research groups via OpenAlex.
    """
    all_papers = {}

    for gid, gcfg in groups_config.items():
        oa_id = gcfg.get("openalex_id")
        if not oa_id:
            logger.warning(f"No OpenAlex ID for group {gid}, skipping")
            continue

        papers = collect_by_author(oa_id, gid, days_back, api_key)
        for p in papers:
            key = p.get("doi") or p.get("openalex_id") or p["title"]
            if key not in all_papers:
                all_papers[key] = p

    result = list(all_papers.values())
    logger.info(f"OpenAlex groups total: {len(result)} papers")
    return result
