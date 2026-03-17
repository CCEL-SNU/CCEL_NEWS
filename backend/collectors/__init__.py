"""
Collectors package.
Provides a unified collect_all() function that merges results from all sources.
"""

import logging
import hashlib
from typing import List

from . import arxiv_collector
from . import s2_collector
from . import rss_collector

logger = logging.getLogger(__name__)


def _make_key(paper: dict) -> str:
    """Generate a unique key for deduplication. Prefer DOI, fall back to title hash."""
    if paper.get("doi"):
        return f"doi:{paper['doi'].lower()}"
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    # Title-based hash as last resort
    title_norm = paper.get("title", "").lower().strip()
    return f"title:{hashlib.md5(title_norm.encode()).hexdigest()}"


def collect_all(config: dict) -> List:
    """
    Run all collectors and merge/deduplicate results.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        List of unique paper dicts, sorted by date (newest first).
    """
    all_papers = {}

    # 1. arXiv
    logger.info("=" * 50)
    logger.info("Collecting from arXiv...")
    keywords = config.get("keywords", {}).get("primary", [])
    arxiv_papers = arxiv_collector.collect(keywords, days_back=3)
    for p in arxiv_papers:
        key = _make_key(p)
        if key not in all_papers:
            all_papers[key] = p

    # 2. Semantic Scholar (group tracking)
    logger.info("=" * 50)
    logger.info("Collecting from Semantic Scholar (groups)...")
    groups = config.get("groups", {})
    s2_key = config.get("semantic_scholar_api_key")
    group_papers = s2_collector.collect_groups(groups, api_key=s2_key)
    for p in group_papers:
        key = _make_key(p)
        if key not in all_papers:
            all_papers[key] = p
        else:
            # Merge group info into existing paper
            if p.get("group"):
                all_papers[key]["group"] = p["group"]
            # Merge citation count if available
            if p.get("citations"):
                all_papers[key]["citations"] = p["citations"]

    # 3. RSS feeds
    logger.info("=" * 50)
    logger.info("Collecting from RSS feeds...")
    feeds = config.get("rss_feeds", [])
    flat_keywords = keywords + config.get("keywords", {}).get("secondary", [])
    rss_papers = rss_collector.collect(feeds, filter_keywords=flat_keywords)
    for p in rss_papers:
        key = _make_key(p)
        if key not in all_papers:
            all_papers[key] = p

    # Sort by date (newest first), then by relevance/citations
    papers = list(all_papers.values())
    papers.sort(key=lambda p: (p.get("date") or "1900-01-01"), reverse=True)

    # Check for CCEL group papers
    ccel_aliases = []
    if "ccel" in groups:
        ccel_aliases = groups["ccel"].get("aliases", [])

    for p in papers:
        # Auto-detect CCEL papers by author name matching
        if not p.get("group"):
            auth_str = p.get("authors_str", "") + " ".join(p.get("authors", []))
            for alias in ccel_aliases:
                if alias.lower() in auth_str.lower():
                    p["group"] = "ccel"
                    p["ccel"] = True
                    break
        if p.get("group") == "ccel":
            p["ccel"] = True

    logger.info("=" * 50)
    logger.info(f"Total collected: {len(papers)} unique papers")
    logger.info(f"  CCEL group: {sum(1 for p in papers if p.get('ccel'))}")
    return papers
