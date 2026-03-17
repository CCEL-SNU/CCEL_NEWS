"""
RSS feed collector v2.
For domain-specific journals (ACS Catal., JACS, EES, etc.),
skips keyword filtering since these journals are already relevant.
Only general-interest journals (Nature, Science) get keyword-filtered.
"""

import urllib.request
import xml.etree.ElementTree as ET
import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

# Journals where ALL papers are potentially relevant (no keyword filter needed)
DOMAIN_SPECIFIC_JOURNALS = {
    "ACS Catalysis",
    "ACS Energy Letters",
    "Energy Environ. Sci.",
    "Joule",
    "Advanced Energy Materials",
    "Nature Catalysis",
    "Nature Energy",
}


def _extract_doi(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'(10\.\d{4,}/[^\s<>"\']+)', text)
    return m.group(1).rstrip(".,;)") if m else None


def _parse_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    for fn in [
        lambda s: parsedate_to_datetime(s).strftime("%Y-%m-%d"),
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%d"),
    ]:
        try:
            return fn(date_str.strip())
        except Exception:
            continue
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%B %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip()[:30], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _clean_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text).strip()


def _get_text(el) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip() if el.text is None else el.text.strip()


def _parse_feed(data: bytes, feed_name: str) -> List[dict]:
    """Parse RSS/Atom/RDF feed data into paper dicts."""
    papers = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        logger.error(f"  XML parse error for {feed_name}: {e}")
        return []

    ns_atom = "http://www.w3.org/2005/Atom"
    ns_dc = "http://purl.org/dc/elements/1.1/"
    ns_prism = "http://prismstandard.org/namespaces/basic/2.0/"
    ns_rss1 = "http://purl.org/rss/1.0/"

    # Find items across all feed formats
    items = root.findall(".//item")
    if not items:
        items = root.findall(f".//{{{ns_atom}}}entry")
    if not items:
        items = root.findall(f".//{{{ns_rss1}}}item")
    if not items:
        items = root.findall("entry")

    for item in items:
        try:
            # Title
            title = None
            for tn in ["title", f"{{{ns_atom}}}title", f"{{{ns_rss1}}}title"]:
                el = item.find(tn)
                if el is not None:
                    title = _clean_html(_get_text(el))
                    break
            if not title:
                continue

            # Link
            link = ""
            for tn in ["link", f"{{{ns_atom}}}link", f"{{{ns_rss1}}}link"]:
                el = item.find(tn)
                if el is not None:
                    link = el.text or el.get("href", "")
                    break
            link = link.strip()

            # Description
            desc = ""
            for tn in ["description", f"{{{ns_atom}}}summary", f"{{{ns_atom}}}content",
                        f"{{{ns_rss1}}}description"]:
                el = item.find(tn)
                if el is not None:
                    desc = _clean_html(_get_text(el))
                    break

            # Date
            date = None
            for tn in ["pubDate", f"{{{ns_atom}}}published", f"{{{ns_atom}}}updated",
                        f"{{{ns_dc}}}date", f"{{{ns_prism}}}publicationDate"]:
                el = item.find(tn)
                if el is not None and _get_text(el):
                    date = _parse_date(_get_text(el))
                    if date:
                        break

            # DOI
            doi = None
            doi_el = item.find(f"{{{ns_prism}}}doi")
            if doi_el is not None:
                doi = _get_text(doi_el)
            if not doi:
                doi = _extract_doi(link)
            if not doi:
                doi = _extract_doi(desc)

            # Authors
            authors_str = ""
            for tn in [f"{{{ns_dc}}}creator", "author", f"{{{ns_atom}}}author"]:
                el = item.find(tn)
                if el is not None:
                    name_el = el.find(f"{{{ns_atom}}}name")
                    authors_str = _get_text(name_el) if name_el is not None else _get_text(el)
                    break

            papers.append({
                "source": feed_name,
                "doi": doi,
                "title": title,
                "authors": [],
                "authors_str": authors_str,
                "abstract": desc[:800] if desc else "",
                "date": date,
                "url": link,
                "pdf_url": None,
                "open_access": False,
            })

        except Exception as e:
            logger.warning(f"  Failed to parse item in {feed_name}: {e}")

    return papers


def collect(feeds_config: List[dict], days_back: int = 30,
            filter_keywords: List[str] = None) -> List[dict]:
    """
    Collect papers from RSS feeds.
    Domain-specific journals skip keyword filtering.
    General journals (Nature, Science, JACS) use keyword filtering.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_papers = {}

    # Build keyword match set for general journals
    match_words = set()
    if filter_keywords:
        for kw in filter_keywords:
            for word in kw.lower().split():
                if len(word) >= 4:
                    match_words.add(word)

    for feed in feeds_config:
        fname = feed["name"]
        furl = feed["url"]
        is_domain_specific = fname in DOMAIN_SPECIFIC_JOURNALS

        logger.info(f"RSS fetch: {fname} ({'no filter' if is_domain_specific else 'keyword-filtered'})")

        try:
            req = urllib.request.Request(furl, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CCEL-DailyNews/1.0",
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()

            papers = _parse_feed(data, fname)
            accepted = 0

            for paper in papers:
                # Date filter
                if paper["date"] and paper["date"] < cutoff:
                    continue

                # Keyword filter ONLY for general journals
                if not is_domain_specific and match_words:
                    text = (paper["title"] + " " + paper.get("abstract", "")).lower()
                    if not any(w in text for w in match_words):
                        continue

                key = paper["doi"] or paper["title"].lower()[:80]
                if key not in all_papers:
                    all_papers[key] = paper
                    accepted += 1

            logger.info(f"  {fname}: {len(papers)} parsed, {accepted} accepted")

        except urllib.error.HTTPError as e:
            logger.error(f"  RSS HTTP error for {fname}: {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"  RSS error for {fname}: {e}")

        time.sleep(1)

    papers = list(all_papers.values())
    logger.info(f"RSS total: {len(papers)} papers")
    return papers