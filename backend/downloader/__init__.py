"""
PDF Downloader.
Downloads paper PDFs using SNU institutional IP access.
Includes rate limiting, retry logic, and publisher-specific URL resolution.
"""

import os
import re
import time
import urllib.request
import urllib.error
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Publisher-specific PDF URL patterns
# Maps DOI prefixes to functions that construct PDF download URLs
PUBLISHER_PATTERNS = {
    # ACS (JACS, ACS Catal., ACS Energy Lett., etc.)
    "10.1021": lambda doi: f"https://pubs.acs.org/doi/pdf/{doi}",
    # Wiley (Angew. Chem., Adv. Energy Mater., Small, etc.)
    "10.1002": lambda doi: f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
    # Elsevier (Joule, Applied Catalysis B, etc.)
    "10.1016": lambda doi: f"https://www.sciencedirect.com/science/article/pii/{{pii}}/pdfft",
    # RSC (Energy Environ. Sci., J. Mater. Chem. A, etc.)
    "10.1039": lambda doi: f"https://pubs.rsc.org/en/content/articlepdf/{doi.split('/')[-1]}",
    # Nature group (Nature, Nat. Catal., Nat. Energy, etc.)
    "10.1038": lambda doi: f"https://www.nature.com/articles/{doi.split('/')[-1]}.pdf",
    # Science / AAAS
    "10.1126": lambda doi: f"https://www.science.org/doi/pdf/{doi}",
    # APS (Phys. Rev. Lett., Phys. Rev. B, etc.)
    "10.1103": lambda doi: f"https://journals.aps.org/prl/pdf/{doi}",
    # Springer
    "10.1007": lambda doi: f"https://link.springer.com/content/pdf/{doi}.pdf",
    # IOP
    "10.1088": lambda doi: f"https://iopscience.iop.org/article/{doi}/pdf",
}


class PaperDownloader:
    """Downloads PDFs with rate limiting and SNU IP institutional access."""

    def __init__(self, config: dict):
        dl_cfg = config.get("downloader", {})
        self.max_per_day = dl_cfg.get("max_papers_per_day", 30)
        self.delay = dl_cfg.get("delay_between_requests_sec", 8)
        self.timeout = dl_cfg.get("timeout_sec", 30)
        self.pdf_dir = Path(dl_cfg.get("pdf_directory", "./data/papers"))
        self.user_agent = dl_cfg.get("user_agent", "CCEL-DailyNews/1.0")
        self.downloaded_today = 0

        # Create PDF directory
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_pdf_url(self, paper: dict) -> Optional[str]:
        """Resolve the PDF download URL for a paper."""
        # 1. Direct PDF URL already available (arXiv, OA)
        if paper.get("pdf_url"):
            return paper["pdf_url"]

        # 2. arXiv papers
        if paper.get("arxiv_id"):
            return f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"

        # 3. DOI-based resolution
        doi = paper.get("doi")
        if not doi:
            return None

        # Match DOI prefix to publisher
        for prefix, url_fn in PUBLISHER_PATTERNS.items():
            if doi.startswith(prefix):
                return url_fn(doi)

        # 4. Fallback: try Unpaywall API for OA copies
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=ccel@snu.ac.kr"
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                data = json.loads(resp.read())
                oa_loc = data.get("best_oa_location")
                if oa_loc and oa_loc.get("url_for_pdf"):
                    return oa_loc["url_for_pdf"]
        except Exception:
            pass

        # 5. Generic DOI redirect (works with SNU IP for subscribed journals)
        return f"https://doi.org/{doi}"

    def _safe_filename(self, paper: dict) -> str:
        """Generate a safe filename for the PDF."""
        # Use DOI or arxiv_id as base
        if paper.get("doi"):
            name = paper["doi"].replace("/", "_").replace(".", "_")
        elif paper.get("arxiv_id"):
            name = f"arxiv_{paper['arxiv_id'].replace('/', '_')}"
        else:
            # Title hash
            import hashlib
            name = hashlib.md5(paper["title"].encode()).hexdigest()[:16]
        return f"{name}.pdf"

    def download_one(self, paper: dict) -> Optional[str]:
        """
        Download a single paper's PDF.

        Returns:
            Local file path if successful, None otherwise.
        """
        if self.downloaded_today >= self.max_per_day:
            logger.warning(f"Daily download limit reached ({self.max_per_day})")
            return None

        filename = self._safe_filename(paper)
        filepath = self.pdf_dir / filename

        # Skip if already downloaded
        if filepath.exists() and filepath.stat().st_size > 1000:
            logger.info(f"  Already exists: {filename}")
            paper["local_pdf"] = str(filepath)
            return str(filepath)

        # Resolve URL
        pdf_url = self._resolve_pdf_url(paper)
        if not pdf_url:
            logger.warning(f"  No PDF URL for: {paper['title'][:60]}")
            return None

        logger.info(f"  Downloading: {filename}")
        logger.info(f"    URL: {pdf_url}")

        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,*/*",
                "Referer": paper.get("url", ""),
            }
            req = urllib.request.Request(pdf_url, headers=headers)

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")

                # Check if we got a PDF (not an HTML login page)
                if "html" in content_type.lower() and "pdf" not in content_type.lower():
                    # Might be a redirect to login page
                    # Try following the response URL for DOI redirects
                    final_url = resp.url
                    if final_url != pdf_url:
                        logger.info(f"    Redirected to: {final_url}")
                        # Try to find PDF link on the page
                        # For now, skip these
                        logger.warning(f"    Got HTML instead of PDF, skipping")
                        return None

                data = resp.read()

                # Verify it looks like a PDF
                if len(data) < 1000 or not data[:5] == b'%PDF-':
                    logger.warning(f"    Downloaded file is not a valid PDF ({len(data)} bytes)")
                    return None

                with open(filepath, "wb") as f:
                    f.write(data)

            self.downloaded_today += 1
            paper["local_pdf"] = str(filepath)
            logger.info(f"    Success: {len(data)} bytes")

            # Rate limiting
            time.sleep(self.delay)
            return str(filepath)

        except urllib.error.HTTPError as e:
            if e.code == 403:
                logger.warning(f"    Access denied (403) - may not be subscribed")
            elif e.code == 404:
                logger.warning(f"    Not found (404)")
            elif e.code == 429:
                logger.warning(f"    Rate limited (429) - waiting 60s")
                time.sleep(60)
            else:
                logger.error(f"    HTTP error {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"    Download failed: {e}")
            return None

    def download_batch(self, papers: list) -> dict:
        """
        Download PDFs for a batch of papers.

        Args:
            papers: List of paper dicts.

        Returns:
            Dict with download statistics.
        """
        stats = {"total": len(papers), "downloaded": 0, "skipped": 0, "failed": 0, "limit_reached": False}

        # Prioritize: CCEL papers first, then by relevance
        sorted_papers = sorted(papers, key=lambda p: (
            not p.get("ccel", False),
            -p.get("relevance", 0),
        ))

        for paper in sorted_papers:
            if self.downloaded_today >= self.max_per_day:
                stats["limit_reached"] = True
                logger.warning("Daily limit reached, stopping downloads")
                break

            result = self.download_one(paper)
            if result:
                stats["downloaded"] += 1
            elif paper.get("local_pdf"):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"Download stats: {stats}")
        return stats
