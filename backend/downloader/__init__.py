"""
PDF Downloader with Selenium support.
Uses Selenium (headless Chrome) for publishers that block urllib requests (ACS, Wiley, Science, RSC).
Falls back to urllib for open-access sources (arXiv, Nature).

Requirements:
    pip install selenium
    Chrome browser installed on the system
    ChromeDriver matching Chrome version (or use selenium 4.6+ with auto driver management)

Setup on lab server:
    pip install selenium --break-system-packages
    # Chrome is usually pre-installed on Windows
    # Selenium 4.6+ auto-downloads ChromeDriver
"""

import os
import re
import time
import urllib.request
import urllib.error
import logging
import hashlib
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Publishers that need Selenium (block direct urllib requests)
SELENIUM_PUBLISHERS = {"10.1021", "10.1002", "10.1039", "10.1126"}

# Publishers where urllib works fine
URLLIB_PUBLISHERS = {"10.1038"}  # Nature

# Publisher-specific PDF URL patterns
PUBLISHER_PDF_URLS = {
    # ACS (JACS, ACS Catal., ACS Energy Lett., etc.)
    "10.1021": lambda doi: f"https://pubs.acs.org/doi/pdf/{doi}",
    # Wiley (Angew. Chem., Adv. Energy Mater., Small, etc.)
    "10.1002": lambda doi: f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
    # RSC (Energy Environ. Sci., J. Mater. Chem. A, etc.)
    "10.1039": lambda doi: f"https://pubs.rsc.org/en/content/articlepdf/{doi.split('/')[-1]}",
    # Nature group
    "10.1038": lambda doi: f"https://www.nature.com/articles/{doi.split('/')[-1]}.pdf",
    # Science / AAAS
    "10.1126": lambda doi: f"https://www.science.org/doi/pdf/{doi}",
    # Elsevier
    "10.1016": lambda doi: f"https://doi.org/{doi}",
    # APS
    "10.1103": lambda doi: f"https://journals.aps.org/prl/pdf/{doi}",
    # Springer
    "10.1007": lambda doi: f"https://link.springer.com/content/pdf/{doi}.pdf",
    # IOP
    "10.1088": lambda doi: f"https://iopscience.iop.org/article/{doi}/pdf",
}


class PaperDownloader:
    """Downloads PDFs with Selenium for paywalled publishers, urllib for open access."""

    def __init__(self, config: dict):
        dl_cfg = config.get("downloader", {})
        self.max_per_day = dl_cfg.get("max_papers_per_day", 30)
        self.delay = dl_cfg.get("delay_between_requests_sec", 8)
        self.timeout = dl_cfg.get("timeout_sec", 30)
        self.pdf_dir = Path(dl_cfg.get("pdf_directory", "./data/papers"))
        self.user_agent = dl_cfg.get("user_agent", "CCEL-DailyNews/1.0")
        self.downloaded_today = 0
        self._driver = None
        self._selenium_available = None

        # Create PDF directory
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def _check_selenium(self) -> bool:
        """Check if Selenium is available."""
        if self._selenium_available is not None:
            return self._selenium_available
        try:
            from selenium import webdriver
            self._selenium_available = True
            logger.info("Selenium is available")
        except ImportError:
            self._selenium_available = False
            logger.warning("Selenium not installed. Install with: pip install selenium")
            logger.warning("Paywalled PDFs (ACS, Wiley, Science, RSC) will be skipped.")
        return self._selenium_available

    def _get_driver(self):
        """Get or create a Selenium Chrome WebDriver."""
        if self._driver is not None:
            return self._driver

        if not self._check_selenium():
            return None

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

        # Configure download directory
        prefs = {
            "download.default_directory": str(self.pdf_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,  # Don't open PDF in browser
        }
        options.add_experimental_option("prefs", prefs)

        try:
            # Selenium 4.6+ auto-manages ChromeDriver
            self._driver = webdriver.Chrome(options=options)
            self._driver.set_page_load_timeout(self.timeout)
            logger.info("Chrome WebDriver initialized (headless)")
            return self._driver
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            logger.error("Make sure Chrome is installed and up to date.")
            self._selenium_available = False
            return None

    def _close_driver(self):
        """Close the Selenium driver."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def _get_doi_prefix(self, doi: str) -> str:
        """Extract publisher prefix from DOI."""
        if not doi:
            return ""
        return doi.split("/")[0] if "/" in doi else ""

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

        prefix = self._get_doi_prefix(doi)
        if prefix in PUBLISHER_PDF_URLS:
            return PUBLISHER_PDF_URLS[prefix](doi)

        # 4. Generic DOI redirect
        return f"https://doi.org/{doi}"

    def _safe_filename(self, paper: dict) -> str:
        """Generate a safe filename for the PDF."""
        if paper.get("doi"):
            name = paper["doi"].replace("/", "_").replace(".", "_")
        elif paper.get("arxiv_id"):
            name = f"arxiv_{paper['arxiv_id'].replace('/', '_')}"
        else:
            name = hashlib.md5(paper["title"].encode()).hexdigest()[:16]
        return f"{name}.pdf"

    def _needs_selenium(self, paper: dict) -> bool:
        """Check if this paper's publisher requires Selenium."""
        doi = paper.get("doi", "")
        prefix = self._get_doi_prefix(doi)
        return prefix in SELENIUM_PUBLISHERS

    def _download_with_urllib(self, paper: dict, pdf_url: str, filepath: Path) -> Optional[str]:
        """Download PDF using urllib (for open-access / Nature)."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*",
                "Referer": paper.get("url", ""),
            }
            req = urllib.request.Request(pdf_url, headers=headers)

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")

                if "html" in content_type.lower() and "pdf" not in content_type.lower():
                    logger.warning(f"    Got HTML instead of PDF (urllib), skipping")
                    return None

                data = resp.read()

                if len(data) < 1000 or not data[:5] == b'%PDF-':
                    logger.warning(f"    Not a valid PDF ({len(data)} bytes)")
                    return None

                with open(filepath, "wb") as f:
                    f.write(data)

            paper["local_pdf"] = str(filepath)
            logger.info(f"    Success (urllib): {len(data)} bytes")
            return str(filepath)

        except urllib.error.HTTPError as e:
            if e.code == 403:
                logger.warning(f"    Access denied (403/urllib)")
            elif e.code == 404:
                logger.warning(f"    Not found (404/urllib)")
            else:
                logger.error(f"    HTTP error {e.code} (urllib)")
            return None
        except Exception as e:
            logger.error(f"    urllib download failed: {e}")
            return None

    _ERROR_PATTERNS = [
        "page not found",
        "404 not found",
        "this page is not available",
        "we couldn't find the page",
        "couldn't find the page you were looking for",
        "sorry, we couldn't find",
        "the page you requested",
        "page does not exist",
        "article not found",
        "no longer available",
        "has been removed",
        "you do not have access",
        "institutional login required",
    ]

    _TITLE_ERROR_PATTERNS = ["404", "not found", "error", "page not found", "access denied"]

    def _is_error_page(self, driver) -> bool:
        """Check if the current browser page is an error/login page."""
        current_url = driver.current_url.lower()

        url_patterns = ["/404", "/403", "error=", "errorpage"]
        if any(p in current_url for p in url_patterns):
            return True

        title = ""
        try:
            title = driver.title.lower()
        except Exception:
            pass

        if title and any(p in title for p in self._TITLE_ERROR_PATTERNS):
            return True

        try:
            page_text = driver.page_source[:3000].lower() if driver.page_source else ""
        except Exception:
            return False

        if any(p in page_text for p in self._ERROR_PATTERNS):
            return True

        return False

    def _wait_for_download(self, existing_files: set, max_wait: int = 15) -> Optional[Path]:
        """Wait for a new PDF to appear in the download directory."""
        for _ in range(max_wait):
            time.sleep(1)
            for f in self.pdf_dir.iterdir():
                if f.name.endswith(".crdownload") or f.name.endswith(".tmp"):
                    continue
                if f.suffix == ".pdf" and f.name not in existing_files:
                    if f.stat().st_size > 1000:
                        return f
        return None

    def _download_with_selenium(self, paper: dict, pdf_url: str, filepath: Path) -> Optional[str]:
        """Download PDF using Selenium headless Chrome (for paywalled publishers)."""
        driver = self._get_driver()
        if not driver:
            logger.warning(f"    Selenium not available, skipping paywalled PDF")
            return None

        try:
            existing_files = {f.name for f in self.pdf_dir.iterdir() if f.suffix == ".pdf"}

            logger.info(f"    Selenium navigating to PDF...")
            driver.get(pdf_url)
            time.sleep(3)

            # Check for quick auto-download first (PDF may already be downloading)
            downloaded = self._wait_for_download(existing_files, max_wait=5)
            if downloaded:
                if downloaded.name != filepath.name:
                    if filepath.exists():
                        filepath.unlink()
                    downloaded.rename(filepath)
                paper["local_pdf"] = str(filepath)
                logger.info(f"    Success (selenium/auto-download): {filepath.stat().st_size} bytes")
                return str(filepath)

            # No quick download — check if this is an error page
            if self._is_error_page(driver):
                logger.warning(f"    Error page detected, skipping")
                return None

            # Wait longer for slower downloads
            downloaded = self._wait_for_download(existing_files, max_wait=10)
            if downloaded:
                if downloaded.name != filepath.name:
                    if filepath.exists():
                        filepath.unlink()
                    downloaded.rename(filepath)
                paper["local_pdf"] = str(filepath)
                logger.info(f"    Success (selenium/auto-download): {filepath.stat().st_size} bytes")
                return str(filepath)

            # Fallback: use cookies from Selenium session for direct HTTP download
            try:
                cookies = driver.get_cookies()
                cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

                req = urllib.request.Request(pdf_url, headers={
                    "User-Agent": driver.execute_script("return navigator.userAgent"),
                    "Cookie": cookie_str,
                    "Accept": "application/pdf,*/*",
                    "Referer": paper.get("url", pdf_url),
                })

                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()

                if len(data) > 1000 and data[:5] == b'%PDF-':
                    with open(filepath, "wb") as f:
                        f.write(data)
                    paper["local_pdf"] = str(filepath)
                    logger.info(f"    Success (selenium/cookies): {len(data)} bytes")
                    return str(filepath)
                else:
                    logger.warning(f"    Cookie-based download returned non-PDF ({len(data)} bytes)")
            except Exception as e2:
                logger.debug(f"    Cookie-based download failed: {e2}")

            logger.warning(f"    Selenium could not retrieve PDF")
            return None

        except Exception as e:
            logger.error(f"    Selenium download failed: {e}")
            return None

    def _validate_pdf(self, filepath: Path) -> bool:
        """Validate that a downloaded PDF contains actual paper content."""
        if not filepath.exists() or filepath.stat().st_size < 1000:
            return False

        with open(filepath, "rb") as f:
            header = f.read(5)
        if header != b'%PDF-':
            logger.warning(f"    Validation failed: not a PDF file")
            return False

        try:
            import fitz
            doc = fitz.open(str(filepath))
            if len(doc) == 0:
                doc.close()
                return False
            text = doc[0].get_text()[:500].lower()
            doc.close()

            error_indicators = [
                "page not found", "not found",
                "we couldn't find", "couldn't find the page",
                "page you were looking for",
                "this page is not available",
                "access denied", "forbidden",
            ]
            if any(ind in text for ind in error_indicators):
                logger.warning(f"    Validation failed: error page content detected")
                return False
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"    PDF validation (fitz) skipped: {e}")

        return True

    def download_one(self, paper: dict) -> Optional[str]:
        """Download a single paper's PDF."""
        if self.downloaded_today >= self.max_per_day:
            logger.warning(f"Daily download limit reached ({self.max_per_day})")
            return None

        filename = self._safe_filename(paper)
        filepath = self.pdf_dir / filename

        # Skip if already downloaded and valid
        if filepath.exists() and filepath.stat().st_size > 1000:
            if self._validate_pdf(filepath):
                logger.info(f"  Already exists: {filename}")
                paper["local_pdf"] = str(filepath)
                return str(filepath)
            else:
                logger.info(f"  Removing invalid existing file: {filename}")
                filepath.unlink()

        # Resolve URL
        pdf_url = self._resolve_pdf_url(paper)
        if not pdf_url:
            logger.warning(f"  No PDF URL for: {paper['title'][:60]}")
            return None

        logger.info(f"  Downloading: {filename}")
        logger.info(f"    URL: {pdf_url}")

        # Choose download method based on publisher
        if self._needs_selenium(paper):
            result = self._download_with_selenium(paper, pdf_url, filepath)
        else:
            result = self._download_with_urllib(paper, pdf_url, filepath)

        if result and not self._validate_pdf(filepath):
            logger.warning(f"    Post-download validation failed, removing: {filename}")
            filepath.unlink(missing_ok=True)
            paper.pop("local_pdf", None)
            result = None

        if result:
            self.downloaded_today += 1
            time.sleep(self.delay)

        return result

    def download_batch(self, papers: list) -> dict:
        """Download PDFs for a batch of papers."""
        stats = {
            "total": len(papers),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "selenium_used": 0,
            "limit_reached": False,
        }

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

            if self._needs_selenium(paper):
                stats["selenium_used"] += 1

            result = self.download_one(paper)
            if result:
                stats["downloaded"] += 1
            elif paper.get("local_pdf"):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        # Clean up Selenium driver
        self._close_driver()

        logger.info(f"Download stats: {stats}")
        return stats