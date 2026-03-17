"""
CCEL Daily News - Main Pipeline Runner
=======================================
Orchestrates the daily collection, download, summarization, and deployment.

Usage:
    python run_daily.py              # Full daily pipeline
    python run_daily.py --collect    # Only collect metadata
    python run_daily.py --summarize  # Only summarize (skip collection)
    python run_daily.py --digest     # Generate weekly digest + category trends
    python run_daily.py --deploy     # Only push to GitHub
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from collectors import collect_all
from downloader import PaperDownloader
from summarizer import summarize_batch, generate_weekly_digest, generate_category_trends

# ---- Logging setup ----
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ccel-news")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_existing_data(path: str) -> dict:
    """Load existing news.json data if it exists. Returns full data dict."""
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_output(papers: list, digest: str, category_trends: dict, config: dict):
    """Save papers, digest, and category trends to news.json."""
    output_path = Path(config["output"]["json_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data to preserve fields not being updated
    existing = load_existing_data(str(output_path))

    # Also save to history
    history_dir = Path(config["output"]["history_dir"])
    history_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_papers": len(papers),
        "ccel_papers": sum(1 for p in papers if p.get("ccel")),
        "weekly_digest": digest or existing.get("weekly_digest", ""),
        "category_trends": category_trends or existing.get("category_trends", {}),
        "papers": papers,
    }

    # Clean papers for JSON serialization
    for p in output["papers"]:
        p.pop("local_pdf", None)
        for key in list(p.keys()):
            if isinstance(p[key], set):
                p[key] = list(p[key])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(papers)} papers to {output_path}")

    # Save daily snapshot to history
    hist_file = history_dir / f"news_{datetime.now():%Y%m%d}.json"
    shutil.copy2(output_path, hist_file)
    logger.info(f"History saved to {hist_file}")


def git_push(config: dict):
    """Push news.json to the GitHub repository."""
    output_path = config["output"]["json_path"]

    try:
        # Copy news.json to frontend public directory
        frontend_data = Path(__file__).parent.parent / "frontend" / "public" / "data"
        frontend_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, frontend_data / "news.json")
        logger.info(f"Copied news.json to frontend/public/data/")

        # Git operations
        repo_root = Path(__file__).parent.parent
        cmds = [
            ["git", "-C", str(repo_root), "add", "-A"],
            ["git", "-C", str(repo_root), "commit", "-m",
             f"Daily update: {datetime.now():%Y-%m-%d %H:%M}"],
            ["git", "-C", str(repo_root), "push", "origin", "main"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.warning(f"Git command failed: {' '.join(cmd)}")
                logger.warning(f"  stderr: {result.stderr}")
            else:
                logger.info(f"Git: {' '.join(cmd[-2:])}")

        logger.info("Successfully pushed to GitHub")
    except Exception as e:
        logger.error(f"Git push failed: {e}")


def run_collect(config: dict) -> list:
    """Step 1-2: Collect metadata and download PDFs."""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting paper metadata...")
    logger.info("=" * 60)

    papers = collect_all(config)
    logger.info(f"Collected {len(papers)} papers")

    logger.info("=" * 60)
    logger.info("STEP 2: Downloading PDFs...")
    logger.info("=" * 60)

    downloader = PaperDownloader(config)
    stats = downloader.download_batch(papers)
    logger.info(f"Download complete: {stats}")

    return papers


def run_summarize(papers: list, config: dict) -> list:
    """Step 3: AI summarization."""
    logger.info("=" * 60)
    logger.info("STEP 3: AI Summarization (Gemini API)...")
    logger.info("=" * 60)

    papers = summarize_batch(papers, config)
    logger.info(f"Summarized {len(papers)} papers")
    return papers


def run_digest(papers: list, config: dict) -> tuple:
    """
    Generate weekly digest and category trends.
    Returns (digest_text, category_trends_dict).
    Runs on the configured digest day (default: Sunday) or when --digest is used.
    """
    today = datetime.now().strftime("%A")
    digest_day = config.get("schedule", {}).get("weekly_digest_day", "Sunday")

    # Load this week's papers from history
    history_dir = Path(config["output"]["history_dir"])
    week_papers = []
    for i in range(7):
        d = datetime.now() - timedelta(days=i)
        hist_file = history_dir / f"news_{d:%Y%m%d}.json"
        if hist_file.exists():
            with open(hist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                week_papers.extend(data.get("papers", []))

    if not week_papers:
        week_papers = papers  # Fallback to current papers

    digest = ""
    category_trends = {}

    if today == digest_day or not papers:
        logger.info("=" * 60)
        logger.info("Generating weekly digest...")
        logger.info("=" * 60)
        digest = generate_weekly_digest(week_papers, config)

        logger.info("=" * 60)
        logger.info("Generating category trend columns...")
        logger.info("=" * 60)
        category_trends = generate_category_trends(week_papers, config)
    else:
        # Load previous digest and trends if not digest day
        output_path = Path(config["output"]["json_path"])
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                digest = data.get("weekly_digest", "")
                category_trends = data.get("category_trends", {})

    return digest, category_trends


def main():
    parser = argparse.ArgumentParser(description="CCEL Daily News Pipeline")
    parser.add_argument("--collect", action="store_true", help="Only collect papers")
    parser.add_argument("--summarize", action="store_true", help="Only summarize")
    parser.add_argument("--digest", action="store_true", help="Generate weekly digest + category trends")
    parser.add_argument("--deploy", action="store_true", help="Only push to GitHub")
    args = parser.parse_args()

    config = load_config()
    logger.info("CCEL Daily News Pipeline started")
    logger.info(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")

    if args.deploy:
        git_push(config)
        return

    # Full pipeline or individual steps
    if args.collect or not (args.summarize or args.digest):
        papers = run_collect(config)
    else:
        # Load from existing data
        existing = load_existing_data(config["output"]["json_path"])
        papers = existing.get("papers", [])
        if not papers:
            logger.warning("No existing data found, running collection first")
            papers = run_collect(config)

    # Summarize only when explicitly requested or in full pipeline
    # --digest and --collect should NOT trigger summarization
    if args.summarize or not (args.collect or args.digest or args.deploy):
        papers = run_summarize(papers, config)

    digest = ""
    category_trends = {}

    if args.digest:
        # Force generation regardless of day
        logger.info("Forced digest + category trends generation")
        week_papers = papers
        history_dir = Path(config["output"]["history_dir"])
        for i in range(7):
            d = datetime.now() - timedelta(days=i)
            hist_file = history_dir / f"news_{d:%Y%m%d}.json"
            if hist_file.exists():
                with open(hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    week_papers = week_papers + data.get("papers", [])

        digest = generate_weekly_digest(week_papers, config)
        category_trends = generate_category_trends(week_papers, config)
    elif not (args.collect or args.summarize):
        digest, category_trends = run_digest(papers, config)

    # Save output
    save_output(papers, digest, category_trends, config)

    # Deploy
    if not (args.collect or args.summarize or args.digest):
        git_push(config)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Papers: {len(papers)}")
    logger.info(f"  CCEL: {sum(1 for p in papers if p.get('ccel'))}")
    logger.info(f"  Digest: {'Yes' if digest else 'No'}")
    logger.info(f"  Category trends: {len(category_trends)} categories")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()