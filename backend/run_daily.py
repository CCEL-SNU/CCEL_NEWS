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
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from collectors import collect_all
from downloader import PaperDownloader
from summarizer import (
    summarize_batch,
    generate_weekly_digest,
    generate_category_trends,
    generate_group_digests,
    _dedupe_papers_by_doi_or_title,
)

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


def save_output(
    papers: list,
    digest,
    category_trends: dict,
    config: dict,
    group_digests: dict = None,
    weekly_digest_in_group=None,
    weekly_digest_out_group=None,
):
    """Save papers, digest(s), category trends, and group digests to news.json."""
    output_path = Path(config["output"]["json_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_data(str(output_path))

    history_dir = Path(config["output"]["history_dir"])
    history_dir.mkdir(parents=True, exist_ok=True)

    groups_meta = [
        {"id": gid, "name": gcfg.get("name", gid), "pi": gcfg.get("pi", "")}
        for gid, gcfg in config.get("groups", {}).items()
    ]

    wd = digest if digest is not None else existing.get("weekly_digest", "")
    wd_in = weekly_digest_in_group if weekly_digest_in_group is not None else existing.get("weekly_digest_in_group", {})
    wd_out = weekly_digest_out_group if weekly_digest_out_group is not None else existing.get("weekly_digest_out_group", {})

    output = {
        "generated_at": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_papers": len(papers),
        "ccel_papers": sum(1 for p in papers if p.get("ccel")),
        "weekly_digest": wd,
        "weekly_digest_in_group": wd_in,
        "weekly_digest_out_group": wd_out,
        "category_trends": category_trends or existing.get("category_trends", {}),
        "group_digests": group_digests or existing.get("group_digests", {}),
        "groups": groups_meta,
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

    copy_news_json_to_frontend(config)


def copy_news_json_to_frontend(config: dict) -> None:
    """Copy backend news.json to frontend/public/data so local UI stays in sync (e.g. after --digest)."""
    output_path = Path(config["output"]["json_path"])
    if not output_path.is_file():
        return
    frontend_data = Path(__file__).parent.parent / "frontend" / "public" / "data"
    frontend_data.mkdir(parents=True, exist_ok=True)
    dest = frontend_data / "news.json"
    shutil.copy2(output_path, dest)
    logger.info(f"Copied news.json to {dest}")


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
    """Step 1-2: Collect metadata, download PDFs, and filter out papers without PDFs."""
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

    # Filter: only keep papers with successfully downloaded PDFs
    before = len(papers)
    papers = [p for p in papers if p.get("local_pdf")]
    after = len(papers)
    removed = before - after
    logger.info("=" * 60)
    logger.info(f"PDF filter: {before} -> {after} papers ({removed} removed without PDF)")
    logger.info("=" * 60)

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
    Generate weekly digests (all / in-group / out-group) and category trends.
    Returns (digest_dict, digest_in_group, digest_out_group, category_trends_dict).
    Runs on the configured digest day (default: Sunday) or when --digest is used.
    """
    today = datetime.now().strftime("%A")
    digest_day = config.get("schedule", {}).get("weekly_digest_day", "Sunday")

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
        week_papers = papers

    week_papers = _dedupe_papers_by_doi_or_title(week_papers)

    digest = {}
    digest_in_group = {}
    digest_out_group = {}
    category_trends = {}

    if today == digest_day or not papers:
        logger.info("=" * 60)
        logger.info("Generating weekly digest (all + group split)...")
        logger.info("=" * 60)
        digest = generate_weekly_digest(week_papers, config, subset=None)
        time.sleep(2)
        digest_in_group = generate_weekly_digest(week_papers, config, subset="in_group")
        time.sleep(2)
        digest_out_group = generate_weekly_digest(week_papers, config, subset="out_group")

        logger.info("=" * 60)
        logger.info("Generating category trend columns...")
        logger.info("=" * 60)
        category_trends = generate_category_trends(week_papers, config)
    else:
        output_path = Path(config["output"]["json_path"])
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                digest = data.get("weekly_digest", {})
                digest_in_group = data.get("weekly_digest_in_group", {})
                digest_out_group = data.get("weekly_digest_out_group", {})
                category_trends = data.get("category_trends", {})

    return digest, digest_in_group, digest_out_group, category_trends


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

    digest = None
    digest_in_group = None
    digest_out_group = None
    category_trends = {}
    group_digests = {}

    if args.digest:
        logger.info("Forced digest + category trends + group digests generation")
        week_papers = papers
        history_dir = Path(config["output"]["history_dir"])
        for i in range(7):
            d = datetime.now() - timedelta(days=i)
            hist_file = history_dir / f"news_{d:%Y%m%d}.json"
            if hist_file.exists():
                with open(hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    week_papers = week_papers + data.get("papers", [])

        week_papers = _dedupe_papers_by_doi_or_title(week_papers)

        digest = generate_weekly_digest(week_papers, config, subset=None)
        time.sleep(2)
        digest_in_group = generate_weekly_digest(week_papers, config, subset="in_group")
        time.sleep(2)
        digest_out_group = generate_weekly_digest(week_papers, config, subset="out_group")
        category_trends = generate_category_trends(week_papers, config)
        group_digests = generate_group_digests(week_papers, config)
    elif not (args.collect or args.summarize):
        digest, digest_in_group, digest_out_group, category_trends = run_digest(papers, config)
        group_digests = generate_group_digests(papers, config) if digest else {}

    save_output(
        papers,
        digest,
        category_trends,
        config,
        group_digests,
        weekly_digest_in_group=digest_in_group,
        weekly_digest_out_group=digest_out_group,
    )

    # Deploy
    if not (args.collect or args.summarize or args.digest):
        git_push(config)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Papers: {len(papers)}")
    logger.info(f"  CCEL: {sum(1 for p in papers if p.get('ccel'))}")
    logger.info(f"  Digest: {'Yes' if digest else 'No'} (in-group / out-group splits saved when generated)")
    logger.info(f"  Category trends: {len(category_trends)} categories")
    logger.info(f"  Group digests: {len(group_digests)} groups")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()