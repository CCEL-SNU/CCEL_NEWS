"""
CCEL Daily News - Summarizer Test (Gemini version)
====================================================
Run from backend/ directory:
    $env:GOOGLE_API_KEY = "your-gemini-api-key"
    python run_daily_test.py
"""

import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from summarizer import summarize_paper, generate_weekly_digest

TEST_COUNT = 5
MIN_ABSTRACT_LEN = 150


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Run this first:")
        print('  $env:GOOGLE_API_KEY = "your-gemini-api-key"')
        sys.exit(1)
    print(f"API key: ...{api_key[-8:]}")

    # Load papers
    news_path = Path("data/news.json")
    if not news_path.exists():
        print(f"ERROR: {news_path} not found. Run --collect first.")
        sys.exit(1)

    with open(news_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    print(f"Total papers in news.json: {len(papers)}")

    # Filter papers with good abstracts
    good = [p for p in papers if len(p.get("abstract") or "") >= MIN_ABSTRACT_LEN]
    print(f"Papers with abstract >= {MIN_ABSTRACT_LEN} chars: {len(good)}")

    if not good:
        print("Falling back to papers with any abstract...")
        good = [p for p in papers if p.get("abstract")]
        if not good:
            print("No papers with abstracts. Exiting.")
            sys.exit(1)

    # Select diverse test papers
    selected = []
    sources_seen = set()
    for p in good:
        src = p.get("source", "")
        if src not in sources_seen and len(selected) < TEST_COUNT:
            selected.append(p)
            sources_seen.add(src)
    for p in good:
        if p not in selected and len(selected) < TEST_COUNT:
            selected.append(p)

    print(f"\nTesting with {len(selected)} papers:")
    print("-" * 60)

    categories = config.get("categories", [])
    results = []

    for i, paper in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] {paper['title'][:70]}...")
        print(f"  Source: {paper.get('source', '?')}")
        print(f"  Abstract: {len(paper.get('abstract') or '')} chars")
        print(f"  Calling Gemini API...")

        result = summarize_paper(paper, categories, config)

        print(f"  Category: {result.get('category', '?')}")
        print(f"  Relevance: {result.get('relevance', '?')}")
        print(f"  Summary:")
        summary = result.get("summary", "")
        for line in summary.split("\n"):
            if line.strip():
                print(f"    {line.strip()}")

        paper.update(result)
        results.append(paper)

    # Save test results
    out_path = Path("data/test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Test results saved to {out_path}")

    # Test weekly digest
    print(f"\n{'=' * 60}")
    print("Testing weekly digest generation...")
    digest = generate_weekly_digest(results, config)
    print(f"\nWeekly Digest:")
    print("-" * 60)
    print(digest)

    print(f"\n{'=' * 60}")
    print("Test complete!")


if __name__ == "__main__":
    main()