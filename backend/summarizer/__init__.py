"""
AI Summarizer using Google Gemini API.
Generates Korean summaries, categorizes papers (multi-label), scores relevance,
produces weekly digests, and category trend columns.
"""

import json
import os
import logging
import time
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try google-generativeai SDK
try:
    import google.generativeai as genai
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    import urllib.request


def _call_gemini(prompt: str, system: str = "", config: dict = None) -> Optional[str]:
    """Call Gemini API and return the text response."""
    cfg = config or {}
    scfg = cfg.get("summarizer", {})
    model_name = scfg.get("model", "gemini-2.5-pro")
    temperature = scfg.get("temperature", 0.1)
    api_key = cfg.get("gemini_api_key") or os.environ.get("GOOGLE_API_KEY", "")

    if not api_key:
        logger.error("gemini_api_key not found in config.yaml and GOOGLE_API_KEY env var not set")
        return None

    if HAS_SDK:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system if system else None,
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=100000,
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini SDK error: {e}")
            return None
    else:
        # Raw HTTP fallback
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            body = json.dumps({
                "system_instruction": {"parts": [{"text": system}]} if system else None,
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 10000,
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini HTTP error: {e}")
            return None


def _extract_text_from_pdf(pdf_path: str, max_chars: int = 8000) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                break
        doc.close()
        return text[:max_chars]
    except ImportError:
        pass

    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                if len(text) > max_chars:
                    break
        return text[:max_chars]
    except ImportError:
        pass

    logger.warning("No PDF reader available (install PyMuPDF or pdfplumber)")
    return ""


def _get_paper_cats(paper: dict) -> List[str]:
    """Get categories list from paper (backward compatible)."""
    cats = paper.get("categories")
    if cats and isinstance(cats, list) and len(cats) > 0:
        return cats
    cat = paper.get("category", "")
    return [cat] if cat else []


def _load_history_papers(config: dict) -> List[dict]:
    """Load all papers from history directory for long-term trend analysis."""
    history_dir = Path(config.get("output", {}).get("history_dir", "./data/history"))
    all_papers = {}

    if not history_dir.exists():
        return []

    for hist_file in sorted(history_dir.glob("news_*.json")):
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for p in data.get("papers", []):
                    key = p.get("doi") or p.get("title", "")
                    if key and key not in all_papers:
                        all_papers[key] = p
        except Exception as e:
            logger.warning(f"Failed to load history file {hist_file}: {e}")

    papers = list(all_papers.values())
    logger.info(f"Loaded {len(papers)} unique papers from history")
    return papers


def summarize_paper(paper: dict, categories: List, config: dict = None) -> dict:
    """
    Generate a Korean summary, categories (multi-label), and relevance score for a paper.
    """
    context = paper.get("abstract") or ""
    if paper.get("local_pdf") and os.path.exists(paper["local_pdf"]):
        pdf_text = _extract_text_from_pdf(paper["local_pdf"])
        if pdf_text:
            context = pdf_text

    if not context:
        context = paper.get("title") or "No content available"

    cat_list = "\n".join(
        f"- {c['id']}: {c['label']} (keywords: {', '.join(c['keywords'][:5])})"
        for c in categories
    )

    system = """You are a research assistant for CCEL (Computational Catalysis and Emerging Materials Laboratory) at Seoul National University.
The lab's core research areas are: DFT/computational chemistry, heterogeneous catalysis, electrochemistry/fuel cells, batteries, and ML/MLIP.
You summarize academic papers concisely in Korean for lab members."""

    prompt = f"""Analyze this paper and respond in the exact JSON format below.

Paper title: {paper['title']}
Authors: {paper.get('authors_str') or 'Unknown'}
Source: {paper.get('source') or 'Unknown'}

Content:
{context[:6000]}

Available categories:
{cat_list}

Respond ONLY with valid JSON (no markdown, no explanation, no backticks):
{{
  "summary": "5-line Korean summary covering: (1) research objective, (2) methods used, (3) key results, (4) significance, (5) relevance to CCEL. Be specific about materials, techniques, and quantitative results.",
  "categories": ["primary_category_id", "secondary_category_id"],
  "relevance": 85,
  "relevance_reason": "Brief Korean explanation of why this is relevant (or not) to CCEL"
}}

IMPORTANT rules for categories:
- Assign 1 to 3 categories from the list above, ordered by relevance (primary first).
- The FIRST category should be the primary research topic (e.g., "battery" for a battery paper).
- Additional categories should reflect the methodology or secondary aspects.
  Examples:
  - A paper using DFT to study catalysis -> ["catalysis", "dft"]
  - A paper using ML to discover battery materials -> ["battery", "ml"]
  - A paper on fuel cell catalysts studied with DFT -> ["electrochemistry", "catalysis", "dft"]
  - A pure policy paper -> ["policy"]
- Only assign categories that are genuinely relevant. Do NOT pad with weakly related ones.

The relevance score (0-100) should reflect how relevant this paper is to CCEL's research:
- 95-100: Directly related to CCEL's active research topics
- 80-94: Highly relevant methodology or application area
- 60-79: Related field, useful background
- 40-59: Tangentially related
- 0-39: Not very relevant"""

    response = _call_gemini(prompt, system, config)
    if not response:
        return {
            "summary": (paper.get("abstract") or paper.get("title") or "")[:200],
            "categories": [paper.get("category", "dft")],
            "relevance": 50,
        }

    try:
        cleaned = response.strip()
        import re
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)

        cats = result.get("categories")
        if not cats:
            cat = result.get("category", "dft")
            cats = [cat] if isinstance(cat, str) else cat

        if isinstance(cats, str):
            cats = [cats]
        cats = [c for c in cats if isinstance(c, str) and c.strip()]
        if not cats:
            cats = ["dft"]

        return {
            "summary": result.get("summary", ""),
            "categories": cats,
            "category": cats[0],
            "relevance": min(100, max(0, int(result.get("relevance", 50)))),
            "relevance_reason": result.get("relevance_reason", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse Gemini response: {e}")
        logger.warning(f"Raw response: {response[:300]}")
        return {"summary": response[:300], "categories": ["dft"], "category": "dft", "relevance": 50}


def generate_weekly_digest(papers: List, config: dict = None):
    """
    Generate a structured weekly research digest.
    Returns a dict with sectioned content, or a plain string as fallback.
    """
    if not papers:
        return {"sections": [], "hot_issues": [], "generated_at": datetime.now().isoformat()}

    by_cat = {}
    for p in papers:
        cats = _get_paper_cats(p)
        for cat in cats:
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(p)

    papers_summary = ""
    for cat, cat_papers in by_cat.items():
        papers_summary += f"\n[{cat}] ({len(cat_papers)} papers)\n"
        for p in sorted(cat_papers, key=lambda x: -(x.get("relevance") or 0))[:8]:
            summary_short = (p.get("summary") or p.get("abstract") or "")[:150]
            papers_summary += f"  - {p['title'][:100]} (rel: {p.get('relevance', '?')})\n    {summary_short}\n"

    ccel_papers = [p for p in papers if p.get("ccel")]
    ccel_section = ""
    if ccel_papers:
        ccel_section = "\nCCEL group papers:\n"
        for p in ccel_papers:
            ccel_section += f"  - {p['title'][:100]}\n"

    system = """You are a research trend analyst writing for researchers at CCEL (Computational Catalysis and Emerging Materials Laboratory), Seoul National University.

Rules:
- Write in Korean.
- Use a plain, objective, concise tone. No flattery or hype ("매우 주목할 만한", "획기적인" 등은 금지).
- Evaluate novelty critically: what is genuinely new vs. incremental improvement.
- Focus on what problems the field is trying to solve and what approaches are gaining traction.
- Be specific: mention materials, methods, and quantitative results where available."""

    prompt = f"""Analyze the following papers collected this week and produce a structured research digest.

{papers_summary}
{ccel_section}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "hot_issues": [
    {{
      "topic": "짧은 이슈 제목 (Korean, 10자 내외)",
      "description": "이 이슈가 왜 중요한지, 현재 어떤 접근법이 시도되고 있는지 2-3문장으로 객관적으로 설명"
    }}
  ],
  "sections": [
    {{
      "title": "섹션 제목",
      "content": "해당 섹션 내용 (2-4문장, 구체적 논문/방법론 언급)"
    }}
  ]
}}

Required sections (in order):
1. "핵심 연구 동향" - 이번 주 가장 활발한 연구 방향 2-3개를 짧게 정리. 각 방향에서 어떤 문제를 해결하려 하는지 명시.
2. "주목할 논문" - 실제 novelty가 있는 논문 2-3편을 선별하고, 왜 새로운지 비판적으로 평가. 단순 성능 개선은 낮게, 새로운 방법론이나 관점은 높게 평가.
3. "CCEL 관련성" - CCEL의 연구(DFT, 촉매, 전기화학, 배터리, MLIP)와 직접 연결되는 시사점을 1-2문장으로 담백하게 정리.
4. "산업/정책 동향" - 관련 산업/정책 뉴스가 있으면 1-2문장. 없으면 이 섹션은 빈 문자열로 남겨도 됨.

hot_issues: 이번 주 논문들에서 반복적으로 등장하는 미해결 과제나 핫한 연구 질문 3-5개."""

    response = _call_gemini(prompt, system, config)
    if not response:
        return {"sections": [], "hot_issues": [], "generated_at": datetime.now().isoformat()}

    try:
        import re
        cleaned = response.strip()
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()
        result = json.loads(cleaned)
        result["generated_at"] = datetime.now().isoformat()
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse digest JSON: {e}")
        return {"sections": [{"title": "Weekly Digest", "content": response[:2000]}], "hot_issues": [], "generated_at": datetime.now().isoformat()}


def generate_group_digests(papers: List, config: dict = None) -> Dict:
    """
    Generate per-group research digests.
    Returns a dict keyed by group_id, each containing structured analysis.
    """
    groups = config.get("groups", {}) if config else {}
    if not groups:
        return {}

    by_group = {}
    for p in papers:
        gid = p.get("group")
        if gid:
            by_group.setdefault(gid, []).append(p)

    digests = {}

    system = """You are a research trend analyst. Write concise, objective analysis in Korean.
Rules:
- No hype or flattery. No "매우 주목할 만한", "획기적인" etc.
- Evaluate novelty critically: distinguish genuine advances from incremental work.
- Be specific about methods, materials, and results.
- Focus on what problems are being tackled and what approaches are gaining traction."""

    for gid, gcfg in groups.items():
        group_papers = by_group.get(gid, [])
        if not group_papers:
            continue

        pi_name = gcfg.get("pi", gid)
        group_name = gcfg.get("name", gid)

        paper_list = ""
        for p in sorted(group_papers, key=lambda x: -(x.get("relevance") or 0)):
            summary_short = (p.get("summary") or p.get("abstract") or "")[:200]
            paper_list += f"- [{p.get('date','')}] {p['title'][:120]}\n  {summary_short}\n"

        prompt = f"""Analyze the recent papers from the research group "{group_name}" (PI: {pi_name}).

Papers ({len(group_papers)} total):
{paper_list}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "focus_areas": ["이 그룹이 현재 집중하는 연구 주제 2-3개, 각 5-10자"],
  "summary": "이 그룹의 최근 연구 방향을 2-3문장으로 객관적으로 요약. 어떤 문제를 풀고 있고, 어떤 방법론을 사용하는지.",
  "notable_papers": [
    {{
      "title": "논문 제목 (원문 그대로)",
      "reason": "왜 주목할 만한지 1문장 (비판적 평가 포함)"
    }}
  ]
}}

notable_papers: novelty가 있는 논문만 최대 2편. 없으면 빈 배열."""

        logger.info(f"Generating group digest: {group_name} ({len(group_papers)} papers)")
        response = _call_gemini(prompt, system, config)

        if response:
            try:
                import re
                cleaned = response.strip()
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
                cleaned = cleaned.strip()
                result = json.loads(cleaned)
                digests[gid] = {
                    "group_name": group_name,
                    "pi": pi_name,
                    "paper_count": len(group_papers),
                    "focus_areas": result.get("focus_areas", []),
                    "summary": result.get("summary", ""),
                    "notable_papers": result.get("notable_papers", []),
                    "generated_at": datetime.now().isoformat(),
                }
                logger.info(f"  {group_name}: {len(group_papers)} papers, focus={result.get('focus_areas', [])}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse group digest for {gid}: {e}")
                digests[gid] = {
                    "group_name": group_name,
                    "pi": pi_name,
                    "paper_count": len(group_papers),
                    "focus_areas": [],
                    "summary": response[:500],
                    "notable_papers": [],
                    "generated_at": datetime.now().isoformat(),
                }
        else:
            digests[gid] = {
                "group_name": group_name,
                "pi": pi_name,
                "paper_count": len(group_papers),
                "focus_areas": [],
                "summary": "Digest generation failed.",
                "notable_papers": [],
                "generated_at": datetime.now().isoformat(),
            }

        time.sleep(4)

    return digests


def generate_category_trends(current_papers: List, config: dict = None) -> Dict:
    """
    Generate per-category trend columns (1-month and long-term).
    Returns a dict with category IDs as keys, each containing
    monthly_trend and yearly_trend text.
    """
    # Load history for long-term trends
    history_papers = _load_history_papers(config) if config else []

    # Merge current + history (dedup by DOI/title)
    all_papers_map = {}
    for p in history_papers:
        key = p.get("doi") or p.get("title", "")
        if key:
            all_papers_map[key] = p
    for p in current_papers:
        key = p.get("doi") or p.get("title", "")
        if key:
            all_papers_map[key] = p
    all_papers = list(all_papers_map.values())

    # Split by time window
    now = datetime.now()
    one_month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    one_year_ago = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    monthly_papers = [p for p in all_papers if (p.get("date") or "") >= one_month_ago]
    yearly_papers = [p for p in all_papers if (p.get("date") or "") >= one_year_ago]

    # Group by category
    cat_ids = ["dft", "catalysis", "electrochemistry", "battery", "ml", "policy"]
    cat_labels = {
        "dft": "DFT / Computational Chemistry",
        "catalysis": "Catalysis",
        "electrochemistry": "Electrochemistry / Fuel Cells",
        "battery": "Battery",
        "ml": "ML / MLIP",
        "policy": "Industry / Policy",
    }

    trends = {}

    system = """You are a research trend analyst for CCEL (Computational Catalysis and Emerging Materials Laboratory) at Seoul National University.
You write concise, insightful trend analysis in Korean for lab members.
Focus on specific research themes, methodologies, and notable findings.
Be concrete with paper titles and findings, not vague generalizations."""

    for cat_id in cat_ids:
        label = cat_labels[cat_id]
        logger.info(f"Generating trend for: {label}")

        # Filter papers for this category
        monthly_cat = [p for p in monthly_papers if cat_id in _get_paper_cats(p)]
        yearly_cat = [p for p in yearly_papers if cat_id in _get_paper_cats(p)]

        # Build paper lists for prompt
        def format_papers(papers, max_count=100):
            sorted_p = sorted(papers, key=lambda p: -(p.get("relevance") or 0))[:max_count]
            lines = []
            for p in sorted_p:
                rel = p.get("relevance", "?")
                summary_short = (p.get("summary") or p.get("abstract") or "")[:200]
                lines.append(f"- [{p.get('date','')}] {p['title'][:120]} (rel:{rel})\n  {summary_short}")
            return "\n".join(lines) if lines else "(No papers)"

        monthly_list = format_papers(monthly_cat, 100)
        yearly_list = format_papers(yearly_cat, 100)

        prompt = f"""Analyze the research trends for the "{label}" category and write two trend columns in Korean.

=== Recent 1 month ({len(monthly_cat)} papers) ===
{monthly_list}

=== Past 1 year ({len(yearly_cat)} papers) ===
{yearly_list}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "monthly_trend": "2-3 paragraphs analyzing the past month's trends in this category. Mention specific research themes, notable papers, emerging methods, and what CCEL should pay attention to. Be concrete and specific.",
  "yearly_trend": "2-3 paragraphs analyzing the broader yearly trends. Identify major shifts, growing/declining topics, breakthrough papers, and strategic implications for CCEL's research direction.",
  "hot_topics": ["topic1", "topic2", "topic3"],
  "paper_count_monthly": {len(monthly_cat)},
  "paper_count_yearly": {len(yearly_cat)}
}}

Guidelines:
- Write in Korean, professional but accessible tone
- Be specific: mention actual paper topics, methods, materials
- For CCEL relevance: connect trends to DFT, catalysis, electrochemistry, batteries, or ML research
- If few papers exist, note this and focus on available data
- Do NOT use markdown formatting in the text fields
- hot_topics should be 3-5 short Korean keywords/phrases for this category"""

        response = _call_gemini(prompt, system, config)

        if response:
            try:
                import re
                cleaned = response.strip()
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
                cleaned = cleaned.strip()

                result = json.loads(cleaned)
                trends[cat_id] = {
                    "label": label,
                    "monthly_trend": result.get("monthly_trend", ""),
                    "yearly_trend": result.get("yearly_trend", ""),
                    "hot_topics": result.get("hot_topics", []),
                    "paper_count_monthly": len(monthly_cat),
                    "paper_count_yearly": len(yearly_cat),
                    "generated_at": datetime.now().isoformat(),
                }
                logger.info(f"  {label}: monthly={len(monthly_cat)}, yearly={len(yearly_cat)}, topics={result.get('hot_topics', [])}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse trend for {cat_id}: {e}")
                trends[cat_id] = {
                    "label": label,
                    "monthly_trend": response[:500],
                    "yearly_trend": "",
                    "hot_topics": [],
                    "paper_count_monthly": len(monthly_cat),
                    "paper_count_yearly": len(yearly_cat),
                    "generated_at": datetime.now().isoformat(),
                }
        else:
            trends[cat_id] = {
                "label": label,
                "monthly_trend": "Trend generation failed.",
                "yearly_trend": "",
                "hot_topics": [],
                "paper_count_monthly": len(monthly_cat),
                "paper_count_yearly": len(yearly_cat),
                "generated_at": datetime.now().isoformat(),
            }

        # Rate limiting
        time.sleep(4)

    return trends


def _load_existing_summaries(config: dict) -> dict:
    """Load summaries from the most recent history file, keyed by DOI or title."""
    history_dir = Path(config.get("output", {}).get("history_dir", "./data/history"))
    if not history_dir.exists():
        return {}

    hist_files = sorted(history_dir.glob("news_*.json"), reverse=True)
    if not hist_files:
        return {}

    cache = {}
    for hist_file in hist_files[:3]:
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("papers", []):
                if not p.get("summary"):
                    continue
                key = p.get("doi") or p.get("title", "")
                if key and key not in cache:
                    cache[key] = {
                        "summary": p["summary"],
                        "categories": p.get("categories", []),
                        "category": p.get("category", ""),
                        "relevance": p.get("relevance", 50),
                        "relevance_reason": p.get("relevance_reason", ""),
                    }
        except Exception as e:
            logger.warning(f"Failed to load history for cache: {e}")

    logger.info(f"Loaded {len(cache)} existing summaries from history")
    return cache


def summarize_batch(papers: List, config: dict) -> List:
    """Summarize a batch of papers, reusing existing summaries from history."""
    categories = config.get("categories", [])
    cache = _load_existing_summaries(config)

    skipped = 0
    summarized = 0

    for i, paper in enumerate(papers):
        key = paper.get("doi") or paper.get("title", "")

        if key and key in cache:
            cached = cache[key]
            paper["summary"] = cached["summary"]
            paper["categories"] = cached["categories"]
            paper["category"] = cached.get("category", cached["categories"][0] if cached["categories"] else "dft")
            paper["relevance"] = cached["relevance"]
            if cached.get("relevance_reason"):
                paper["relevance_reason"] = cached["relevance_reason"]
            paper.setdefault("ccel", paper.get("group") == "ccel")
            skipped += 1
            logger.info(f"Summarizing [{i+1}/{len(papers)}]: (cached) {paper['title'][:60]}...")
            continue

        logger.info(f"Summarizing [{i+1}/{len(papers)}]: {paper['title'][:60]}...")
        result = summarize_paper(paper, categories, config)
        paper["summary"] = result["summary"]
        paper["categories"] = result["categories"]
        paper["category"] = result.get("category", result["categories"][0] if result["categories"] else "dft")
        paper["relevance"] = result["relevance"]
        paper.setdefault("ccel", paper.get("group") == "ccel")
        summarized += 1

        time.sleep(4)

    logger.info(f"Summary complete: {summarized} new, {skipped} cached (reused)")

    papers.sort(key=lambda p: (-p.get("relevance", 0), p.get("date") or ""), reverse=False)
    papers.sort(key=lambda p: p.get("date") or "", reverse=True)

    return papers