"""
AI Summarizer using Google Gemini API.
Generates Korean summaries, categorizes papers, scores relevance,
and produces weekly digests.
"""

import json
import os
import logging
import time
from typing import List, Optional
from pathlib import Path

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
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    if not api_key:
        logger.error("GOOGLE_API_KEY not set")
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


def summarize_paper(paper: dict, categories: List, config: dict = None) -> dict:
    """
    Generate a Korean summary, category, and relevance score for a paper.
    """
    # Build context from abstract and/or PDF text
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
  "summary": "3-line Korean summary of the paper's key findings and significance for CCEL research. Be specific about methods and results.",
  "category": "category_id from the list above",
  "relevance": 85,
  "relevance_reason": "Brief Korean explanation of why this is relevant (or not) to CCEL"
}}

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
            "category": "dft",
            "relevance": 50,
        }

    try:
        # Clean response (remove potential markdown fences)
        cleaned = response.strip()
        # Remove all markdown fences (```json ... ``` or ``` ... ```)
        import re
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        return {
            "summary": result.get("summary", ""),
            "category": result.get("category", "dft"),
            "relevance": min(100, max(0, int(result.get("relevance", 50)))),
            "relevance_reason": result.get("relevance_reason", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse Gemini response: {e}")
        logger.warning(f"Raw response: {response[:300]}")
        return {"summary": response[:300], "category": "dft", "relevance": 50}


def generate_weekly_digest(papers: List, config: dict = None) -> str:
    """Generate a weekly research digest summarizing trends and highlights."""
    if not papers:
        return "This week no relevant papers were found."

    by_cat = {}
    for p in papers:
        cat = p.get("category", "other")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(p)

    papers_summary = ""
    for cat, cat_papers in by_cat.items():
        papers_summary += f"\n[{cat}] ({len(cat_papers)} papers)\n"
        for p in cat_papers[:5]:
            papers_summary += f"  - {p['title'][:80]} (rel: {p.get('relevance', '?')})\n"

    ccel_papers = [p for p in papers if p.get("ccel")]
    ccel_section = ""
    if ccel_papers:
        ccel_section = "\nCCEL group papers this week:\n"
        for p in ccel_papers:
            ccel_section += f"  - {p['title'][:80]}\n"

    system = """You are a research analyst for CCEL at Seoul National University.
Write a concise weekly research digest in Korean that highlights trends,
important findings, and their relevance to CCEL's research."""

    prompt = f"""Write a weekly research digest based on these papers collected this week.

{papers_summary}
{ccel_section}

Write 3-4 paragraphs in Korean:
1. Overall trends this week (which topics are hot, any emerging patterns)
2. Highlight 2-3 most important papers and why they matter for CCEL
3. Any relevant policy/industry news
4. Brief outlook or suggested actions for the lab

Keep it concise and actionable. Write in a professional but accessible tone.
Do NOT use markdown formatting. Just plain text paragraphs."""

    digest = _call_gemini(prompt, system, config)
    return digest or "Weekly digest generation failed."


def summarize_batch(papers: List, config: dict) -> List:
    """Summarize a batch of papers."""
    categories = config.get("categories", [])

    for i, paper in enumerate(papers):
        logger.info(f"Summarizing [{i+1}/{len(papers)}]: {paper['title'][:60]}...")

        result = summarize_paper(paper, categories, config)
        paper["summary"] = result["summary"]
        paper["category"] = result["category"]
        paper["relevance"] = result["relevance"]
        paper.setdefault("ccel", paper.get("group") == "ccel")

        # Rate limiting for Gemini API (free tier: 15 RPM for 2.5 Pro)
        time.sleep(4)

    papers.sort(key=lambda p: (-p.get("relevance", 0), p.get("date") or ""), reverse=False)
    papers.sort(key=lambda p: p.get("date") or "", reverse=True)

    return papers