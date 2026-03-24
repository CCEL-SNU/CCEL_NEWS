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


def _call_gemini(
    prompt: str,
    system: str = "",
    config: dict = None,
    *,
    log_finish_reason: bool = False,
) -> Optional[str]:
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
            if log_finish_reason and getattr(response, "candidates", None):
                c0 = response.candidates[0]
                fr = getattr(c0, "finish_reason", None)
                # 1=STOP, 2=MAX_TOKENS, ... (names vary by SDK version)
                logger.warning(
                    "Gemini response finish_reason=%r (if MAX_TOKENS, JSON may be truncated)",
                    fr,
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


def _paper_date_ymd(paper: dict) -> str:
    """Return YYYY-MM-DD from paper['date'] for comparison, or '' if missing/invalid."""
    d = paper.get("date") or ""
    if not d or not isinstance(d, str):
        return ""
    d = d.strip()
    if len(d) >= 10 and d[4] == "-" and d[7] == "-":
        return d[:10]
    return ""


def _dedupe_papers_by_doi_or_title(papers: List) -> List:
    """Stable dedupe across merged history snapshots."""
    seen = set()
    out = []
    for p in papers:
        key = (p.get("doi") or p.get("arxiv_id") or p.get("title") or "").strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _filter_papers_by_rolling_days(papers: List, days: int) -> List:
    """Keep papers whose `date` (YYYY-MM-DD) is >= (today - days), inclusive."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    kept = []
    for p in papers:
        dk = _paper_date_ymd(p)
        if dk and dk >= cutoff:
            kept.append(p)
    return kept


def _normalize_weekly_digest_subset(subset: Optional[str]) -> str:
    """Return 'all' | 'in_group' | 'out_group'."""
    if subset in (None, "", "all"):
        return "all"
    if subset == "in_group":
        return "in_group"
    if subset == "out_group":
        return "out_group"
    return "all"


def _filter_papers_by_weekly_subset(papers: List, subset: str) -> List:
    """After rolling window: keep all, only tracked-group papers, or only non-group papers."""
    if subset == "all":
        return papers
    if subset == "in_group":
        return [p for p in papers if p.get("group")]
    if subset == "out_group":
        return [p for p in papers if not p.get("group")]
    return papers


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
  "relevance_reason": "2-3문장 한국어. (1) 이 분야/학계가 직면한 미해결 문제·한계·핵심 질문이 무엇인지 (2) 본 논문이 그에 대해 어떤 기여·시도를 하는지 (3) CCEL 관점에서 왜 의미 있는지(또는 관련이 적은 이유). 단순 칭찬이나 한 줄 가치 판단 금지."
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


def generate_weekly_digest(papers: List, config: dict = None, subset: Optional[str] = None):
    """
    Generate a structured weekly research digest.
    Only papers with `date` >= (today - weekly_digest_days) are included (rolling window).
    subset: None/'all' (전체), 'in_group' (group 필드 있는 논문만), 'out_group' (group 없음만).
    Returns a dict with sectioned content, or a plain string as fallback.
    """
    cfg = config or {}
    digest_subset = _normalize_weekly_digest_subset(subset)
    window_days = int(cfg.get("summarizer", {}).get("weekly_digest_days", 7))
    raw_n = len(papers or [])
    papers = _dedupe_papers_by_doi_or_title(papers or [])
    papers = _filter_papers_by_rolling_days(papers, window_days)
    papers = _filter_papers_by_weekly_subset(papers, digest_subset)
    logger.info(
        "Weekly digest [%s]: %s papers in last %s days (by date), %s rows before dedupe/window/subset",
        digest_subset,
        len(papers),
        window_days,
        raw_n,
    )

    if not papers:
        return {
            "sections": [],
            "hot_issues": [],
            "generated_at": datetime.now().isoformat(),
            "weekly_digest_window_days": window_days,
            "weekly_digest_paper_count": 0,
            "digest_subset": digest_subset,
        }

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
            dshow = _paper_date_ymd(p) or "?"
            papers_summary += f"  - {p['title'][:100]} (date: {dshow}, rel: {p.get('relevance', '?')})\n    {summary_short}\n"

    ccel_papers = [p for p in papers if p.get("ccel")]
    ccel_section = ""
    if ccel_papers:
        ccel_section = "\nCCEL group papers:\n"
        for p in ccel_papers:
            ccel_section += f"  - {p['title'][:100]}\n"

    scope_line = {
        "all": "범위: 위 목록은 날짜 윈도 내 전체 논문입니다.",
        "in_group": "범위: 추적 연구 그룹 소속 논문만 포함합니다(`group` 필드가 있는 항목). 그룹 외 논문은 목록에 없습니다.",
        "out_group": "범위: 연구 그룹 미지정 논문만 포함합니다(`group` 필드가 없는 항목). 그룹 소속 논문은 목록에 없습니다.",
    }.get(digest_subset, "범위: 위 목록은 날짜 윈도 내 전체 논문입니다.")

    system = """You are a research trend analyst writing for researchers at CCEL (Computational Catalysis and Emerging Materials Laboratory), Seoul National University.

Rules:
- Write in Korean.
- Use a plain, objective, concise tone. No flattery or hype ("매우 주목할 만한", "획기적인" 등은 금지).
- Evaluate novelty critically: what is genuinely new vs. incremental improvement.
- Focus on what problems the field is trying to solve and what approaches are gaining traction.
- Be specific: mention materials, methods, and quantitative results where available.
- If the paper list is small, generalize only cautiously; do not invent trends not supported by the list."""

    prompt = f"""아래 목록은 각 논문의 `date` 필드(YYYY-MM-DD, 발행·게재일)가 오늘 기준 지난 {window_days}일 이내인 항목만 포함합니다. {scope_line} 이 목록만 근거로 작성하세요. 목록에 없는 논문은 언급하지 마세요.

Analyze the following papers from the last {window_days} days (by date) and produce a structured research digest.

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
2. "주목할 논문" - 실제 novelty가 있는 논문 2-3편을 선별하되, 단순히 "가치 있다"고 말하지 말 것. 각 논문마다 **학계·분야가 직면한 병목·미해결 질문**이 무엇이었고, **그 논문이 그 문제를 어떻게 다루거나 완화하는지**(방법·관점·증거)를 비판적으로 서술. 단순 수치 개선만 강조하지 말 것.
3. "CCEL 관련성" - CCEL의 연구(DFT, 촉매, 전기화학, 배터리, MLIP)와 직접 연결되는 시사점을 1-2문장으로 담백하게 정리.
4. "산업/정책 동향" - 관련 산업/정책 뉴스가 있으면 1-2문장. 없으면 이 섹션은 빈 문자열로 남겨도 됨.

hot_issues: 이번 주 논문들에서 반복적으로 등장하는 미해결 과제나 핫한 연구 질문 3-5개."""

    response = _call_gemini(prompt, system, config)
    if not response:
        return {
            "sections": [],
            "hot_issues": [],
            "generated_at": datetime.now().isoformat(),
            "weekly_digest_window_days": window_days,
            "weekly_digest_paper_count": len(papers),
            "digest_subset": digest_subset,
        }

    result = _parse_gemini_json(response)
    if result:
        result["generated_at"] = datetime.now().isoformat()
        result["weekly_digest_window_days"] = window_days
        result["weekly_digest_paper_count"] = len(papers)
        result["digest_subset"] = digest_subset
        return result

    logger.warning("Weekly digest: failed to parse JSON from Gemini")
    return {
        "sections": [
            {
                "title": "Weekly Digest",
                "content": "주간 digest JSON 파싱에 실패했습니다. 다음 실행 시 재시도됩니다.",
            }
        ],
        "hot_issues": [],
        "generated_at": datetime.now().isoformat(),
        "weekly_digest_window_days": window_days,
        "weekly_digest_paper_count": len(papers),
        "digest_subset": digest_subset,
    }


_DIGEST_SYSTEM = """You are a research trend analyst writing for researchers at CCEL (Computational Catalysis and Emerging Materials Laboratory), Seoul National University.

Rules:
- Write in Korean.
- Use a plain, objective, concise tone. No flattery or hype ("매우 주목할 만한", "획기적인", "놀라운", "아주 가치가 있다" 등은 금지).
- Evaluate novelty critically: what is genuinely new vs. incremental improvement. Rank new methodology/perspective higher than performance gains.
- Focus on what problems the field is trying to solve and what approaches are gaining traction.
- Be specific: mention materials, methods, and quantitative results where available.
- For notable_papers, each "reason" must state which academic/technical gap or open problem the work addresses and how—do not give generic one-line praise."""


def _extract_balanced_json_object(s: str) -> Optional[str]:
    """Return the first top-level JSON object substring using brace matching (respects strings)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    i = start
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
        i += 1
    return None


def _parse_gemini_json(response: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON; tolerate extra prose via balanced-object extraction."""
    if not response:
        return None
    import re

    cleaned = response.strip()
    for _ in range(5):
        nxt = re.sub(r"^```\w*\n?", "", cleaned)
        nxt = re.sub(r"\n?```\s*$", "", nxt).strip()
        if nxt == cleaned:
            break
        cleaned = nxt

    last_err: Optional[Exception] = None
    for blob in (cleaned, _extract_balanced_json_object(cleaned) or ""):
        if not blob:
            continue
        try:
            return json.loads(blob)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    logger.warning("Failed to parse Gemini JSON: %s", last_err)
    return None


def _format_paper_list(papers: List, max_count: int = 50, summary_max_chars: int = 200) -> str:
    """Format papers into a text list for prompts."""
    sorted_p = sorted(papers, key=lambda p: -(p.get("relevance") or 0))[:max_count]
    lines = []
    for p in sorted_p:
        rel = p.get("relevance", "?")
        summary_short = (p.get("summary") or p.get("abstract") or "")[:summary_max_chars]
        lines.append(f"- [{p.get('date','')}] {p['title'][:120]} (rel:{rel})\n  {summary_short}")
    return "\n".join(lines) if lines else "(No papers)"


_DIGEST_COMPACT_SUFFIX = """

중요: 응답이 잘리지 않도록 hot_issues·sections·notable_papers의 각 문자열은 짧게(설명은 1~2문장) 유지하세요. 마크다운 없이 순수 JSON만 출력하세요."""


def _build_structured_digest_prompt(
    n: int,
    paper_list: str,
    context_intro: str,
    context_type: str,
    *,
    full_template: bool,
    compact: bool,
) -> str:
    suffix = _DIGEST_COMPACT_SUFFIX if compact else ""
    if full_template:
        return f"""Analyze the following papers from {context_intro} and produce a structured research digest.

Papers ({n} total):
{paper_list}

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
  ],
  "notable_papers": [
    {{
      "title": "논문 제목 (원문 그대로)",
      "reason": "1-2문장(한국어). 학계·분야의 어떤 문제·병목·공백을 겨냥하는지 먼저 밝히고, 본 논문이 그에 대해 무엇을 제시하는지(방법·결과)를 비판적으로. 단순 칭찬·한 줄 가치 판단 금지."
    }}
  ]
}}

Required sections (in order):
1. "핵심 연구 동향" - 가장 활발한 연구 방향 2-3개를 짧게 정리. 각 방향에서 어떤 문제를 해결하려 하는지 명시.
2. "미해결 과제" - 이 {context_type}에서 아직 풀리지 않은 핵심 질문이나 기술적 병목. 구체적으로 어떤 시도가 실패했거나 한계가 있는지.
3. "CCEL 시사점" - CCEL 연구(DFT, 촉매, 전기화학, 배터리, MLIP)와 연결되는 함의를 1-2문장으로 담백하게.

hot_issues: 반복 등장하는 미해결 과제나 핫한 연구 질문 3-5개.
notable_papers: 실제 novelty가 있는 논문만 2-3편. 단순 성능 개선은 제외. reason은 반드시 문제→기여 구조.{suffix}"""
    return f"""Analyze the following papers from {context_intro} and produce a brief research digest.

Papers ({n} total):
{paper_list}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "hot_issues": [
    {{
      "topic": "짧은 이슈 제목 (Korean, 10자 내외)",
      "description": "이 이슈의 핵심을 1문장으로 설명"
    }}
  ],
  "sections": [
    {{
      "title": "핵심 동향",
      "content": "이 {context_type}의 최근 연구 방향을 2-3문장으로 객관적으로 요약. 어떤 문제를 풀고 있고, 어떤 방법론이 사용되는지."
    }}
  ],
  "notable_papers": [
    {{
      "title": "논문 제목 (원문 그대로)",
      "reason": "1-2문장. 어떤 학문적·기술적 공백을 다루는지와 기여를 문제→해결 형태로."
    }}
  ]
}}

hot_issues: 1-2개만. notable_papers: novelty가 있는 논문만 최대 2편. 없으면 빈 배열. reason은 칭찬 한 줄 금지.{suffix}"""


def _digest_failed_fallback(
    depth: str,
    context_label: str,
    *,
    api_gave_text: bool,
) -> dict:
    """User-visible message when digest JSON cannot be produced (no raw model dump)."""
    if api_gave_text:
        body = (
            f"「{context_label}」digest 응답을 JSON으로 처리하지 못했습니다. "
            "논문이 많을 때 모델 출력이 잘리거나 형식이 어긋날 수 있습니다. "
            "더 적은 논문·짧은 설명으로 자동 재시도했으나 모두 실패했습니다. 다음 파이프라인에서 다시 시도됩니다."
        )
    else:
        body = (
            f"「{context_label}」digest 생성 중 Gemini API 응답이 없었습니다. "
            "API 키, 네트워크, 할당량을 확인한 뒤 다시 실행하세요."
        )
    return {
        "depth": depth,
        "parse_failed": True,
        "hot_issues": [],
        "sections": [{"title": "AI 요약을 불러오지 못했습니다", "content": body}],
        "notable_papers": [],
        "generated_at": datetime.now().isoformat(),
    }


def _generate_structured_digest(
    papers: List,
    context_type: str,
    context_label: str,
    context_detail: str = "",
    config: dict = None,
) -> dict:
    """
    Shared digest generator with adaptive depth.
    context_type: "category" | "group"
    Returns unified schema: {depth, hot_issues, sections, notable_papers, generated_at}
    On JSON parse failure: retries with fewer papers / compact instructions, then a safe error section.
    """
    n = len(papers)

    if n < 10:
        return {
            "depth": "skip",
            "hot_issues": [],
            "sections": [],
            "notable_papers": [],
            "generated_at": datetime.now().isoformat(),
        }

    depth = "full" if n >= 30 else "light"

    context_intro = {
        "category": f'"{context_label}" 분야',
        "group": f'"{context_label}" 연구 그룹 (PI: {context_detail})',
    }.get(context_type, context_label)

    if depth == "full":
        attempts = [
            # max_papers, full_template, compact, summary_chars, log_finish_reason
            (80, True, False, 200, False),
            (45, True, True, 120, True),
            (30, False, True, 120, True),
        ]
    else:
        attempts = [
            (30, False, False, 200, False),
            (18, False, True, 120, True),
        ]

    any_text = False
    for attempt_i, (max_papers, full_template, compact, summary_chars, log_fr) in enumerate(attempts):
        paper_list = _format_paper_list(papers, max_count=max_papers, summary_max_chars=summary_chars)
        prompt = _build_structured_digest_prompt(
            n,
            paper_list,
            context_intro,
            context_type,
            full_template=full_template,
            compact=compact,
        )
        response = _call_gemini(prompt, _DIGEST_SYSTEM, config, log_finish_reason=log_fr)
        if response:
            any_text = True
        result = _parse_gemini_json(response) if response else None
        if result:
            result["depth"] = depth
            result.setdefault("hot_issues", [])
            result.setdefault("sections", [])
            result.setdefault("notable_papers", [])
            result["generated_at"] = datetime.now().isoformat()
            return result
        logger.warning(
            "Structured digest parse failed (%s/%s): max_papers=%s full_template=%s compact=%s",
            attempt_i + 1,
            len(attempts),
            max_papers,
            full_template,
            compact,
        )

    return _digest_failed_fallback(depth, context_label, api_gave_text=any_text)


def generate_group_digests(papers: List, config: dict = None) -> Dict:
    """
    Generate per-group structured digests with adaptive depth.
    Papers are deduped (DOI/title) and limited to a rolling date window
    (summarizer.weekly_digest_days, default 7) so counts match "recent" scope
    and --digest history merges do not inflate totals.
    Returns a dict keyed by group_id, each containing metadata + digest.
    """
    groups = config.get("groups", {}) if config else {}
    if not groups:
        return {}

    cfg = config or {}
    window_days = int(cfg.get("summarizer", {}).get("weekly_digest_days", 7))
    raw_n = len(papers or [])
    scoped = _dedupe_papers_by_doi_or_title(papers or [])
    scoped = _filter_papers_by_rolling_days(scoped, window_days)
    logger.info(
        "Group digests: %s unique papers in last %s days by date (from %s raw rows)",
        len(scoped),
        window_days,
        raw_n,
    )

    by_group = {}
    for p in scoped:
        gid = p.get("group")
        if gid:
            by_group.setdefault(gid, []).append(p)

    digests = {}

    for gid, gcfg in groups.items():
        group_papers = by_group.get(gid, [])
        if not group_papers:
            continue

        pi_name = gcfg.get("pi", gid)
        group_name = gcfg.get("name", gid)

        logger.info(f"Generating group digest: {group_name} ({len(group_papers)} papers in window)")

        digest = _generate_structured_digest(
            group_papers,
            context_type="group",
            context_label=group_name,
            context_detail=pi_name,
            config=config,
        )

        digests[gid] = {
            "group_name": group_name,
            "pi": pi_name,
            "paper_count": len(group_papers),
            "paper_count_window_days": window_days,
            "digest": digest,
        }

        logger.info(f"  {group_name}: depth={digest['depth']}, issues={len(digest.get('hot_issues', []))}")

        if digest["depth"] != "skip":
            time.sleep(4)

    return digests


def generate_category_trends(current_papers: List, config: dict = None) -> Dict:
    """
    Generate per-category structured digests with adaptive depth.
    Returns a dict keyed by category ID, each containing metadata + digest.
    """
    history_papers = _load_history_papers(config) if config else []

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

    now = datetime.now()
    one_month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    one_year_ago = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    monthly_papers = [p for p in all_papers if (p.get("date") or "") >= one_month_ago]
    yearly_papers = [p for p in all_papers if (p.get("date") or "") >= one_year_ago]

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

    for cat_id in cat_ids:
        label = cat_labels[cat_id]
        monthly_cat = [p for p in monthly_papers if cat_id in _get_paper_cats(p)]
        yearly_cat = [p for p in yearly_papers if cat_id in _get_paper_cats(p)]

        logger.info(f"Generating category digest: {label} (monthly={len(monthly_cat)}, yearly={len(yearly_cat)})")

        digest_monthly = _generate_structured_digest(
            monthly_cat,
            context_type="category",
            context_label=f"{label} (최근 1개월)",
            config=config,
        )
        logger.info(f"  {label} monthly: depth={digest_monthly['depth']}, issues={len(digest_monthly.get('hot_issues', []))}")
        if digest_monthly["depth"] != "skip":
            time.sleep(4)

        digest_yearly = _generate_structured_digest(
            yearly_cat,
            context_type="category",
            context_label=f"{label} (최근 1년)",
            config=config,
        )
        logger.info(f"  {label} yearly: depth={digest_yearly['depth']}, issues={len(digest_yearly.get('hot_issues', []))}")
        if digest_yearly["depth"] != "skip":
            time.sleep(4)

        trends[cat_id] = {
            "label": label,
            "paper_count_monthly": len(monthly_cat),
            "paper_count_yearly": len(yearly_cat),
            "digest_monthly": digest_monthly,
            "digest_yearly": digest_yearly,
        }

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
        paper["relevance_reason"] = result.get("relevance_reason", "")
        paper.setdefault("ccel", paper.get("group") == "ccel")
        summarized += 1

        time.sleep(4)

    logger.info(f"Summary complete: {summarized} new, {skipped} cached (reused)")

    papers.sort(key=lambda p: (-p.get("relevance", 0), p.get("date") or ""), reverse=False)
    papers.sort(key=lambda p: p.get("date") or "", reverse=True)

    return papers