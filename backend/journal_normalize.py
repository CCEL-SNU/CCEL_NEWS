"""
Canonical journal names for merging RSS vs API spelling variants (same venue, different strings).
"""

from __future__ import annotations

import re
from typing import Dict

# Lowercase trimmed key -> canonical `source` stored in news.json
SOURCE_ALIASES: Dict[str, str] = {
    "journal of the american chemical society": "JACS",
    "j. am. chem. soc.": "JACS",
    "j am chem soc": "JACS",
    "jacs": "JACS",
}


def normalize_journal_source(name: str | None) -> str:
    """Return a single canonical journal label for filtering and breakdown."""
    if name is None:
        return "Unknown"
    raw = str(name).strip()
    if not raw:
        return "Unknown"
    if re.match(r"^ar[Xx]iv", raw):
        return "arXiv"
    key = raw.lower()
    return SOURCE_ALIASES.get(key, raw)
