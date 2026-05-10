"""Hidden intent tags for retrieval / ranking tone (deterministic hints)."""


from __future__ import annotations

import re
from typing import List


def infer_contextual_tags(query: str) -> List[str]:
    ql = (query or "").strip().lower()
    out: List[str] = []
    if not ql:
        return out
    if re.search(r"\binfluencers?\b|\binstagram\b|\bwow\b", ql):
        out.append("modern flagship aesthetic")
        out.append("lifestyle-social visibility")
    if re.search(r"\bclients\b.*\b(wow|impress)|\bimpress\b|\bpresence\b|\bgrand\s+entrance\b", ql):
        out.append("VIP arrival presence")
        out.append("stand-up cabin headline")
    if re.search(r"\b(ceos?\s+young|young\s+money|\bstartup\b|\b exits?\b|\b unicorn\b)", ql):
        out.append("new-wealth pacing")
    return list(dict.fromkeys(out))[-20:]
