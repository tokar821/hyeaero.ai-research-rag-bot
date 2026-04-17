"""
Heuristic notes when the user names a likely non-existent aircraft marketing type.

Used to steer the consultant away from hallucinating specs/listings for fake model strings.
"""

from __future__ import annotations

import re
from typing import Optional

_FALCON_NUM = re.compile(r"\bfalcon\s*(\d{3,5})\b", re.I)
# Series numbers used in Dassault Falcon branding (not every sub-variant).
_VALID_FALCON_SERIES_NUMS = frozenset({"10", "20", "50", "100", "200", "900", "2000"})

_G6500 = re.compile(r"\bg\s*[-.]?\s*6500\b", re.I)
_G750 = re.compile(r"\bgulfstream\s*g\s*[-.]?\s*750\b|\bg\s*[-.]?\s*750\b", re.I)
_GLOBAL_10000 = re.compile(r"\bglobal\s*10000\b", re.I)


def consultant_suspicious_aircraft_model_note(query: str) -> Optional[str]:
    """
    Return a short internal-facing note for the system prompt, or None if no strong hit.

    Keep messages user-safe: the assistant should repeat this guidance in plain language,
    not claim the fake model exists.
    """
    q = (query or "").strip()
    if not q:
        return None

    if _G6500.search(q):
        return (
            "There is no **Gulfstream G6500**. The user likely means **G650** or **G650ER**."
        )

    if _G750.search(q):
        return (
            "There is no **Gulfstream G750**. The user likely means **G700**, **G650ER**, or **G800**."
        )

    if _GLOBAL_10000.search(q):
        return (
            "There is no **Global 10000**. Bombardier Global line includes **Global 5000/6000/6500/7500/8000** "
            "and related variants — clarify which they mean."
        )

    m = _FALCON_NUM.search(q)
    if not m:
        return None
    num = m.group(1)
    if num == "9000":
        return (
            "There is no **Falcon 9000** in Dassault's lineup. Likely intents: **Falcon 900**, **Falcon 2000**, "
            "or **Falcon 7X / 8X / 6X / 10X**."
        )
    if num not in _VALID_FALCON_SERIES_NUMS and num.isdigit():
        n = int(num)
        if n >= 3000 or n in (600, 800, 6000, 8000):
            return (
                f"There is no well-known **Falcon {num}** production model. Common families: **Falcon 900**, "
                "**Falcon 2000**, **Falcon 7X / 8X / 6X**, and older **Falcon 10/100/200**."
            )
    return None
