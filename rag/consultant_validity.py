"""
Aircraft validity firewall.

Purpose: prevent the consultant from answering as if a non-existent marketing model is real.
Deterministic; used both in prompt routing and last-mile answer enforcement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AircraftValidityResult:
    status: str  # "ok" | "invalid_model"
    message: str
    suggestions: List[str]
    invalid_name: Optional[str] = None


# Expand over time; keep conservative (only flag when we're confident it's fake).
_INVALID_MODEL_PATTERNS: Tuple[Tuple[re.Pattern, str, List[str]], ...] = (
    (re.compile(r"\bfalcon\s*9000\b", re.I), "Falcon 9000", ["Falcon 900", "Falcon 8X", "Falcon 7X", "Falcon 6X"]),
    (re.compile(r"\bgulfstream\s+g6500\b|\bg\s*[-.]?\s*6500\b", re.I), "Gulfstream G6500", ["Gulfstream G650", "Gulfstream G650ER", "Gulfstream G700"]),
    (re.compile(r"\bgulfstream\s+g750\b|\bg\s*[-.]?\s*750\b", re.I), "Gulfstream G750", ["Gulfstream G700", "Gulfstream G650ER", "Gulfstream G800"]),
    (re.compile(r"\bglobal\s*10000\b", re.I), "Global 10000", ["Global 7500", "Global 8000", "Global 6500", "Global 6000"]),
)


def validate_aircraft_model(text: str) -> Optional[AircraftValidityResult]:
    """
    Return an invalid-model payload if the text contains a known-fake model string, else None.
    """
    t = (text or "").strip()
    if not t:
        return None
    for pat, label, sugg in _INVALID_MODEL_PATTERNS:
        if pat.search(t):
            return AircraftValidityResult(
                status="invalid_model",
                message=f"No aircraft called {label} exists.",
                suggestions=list(sugg),
                invalid_name=label,
            )
    return None


def count_known_model_mentions(answer: str) -> int:
    """
    Heuristic: count mentions of known model labels we already detect in query expansion.
    Used to enforce "recommendations required" in advisory outputs.
    """
    try:
        from rag.consultant_query_expand import _detect_models
    except Exception:
        return 0
    # _detect_models returns canonical labels; re-run it on the answer to approximate "named real aircraft".
    return len(_detect_models(answer or ""))


def build_invalid_model_user_facing_reply(v: AircraftValidityResult) -> str:
    s = (v.invalid_name or "that model").strip()
    sugg = [x for x in (v.suggestions or []) if (x or "").strip()]
    lines = [f"There’s **no aircraft called {s}** in production.", ""]
    if sugg:
        lines.append("Closest real options:")
        for x in sugg[:6]:
            lines.append(f"- {x}")
        lines.append("")
    lines.append("If you tell me passengers + longest leg, I’ll steer you to the closest real fit.")
    lines.append("")
    lines.append("Consultant Insight: Model-name confusion is how buyers end up comparing the wrong listings and the wrong photos—locking the exact variant early saves time and money.")
    return "\n".join(lines).strip()

