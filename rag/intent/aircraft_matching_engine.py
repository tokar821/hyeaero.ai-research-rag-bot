"""
Aircraft Matching Engine — maps user intent to **correct aircraft class** peers.

Ultra-large-cabin / flagship anchors (**G650**, **Global 7500**, close kin) may only map to
the approved same-class shortlist — **never** light jets, **Eclipse**, **CJ2**, etc.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# User-requested anchor family (query / intent text).
_ULR_FLAGSHIP_ANCHOR = re.compile(
    r"(?is)"
    r"\b(g-?\s*650|g650|gulfstream\s*g-?\s*650|gulfstream\s*650)\b|"
    r"\b(g-?\s*700|g700|gulfstream\s*g-?\s*700)\b|"
    r"\b(global\s*7500|bd-?700)\b|"
    r"\b(global\s*8000|global\s*8(?:k|000))\b"
)

# ONLY these when ULR flagship anchor is present (exact policy list + close variants).
_ULR_FLAGSHIP_PEER_MODELS: List[str] = [
    "Bombardier Challenger 650",
    "Dassault Falcon 7X",
    "Dassault Falcon 8X",
    "Bombardier Global 5000",
    "Bombardier Global 6000",
]

# NEVER suggest these when ULR flagship anchor matched (hard fail if they appear in proposals).
_DOWNGRADE_FORBIDDEN = re.compile(
    r"(?is)\b("
    r"eclipse\s*500|ea\s*500|eclipse|"
    r"cj\s*2|citation\s*cj\s*2|citationjet|mustang|"
    r"phenom\s*100|phenom\s*300|"
    r"learjet\s*3[15]|learjet\s*40|"
    r"hawker\s*400|beechjet|"
    r"vision\s*jet|cirrus\s*sf50|"
    r"hondajet|"
    r"light\s*jet"
    r")\b",
)


def _intent_query_blob(user_query: str, normalized_intent: Optional[Dict[str, Any]]) -> str:
    """Current query + intent fields only (no history) — used for downgrade-forbidden scan."""
    parts: List[str] = []
    if normalized_intent and isinstance(normalized_intent, dict):
        for k in ("aircraft", "category", "visual_focus", "intent_type"):
            parts.append(str(normalized_intent.get(k) or ""))
        c = normalized_intent.get("constraints")
        if isinstance(c, dict):
            parts.append(str(c.get("comparison_target") or ""))
            parts.append(str(c.get("style") or ""))
    parts.append(user_query or "")
    return " ".join(parts).strip()


def _text_blob(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    normalized_intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Query + intent + recent user lines — anchor detection (may need thread for 'that jet')."""
    parts: List[str] = [_intent_query_blob(user_query, normalized_intent)]
    if history:
        for h in history[-6:]:
            if isinstance(h, dict) and (h.get("role") or "").strip().lower() == "user":
                parts.append(str(h.get("content") or ""))
    return " ".join(parts).strip()


def _ulr_flagship_anchor_present(blob: str) -> bool:
    return bool(_ULR_FLAGSHIP_ANCHOR.search(blob))


def _forbidden_downgrade_in_text(blob: str) -> Optional[str]:
    m = _DOWNGRADE_FORBIDDEN.search(blob)
    if m:
        return m.group(0).strip()
    return None


def run_aircraft_matching_engine(
    user_query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    normalized_intent: Optional[Dict[str, Any]] = None,
    proposed_candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return ``aircraft_candidates``, ``reasoning``, and optional hard-fail flags.

    If ``proposed_candidates`` is supplied and violates class rules for the detected anchor,
    sets ``hard_fail`` True and clears ``aircraft_candidates``.
    """
    blob = _text_blob(user_query or "", history, normalized_intent)
    anchor_ulr = _ulr_flagship_anchor_present(blob)
    current_only = _intent_query_blob(user_query or "", normalized_intent)

    hard_fail = False
    hard_fail_reason: Optional[str] = None
    candidates: List[str] = []
    reasoning = ""

    if anchor_ulr:
        forbidden_hit = _forbidden_downgrade_in_text(current_only)
        if forbidden_hit and proposed_candidates is None:
            # User text itself asks for a downgraded class alongside anchor — policy violation.
            hard_fail = True
            hard_fail_reason = (
                f"HARD FAIL: anchor is ultra-large-cabin flagship but text references forbidden class "
                f"({forbidden_hit})."
            )
            reasoning = hard_fail_reason
        else:
            candidates = list(_ULR_FLAGSHIP_PEER_MODELS)
            reasoning = (
                "User intent references an ultra-long-range / large-cabin flagship (e.g. Gulfstream G650 "
                "or Global 7500). Same-class alternatives stay in **large cabin / long-range** peers: "
                "Challenger 650, Falcon 7X/8X, Global 5000/6000 — not light jets or VLJs."
            )

        if proposed_candidates is not None:
            joined = " | ".join(str(x) for x in proposed_candidates)
            bad = _forbidden_downgrade_in_text(joined)
            if bad:
                hard_fail = True
                hard_fail_reason = (
                    f"HARD FAIL: proposed candidate list includes class-incompatible type ({bad}) "
                    f"for ULR flagship anchor."
                )
                candidates = []
                reasoning = hard_fail_reason
    else:
        # No ULR flagship anchor — engine does not impose the ULR-only shortlist.
        reasoning = (
            "No Gulfstream G650 / Global 7500-class flagship anchor detected in query or intent; "
            "class-specific peer shortlist not forced. Use mission, budget, and category from intent elsewhere."
        )
        if proposed_candidates:
            candidates = [str(x).strip() for x in proposed_candidates if str(x).strip()]
        else:
            candidates = []

    return {
        "aircraft_candidates": candidates,
        "reasoning": reasoning,
        "hard_fail": hard_fail,
        "hard_fail_reason": hard_fail_reason,
    }


def validate_ulr_peer_list(models: List[str]) -> Dict[str, Any]:
    """
    Return hard_fail True if any model string matches forbidden downgrade tokens
    while the list is intended as ULR-flagship peers (caller responsibility).
    """
    joined = " | ".join(str(m) for m in models or [])
    bad = _forbidden_downgrade_in_text(joined)
    if bad:
        return {
            "aircraft_candidates": [],
            "reasoning": f"HARD FAIL: forbidden downgrade token present ({bad}).",
            "hard_fail": True,
            "hard_fail_reason": "class_mismatch",
        }
    allowed_l = {m.lower() for m in _ULR_FLAGSHIP_PEER_MODELS}
    out: List[str] = []
    unknown: List[str] = []
    for m in models or []:
        ms = str(m).strip()
        if not ms:
            continue
        if ms.lower() in allowed_l or any(a in ms.lower() for a in ("challenger 650", "falcon 7x", "falcon 8x", "global 5000", "global 6000")):
            out.append(ms)
        else:
            unknown.append(ms)
    if unknown:
        return {
            "aircraft_candidates": [],
            "reasoning": (
                f"HARD FAIL: models not in approved ULR flagship peer set: {', '.join(unknown)}."
            ),
            "hard_fail": True,
            "hard_fail_reason": "peer_not_in_allowlist",
        }
    return {
        "aircraft_candidates": out,
        "reasoning": "All candidates are in the approved large-cabin / ULR-adjacent peer set.",
        "hard_fail": False,
        "hard_fail_reason": None,
    }
