"""Deterministic polish for short refinement follow-ups (style, size, view)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _refinement_type(data_used: Optional[Dict[str, Any]]) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    rt = str(du.get("consultant_refinement_type") or "").strip().lower()
    if rt:
        return rt
    ip = du.get("intent_persistence")
    if isinstance(ip, dict):
        ri = ip.get("resolved_intent")
        if isinstance(ri, dict):
            pass
    cont = du.get("consultant_continuity_state")
    if isinstance(cont, dict):
        lr = cont.get("last_refinement")
        if isinstance(lr, dict) and lr.get("type"):
            return str(lr["type"]).strip().lower()
    return ""


def _anchor_aircraft(data_used: Optional[Dict[str, Any]]) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    for key in (
        "consultant_gallery_marketing_anchor",
        "consultant_gallery_aircraft",
    ):
        v = str(du.get(key) or "").strip()
        if v:
            return v
    cs = du.get("consultant_conversation_state")
    if isinstance(cs, dict):
        v = str(cs.get("current_aircraft_reference") or "").strip()
        if v:
            return v
        mem = cs.get("conversation_memory")
        if isinstance(mem, dict):
            v = str(mem.get("active_aircraft") or "").strip()
            if v:
                return v
    return "this aircraft"


def _refinement_context_reset(answer: str, *, query: str, data_used: Optional[Dict[str, Any]]) -> bool:
    """True when the model re-opened the whole shopping brief instead of refining the thread."""
    ref = _refinement_type(data_used)
    if ref not in ("style_shift", "size_upgrade"):
        return False
    al = (answer or "").lower()
    ql = (query or "").lower()
    if len(ql) > 80:
        return False
    reset_markers = (
        r"\bfor a modern cabin under\b",
        r"\bif you(?:'re| are) looking for a modern cabin\b",
        r"\bgot it!?\s+for a modern cabin\b",
        r"\bconsider the (?:embraer )?phenom\b.*\bunder \$10",
        r"\bhere are (?:some )?options to consider\b",
    )
    if any(re.search(p, al) for p in reset_markers):
        return True
    if ref == "style_shift" and re.search(r"\bless\s+corporate\b", ql):
        if re.search(r"\bphenom\s*300\b|\bcj3\+?\b|\bcitation\s+cj\b", al) and _anchor_aircraft(data_used).lower() not in al:
            return True
    return False


def enforce_style_shift_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Ensure 'less corporate' / style refinements address aesthetic, not brochure specs."""
    q = (query or "").strip()
    ql = q.lower()
    ref = _refinement_type(data_used)
    if ref != "style_shift" and not re.search(
        r"\b(less\s+corporate|more\s+modern|not\s+that\s+corporate|younger\s+feeling)\b", ql
    ):
        return answer

    a = (answer or "").strip()
    al = a.lower()
    if _refinement_context_reset(a, query=q, data_used=data_used):
        a = ""
        al = ""
    style_cues = (
        "less corporate",
        "more residential",
        "residential",
        "boutique",
        "hotel",
        "warm",
        "softer",
        "lifestyle",
        "boardroom",
        "beige",
        "matte",
        "welcoming",
    )
    if any(c in al for c in style_cues):
        return a

    anchor = _anchor_aircraft(data_used)
    if re.search(r"\bless\s+corporate\b", ql):
        lead = (
            f"Understood — warmer and less boardroom. The gallery leans toward a more residential "
            f"take on the {anchor} cabin: softer materials, less conference-table layout, more "
            f"lounge than meeting room."
        )
    elif re.search(r"\bmore\s+modern\b", ql):
        lead = (
            f"Got it — more contemporary. The {anchor} images skew cleaner lines and updated "
            f"completions rather than traditional cream-and-gold."
        )
    else:
        lead = (
            f"Noted on the aesthetic direction. The gallery reflects that styling on the {anchor} cabin."
        )

    if not a or re.search(r"\b(asking price|registry|market data|\$\d)\b", al):
        return lead
    return f"{lead}\n\n{a}".strip()


def enforce_size_upgrade_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Keep 'bigger' inside budget band when a cap is known."""
    if _refinement_type(data_used) != "size_upgrade" and not re.match(
        r"^\s*bigger\s*[\.\!]?\s*$", (query or "").strip(), re.I
    ):
        return answer
    du = data_used if isinstance(data_used, dict) else {}
    cs = du.get("consultant_conversation_state") if isinstance(du.get("consultant_conversation_state"), dict) else {}
    budget = str(cs.get("current_budget") or "").strip()
    mem = cs.get("conversation_memory") if isinstance(cs.get("conversation_memory"), dict) else {}
    if not budget and mem.get("active_budget_usd"):
        try:
            budget = f"${int(float(mem['active_budget_usd']) / 1_000_000)}M"
        except (TypeError, ValueError):
            pass
    al = (answer or "").lower()
    anchor = _anchor_aircraft(data_used)
    if _refinement_context_reset(answer, query=query, data_used=data_used):
        answer = ""
        al = ""
    if budget and "10" in budget:
        if not al or re.search(r"\bg650\b|\bg700\b|\bglobal\s*7500\b", al):
            return (
                f"Within roughly $10M, bigger than the {anchor} usually means a strong super-midsize "
                "(Challenger 650 or Legacy 650 class) — not an ultra-long-range flagship. "
                "The gallery shows the next cabin size up in that band."
            )
        if re.search(r"\bg650\b|\bg700\b|\bglobal\s*7500\b", al) and not re.search(
            r"\b(latitude|legacy\s*650|challenger\s*650)\b", al
        ):
            return (
                f"Within roughly $10M, bigger than the {anchor} usually means stepping up to a strong super-midsize "
                "(Challenger 650 or Legacy 650 class) — not an ultra-long-range flagship. "
                "The gallery shows the next cabin size up in that band."
            )
    if not al and re.match(r"^\s*bigger\s*[\.\!]?\s*$", (query or "").strip(), re.I):
        return (
            f"Stepping up from the {anchor}: the gallery shifts to the next cabin size in the same band — "
            f"still aligned with your prior cabin direction."
        )
    return answer
