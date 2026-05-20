"""
Deterministic buyer-journey answers for the elite cabin-shopping thread.

Maps user turns to short, gallery-first copy (vibe over specs). Used as last-mile
enforcement when the LLM drifts into broker templates, spec dumps, or context resets.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# --- Canonical model shortlists (gallery / copy) ---

MODERN_CABIN_UNDER_10M: Tuple[str, ...] = (
    "Challenger 350",
    "Praetor 500",
    "Citation Latitude",
    "Legacy 450",
)

LESS_CORPORATE_INTERIORS: Tuple[str, ...] = (
    "Praetor 600",
    "Falcon 8X",
    "Global 6500",
)

BIGGER_MODERN_CABIN: Tuple[str, ...] = (
    "Global 6000",
    "Falcon 8X",
    "G500",
)

CABIN_FEEL_PRIORITY: Tuple[str, ...] = (
    "Falcon 8X",
    "Global 7500",
    "Praetor 600",
    "Challenger 650",
)

_SPEC_NOISE_RE = re.compile(
    r"(?im)^\s*(?:"
    r".*\b\d[\d,]*\s*(?:nm|nautical|knots?|ktas|mach)\b.*|"
    r".*\bcabin\s+pressure\b.*|"
    r".*\bbaggage\s+volume\b.*|"
    r".*\bclimb\s+rate\b.*|"
    r".*\bdispatch\s+reliability\b.*|"
    r".*\brunway\s+performance\b.*|"
    r".*\bfaa\b.*|"
    r".*\bregistry\b.*"
    r")\s*$"
)

_BROKER_RESET_RE = re.compile(
    r"(?i)\b("
    r"what do you mean|could you clarify|which aircraft|let me know if you have questions|"
    r"to effectively meet your needs|based on your requirements|"
    r"how many passengers|passenger count|tell me your route|"
    r"mission analysis|let's start fresh|start over"
    r")\b",
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def _refinement_type(data_used: Optional[Dict[str, Any]]) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    rt = str(du.get("consultant_refinement_type") or "").strip().lower()
    if rt:
        return rt
    ip = du.get("intent_persistence")
    if isinstance(ip, dict):
        ri = ip.get("resolved_intent")
        if isinstance(ri, dict):
            t = str(ri.get("last_refinement_type") or "").strip().lower()
            if t:
                return t
    return ""


def _comparison_pair(data_used: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    du = data_used if isinstance(data_used, dict) else {}
    target = ""
    cs = du.get("consultant_conversation_state")
    if isinstance(cs, dict):
        mem = cs.get("conversation_memory")
        if isinstance(mem, dict):
            target = str(mem.get("comparison_target") or "")
    if not target:
        ip = du.get("intent_persistence")
        if isinstance(ip, dict):
            ri = ip.get("resolved_intent")
            if isinstance(ri, dict):
                target = str(ri.get("comparison_target") or "")
    if not target or not re.search(r"\bvs\.?\b", target, re.I):
        ql = (du.get("consultant_query") or "").lower() if isinstance(du.get("consultant_query"), str) else ""
        m = re.search(r"\b(g700|global\s*7500)\b.*\b(g700|global\s*7500)\b", ql, re.I)
        if m:
            return ("G700", "Global 7500")
        return None
    parts = [p.strip() for p in re.split(r"\s+vs\.?\s+", target, flags=re.I) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _has_gallery(data_used: Optional[Dict[str, Any]]) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    imgs = du.get("aircraft_images")
    return isinstance(imgs, list) and len(imgs) > 0


def _truncate_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = [p for p in parts if p.strip()][: max(1, max_sentences)]
    return " ".join(kept).strip()


def _strip_spec_noise(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if _SPEC_NOISE_RE.match(line.strip()):
            continue
        if re.search(r"\b\d[\d,]*\s*nm\b", line, re.I) and len(line) > 80:
            continue
        lines.append(line)
    s = "\n".join(lines).strip()
    s = _URL_RE.sub("", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def modern_cabin_under_10m_copy() -> str:
    return (
        "Best modern-feeling cabins under $10M right now are Challenger 350 and Praetor 500. "
        "Latitude and Legacy 450 are also strong if you want a slightly different layout feel. "
        "The gallery shows representative interiors in that band."
    )


def less_corporate_copy() -> str:
    return (
        "Then I'd lean toward newer Praetor, Global, or Falcon interiors rather than older "
        "Gulfstream wood finishes — white cabins, softer lighting, and a younger lounge feel. "
        "The gallery shifts to Praetor 600, Falcon 8X, and Global 6500 style cabins."
    )


def bigger_modern_copy() -> str:
    models = ", ".join(BIGGER_MODERN_CABIN[:-1]) + f", and {BIGGER_MODERN_CABIN[-1]}"
    return (
        f"If you want bigger while keeping the modern feel, look at {models}. "
        "The gallery shows the next step up in cabin volume and presence."
    )


def g700_vs_global_7500_copy() -> str:
    return (
        "G700 feels more dramatic and prestige-focused. Global 7500 feels more refined, "
        "spacious, and practical for long missions. G700 wins on presence. "
        "Global 7500 wins on cabin layout and comfort. "
        "The gallery lines up side-by-side interiors so you can feel the difference."
    )


def comparison_cockpit_copy() -> str:
    return (
        "G700 cockpit feels cleaner and more futuristic. Global 7500 feels more operational and refined. "
        "Both are in the gallery."
    )


def cabin_feel_over_speed_copy() -> str:
    return (
        "Then I'd prioritize cabin atmosphere over performance specs. "
        "Falcon 8X, Global 7500, Praetor 600, and Challenger 650 are the emotional winners — "
        "quieter, wider, softer lighting, less boardroom, more lounge. "
        "The gallery leans into that cabin-first direction."
    )


def enforce_buyer_journey_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply journey-specific templates and strip unacceptable drift."""
    q = (query or "").strip()
    ql = q.lower()
    du = data_used if isinstance(data_used, dict) else {}
    a = (answer or "").strip()
    al = a.lower()
    ref = _refinement_type(du)
    has_gallery = _has_gallery(du)

    # Never allow broker reset / clarification loops on short refinements.
    if len(q) < 100 and _BROKER_RESET_RE.search(a):
        a = ""
        al = ""

    # --- Turn: modern cabin under $10M (shopping pivot) ---
    if du.get("consultant_shopping_pivot") or (
        re.search(r"\bmodern\s+cabin\b", ql) and re.search(r"\b10\s*m|\$10", ql, re.I)
    ):
        if (
            not a
            or len(a) > 320
            or re.search(r"\b(nm|knots?|mach|passengers?|range:)\b", al)
            or not re.search(r"\bchallenger|praetor|latitude|legacy\b", al)
        ):
            a = modern_cabin_under_10m_copy()
        else:
            a = _truncate_sentences(_strip_spec_noise(a), 3)
        if has_gallery:
            a = _truncate_sentences(a, 2)
        return _URL_RE.sub("", a).strip()

    # --- Turn: less corporate ---
    if ref == "style_shift" or re.search(r"\bless\s+corporate\b", ql):
        if (
            not a
            or _BROKER_RESET_RE.search(a)
            or re.search(r"\bfor a modern cabin under\b", al)
            or not re.search(r"\b(praetor|falcon|global|wood|corporate|lighting|lounge)\b", al)
        ):
            a = less_corporate_copy()
        else:
            a = _truncate_sentences(_strip_spec_noise(a), 3)
        if has_gallery:
            a = _truncate_sentences(a, 2)
        return _URL_RE.sub("", a).strip()

    # --- Turn: bigger ---
    if ref == "size_upgrade" or re.match(r"^\s*bigger\s*[\.\!]?\s*$", q, re.I):
        if (
            not a
            or _BROKER_RESET_RE.search(a)
            or re.search(r"\bhow many passengers\b", al)
            or not re.search(r"\b(global|falcon|g500|g6000)\b", al)
        ):
            a = bigger_modern_copy()
        else:
            a = _truncate_sentences(_strip_spec_noise(a), 3)
        if has_gallery:
            a = _truncate_sentences(a, 2)
        return _URL_RE.sub("", a).strip()

    # --- Turn: G700 vs Global 7500 ---
    if re.search(r"\bg700\b", ql) and re.search(r"\bglobal\s*7500\b", ql):
        if (
            not a
            or len(a) > 420
            or re.search(r"\b\d[\d,]*\s*nm\b", al)
            or re.search(r"\bmach\b", al)
            or not re.search(r"\b(presence|comfort|refined|dramatic|layout)\b", al)
        ):
            a = g700_vs_global_7500_copy()
        else:
            a = _truncate_sentences(_strip_spec_noise(a), 5)
        return _URL_RE.sub("", a).strip()

    # --- Turn: cockpit (comparison thread) ---
    if ref == "view_change" and re.search(r"\bcockpit\b", ql):
        pair = _comparison_pair(du)
        if pair and re.search(r"\bg700\b", " ".join(pair).lower()) and re.search(
            r"\bglobal\b", " ".join(pair).lower()
        ):
            if not a or len(a) > 200 or _BROKER_RESET_RE.search(a):
                a = comparison_cockpit_copy()
            else:
                a = _truncate_sentences(_strip_spec_noise(a), 2)
            return _URL_RE.sub("", a).strip()

    # --- Turn: cabin feel > speed ---
    if re.search(r"\bcabin\s+feel\b", ql) and re.search(r"\bthan\s+speed\b|\bover\s+speed\b", ql):
        if (
            not a
            or re.search(r"\bmach\b|\bknots?\b|\brunway\b|\bclimb\b|\bdispatch\b", al)
            or not re.search(r"\b(quiet|comfort|lighting|lounge|atmosphere|feel)\b", al)
        ):
            a = cabin_feel_over_speed_copy()
        else:
            a = _truncate_sentences(_strip_spec_noise(a), 4)
        return _URL_RE.sub("", a).strip()

    return _URL_RE.sub("", _strip_spec_noise(a)).strip() if a else a
