"""
HyeAero.AI — **Buyer Psychology Engine** (advisory layer for the consultant LLM).

Infers hidden intent, buyer archetype, want-vs-budget gap, and **response_guidance** for the main model.
Does **not** answer the user directly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ASPIRATION_MODELS = re.compile(
    r"\b(g650|g700|g800|global\s*7500|global\s*8000|falcon\s*10x|falcon\s*8x|falcon\s*7x)\b",
    re.I,
)
_VALUE_LEX = re.compile(
    r"\b(cheaper|cheap|deal|budget|value|pre-?owned|used|under\s*\$|save\s+money|lowest)\b",
    re.I,
)
_PERF_LEX = re.compile(
    r"\b(fast|speed|mach|cruise|range|non-?stop|nonstop|transcon|transatlantic|mach\s*0\.?\d+)\b",
    re.I,
)
_LIFESTYLE_LEX = re.compile(
    r"\b(hotel|lounge|like\s+a\s+hotel|interior|cabin|show\s+me|photos?|pictures?|"
    r"galley|divan|seating\s+layout|wide\s+cabin|quiet\s+cabin)\b",
    re.I,
)
_INVESTOR_LEX = re.compile(
    r"\b(resale|depreciation|hold\s+value|invest|investor|exit|liquid|charter\s+revenue)\b",
    re.I,
)
_RISK_LEX = re.compile(
    r"\b(good\s+deal|too\s+good|should\s+i\s+stretch|stretch\s+the\s+budget|"
    r"worried|concerned|red\s+flag|hidden\s+costs?)\b",
    re.I,
)
_ROUTE_LEX = re.compile(
    r"\b(nyc|lax|mia|tpa|ord|dfw|bos|sea|iad|lon|par|dub|dxb|hkg|syd|gva)\b.*\b(to|→|-|–)\b",
    re.I,
)


def consultant_buyer_psychology_enabled() -> bool:
    return (os.getenv("CONSULTANT_BUYER_PSYCHOLOGY") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _merge_conversation_text(
    latest_query: str,
    history: Optional[List[Dict[str, str]]],
    max_chars: int = 6000,
) -> str:
    parts: List[str] = []
    if history:
        for turn in history[-16:]:
            if not isinstance(turn, dict):
                continue
            r = str(turn.get("role") or "").strip().lower()
            c = str(turn.get("content") or "").strip()
            if not c:
                continue
            if r in ("user", "assistant"):
                parts.append(c)
    parts.append(latest_query or "")
    blob = "\n".join(parts).strip()
    if len(blob) > max_chars:
        return blob[-max_chars:]
    return blob


def _phly_marketing_strings(phly_rows: Optional[List[Dict[str, Any]]], limit: int = 4) -> List[str]:
    out: List[str] = []
    for r in phly_rows or []:
        if not isinstance(r, dict):
            continue
        man = str(r.get("manufacturer") or "").strip()
        mdl = str(r.get("model") or "").strip()
        try:
            from services.searchapi_aircraft_images import compose_manufacturer_model_phrase, normalize_aircraft_name

            mm = compose_manufacturer_model_phrase(man, mdl).strip()
            mm = normalize_aircraft_name(mm) if mm else ""
        except Exception:
            mm = " ".join(x for x in (man, mdl) if x).strip()
        if mm and mm not in out:
            out.append(mm)
        if len(out) >= limit:
            break
    return out


def _class_typical_floor_millions(jet_class: str) -> float:
    """Rough illustrative pre-owned *entry* floors by class (internal heuristic only)."""
    return {
        "light": 2.0,
        "midsize": 3.5,
        "super_midsize": 7.0,
        "large": 14.0,
        "ultra": 28.0,
        "unknown": 5.0,
    }.get(jet_class or "unknown", 5.0)


def _infer_aspiration_class_from_text(blob: str, models: List[str]) -> str:
    try:
        from services.aircraft_decision_engine import infer_jet_class
    except Exception:
        infer_jet_class = lambda s: "unknown"  # type: ignore[misc,assignment]
    best = "unknown"
    best_rank = -1
    rank = {"ultra": 4, "large": 3, "super_midsize": 2, "midsize": 1, "light": 0, "unknown": -1}
    if _ASPIRATION_MODELS.search(blob):
        return "ultra"
    for m in models:
        cls = infer_jet_class(m)
        if rank.get(cls, -1) > best_rank:
            best_rank = rank.get(cls, -1)
            best = cls
    if best == "unknown":
        cls2 = infer_jet_class(blob[:500])
        if cls2 != "unknown":
            best = cls2
    return best


def _budget_signal_text(
    blob: str,
    budget_m: Optional[float],
    aspiration_class: str,
) -> str:
    if budget_m is not None:
        return f"explicit ~${budget_m:.1f}M (parsed)"
    if _VALUE_LEX.search(blob):
        return "implicit: price-sensitive / seeking value"
    if _ASPIRATION_MODELS.search(blob) or aspiration_class == "ultra":
        return "implicit: high aspiration (ultra / flagship language or models)"
    return "not clearly stated"


def _mission_signal_text(mission: Dict[str, Any], blob: str) -> str:
    bits: List[str] = []
    if mission.get("passengers"):
        bits.append(f"~{mission['passengers']} pax")
    if mission.get("longest_leg_nm"):
        bits.append(f"~{mission['longest_leg_nm']:.0f} nm leg")
    if mission.get("usage"):
        bits.append(str(mission["usage"]))
    if mission.get("typical_routes_hint"):
        bits.append("route pairs hinted")
    elif _ROUTE_LEX.search(blob):
        bits.append("routes mentioned")
    if mission.get("missing_fields"):
        bits.append(f"gaps: {', '.join(mission['missing_fields'][:4])}")
    return "; ".join(bits) if bits else "not specified"


def _preference_signal(blob: str) -> str:
    if _LIFESTYLE_LEX.search(blob):
        return "premium cabin / visual / experience-oriented"
    if _PERF_LEX.search(blob):
        return "performance / range / speed-oriented"
    if _VALUE_LEX.search(blob):
        return "value / deal-oriented"
    if _INVESTOR_LEX.search(blob):
        return "resale / economics-oriented"
    return "neutral / unspecified"


def _risk_signal(blob: str) -> str:
    if re.search(r"\bshould\s+i\s+stretch\b|\bstretch\b", blob, re.I):
        return "stretching budget / overreach risk"
    if _RISK_LEX.search(blob):
        return "cautious / deal validation"
    return "low explicit risk language"


def _classify_buyer_type(blob: str, mission: Dict[str, Any], aspiration_class: str) -> Tuple[str, float]:
    tags: List[str] = []
    conf = 0.55
    if _ASPIRATION_MODELS.search(blob) or aspiration_class in ("ultra", "large"):
        tags.append("ASPIRATIONAL")
        conf += 0.12
    if _VALUE_LEX.search(blob) or mission.get("budget_millions_usd"):
        tags.append("VALUE")
        conf += 0.08
    if _PERF_LEX.search(blob):
        tags.append("PERFORMANCE")
        conf += 0.08
    if _LIFESTYLE_LEX.search(blob):
        tags.append("LIFESTYLE")
        conf += 0.1
    if _INVESTOR_LEX.search(blob):
        tags.append("INVESTOR")
        conf += 0.07
    if not tags:
        tags.append("GENERAL_BUYER")
        conf = 0.45
    # Prefer a single primary + optional secondary for readability
    primary_order = (
        "ASPIRATIONAL",
        "LIFESTYLE",
        "VALUE",
        "PERFORMANCE",
        "INVESTOR",
        "GENERAL_BUYER",
    )
    ordered = [t for t in primary_order if t in tags]
    if len(ordered) >= 2:
        label = f"{ordered[0]} + {ordered[1]}"
    else:
        label = ordered[0] if ordered else "GENERAL_BUYER"
    return label, max(0.35, min(0.95, conf))


def _gap_and_strategy(
    aspiration_class: str,
    budget_m: Optional[float],
    blob: str,
) -> Tuple[str, str, str]:
    floor = _class_typical_floor_millions(aspiration_class)
    if budget_m is None:
        if _VALUE_LEX.search(blob) and aspiration_class in ("ultra", "large"):
            return (
                "Budget not stated; language mixes flagship-class interest with value-seeking — mismatch risk.",
                "MEDIUM",
                "Clarify budget and longest nonstop leg; then offer 2–3 realistic classes with tradeoffs.",
            )
        return (
            "Insufficient numbers to quantify gap; keep recommendations conditional on budget + mission.",
            "MEDIUM",
            "Ask one tight clarifying question (budget + typical mission), then recommend within that envelope.",
        )
    ratio = budget_m / floor if floor > 0 else 1.0
    if ratio >= 0.92:
        gap = f"Stated ~${budget_m:.1f}M is broadly compatible with **{aspiration_class.replace('_', ' ')}**-class shopping (rule-of-thumb floor ~${floor:.0f}M+ used entry — wide market variance)."
        strat = "SMALL"
        strat_detail = "Recommend directly within class; still name operating-cost and inspection blind spots."
    elif ratio >= 0.5:
        gap = f"Stated ~${budget_m:.1f}M vs **{aspiration_class.replace('_', ' ')}**-class typical floor ~${floor:.0f}M+ suggests a **moderate** stretch or a step-down cabin/range tradeoff."
        strat = "MEDIUM"
        strat_detail = "Offer 2–3 alternatives (adjacent class or older large-cabin) with honest tradeoffs."
    else:
        gap = (
            f"Stated ~${budget_m:.1f}M is **well below** typical **{aspiration_class.replace('_', ' ')}**-class "
            f"used-entry band (~${floor:.0f}M+ rule-of-thumb) — **large gap** unless they shift class or mission."
        )
        strat = "LARGE"
        strat_detail = (
            "Acknowledge the aspiration first, then redirect: closest *experience* (cabin height/seat count) "
            "in a lower class — never pretend flagship economics fit a mid-market budget."
        )
    return gap, strat, strat_detail


def _response_guidance(
    buyer_type: str,
    strategy: str,
    strategy_detail: str,
    aspiration_class: str,
    models: List[str],
) -> str:
    mpreview = ", ".join(models[:3]) if models else "their stated type"
    lines = [
        f"Buyer profile: **{buyer_type}**. Strategy: **{strategy}** — {strategy_detail}",
        f"Aspiration / class anchor: **{aspiration_class.replace('_', ' ')}**; aircraft mentions: {mpreview}.",
    ]
    if "LIFESTYLE" in buyer_type or "ASPIRATIONAL" in buyer_type:
        lines.append(
            "Tone: warm broker — validate taste, then tighten to realistic options; prefer cabin/experience framing over spec dumps."
        )
    if "VALUE" in buyer_type:
        lines.append("Emphasize total cost of ownership and deal discipline; warn on deferred maintenance / pedigree without fear-mongering.")
    if "PERFORMANCE" in buyer_type:
        lines.append("Lead with range/speed/payload fit to stated mission; cabin second.")
    if "INVESTOR" in buyer_type:
        lines.append("Surface liquidity/resale drivers at class level; avoid implying specific investment returns.")
    if strategy == "LARGE":
        lines.append(
            "Mandatory: one sentence that ultra-flagship ask at their budget is not realistic; pivot to closest real alternatives."
        )
    return " ".join(lines)


def run_buyer_psychology_engine(
    *,
    latest_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    detected_aircraft_interest: Optional[List[str]] = None,
    budget_hint: Optional[str] = None,
    mission_hint: Optional[str] = None,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable bundle for ``data_used["buyer_psychology"]`` and optional system prompt injection.
    """
    blob = _merge_conversation_text(latest_query, conversation_history)
    low = blob.lower()

    try:
        from services.aircraft_decision_engine import extract_mission_profile

        mission = extract_mission_profile(latest_query)
    except Exception:
        mission = {}

    if mission_hint:
        mission = {**mission, "hint": mission_hint}

    models = list(detected_aircraft_interest or [])
    for pm in _phly_marketing_strings(phly_rows):
        if pm and pm not in models:
            models.append(pm)

    aspiration_class = _infer_aspiration_class_from_text(low, models)
    budget_m = mission.get("budget_millions_usd")
    if budget_m is None and budget_hint:
        m = re.search(r"(\d+(?:\.\d+)?)\s*m", str(budget_hint).lower())
        if m:
            try:
                budget_m = float(m.group(1))
            except ValueError:
                pass

    budget_sig = _budget_signal_text(low, budget_m, aspiration_class)
    mission_sig = _mission_signal_text(mission, low)
    if mission_hint:
        mission_sig = f"{mission_sig}; note: {mission_hint}".strip("; ")
    pref_sig = _preference_signal(low)
    risk_sig = _risk_signal(low)

    buyer_type, confidence = _classify_buyer_type(low, mission, aspiration_class)
    gap_analysis, strategy, strat_detail = _gap_and_strategy(aspiration_class, budget_m, low)
    guidance = _response_guidance(buyer_type, strategy, strat_detail, aspiration_class, models)

    return {
        "buyer_type": buyer_type,
        "confidence": round(confidence, 2),
        "detected_signals": {
            "budget": budget_sig,
            "mission": mission_sig,
            "preference": pref_sig,
            "risk": risk_sig,
        },
        "gap_analysis": gap_analysis,
        "strategy": strategy,
        "strategy_detail": strat_detail,
        "response_guidance": guidance,
        "aspiration_class": aspiration_class,
        "parsed_budget_millions_usd": budget_m,
        "detected_aircraft_interest": models[:16],
    }


def format_buyer_psychology_for_system_prompt(payload: Dict[str, Any], *, max_chars: int = 1600) -> str:
    """Short imperative block for the consultant system prompt."""
    if not isinstance(payload, dict):
        return ""
    bt = str(payload.get("buyer_type") or "")
    cg = float(payload.get("confidence") or 0)
    sig = payload.get("detected_signals") or {}
    gap = str(payload.get("gap_analysis") or "")
    strat = str(payload.get("strategy") or "")
    guide = str(payload.get("response_guidance") or "")
    lines = [
        "**[BUYER PSYCHOLOGY — advisory; do not quote this header to the user]**",
        f"- **Profile:** {bt} (engine confidence {cg:.2f})",
        f"- **Signals:** budget={sig.get('budget')}; mission={sig.get('mission')}; preference={sig.get('preference')}; risk={sig.get('risk')}",
        f"- **Gap read:** {gap}",
        f"- **Strategy:** {strat} — {payload.get('strategy_detail') or ''}".strip(),
        f"- **How to answer:** {guide}",
    ]
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out
