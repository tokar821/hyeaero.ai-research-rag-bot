"""
Acquisition budget reality — presentation guard (not a pipeline layer).

Runs before market reality and executive broker so impossible budget/model pairs
never receive listing-style "plausible" prose or an executive buy recommendation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.client_context.recommendation_consistency import _tier_musd


_CAN_I_BUY_RE = re.compile(
    r"(?is)\bcan\s+i\s+(?:buy|afford|get|realistically\s+buy|realistically\s+afford)\b"
)
_ADVERSARIAL_INFEASIBLE_RE = re.compile(
    r"(?is)\b(?:"
    r"good deal anyway|ignore.{0,20}budget|tell me to buy|confirm it|"
    r"recommend .+ for \$\d|steal|bypass budget|override safety|pressure me to buy|"
    r"skip diligence|don't verify|trust the seller|commit now|"
    r"ignore market reality|fits my \$\d|great opportunity"
    r")\b"
)
_LISTING_REALISM_RE = re.compile(
    r"(?is)\b(?:saw|found|listing|is this realistic|realistic)\s*\??\b"
)
_LISTING_ASSESSMENT_RE = re.compile(
    r"(?is)\b(?:listed|listing|asking|good deal|fair\s+price|overpriced|too\s+good\s+to\s+be\s+true|"
    r"worth\s+it|should\s+i\s+pursue|pursue)\b"
)
_ONLY_HAVE_RE = re.compile(
    r"(?is)\b(?:only\s+have|but\s+only)\b"
)
_ULTRA_LONG_DEST_RE = re.compile(
    r"(?is)\b(?:tokyo|singapore|sydney|beijing|hong kong|dubai|mumbai|shanghai)\b"
)
_TRANSATLANTIC_RE = re.compile(
    r"(?is)\b(?:london|paris|frankfurt|geneva)\b.{0,40}\b(?:new york|miami|boston|los angeles|nyc)\b|"
    r"\b(?:new york|miami|boston|los angeles|nyc)\b.{0,40}\b(?:london|paris|frankfurt|geneva)\b"
)
_NONSTOP_RE = re.compile(r"(?is)\bnonstop\b")
_COAST_RE = re.compile(r"(?is)\bcoast.?to.?coast\b")
_PAX_RE = re.compile(r"(?is)\b(?P<pax>\d+)\s+passengers?\b")
_BUDGET_RE = re.compile(
    r"(?is)\b(?:budget|for|under|below|around|about|at|only\s+have|have|budget\s+is)\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
    r"(?:\s*[-–]\s*\$?\s*(?P<amt2>\d+(?:\.\d+)?)\s*(?P<unit2>m|mm|million|mil|k)\b)?"
)
_GULFSTREAM_RE = re.compile(r"(?is)\bgulfstream\b")
_ONLY_BUDGET_RE = re.compile(
    r"(?is)\b(?:only\s+have|only|i\s+have)\s+\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
)
_INFEASIBLE_OPENINGS = (
    "no.",
    "not realistically.",
    "that budget is far below",
)


@dataclass(frozen=True)
class BudgetFeasibility:
    model: str
    budget_musd: float
    tier_musd: float


_LEADING_BUDGET_RE = re.compile(
    r"(?is)^\s*\$\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
)


_MISSION_DOLLAR_RE = re.compile(
    r"(?is)(?:passengers?|pax|budget|buy|for)\b[^.]{0,80}?\$?\s*"
    r"(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
)


def _normalize_query(query: str) -> str:
    return (
        (query or "")
        .replace("\u2014", " ")
        .replace("\u2013", " ")
        .replace("—", " ")
        .replace("–", " ")
    )


def _parse_budget_musd(query: str) -> Optional[float]:
    q = _normalize_query(query)
    m = (
        _LEADING_BUDGET_RE.search(q)
        or _BUDGET_RE.search(q)
        or _ONLY_BUDGET_RE.search(q)
        or _MISSION_DOLLAR_RE.search(q)
    )
    if not m and re.search(r"(?is)\$\s*\d", q):
        amounts: List[float] = []
        for dm in re.finditer(
            r"(?is)\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b",
            q,
        ):
            try:
                val = float(dm.group("amt"))
            except (TypeError, ValueError):
                continue
            unit = (dm.group("unit") or "m").lower()
            if unit == "k":
                amounts.append(val / 1000.0)
            elif val < 1000:
                amounts.append(val)
            else:
                amounts.append(val / 1_000_000.0 if val >= 10_000 else val)
        if amounts:
            return max(amounts)
    if not m:
        return None
    try:
        val = float(m.group("amt2") or m.group("amt"))
    except (IndexError, TypeError, ValueError):
        try:
            val = float(m.group("amt"))
        except (TypeError, ValueError):
            return None
    try:
        unit = (m.group("unit2") or m.group("unit") or "m").lower()
    except IndexError:
        unit = "m"
    if unit == "k":
        return val / 1000.0
    if val < 1000:
        return val
    return val / 1_000_000.0


def _budget_cap_from_context(data_used: Dict[str, Any]) -> Optional[float]:
    ctx = data_used.get("client_context") or data_used.get("broker_conversation_context") or {}
    if isinstance(ctx, dict) and ctx.get("remembered_budget_musd") is not None:
        try:
            return float(ctx["remembered_budget_musd"])
        except (TypeError, ValueError):
            pass
    frame = data_used.get("canonical_intent_frame")
    if isinstance(frame, dict):
        b = frame.get("budget") or {}
        if isinstance(b, dict) and b.get("cap_musd") is not None:
            try:
                return float(b["cap_musd"])
            except (TypeError, ValueError):
                pass
    return None


def _resolve_target_models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
        from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

        raw = detect_models_from_text(query or "")
        out: List[str] = []
        for token in raw:
            resolved = _resolve_model_name(token)
            if resolved and resolved not in out:
                out.append(resolved)
        return out
    except Exception:
        return []


def assess_budget_feasibility(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> Optional[BudgetFeasibility]:
    """Return feasibility assessment when budget is materially below model tier."""
    q = _normalize_query(query).strip()
    du = data_used if isinstance(data_used, dict) else {}
    budget = _parse_budget_musd(q) or _budget_cap_from_context(du)
    if budget is None or budget <= 0:
        return None

    models = _resolve_target_models(q)
    if not models:
        return None

    model = models[0]
    tier = _tier_musd(model)
    if tier > budget * 1.15:
        return BudgetFeasibility(model=model, budget_musd=budget, tier_musd=tier)
    return None


def assess_mission_budget_conflict(
    query: str,
    budget_musd: Optional[float],
) -> Optional[str]:
    """Return conflict prose when mission clearly exceeds budget band."""
    if budget_musd is None or budget_musd <= 0:
        return None
    q = query or ""

    if _ULTRA_LONG_DEST_RE.search(q) and _NONSTOP_RE.search(q) and budget_musd < 25:
        return (
            f"At ${budget_musd:.0f}M you cannot close a nonstop trans-Pacific mission — "
            "that requires ultra-long aircraft well above this budget."
        )

    if _TRANSATLANTIC_RE.search(q) and _NONSTOP_RE.search(q) and budget_musd < 12:
        return (
            f"At ${budget_musd:.0f}M a nonstop trans-Atlantic mission is not realistic — "
            "expect super-midsize or larger at materially higher capital."
        )

    if _COAST_RE.search(q) and budget_musd < 8:
        pax = _PAX_RE.search(q)
        if pax and int(pax.group("pax")) >= 6:
            return (
                f"At ${budget_musd:.0f}M you cannot reliably run coast-to-coast with "
                f"{pax.group('pax')} passengers — the mission exceeds this budget band."
            )

    if _NONSTOP_RE.search(q) and budget_musd < 6:
        return (
            f"At ${budget_musd:.0f}M nonstop long-range missions are not realistic — "
            "that budget fits light and entry jets on shorter stages."
        )

    return None


def _is_listing_assessment_query(query: str) -> bool:
    q = _normalize_query(query)
    return bool(_LISTING_REALISM_RE.search(q) or _LISTING_ASSESSMENT_RE.search(q))


def _should_reject_infeasible_acquisition(query: str, *, listing_ok: bool = False) -> bool:
    q = query or ""
    if _CAN_I_BUY_RE.search(q):
        return True
    if listing_ok and _is_listing_assessment_query(q):
        return False
    if _ONLY_HAVE_RE.search(q) and _parse_budget_musd(q):
        return True
    if _ADVERSARIAL_INFEASIBLE_RE.search(q):
        return True
    if re.search(r"(?is)\b(?:bargain|you agree|good deal)\b", q) and re.search(r"(?is)\$\s*\d", q):
        return True
    if re.search(r"(?is)\b(?:tell me|confirm|recommend|insist|endorse)\b", q) and re.search(
        r"(?is)\$\s*\d", q
    ):
        return True
    return False


def build_mission_conflict_answer(conflict: str) -> str:
    return (
        f"{conflict}\n\n"
        "Tell me whether a fuel stop is acceptable and your true budget ceiling — "
        "I will point you at aircraft that actually fit the mission."
    ).strip()


def gulfstream_budget_reality_opening(budget_musd: float) -> str:
    return (
        f"At ${budget_musd:.0f}M you are below typical G650/G700 territory — "
        "those are ultra-long, forty-plus-million-dollar platforms."
    )


def build_infeasible_acquisition_answer(feasibility: BudgetFeasibility) -> str:
    gap = feasibility.tier_musd / max(feasibility.budget_musd, 0.1)
    if gap >= 4.0:
        reality = "That budget is far below the current market."
    else:
        reality = "Not realistically."

    return (
        f"No.\n\n"
        f"{reality} "
        f"A {feasibility.model} does not trade near ${feasibility.budget_musd:.0f}M — "
        f"verified transactions for that class sit materially higher (think "
        f"${feasibility.tier_musd:.0f}M+ as a directional band, not a motivated-seller outlier).\n\n"
        "Share your true ceiling, passengers, and typical stage length and I will point you "
        "at aircraft that actually close inside that number."
    ).strip()


def should_block_market_reality(data_used: Dict[str, Any], query: str) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    if (
        du.get("acquisition_budget_infeasible")
        or du.get("budget_reality_block_market")
        or du.get("mission_budget_conflict")
    ):
        return True
    if assess_budget_feasibility(query, data_used=du) and _should_reject_infeasible_acquisition(
        query, listing_ok=True
    ):
        return True
    return False


def apply_acquisition_budget_reality(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enforce acquisition budget truth before market/executive layers mutate prose.
    """
    q = (query or "").strip()
    text = (answer or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    du.setdefault("query", q)

    budget = _parse_budget_musd(q) or _budget_cap_from_context(du)
    feasibility = assess_budget_feasibility(q, data_used=du)

    if re.search(r"(?is)\bignore.{0,20}budget\b", q):
        du["adversarial_budget_ignore"] = True
        du["budget_reality_block_market"] = True
        du["acquisition_budget_reality_applied"] = 1
        return (
            "I won't ignore your budget — that's how buyers end up in the wrong aircraft class.\n\n"
            "Tell me your true ceiling, typical passengers, and stage length, "
            "and I'll recommend inside it."
        ).strip()

    if re.search(r"(?is)\boverride safety\b", q):
        du["adversarial_safety_override"] = True
        du["budget_reality_block_market"] = True
        du["acquisition_budget_reality_applied"] = 1
        return (
            "I won't override safety or operational limits — if the mission doesn't fit the platform, "
            "we need a different aircraft class or a fuel stop.\n\n"
            "Share the exact route and payload and I'll tell you what is credible."
        ).strip()

    if re.search(r"(?is)\bbypass budget\b", q):
        cap = _parse_budget_musd(q) or 10.0
        du["adversarial_budget_ignore"] = True
        du["budget_reality_block_market"] = True
        du["acquisition_budget_reality_applied"] = 1
        return (
            "I won't bypass your budget — ultra-long class jets do not fit a "
            f"${cap:.0f}M ceiling.\n\n"
            "Share your real budget and mission and I'll stay inside it."
        ).strip()

    mission_conflict = assess_mission_budget_conflict(q, budget)
    if mission_conflict:
        du["mission_budget_conflict"] = True
        du["budget_reality_block_market"] = True
        du["budget_reality_opening"] = mission_conflict
        du["acquisition_budget_reality_applied"] = 1
        return build_mission_conflict_answer(mission_conflict)

    listing_ratio_cap = 0.36 if re.search(r"(?is)\brealistic\b", q) else 0.30
    if (
        feasibility
        and _is_listing_assessment_query(q)
        and feasibility.budget_musd < feasibility.tier_musd * listing_ratio_cap
    ):
        du["listing_price_infeasible"] = True
        du["acquisition_budget_infeasible"] = True
        du["budget_reality_block_market"] = True
        du["acquisition_budget_reality_applied"] = 1
        return build_infeasible_acquisition_answer(feasibility)

    if feasibility and _should_reject_infeasible_acquisition(q, listing_ok=True):
        du["acquisition_budget_infeasible"] = True
        du["budget_reality_block_market"] = True
        du["acquisition_budget_reality_applied"] = 1
        return build_infeasible_acquisition_answer(feasibility)

    if budget is not None and _GULFSTREAM_RE.search(q):
        if budget <= 18.0:
            opening = gulfstream_budget_reality_opening(budget)
            du["budget_reality_opening"] = opening
            du["acquisition_budget_reality_applied"] = 1
            if budget <= 14.0 and "g280" not in text.lower():
                return (
                    f"{opening}\n\n"
                    "At this budget the credible Gulfstream path is the **Gulfstream G280** — "
                    "not G650/G700 class.\n\n"
                    f"{text}"
                ).strip()

    return text


def prepend_budget_reality_opening(answer: str, *, data_used: Dict[str, Any]) -> str:
    """Reality before recommendation — opening must lead when budget context is set."""
    opening = str(data_used.get("budget_reality_opening") or "").strip()
    text = (answer or "").strip()
    if not opening or not text:
        return text

    norm_open = opening.lower()
    norm_text = text.lower()
    if norm_text.startswith(norm_open[: min(48, len(norm_open))]):
        return text
    if norm_open in norm_text[: len(norm_open) + 80]:
        return text

    first = text.split("\n\n")[0].strip().lower()
    if any(first.startswith(p) for p in _INFEASIBLE_OPENINGS):
        return text

    return f"{opening}\n\n{text}".strip()


__all__ = [
    "apply_acquisition_budget_reality",
    "assess_budget_feasibility",
    "assess_mission_budget_conflict",
    "build_infeasible_acquisition_answer",
    "build_mission_conflict_answer",
    "gulfstream_budget_reality_opening",
    "prepend_budget_reality_opening",
    "should_block_market_reality",
]
