"""Build structured BrokerDecision from query and pipeline metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.broker_decision.alternative_engine import resolve_alternatives
from services.broker_decision.budget_matcher import match_budget_opportunities
from services.broker_decision.decision_intent_detector import (
    DecisionIntent,
    detect_decision_intent,
    extract_price_context,
)
from services.broker_reasoning.broker_reasoning_layer import infer_buy_fields


@dataclass
class BrokerDecision:
    primary_question: str
    answer_type: str  # yes_no | guidance | opportunities | alternatives | timing
    direct_answer: str
    key_risk: str = ""
    recommended_action: str = ""
    alternatives: List[Dict[str, str]] = field(default_factory=list)
    supporting_points: List[str] = field(default_factory=list)
    decision_intent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_question": self.primary_question,
            "answer_type": self.answer_type,
            "direct_answer": self.direct_answer,
            "key_risk": self.key_risk,
            "recommended_action": self.recommended_action,
            "alternatives": list(self.alternatives),
            "supporting_points": list(self.supporting_points),
            "decision_intent": self.decision_intent,
        }


def _resolve_model(phrase: str) -> Optional[str]:
    from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

    return _resolve_model_name(phrase)


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    return float(_ACQUISITION_TIER_MUSD.get(model, 30.0))


def build_broker_decision(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    raw_answer: str = "",
) -> Optional[BrokerDecision]:
    """Construct a broker decision when the query is acquisition-oriented."""
    q = (query or "").strip()
    if not q:
        return None

    du = data_used or {}
    intent = detect_decision_intent(q, data_used=du)
    if intent == DecisionIntent.NONE:
        return None

    ctx = extract_price_context(q)
    buy_hint = infer_buy_fields(q) or (du.get("broker_reasoning") or {}).get("buy_parse_hint") or {}
    br = du.get("broker_reasoning") or {}
    mission = br.get("mission") or {}

    model = _resolve_model(ctx.get("model_phrase") or buy_hint.get("model") or "")
    cc = du.get("client_context") or du.get("broker_conversation_context") or {}

    budget_m = (
        ctx.get("budget_musd")
        or buy_hint.get("budget_musd")
        or mission.get("acquisition_budget_musd")
    )
    if budget_m is None and isinstance(cc, dict) and cc.get("remembered_budget_musd") is not None:
        budget_m = float(cc["remembered_budget_musd"])
    ask_m = ctx.get("ask_musd")

    if intent == DecisionIntent.REALISTICITY_CHECK:
        return _build_realisticity(q, model, budget_m, du)

    pref_mfrs = (
        list(cc.get("preferred_manufacturers") or [])
        if isinstance(cc, dict)
        else []
    )

    if intent == DecisionIntent.BUDGET_MATCH and budget_m is not None:
        mfr = ctx.get("manufacturer")
        if not mfr and re.search(r"(?is)\bfalcon\b", q):
            mfr = "Dassault"
        if not mfr and re.search(r"(?is)\bchallenger\b", q):
            mfr = "Bombardier"
        if not mfr and re.search(r"(?is)\bcitation\b", q):
            mfr = "Cessna"
        return _build_budget_match(
            q,
            float(budget_m),
            mfr,
            preferred_manufacturers=pref_mfrs,
        )

    if intent == DecisionIntent.ALTERNATIVE_DISCOVERY:
        ref = (
            br.get("intent_expansion", {}).get("reference_model")
            or br.get("alternatives", {}).get("reference_model")
            or model
            or "Citation Longitude"
        )
        return _build_alternatives(q, str(ref), budget_m)

    if intent == DecisionIntent.OVERPAY_CHECK:
        return _build_overpay(q, model, ask_m, raw_answer, du)

    if intent == DecisionIntent.BUY_OR_WAIT:
        return _build_buy_or_wait(q, model)

    if intent == DecisionIntent.STRETCH_BUDGET:
        return _build_stretch(q, model, budget_m)

    if intent == DecisionIntent.GENERAL_ACQUISITION and budget_m:
        return _build_budget_match(
            q,
            float(budget_m),
            ctx.get("manufacturer"),
            preferred_manufacturers=pref_mfrs,
        )

    return None


def _build_realisticity(
    query: str,
    model: Optional[str],
    budget_m: Optional[float],
    du: Dict[str, Any],
) -> BrokerDecision:
    adv = du.get("adversarial") or {}
    infeasible = adv.get("budget_feasibility") == "INFEASIBLE"

    if not model and budget_m:
        low = query.lower()
        if "gulfstream" in low:
            supporting = [
                f"At ${budget_m:.0f}M you are not in Gulfstream large-cabin territory — you are in light or entry super-mid.",
                "G650 and G700 trades are multiples of that budget.",
            ]
            return BrokerDecision(
                primary_question="Can I get a Gulfstream at this budget?",
                answer_type="yes_no",
                direct_answer="No.",
                key_risk="Wasting diligence on mispriced or misrepresented listings.",
                recommended_action="Start with G280-class or older super-mid if Gulfstream branding matters at this cap.",
                supporting_points=supporting,
                decision_intent=DecisionIntent.REALISTICITY_CHECK.value,
            )
        return BrokerDecision(
            primary_question="Can this budget buy the aircraft class requested?",
            answer_type="yes_no",
            direct_answer="No — that budget and aircraft class do not line up.",
            key_risk="Wasting time on listings that cannot close at the stated cap.",
            recommended_action="Reset the search to aircraft that trade inside your budget band.",
            decision_intent=DecisionIntent.REALISTICITY_CHECK.value,
        )

    tier = _tier_musd(model) if model else 50.0
    cap = float(budget_m or 0)

    if infeasible or (model and cap and tier > cap * 1.25):
        direct = "No."
        supporting = [
            f"Current {model} pricing is far above ${cap:.0f}M.",
            "At that budget you are not negotiating a discount — you are in a different aircraft category.",
        ]
        if cap <= 8:
            supporting.append(
                "If that is a hard cap, stop looking at ultra-long jets and focus on super-mid or older large-cabin aircraft."
            )
        elif model and "Gulfstream" in model and cap < 15:
            supporting.append(
                "If Gulfstream is the goal at this budget, focus on G280-class or older large-cabin tails — not G650/G700."
            )
        return BrokerDecision(
            primary_question=f"Can I get a {model} for ${cap:.0f}M?",
            answer_type="yes_no",
            direct_answer=direct,
            key_risk="Chasing impossible listings burns diligence budget and creates false anchors.",
            recommended_action="Reframe the search by budget first, then narrow model.",
            supporting_points=supporting,
            decision_intent=DecisionIntent.REALISTICITY_CHECK.value,
        )

    return BrokerDecision(
        primary_question=f"Is {model} realistic at ${cap:.0f}M?" if cap else f"Is {model} realistic?",
        answer_type="guidance",
        direct_answer="Possibly — but only with the right year, hours, and maintenance history.",
        key_risk="Thin listings at aggressive prices often have deferred maintenance or damage history.",
        recommended_action="Verify airframe hours, engine program status, and logbooks before treating any ask as market.",
        supporting_points=[
            f"{model} typically trades around ${_tier_musd(model):.0f}M+ for late-model examples.",
        ],
        decision_intent=DecisionIntent.REALISTICITY_CHECK.value,
    )


def _build_budget_match(
    query: str,
    budget_m: float,
    manufacturer: Optional[str],
    *,
    preferred_manufacturers: Optional[List[str]] = None,
) -> BrokerDecision:
    mfr = manufacturer
    if not mfr and preferred_manufacturers:
        mfr = preferred_manufacturers[0]
    opps = match_budget_opportunities(budget_m, manufacturer=mfr, query=query)
    alts = [{"model": o.model, "rationale": o.value_reason} for o in opps]
    names = ", ".join(o.model for o in opps[:3]) if opps else "several super-mid and large-cabin options"

    return BrokerDecision(
        primary_question=f"What can I buy for ${budget_m:.0f}M?",
        answer_type="opportunities",
        direct_answer=(
            f"At ${budget_m:.0f}M, I would focus on {names} — not every jet in the catalog, "
            "but the ones that actually trade in your band with credible mission capability."
        ),
        key_risk="Stretching to the top of budget without reserves for paint, engines, or avionics.",
        recommended_action="Pick two candidates and run logbook + market comps before touring.",
        alternatives=alts,
        decision_intent=DecisionIntent.BUDGET_MATCH.value,
    )


def _build_alternatives(
    query: str,
    reference: str,
    budget_m: Optional[float],
) -> BrokerDecision:
    opps = resolve_alternatives(reference, budget_musd=budget_m)
    alts = [{"model": o.model, "rationale": o.rationale} for o in opps]
    names = ", ".join(o.model for o in opps[:3]) if opps else "lower-tier peers in the same mission band"

    return BrokerDecision(
        primary_question=f"What is like a {reference} but cheaper?",
        answer_type="alternatives",
        direct_answer=(
            f"If you want {reference} capability at lower acquisition cost, start with {names}."
        ),
        key_risk="Stepping down in cabin or range without adjusting mission expectations.",
        recommended_action="Match the alternative to your longest regular leg and passenger load.",
        alternatives=alts,
        decision_intent=DecisionIntent.ALTERNATIVE_DISCOVERY.value,
    )


def _build_overpay(
    query: str,
    model: Optional[str],
    ask_m: Optional[float],
    raw_answer: str,
    du: Dict[str, Any],
) -> BrokerDecision:
    dk = du.get("deal_killer") or {}
    verdict = str(dk.get("verdict") or "").upper()
    if not model:
        from services.consultant.recommendation_engine import detect_models_from_text

        detected = detect_models_from_text(query)
        model = detected[0] if detected else None

    tier = _tier_musd(model) if model else None
    ask = ask_m

    if ask is None and raw_answer:
        m = re.search(r"\$(\d+(?:\.\d+)?)\s*M", raw_answer, re.I)
        if m:
            ask = float(m.group(1))

    if model and ask is not None and tier is not None:
        if ask < tier * 0.55:
            direct = (
                f"A {model} at ${ask:.1f}M would be unusually low — I would verify hours, "
                "maintenance status, and damage history before treating it as legitimate."
            )
            risk = "Too-good-to-be-true pricing often reflects deferred maintenance or title issues."
        elif ask > tier * 1.35:
            direct = f"At ${ask:.1f}M, that {model} ask looks high relative to typical ${tier:.0f}M+ trading levels."
            risk = "Overpaying versus band leaves little margin on resale."
        else:
            direct = f"At ${ask:.1f}M, a {model} can be credible depending on year and program status."
            risk = "Price is only half the story — logs and engine programs drive the real risk."

        if verdict in ("GOOD DEAL", "OVERPRICED", "FAIR DEAL"):
            direct = f"{direct} Market read from synced data: {verdict.replace('_', ' ').title()}."

        return BrokerDecision(
            primary_question=f"Is ${ask:.1f}M realistic for a {model}?",
            answer_type="guidance",
            direct_answer=direct,
            key_risk=risk,
            recommended_action="Get a spec sheet, logbooks, and a broker market pull on comparable tails before LOI.",
            decision_intent=DecisionIntent.OVERPAY_CHECK.value,
        )

    return BrokerDecision(
        primary_question="Is this listing price realistic?",
        answer_type="guidance",
        direct_answer="I need the specific model and year to judge whether the ask is market or noise.",
        recommended_action="Send the model, year, and asking price and I will give you a direct read.",
        decision_intent=DecisionIntent.OVERPAY_CHECK.value,
    )


def _build_buy_or_wait(query: str, model: Optional[str]) -> BrokerDecision:
    rising = "rising" in query.lower() or "trend" in query.lower()
    model_bit = f" on a {model}" if model else ""

    if rising:
        direct = (
            f"I would not rush a buy solely because prices are moving{model_bit} — "
            "confirm whether the move is inventory-driven or true appreciation on the tails you care about."
        )
    else:
        direct = (
            f"Buy when you have a specific tail vetted{model_bit}; "
            "wait when you only have a category preference and no logbooks in hand."
        )

    return BrokerDecision(
        primary_question="Should I buy now or wait?",
        answer_type="timing",
        direct_answer=direct,
        key_risk="Timing the market matters less than buying the wrong aircraft or the wrong maintenance profile.",
        recommended_action="If a credible tail appears, run diligence — do not wait for a perfect macro signal.",
        decision_intent=DecisionIntent.BUY_OR_WAIT.value,
    )


def _build_stretch(
    query: str,
    model: Optional[str],
    budget_m: Optional[float],
) -> BrokerDecision:
    target = model or "that aircraft"
    return BrokerDecision(
        primary_question=f"Should I stretch budget for a {target}?",
        answer_type="guidance",
        direct_answer=(
            f"Stretch only if the {target} solves a mission you cannot cover with the next tier down — "
            "not because of cabin prestige."
        ),
        key_risk="Thin reserves after purchase — engines, paint, and avionics surprises.",
        recommended_action="Model operating cost and a 12-month cash reserve before stretching.",
        supporting_points=[
            "Compare one tier down on your longest regular route before committing extra capital.",
        ],
        decision_intent=DecisionIntent.STRETCH_BUDGET.value,
    )


__all__ = ["BrokerDecision", "build_broker_decision"]
