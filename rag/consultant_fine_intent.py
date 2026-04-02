"""
Fine-grained LLM intent for Ask Consultant + tool routing.

Registry (FAA) SQL runs only for ownership-style questions **and** valid tail patterns.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsultantFineIntent(str, Enum):
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    AVIATION_MISSION = "aviation_mission"
    AIRCRAFT_SPECS = "aircraft_specs"
    AIRCRAFT_COMPARISON = "aircraft_comparison"
    OWNERSHIP_LOOKUP = "ownership_lookup"
    MARKET_QUESTION = "market_question"
    AIRCRAFT_RECOMMENDATION = "aircraft_recommendation"
    GENERAL_QUESTION = "general_question"


@dataclass
class ConsultantFineIntentResult:
    intent: ConsultantFineIntent
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsultantToolRouterState:
    registry_sql: bool
    pinecone_filter: Optional[Dict[str, Any]]
    mission_reasoning_hint: str
    aviation_engines_block: str = ""


def fine_intent_confidence_threshold() -> float:
    try:
        v = float((os.getenv("CONSULTANT_AVIATION_INTENT_MIN_CONFIDENCE") or "0.6").strip())
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, v))


def llm_fine_intent_disabled() -> bool:
    return (os.getenv("CONSULTANT_LLM_FINE_INTENT_DISABLED") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _history_blob(history: Optional[List[Dict[str, str]]], max_messages: int = 10) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for h in history[-max_messages:]:
        role = (h.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        c = (h.get("content") or "").strip()
        if c:
            lines.append(f"{role}: {c}")
    return "\n".join(lines)


_FINE_SYSTEM = """You classify the **latest user message** for **HyeAero.AI** (Hye Aero's aviation intelligence assistant).

Return JSON only:
{
  "intent": "<one of below>",
  "confidence": <0.0-1.0>,
  "entities": {
    "tails": ["N123AB"],
    "models": ["Challenger 601"],
    "icaos": ["KJFK","EGLL"],
    "passengers": null,
    "budget_usd": null,
    "route_description": null
  }
}

**intent values:**
- **greeting** — hi/hello/yo/sup only; no aviation question.
- **small_talk** — thanks, ok, casual, light math, pleasantries.
- **aviation_mission** — can it fly X to Y, range for a route, fuel stops, nonstop feasibility, ocean crossing.
- **aircraft_specs** — engines, pax, speed, range of a **model** (601/604/350 are model numbers, not registrations).
- **aircraft_comparison** — vs / compared to / which is better (two or more aircraft).
- **ownership_lookup** — who owns, registrant, tail lookup when a **registration** is implied (Nxxxxx, G-xxxxx, VH-xxx…).
- **market_question** — price, value, comps, listings, for sale, ask on market.
- **aircraft_recommendation** — mission and/or budget; what jet to buy / options.
- **general_question** — broad aviation or non-aviation knowledge without DB-heavy ownership.

**confidence:** High (0.85+) when clear. Low (<0.6) if ambiguous.

**Rules:**
- "Challenger **601**" → **aircraft_specs** or **aircraft_comparison**, **not** ownership_lookup.
- **greeting** if the message is **only** hello/hi with no aircraft content.
- Use **aircraft_comparison** when "vs" compares two aircraft."""


def _normalize_intent_str(raw: str) -> ConsultantFineIntent:
    t = (raw or "").strip().lower().replace("-", "_")
    aliases = {
        "comparison": ConsultantFineIntent.AIRCRAFT_COMPARISON,
        "specs": ConsultantFineIntent.AIRCRAFT_SPECS,
        "spec": ConsultantFineIntent.AIRCRAFT_SPECS,
        "mission": ConsultantFineIntent.AVIATION_MISSION,
        "ownership": ConsultantFineIntent.OWNERSHIP_LOOKUP,
        "registry": ConsultantFineIntent.OWNERSHIP_LOOKUP,
        "market": ConsultantFineIntent.MARKET_QUESTION,
        "recommendation": ConsultantFineIntent.AIRCRAFT_RECOMMENDATION,
        "general": ConsultantFineIntent.GENERAL_QUESTION,
    }
    if t in aliases:
        return aliases[t]
    try:
        return ConsultantFineIntent(t)
    except ValueError:
        return ConsultantFineIntent.AIRCRAFT_SPECS


def heuristic_fine_intent(
    query: str,
    strict_tails: List[str],
) -> ConsultantFineIntentResult:
    """Deterministic fallback when LLM is disabled or fails."""
    q = (query or "").strip()
    ql = q.lower()
    ent: Dict[str, Any] = {"tails": list(strict_tails)}
    if re.match(r"^\s*(hi|hello|hey|yo|sup)\b[\s!?.…]*$", ql):
        return ConsultantFineIntentResult(ConsultantFineIntent.GREETING, 0.9, ent)
    if len(ql) <= 4 and ql in ("hi", "yo", "sup", "hey"):
        return ConsultantFineIntentResult(ConsultantFineIntent.GREETING, 0.88, ent)
    if re.search(r"\bvs\.?\b|\bversus\b", ql):
        return ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_COMPARISON, 0.82, ent)
    if strict_tails:
        if re.search(
            r"\b(range|spec|specs|fuel|speed|engine|passenger|cabin|how\s+fast|ceiling|mtow)\b",
            ql,
        ):
            return ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_SPECS, 0.8, ent)
        if re.search(r"\b(who\s+owns|ownership|registrant)\b", ql):
            return ConsultantFineIntentResult(ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.88, ent)
        return ConsultantFineIntentResult(ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.82, ent)
    if any(x in ql for x in ("for sale", "asking", "listing", "market value", "comps", "worth")):
        return ConsultantFineIntentResult(ConsultantFineIntent.MARKET_QUESTION, 0.78, ent)
    if any(
        x in ql
        for x in (
            "nonstop",
            "cross",
            "atlantic",
            "fuel stop",
            "can it fly",
            "range from",
            "mission",
        )
    ):
        return ConsultantFineIntentResult(ConsultantFineIntent.AVIATION_MISSION, 0.8, ent)
    if any(x in ql for x in ("recommend", "which jet", "what aircraft should", "budget")):
        return ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_RECOMMENDATION, 0.78, ent)
    return ConsultantFineIntentResult(ConsultantFineIntent.AIRCRAFT_SPECS, 0.72, ent)


def apply_fine_intent_heuristics(
    result: ConsultantFineIntentResult,
    query: str,
    strict_tails: List[str],
) -> ConsultantFineIntentResult:
    """Override LLM when ``vs`` or tail evidence contradicts weak labels."""
    ql = (query or "").lower()
    if re.search(r"\bvs\.?\b|\bversus\b", ql):
        if result.intent != ConsultantFineIntent.AIRCRAFT_COMPARISON:
            result = ConsultantFineIntentResult(
                ConsultantFineIntent.AIRCRAFT_COMPARISON,
                max(result.confidence, 0.85),
                dict(result.entities),
            )
    if result.intent == ConsultantFineIntent.OWNERSHIP_LOOKUP and not strict_tails:
        result = ConsultantFineIntentResult(
            ConsultantFineIntent.AIRCRAFT_SPECS,
            min(result.confidence, 0.55),
            dict(result.entities),
        )
    return result


def classify_consultant_fine_intent_llm(
    query: str,
    history: Optional[List[Dict[str, str]]],
    *,
    api_key: str,
    model: str,
    timeout: float = 16.0,
) -> ConsultantFineIntentResult:
    if not (api_key or "").strip():
        st = __import__("rag.aviation_tail", fromlist=["find_strict_tail_candidates"]).find_strict_tail_candidates(
            query, history
        )
        return apply_fine_intent_heuristics(heuristic_fine_intent(query, st), query, st)
    try:
        import openai

        blob = _history_blob(history)
        user_msg = f"Conversation (oldest last):\n{blob}\n\n---\n\nLatest user message:\n{(query or '').strip()}"
        client = openai.OpenAI(api_key=api_key, timeout=max(6.0, min(35.0, timeout)))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FINE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=220,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        intent = _normalize_intent_str(str(data.get("intent") or ""))
        try:
            conf = float(data.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.75
        conf = max(0.0, min(1.0, conf))
        ent = data.get("entities") if isinstance(data.get("entities"), dict) else {}
        from rag.aviation_tail import find_strict_tail_candidates

        st = find_strict_tail_candidates(query, history)
        merged = dict(ent)
        merged.setdefault("tails", [])
        if isinstance(merged["tails"], list):
            for t in st:
                if t not in merged["tails"]:
                    merged["tails"].append(t)
        out = ConsultantFineIntentResult(intent, conf, merged)
        return apply_fine_intent_heuristics(out, query, st)
    except Exception as e:
        logger.warning("consultant fine intent LLM failed: %s", e)
        from rag.aviation_tail import find_strict_tail_candidates

        st = find_strict_tail_candidates(query, history)
        return apply_fine_intent_heuristics(heuristic_fine_intent(query, st), query, st)


def build_consultant_tool_router(
    fine: ConsultantFineIntentResult,
    query: str,
    strict_tails: List[str],
) -> ConsultantToolRouterState:
    """
    Strict tool routing:
    - FAA / registry (**registration lookup**) SQL **only** when a **valid civil registration**
      is present (see :func:`rag.aviation_tail.find_strict_tail_candidates`).
    - OEM / model numbers like ``601``, ``604``, or ``550-0123`` **do not** enable registry SQL.
    """
    from rag.aviation_engines import build_aviation_engines_block
    from rag.mission_reasoning import build_mission_reasoning_hint
    from rag.pinecone_intent_filter import pinecone_filter_for_fine_intent

    registry = (
        fine.intent == ConsultantFineIntent.OWNERSHIP_LOOKUP and len(strict_tails) > 0
    )
    pf = pinecone_filter_for_fine_intent(fine.intent.value)
    hint = build_mission_reasoning_hint(query, fine.intent.value, fine.entities)
    engines = build_aviation_engines_block(fine, query)
    return ConsultantToolRouterState(
        registry_sql=registry,
        pinecone_filter=pf,
        mission_reasoning_hint=hint,
        aviation_engines_block=engines,
    )


def map_fine_intent_to_legacy_classification(
    fine: ConsultantFineIntent,
):
    """Map to :class:`~rag.intent.schemas.IntentClassification` for context policy compatibility."""
    from rag.intent.schemas import (
        AviationIntent,
        ConsultantIntent,
        ConsultantQueryKind,
        IntentClassification,
        query_kind_from_aviation_intent,
    )

    if fine == ConsultantFineIntent.OWNERSHIP_LOOKUP:
        av = AviationIntent.REGISTRATION_LOOKUP
        pr = ConsultantIntent.REGISTRATION_LOOKUP
    elif fine == ConsultantFineIntent.MARKET_QUESTION:
        av = AviationIntent.MARKET_PRICE
        pr = ConsultantIntent.MARKET_PRICING
    elif fine == ConsultantFineIntent.AIRCRAFT_COMPARISON:
        av = AviationIntent.AIRCRAFT_COMPARISON
        pr = ConsultantIntent.TECHNICAL_SPEC
    elif fine == ConsultantFineIntent.AVIATION_MISSION:
        av = AviationIntent.MISSION_FEASIBILITY
        pr = ConsultantIntent.TECHNICAL_SPEC
    elif fine == ConsultantFineIntent.AIRCRAFT_RECOMMENDATION:
        av = AviationIntent.MISSION_FEASIBILITY
        pr = ConsultantIntent.TECHNICAL_SPEC
    elif fine == ConsultantFineIntent.GENERAL_QUESTION:
        av = AviationIntent.GENERAL_QUESTION
        pr = ConsultantIntent.GENERAL_AVIATION
    else:
        av = AviationIntent.AIRCRAFT_SPECS
        pr = ConsultantIntent.TECHNICAL_SPEC

    qk_map = {
        ConsultantFineIntent.GREETING: ConsultantQueryKind.GENERAL,
        ConsultantFineIntent.SMALL_TALK: ConsultantQueryKind.GENERAL,
        ConsultantFineIntent.AVIATION_MISSION: ConsultantQueryKind.MISSION,
        ConsultantFineIntent.AIRCRAFT_SPECS: ConsultantQueryKind.SPEC,
        ConsultantFineIntent.AIRCRAFT_COMPARISON: ConsultantQueryKind.COMPARISON,
        ConsultantFineIntent.OWNERSHIP_LOOKUP: ConsultantQueryKind.OWNERSHIP,
        ConsultantFineIntent.MARKET_QUESTION: ConsultantQueryKind.MARKET,
        ConsultantFineIntent.AIRCRAFT_RECOMMENDATION: ConsultantQueryKind.MISSION,
        ConsultantFineIntent.GENERAL_QUESTION: ConsultantQueryKind.GENERAL,
    }
    return IntentClassification(
        primary=pr,
        source="fine_intent",
        confidence=1.0,
        notes="",
        aviation_intent=av,
        query_kind=qk_map.get(fine, ConsultantQueryKind.GENERAL),
    )


def should_run_aviation_tools(fine: ConsultantFineIntentResult, threshold: float) -> bool:
    if fine.intent in (ConsultantFineIntent.GREETING, ConsultantFineIntent.SMALL_TALK):
        return False
    if fine.confidence < threshold:
        return False
    return True


def is_conversational_fine_intent(fine: ConsultantFineIntentResult) -> bool:
    return fine.intent in (ConsultantFineIntent.GREETING, ConsultantFineIntent.SMALL_TALK)
