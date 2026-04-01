"""
Fine-grained aviation intents for Ask Consultant (rules + optional LLM).

Resolves overlaps explicitly, e.g. **mission** vs **specs** when both mention *range*.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.consultant_market_lookup import (
    wants_consultant_purchase_market_context,
    wants_consultant_strict_internal_market_sql,
)
from rag.intent.schemas import AviationIntent

logger = logging.getLogger(__name__)


@dataclass
class AviationIntentResult:
    intent: AviationIntent
    confidence: float
    source: str  # "rule" | "llm"
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"intent": self.intent.value, "confidence": round(float(self.confidence), 4)}


def _blob(query: str, history: Optional[List[Dict[str, str]]]) -> str:
    parts: List[str] = []
    if history:
        for h in history[-8:]:
            if (h.get("role") or "").strip().lower() == "user":
                c = (h.get("content") or "").strip()
                if c:
                    parts.append(c)
    parts.append(query or "")
    return " ".join(parts).strip().lower()


# Legal / registrant (use registration_lookup — ties to FAA path when US tail present)
_OWNERSHIP_RE = re.compile(
    r"\b("
    r"who\s+owns|whose\s+(jet|aircraft|plane)|registrant|registration\s+holder|"
    r"legal\s+owner|title\s+holder|faa\s+registrant|u\.?s\.?\s+registrant|"
    r"registered\s+owner|certificate\s+holder|llc\s+owner|trustee"
    r")\b",
    re.I,
)

# US N-number: N + digit + suffix (FAA civil; e.g. N123AB, N1234, N98765). Digit avoids matching "nonstop".
_US_N_TAIL = re.compile(r"\b[Nn]\d[A-Z0-9]{0,4}\b", re.I)

# Non-US hyphenated civil marks (e.g. G-ABCD, F-XXXX).
# Reject short **numeric-only** suffixes (e.g. PC-12, CL-604 series) that are usually model codes.
_ICAO_TAIL = re.compile(r"\b[A-Z]{1,2}-[A-Z0-9]{2,5}\b", re.I)

# Serial / MSN-style tokens (hyphenated OEM + shorthand)
_SERIAL_PHRASE = re.compile(
    r"\b(?:msn|m\.?s\.?n\.?|s/?\s*n|serial(?:\s+no\.?|\s+number)?)\b",
    re.I,
)
# Hyphenated OEM serials / type-series numbers (not bare long digits — too noisy)
_SERIAL_TOKEN = re.compile(
    r"(?:\d{3,4}-[A-Z0-9]{2,6}|[A-Z]{1,3}-\d{3,5})",
    re.I,
)

_MISSION_RE = re.compile(
    r"\b("
    r"fly\s+(to|from|non[-\s]?stop)|flight\s+(to|from|plan)|"
    r"trip\s+to|mission\b|missions\b|route\b|routes\b|"
    r"cross(?:ing)?\s+(the\s+)?(atlantic|pacific|ocean|pond)|"
    r"trans(atlantic|pacific)|can\s+(it|this|the\s+\w+|a\s+\w+)\s+(reach|make|fly)|"
    r"make\s+it\s+to|from\s+.{3,40}\s+to\s+.{3,40}|"
    r"(reach|connect)\s+.{2,25}\s+(non[-\s]?stop|direct)?"
    r")\b",
    re.I,
)

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|v\.|or\s+better|which\s+(is\s+)?(better|faster|cheaper|longer)|"
    r"between\s+the\s+|between\s+\w.+\s+and\s+\w)"
    r"\b",
    re.I,
)

_FOR_SALE_RE = re.compile(
    r"\b(for\s+sale|on\s+the\s+market|listing|listings|buy\s+(this|a|an)\s|"
    r"purchase|available\s+(to\s+)?buy|still\s+for\s+sale)\b",
    re.I,
)

_OPERATOR_RE = re.compile(
    r"\b(who\s+operates|current\s+operator|operator\s+of|"
    r"charter\s+operator|managed\s+by|flown\s+by|"
    r"fleet\s+of|aoc|air\s+operator)\b",
    re.I,
)

_SPECS_RE = re.compile(
    r"\b("
    r"cruise\s+(speed|mach|altitude)|fuel\s+burn|sfc\b|"
    r"passengers\b|pax\b|seats\b|"
    r"runway\b|takeoff\s+(roll|distance)|landing\s+distance|balanced\s+field|"
    r"mtow|m(?:ax)?\s*takeoff|mlw|"
    r"ceiling|service\s+ceiling|max\s+altitude|"
    r"operating\s+empty|oew\b|"
    r"specifications?|specs\b|performance\s+(figures?|data)|"
    r"engine\s+(type|count|thrust)|"
    r"cabin\s+(height|width|length|volume)"
    r")\b",
    re.I,
)


def _has_registration_mark(text: str) -> bool:
    if _US_N_TAIL.search(text):
        return True
    m = _ICAO_TAIL.search(text)
    if not m:
        return False
    suf = m.group(0).split("-", 1)[1]
    if suf.isdigit() and len(suf) <= 3:
        return False
    return True


def _has_serial_signal(text: str) -> bool:
    if _SERIAL_PHRASE.search(text):
        return True
    if _SERIAL_TOKEN.search(text):
        return True
    if re.search(r"\b(\d{5,7})\b", text) and _SERIAL_PHRASE.search(text):
        return True
    return False


def _looks_like_model_comparison(text: str) -> bool:
    if _COMPARE_RE.search(text):
        return True
    # Two manufacturer / family tokens (lightweight)
    families = (
        "challenger",
        "citation",
        "falcon",
        "gulfstream",
        "global",
        "phenom",
        "praetor",
        "learjet",
        "hawker",
        "pilatus",
        "king air",
        "pc-12",
        "longitude",
        "latitude",
        "sovereign",
        "ultra",
    )
    hits = sum(1 for f in families if f in text)
    if hits >= 2:
        return True
    return False


def classify_aviation_intent_detailed(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AviationIntentResult:
    """
    Rule-based aviation intent. Order encodes priority / overlap resolution.
    """
    b = _blob(query, history)
    if not b:
        return AviationIntentResult(
            intent=AviationIntent.GENERAL_QUESTION,
            confidence=0.25,
            source="rule",
            notes="empty",
        )

    if _OWNERSHIP_RE.search(b):
        return AviationIntentResult(
            intent=AviationIntent.REGISTRATION_LOOKUP,
            confidence=0.92,
            source="rule",
            notes="ownership_registrant",
        )

    if _has_registration_mark(b):
        return AviationIntentResult(
            intent=AviationIntent.REGISTRATION_LOOKUP,
            confidence=0.86,
            source="rule",
            notes="registration_pattern",
        )

    if _has_serial_signal(b):
        return AviationIntentResult(
            intent=AviationIntent.SERIAL_LOOKUP,
            confidence=0.84,
            source="rule",
            notes="serial_pattern",
        )

    if _looks_like_model_comparison(b):
        return AviationIntentResult(
            intent=AviationIntent.AIRCRAFT_COMPARISON,
            confidence=0.8,
            source="rule",
            notes="multi_model_or_compare",
        )

    if _MISSION_RE.search(b):
        return AviationIntentResult(
            intent=AviationIntent.MISSION_FEASIBILITY,
            confidence=0.82,
            source="rule",
            notes="route_mission_reach",
        )

    if wants_consultant_purchase_market_context(query, history) or wants_consultant_strict_internal_market_sql(
        query, history
    ):
        if _FOR_SALE_RE.search(b) or re.search(
            r"\b(buy|purchase|for\s+sale|listing|available)\b", b
        ):
            return AviationIntentResult(
                intent=AviationIntent.AIRCRAFT_FOR_SALE,
                confidence=0.81,
                source="rule",
                notes="market_transaction",
            )
        return AviationIntentResult(
            intent=AviationIntent.MARKET_PRICE,
            confidence=0.8,
            source="rule",
            notes="market_price",
        )

    if _OPERATOR_RE.search(b):
        return AviationIntentResult(
            intent=AviationIntent.OPERATOR_LOOKUP,
            confidence=0.78,
            source="rule",
            notes="operator",
        )

    if _SPECS_RE.search(b) or re.search(r"\brange\b", b):
        # Plain "range" without mission phrasing → specs / type capability
        return AviationIntentResult(
            intent=AviationIntent.AIRCRAFT_SPECS,
            confidence=0.76 if _SPECS_RE.search(b) else 0.62,
            source="rule",
            notes="performance_specs" if _SPECS_RE.search(b) else "range_ambiguous_specs",
        )

    if re.search(
        r"\b(what\s+is\s+a|how\s+does|explain|define|difference\s+between|"
        r"types\s+of|categories|regulations?|far\b|easa|certification|training)\b",
        b,
    ):
        return AviationIntentResult(
            intent=AviationIntent.GENERAL_QUESTION,
            confidence=0.72,
            source="rule",
            notes="general_concept",
        )

    if (os.getenv("CONSULTANT_INTENT_LLM") or "").strip().lower() in ("1", "true", "yes"):
        llm = _classify_aviation_intent_llm(query, history)
        if llm is not None:
            return llm

    if _US_N_TAIL.search(b) or _SERIAL_TOKEN.search(b):
        return AviationIntentResult(
            intent=AviationIntent.SERIAL_LOOKUP,
            confidence=0.55,
            source="rule",
            notes="weak_tail_serial_token",
        )

    return AviationIntentResult(
        intent=AviationIntent.GENERAL_QUESTION,
        confidence=0.45,
        source="rule",
        notes="fallback",
    )


def classify_aviation_intent_json(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Public JSON shape: ``{"intent": str, "confidence": float}``."""
    return classify_aviation_intent_detailed(query, history).to_json()


def aviation_to_consultant_coarse(av: AviationIntent):
    """Map fine aviation intent to coarse :class:`~rag.intent.schemas.ConsultantIntent` (ranking / legacy)."""
    from rag.intent.schemas import ConsultantIntent

    if av == AviationIntent.REGISTRATION_LOOKUP:
        return ConsultantIntent.REGISTRATION_LOOKUP
    if av in (
        AviationIntent.SERIAL_LOOKUP,
        AviationIntent.OPERATOR_LOOKUP,
    ):
        return ConsultantIntent.AIRCRAFT_IDENTITY
    if av in (AviationIntent.AIRCRAFT_FOR_SALE, AviationIntent.MARKET_PRICE):
        return ConsultantIntent.MARKET_PRICING
    if av in (
        AviationIntent.AIRCRAFT_SPECS,
        AviationIntent.MISSION_FEASIBILITY,
        AviationIntent.AIRCRAFT_COMPARISON,
    ):
        return ConsultantIntent.TECHNICAL_SPEC
    return ConsultantIntent.GENERAL_AVIATION


def _classify_aviation_intent_llm(
    query: str,
    history: Optional[List[Dict[str, str]]],
) -> Optional[AviationIntentResult]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("CONSULTANT_INTENT_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
    if not api_key or not (query or "").strip():
        return None
    try:
        import json

        import openai

        hist = ""
        if history:
            for h in history[-6:]:
                role = (h.get("role") or "user").strip().lower()
                c = (h.get("content") or "").strip()
                if c and role in ("user", "assistant"):
                    hist += f"{role}: {c}\n"

        allowed = "|".join(m.value for m in AviationIntent)
        sys = f"""Classify the user's latest message for an aircraft broker assistant (Ask Consultant).
Return JSON only: {{"intent": "<one of {allowed}>", "confidence": <0.0-1.0>}}

Definitions:
- aircraft_comparison: comparing two or more aircraft models or families
- aircraft_specs: performance, specs, cruise, fuel, passengers, runway, MTOW (not route feasibility)
- mission_feasibility: can it reach X, routes, fly cross-ocean, trip missions
- registration_lookup: tail / N-number / legal registration or who owns (registrant)
- serial_lookup: MSN / manufacturer serial identity for an airframe
- aircraft_for_sale: listings, buy now, for sale, availability to purchase
- market_price: value, ask, comps, worth — pricing without necessarily buying
- operator_lookup: who operates, charter operator, fleet, AOC (not legal U.S. registrant)
- general_question: broad education / concepts / unclear"""

        user = f"Conversation:\n{hist}\nLatest:\n{query.strip()}"
        client = openai.OpenAI(api_key=api_key, timeout=12.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=96,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        val = (data.get("intent") or "").strip().lower()
        conf = data.get("confidence")
        try:
            cfi = float(conf) if conf is not None else 0.88
        except (TypeError, ValueError):
            cfi = 0.88
        cfi = max(0.0, min(1.0, cfi))
        for m in AviationIntent:
            if m.value == val:
                return AviationIntentResult(intent=m, confidence=cfi, source="llm", notes="llm_json")
    except Exception as e:
        logger.debug("aviation intent LLM skipped: %s", e)
    return None
