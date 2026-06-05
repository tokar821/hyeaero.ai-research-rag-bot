"""
Broker-grade response normalization — final output structure only.

Does not alter routing, classification, dispatch, ranking, or LLM generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_VERDICT_CANONICAL = frozenset(
    {
        "GOOD FIT",
        "CONDITIONAL FIT",
        "NOT A FIT",
        "GOOD DEAL",
        "OVERPRICED",
        "RISKY",
        "VIABLE WITH COMPROMISES",
    }
)

_VERDICT_ALIASES: Dict[str, str] = {
    "HIGH RISK": "RISKY",
    "DO NOT BUY": "NOT A FIT",
    "FAIR DEAL": "CONDITIONAL FIT",
    "VIABLE WITH COMPROMISES": "VIABLE WITH COMPROMISES",
}

_VERDICT_RE = re.compile(
    r"(?:✅|⚠️|❌)?\s*"
    r"(GOOD\s+FIT|CONDITIONAL\s+FIT|NOT\s+A\s+FIT|GOOD\s+DEAL|OVERPRICED|RISKY|"
    r"VIABLE\s+WITH\s+COMPROMISES|FAIR\s+DEAL|HIGH\s+RISK|DO\s+NOT\s+BUY)",
    re.I,
)

_KERNEL_LEAK_RE = re.compile(
    r"(?im)^.*\b(?:operational\s+synthesis|approved\s+shortlist|mission\s+authority\s+kernel|"
    r"ranked\s+aircraft\s+shortlist)\b.*$",
)

_COMPROMISES_RE = re.compile(
    r"(?im)^\s*[*\-]?\s*VIABLE\s+WITH\s+COMPROMISES:?\s*(.*)$"
)

_SECTION_HEADERS = {
    "overview": re.compile(r"(?im)^(?:overview|summary|mission interpretation)\s*:?\s*$"),
    "analysis": re.compile(r"(?im)^(?:analysis|market reality|constraint summary|verified catalog comparison)\s*:?\s*$"),
    "recommendation": re.compile(r"(?im)^(?:recommendation|final verdict|verdict|alternatives)\s*:?\s*$"),
    "risks": re.compile(r"(?im)^(?:risks|red flags|concerns)\s*:?\s*$"),
}


@dataclass
class NormalizedConsultantResponse:
    intent_type: str
    aircraft: List[str] = field(default_factory=list)
    primary_recommendation: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    financial_summary: Dict[str, Any] = field(default_factory=dict)
    mission_fit: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    structured_sections: Dict[str, Any] = field(default_factory=dict)
    data_sources: Dict[str, bool] = field(default_factory=dict)
    answer_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "aircraft": list(self.aircraft),
            "primary_recommendation": dict(self.primary_recommendation),
            "alternatives": list(self.alternatives),
            "financial_summary": dict(self.financial_summary),
            "mission_fit": dict(self.mission_fit),
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 3),
            "notes": list(self.notes),
            "structured_sections": {
                "overview": str((self.structured_sections or {}).get("overview") or ""),
                "analysis": str((self.structured_sections or {}).get("analysis") or ""),
                "recommendation": str((self.structured_sections or {}).get("recommendation") or ""),
                "risks": list((self.structured_sections or {}).get("risks") or []),
            },
            "data_sources": dict(self.data_sources),
        }

    def render_answer(self) -> str:
        if self.answer_text:
            return self.answer_text.strip()
        return _render_structured_answer(self)


def normalize_consultant_response(
    response: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> NormalizedConsultantResponse:
    """Map a raw consultant payload into the broker-grade normalized schema."""
    ctx = context if isinstance(context, dict) else {}
    answer = str((response or {}).get("answer") or "").strip()
    data_used = dict((response or {}).get("data_used") or {})
    query = str(ctx.get("query") or data_used.get("query") or "").strip()

    intent_type = _detect_intent_type(data_used, answer, query)
    cleaned = _sanitize_answer_for_intent(answer, intent_type)

    aircraft = _extract_aircraft(data_used, cleaned, intent_type)
    verdict, confidence = _extract_verdict(data_used, cleaned, intent_type)
    primary, alternatives = _extract_recommendations(data_used, cleaned, intent_type, aircraft)
    financial = _extract_financial_summary(data_used, cleaned, intent_type)
    mission_fit = _extract_mission_fit(data_used, intent_type)
    notes = _extract_notes(data_used, cleaned, intent_type)
    sections = _build_structured_sections(
        cleaned,
        intent_type=intent_type,
        data_used=data_used,
        aircraft=aircraft,
        verdict=verdict,
        alternatives=alternatives,
        financial=financial,
        notes=notes,
    )
    data_sources = _extract_data_sources(data_used, response)

    normalized = NormalizedConsultantResponse(
        intent_type=intent_type,
        aircraft=aircraft,
        primary_recommendation=primary,
        alternatives=alternatives,
        financial_summary=financial,
        mission_fit=mission_fit,
        verdict=verdict,
        confidence=confidence,
        notes=notes,
        structured_sections=sections,
        data_sources=data_sources,
    )
    try:
        from services.broker_execution.output_governance import is_llm_primary_output

        if is_llm_primary_output(data_used):
            normalized.answer_text = cleaned
            return normalized
    except Exception:
        pass
    normalized.answer_text = normalized.render_answer()
    return normalized


def apply_consultant_response_normalization(
    response: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return response dict with normalized answer and schema attached to data_used."""
    if not isinstance(response, dict):
        return response

    normalized = normalize_consultant_response(response, context)
    out = dict(response)
    du = dict(out.get("data_used") or {})
    query = str((context or {}).get("query") or out.get("query") or "").strip()

    try:
        from services.broker_execution.output_governance import apply_governed_client_answer

        broker_answer = apply_governed_client_answer(
            normalized.answer_text,
            query=query,
            data_used=du,
        )
    except Exception:
        broker_answer = normalized.answer_text

    out["answer"] = broker_answer
    du["normalized_response"] = normalized.to_dict()
    du["response_normalization_applied"] = 1
    du["broker_decision_synthesis_applied"] = du.get("broker_decision_synthesis_applied", 0)
    du["broker_conversation_layer_applied"] = 1
    out["data_used"] = du

    try:
        from services.response.ui_render_contract import apply_ui_render_contract_to_response

        out = apply_ui_render_contract_to_response(out, context)
    except Exception:
        pass

    return out


def _detect_intent_type(data_used: Dict[str, Any], answer: str, query: str) -> str:
    dispatch = str(data_used.get("authority_dispatch_kind") or "").strip().lower()
    if dispatch in ("comparison", "alternative", "buy_decision"):
        return dispatch

    if data_used.get("comparison_v2") or data_used.get("comparison_structured_engine"):
        return "comparison"
    if data_used.get("alternative_execution"):
        return "alternative"
    if data_used.get("deal_killer") or data_used.get("buy_decision_dispatch"):
        return "buy_decision"

    qri = str(data_used.get("query_recommendation_intent") or "").strip().lower()
    if qri == "aircraft_comparison":
        return "comparison"
    if qri in ("aircraft_fact", "registry_lookup", "listing_lookup"):
        return "fact"

    if (
        data_used.get("recommendation_pipeline")
        or data_used.get("query_recommendation_requires_pipeline")
        or data_used.get("pre_llm_recommendation")
        or re.search(r"\b(?:mission interpretation|ranked aircraft|operational synthesis)\b", answer, re.I)
    ):
        return "mission"

    if re.search(r"\b(?:compare|versus|vs\.?)\b", query, re.I) and len(_models_from_text(query)) >= 2:
        return "comparison"
    if re.search(r"\balternatives?\s+to\b", query, re.I):
        return "alternative"
    if re.search(r"\b(?:good\s+deal|overpriced|worth\s+it)\b", query, re.I):
        return "buy_decision"

    return "other"


def _sanitize_answer_for_intent(answer: str, intent_type: str) -> str:
    if not answer:
        return ""
    if intent_type not in ("comparison", "alternative", "buy_decision"):
        return answer.strip()

    lines: List[str] = []
    for line in answer.splitlines():
        if _KERNEL_LEAK_RE.match(line):
            continue
        if intent_type in ("comparison", "alternative") and re.search(
            r"\b(?:good\s+fit|approved\s+shortlist)\b", line, re.I
        ):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_aircraft(data_used: Dict[str, Any], answer: str, intent_type: str) -> List[str]:
    found: List[str] = []

    comp = data_used.get("comparison_v2") or {}
    if isinstance(comp, dict):
        found.extend(str(m) for m in (comp.get("models") or []) if m)

    alt = data_used.get("alternative_execution") or {}
    if isinstance(alt, dict):
        target = alt.get("target")
        if target:
            found.append(str(target))
        found.extend(str(c) for c in (alt.get("candidates") or []) if c)

    buy = data_used.get("buy_decision_dispatch") or {}
    if isinstance(buy, dict) and buy.get("model"):
        found.append(str(buy["model"]))

    pipe = data_used.get("recommendation_pipeline") or {}
    if isinstance(pipe, dict):
        found.extend(str(m) for m in (pipe.get("ranked_models") or []) if m)

    if not found:
        found = _models_from_text(answer)

    deduped: List[str] = []
    seen = set()
    for item in found:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped[:8]


def _extract_verdict(
    data_used: Dict[str, Any],
    answer: str,
    intent_type: str,
) -> Tuple[str, float]:
    dk = data_used.get("deal_killer")
    if isinstance(dk, dict) and dk.get("verdict"):
        raw = str(dk["verdict"]).strip().upper()
        canonical = _canonical_verdict(raw)
        conf = float(dk.get("confidence") or 0.7)
        return canonical, conf

    for match in _VERDICT_RE.finditer(answer or ""):
        canonical = _canonical_verdict(match.group(1))
        if canonical:
            return canonical, 0.65

    m = _COMPROMISES_RE.search(answer or "")
    if m:
        return "VIABLE WITH COMPROMISES", 0.6

    if intent_type == "comparison":
        return "", 0.75
    if intent_type == "alternative":
        return "CONDITIONAL FIT", 0.7
    if intent_type == "buy_decision":
        return "CONDITIONAL FIT", 0.5
    if intent_type == "mission":
        return "CONDITIONAL FIT", 0.55
    return "", 0.4


def _canonical_verdict(raw: str) -> str:
    upper = re.sub(r"\s+", " ", (raw or "").strip().upper())
    if upper in _VERDICT_ALIASES:
        return _VERDICT_ALIASES[upper]
    if upper in _VERDICT_CANONICAL:
        return upper
    if "NOT" in upper and "FIT" in upper:
        return "NOT A FIT"
    if "GOOD" in upper and "FIT" in upper:
        return "GOOD FIT"
    if "CONDITIONAL" in upper:
        return "CONDITIONAL FIT"
    return upper


def _extract_recommendations(
    data_used: Dict[str, Any],
    answer: str,
    intent_type: str,
    aircraft: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    primary: Dict[str, Any] = {}
    alternatives: List[str] = []

    alt = data_used.get("alternative_execution") or {}
    if isinstance(alt, dict):
        if alt.get("target"):
            primary = {"model": str(alt["target"]), "role": "replacement_target"}
        alternatives = [str(c) for c in (alt.get("candidates") or []) if c]

    buy = data_used.get("buy_decision_dispatch") or {}
    if isinstance(buy, dict) and buy.get("model"):
        primary = {
            "model": str(buy["model"]),
            "year": buy.get("year"),
            "ask_usd": buy.get("ask_usd"),
        }

    pipe = data_used.get("recommendation_pipeline") or {}
    if isinstance(pipe, dict):
        ranked = [str(m) for m in (pipe.get("ranked_models") or []) if m]
        if ranked:
            primary = {"model": ranked[0], "role": "mission_lead"}
            alternatives = ranked[1:4]

    if not primary and aircraft:
        if intent_type == "comparison":
            primary = {"models": list(aircraft[:2]), "role": "comparison_pair"}
        elif intent_type == "mission" and aircraft:
            primary = {"model": aircraft[0], "role": "mission_lead"}
            alternatives = list(aircraft[1:4])

    if not alternatives and intent_type == "alternative":
        m = re.search(r"alternatives to (?:the )?(.+?) include (.+?)\.", answer, re.I)
        if m:
            primary = primary or {"model": m.group(1).strip(), "role": "replacement_target"}
            alts = re.split(r",\s*|\s+and\s+", m.group(2))
            alternatives = [a.strip(" .") for a in alts if a.strip(" .")]

    return primary, alternatives


def _extract_financial_summary(
    data_used: Dict[str, Any],
    answer: str,
    intent_type: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    buy = data_used.get("buy_decision_dispatch") or {}
    if isinstance(buy, dict):
        if buy.get("ask_usd") is not None:
            summary["ask_usd"] = buy.get("ask_usd")
        if buy.get("year") is not None:
            summary["year"] = buy.get("year")

    dk = data_used.get("deal_killer") or {}
    if isinstance(dk, dict):
        scores = dk.get("scores") or {}
        if isinstance(scores, dict):
            summary["price_score"] = scores.get("price_score")
            summary["condition_score"] = scores.get("condition_score")
        summary["price_position"] = dk.get("price_position")
        echo = dk.get("inputs_echo") or {}
        if isinstance(echo, dict):
            for k in ("ask_price_usd", "market_low_usd", "market_high_usd", "market_avg_usd"):
                if echo.get(k) is not None:
                    summary[k] = echo.get(k)

    if intent_type == "buy_decision" and not summary.get("ask_usd"):
        price_m = re.search(r"Ask:\s*\$?([\d.,]+)\s*([MmKk])?", answer)
        if price_m:
            summary["ask_display"] = price_m.group(0).strip()

    return summary


def _extract_mission_fit(data_used: Dict[str, Any], intent_type: str) -> Dict[str, Any]:
    fit: Dict[str, Any] = {}
    dk = data_used.get("deal_killer") or {}
    if isinstance(dk, dict):
        scores = dk.get("scores") or {}
        if isinstance(scores, dict) and scores.get("mission_fit_score") is not None:
            fit["mission_fit_score"] = scores.get("mission_fit_score")

    mp = data_used.get("mission_preprocessing") or {}
    if isinstance(mp, dict):
        if mp.get("routes"):
            fit["routes"] = list(mp.get("routes") or [])[:3]
        if mp.get("passenger_count") is not None:
            fit["passenger_count"] = mp.get("passenger_count")

    if intent_type == "mission" and not fit:
        fit["status"] = "mission_advisory"
    return fit


def _extract_notes(
    data_used: Dict[str, Any],
    answer: str,
    intent_type: str,
) -> List[str]:
    notes: List[str] = []
    dk = data_used.get("deal_killer") or {}
    if isinstance(dk, dict):
        notes.extend(str(r) for r in (dk.get("key_reasons") or [])[:3] if r)

    if intent_type == "alternative":
        notes.append("Comparable replacements only — not a ranked mission shortlist.")

    broker = str((dk.get("broker_comment") if isinstance(dk, dict) else "") or "").strip()
    if broker and broker not in notes:
        notes.append(broker)

    if not notes and "directional" in answer.lower():
        notes.append("Some details are directional — verify on current records.")

    return notes[:6]


def _build_structured_sections(
    answer: str,
    *,
    intent_type: str,
    data_used: Dict[str, Any],
    aircraft: Sequence[str],
    verdict: str,
    alternatives: Sequence[str],
    financial: Dict[str, Any],
    notes: Sequence[str],
) -> Dict[str, Any]:
    parsed = _parse_answer_sections(answer)

    if intent_type == "comparison":
        pair = " and ".join(aircraft[:2]) if len(aircraft) >= 2 else "the requested models"
        overview = parsed.get("overview") or f"Side-by-side comparison: {pair}."
        analysis = parsed.get("analysis") or _join_bullets(parsed.get("analysis_lines") or _comparison_analysis_lines(answer))
        recommendation = parsed.get("recommendation") or _comparison_recommendation_line(answer, aircraft)
        risks: List[str] = list(parsed.get("risks") or [])
        if not risks:
            risks = ["Validate live market and maintenance status on specific tails before committing."]

    elif intent_type == "alternative":
        target = ""
        alt_exec = data_used.get("alternative_execution") or {}
        if isinstance(alt_exec, dict):
            target = str(alt_exec.get("target") or "").strip()
        overview = parsed.get("overview") or (
            f"Tier-peer alternatives to {target}." if target else "Credible tier-peer alternatives."
        )
        analysis = parsed.get("analysis") or answer
        alts_text = ", ".join(alternatives[:4]) if alternatives else ""
        recommendation = parsed.get("recommendation") or (
            f"Consider verified tier peers: {alts_text}." if alts_text else "Use tier-peer list for like-for-like replacement only."
        )
        risks = list(parsed.get("risks") or [])

    elif intent_type == "buy_decision":
        buy = data_used.get("buy_decision_dispatch") or {}
        model = str((buy.get("model") if isinstance(buy, dict) else "") or (aircraft[0] if aircraft else "")).strip()
        year = buy.get("year") if isinstance(buy, dict) else None
        ask = financial.get("ask_usd")
        overview_parts = [f"Aircraft: {model}" if model else ""]
        if year:
            overview_parts.append(f"Year: {year}")
        if ask is not None:
            overview_parts.append(
                f"Ask: ${float(ask) / 1_000_000:.1f}M" if float(ask) >= 1_000_000 else f"Ask: ${float(ask):,.0f}"
            )
        overview = parsed.get("overview") or "\n".join(p for p in overview_parts if p)
        analysis = parsed.get("analysis") or _extract_block(answer, "Market Reality")
        recommendation = parsed.get("recommendation") or (f"Verdict: {verdict}" if verdict else "Verdict: CONDITIONAL FIT")
        dk = data_used.get("deal_killer") or {}
        risks = list(parsed.get("risks") or [])
        if not risks and isinstance(dk, dict):
            risks = [str(f) for f in (dk.get("red_flags") or [])[:6] if f]

    elif intent_type == "mission":
        overview = parsed.get("overview") or _first_paragraph(answer)
        analysis = parsed.get("analysis") or _extract_mission_analysis(answer)
        recommendation = parsed.get("recommendation") or _extract_mission_recommendation(answer, verdict)
        risks = list(parsed.get("risks") or _extract_risk_bullets(answer))

    else:
        overview = parsed.get("overview") or _first_paragraph(answer)
        analysis = parsed.get("analysis") or answer
        recommendation = parsed.get("recommendation") or (f"Verdict: {verdict}" if verdict else "")
        risks = list(parsed.get("risks") or [])

    return {
        "overview": overview.strip(),
        "analysis": analysis.strip(),
        "recommendation": recommendation.strip(),
        "risks": [r.strip() for r in risks if str(r).strip()][:8],
    }


def _parse_answer_sections(answer: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "overview": "",
        "analysis": "",
        "recommendation": "",
        "risks": [],
        "analysis_lines": [],
    }
    if not answer:
        return result

    current = "overview"
    buckets: Dict[str, List[str]] = {
        "overview": [],
        "analysis": [],
        "recommendation": [],
        "risks": [],
    }

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for key, pattern in _SECTION_HEADERS.items():
            if pattern.match(stripped):
                current = key
                matched = True
                break
        if matched:
            continue
        buckets[current].append(stripped)

    result["overview"] = "\n".join(buckets["overview"]).strip()
    result["analysis"] = "\n".join(buckets["analysis"]).strip()
    result["analysis_lines"] = list(buckets["analysis"])
    result["recommendation"] = "\n".join(buckets["recommendation"]).strip()
    result["risks"] = [
        re.sub(r"^[-*•]\s*", "", r).strip()
        for r in buckets["risks"]
        if r.strip()
    ]
    return result


def _render_structured_answer(normalized: NormalizedConsultantResponse) -> str:
    sections = normalized.structured_sections or {}
    parts: List[str] = []

    overview = str(sections.get("overview") or "").strip()
    if overview:
        parts.append("Overview")
        parts.append(overview)

    analysis = str(sections.get("analysis") or "").strip()
    if analysis:
        parts.append("")
        parts.append("Analysis")
        parts.append(analysis)

    recommendation = str(sections.get("recommendation") or "").strip()
    if recommendation:
        parts.append("")
        parts.append("Recommendation")
        parts.append(recommendation)

    risks = list(sections.get("risks") or [])
    if risks:
        parts.append("")
        parts.append("Risks")
        for risk in risks:
            parts.append(f"- {risk}")

    if normalized.verdict:
        parts.append("")
        parts.append(f"Verdict: {normalized.verdict}")

    for note in normalized.notes[:2]:
        if note and note not in "\n".join(parts):
            parts.append(f"- {note}")

    return "\n".join(parts).strip()


def _extract_data_sources(data_used: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, bool]:
    du = data_used or {}
    return {
        "phly_used": bool(
            du.get("phly_authority")
            or du.get("phly_meta")
            or du.get("phly_rows")
            or du.get("phlydata_lookup")
        ),
        "tavily_used": bool(
            du.get("tavily_hits")
            or du.get("tavily_payload")
            or (response or {}).get("sources")
        ),
        "market_used": bool(
            du.get("deal_killer")
            or du.get("buy_decision_dispatch")
            or du.get("market_block")
            or du.get("listing_rows")
            or du.get("for_sale_listings")
        ),
    }


def _models_from_text(text: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(detect_models_from_text(text or "") or [])
    except Exception:
        return []


def _first_paragraph(text: str) -> str:
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return chunks[0] if chunks else (text or "").strip()[:400]


def _join_bullets(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    return "\n".join(f"- {re.sub(r'^[-*•]\s*', '', str(l))}" for l in lines if str(l).strip())


def _comparison_analysis_lines(answer: str) -> List[str]:
    lines: List[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"(?i)^verified catalog comparison", stripped):
            continue
        if stripped.lower().startswith(("overview", "analysis", "recommendation", "verdict", "risks")):
            continue
        lines.append(stripped)
    return lines[:8]


def _comparison_recommendation_line(answer: str, aircraft: Sequence[str]) -> str:
    for line in (answer or "").splitlines():
        if re.search(r"\b(?:leads|offers more|on verified range|on seating)\b", line, re.I):
            return line.strip().lstrip("- ")
    if len(aircraft) >= 2:
        return f"Compare {aircraft[0]} and {aircraft[1]} on mission length, cabin, and operating economics before selecting."
    return "Use the side-by-side read as a starting point — confirm live market and maintenance status on specific tails."


def _extract_block(answer: str, header: str) -> str:
    lines: List[str] = []
    capture = False
    for line in (answer or "").splitlines():
        if re.match(rf"(?i)^{re.escape(header)}\s*:?\s*$", line.strip()):
            capture = True
            continue
        if capture:
            if re.match(r"(?i)^(red flags|verdict|recommendation|overview|analysis)\s*:?\s*$", line.strip()):
                break
            lines.append(line)
    if lines:
        return "\n".join(lines).strip()
    return ""


def _extract_mission_analysis(answer: str) -> str:
    blocks: List[str] = []
    for header in ("Constraint Summary", "Ranked Aircraft Shortlist", "Mission Interpretation"):
        block = _extract_block(answer, header)
        if block:
            blocks.append(f"{header}\n{block}")
    return "\n\n".join(blocks).strip() or _first_paragraph(answer)


def _extract_mission_recommendation(answer: str, verdict: str) -> str:
    block = _extract_block(answer, "Final Verdict")
    if block:
        return block
    if verdict:
        return f"{verdict}: see analysis above for constraint alignment."
    return _extract_block(answer, "Recommendation") or ""


def _extract_risk_bullets(answer: str) -> List[str]:
    risks = _extract_block(answer, "Red Flags")
    if not risks:
        return []
    return [
        re.sub(r"^[-*•]\s*", "", line).strip()
        for line in risks.splitlines()
        if line.strip()
    ]


__all__ = [
    "NormalizedConsultantResponse",
    "apply_consultant_response_normalization",
    "normalize_consultant_response",
]
