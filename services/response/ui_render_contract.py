"""
Frontend rendering contract — presentation-safe envelope from normalized responses.

Does not alter routing, classification, dispatch, or business logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.response.response_normalizer import NormalizedConsultantResponse

_UI_INTENTS = frozenset({"comparison", "alternative", "buy_decision", "mission", "fact", "other"})
_LAYOUT_TYPES = frozenset(
    {"side_by_side", "ranked_list", "deal_card", "mission_brief", "info_card"}
)
_SECTION_TYPES = frozenset({"overview", "analysis", "recommendation", "risks"})
_RENDER_MODES = frozenset({"text", "bullet", "table"})

_VERDICT_BADGE_INTENTS = frozenset({"buy_decision", "mission", "alternative"})


@dataclass
class UIRenderContract:
    ui_intent: str
    layout_type: str
    primary_cards: List[Dict[str, Any]] = field(default_factory=list)
    secondary_cards: List[Dict[str, Any]] = field(default_factory=list)
    risk_cards: List[Dict[str, Any]] = field(default_factory=list)
    financial_cards: List[Dict[str, Any]] = field(default_factory=list)
    headline: str = ""
    subheadline: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    ui_flags: Dict[str, bool] = field(default_factory=dict)
    render_hints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ui_intent": self.ui_intent,
            "layout_type": self.layout_type,
            "primary_cards": list(self.primary_cards),
            "secondary_cards": list(self.secondary_cards),
            "risk_cards": list(self.risk_cards),
            "financial_cards": list(self.financial_cards),
            "headline": self.headline,
            "subheadline": self.subheadline,
            "sections": list(self.sections),
            "ui_flags": dict(self.ui_flags),
            "render_hints": dict(self.render_hints),
        }


def build_ui_render_contract(
    normalized_response: NormalizedConsultantResponse,
    context: Optional[Dict[str, Any]] = None,
) -> UIRenderContract:
    """Build a deterministic UI rendering contract from a normalized consultant response."""
    ctx = context if isinstance(context, dict) else {}
    intent = _canonical_ui_intent(normalized_response.intent_type)
    layout = _layout_for_intent(intent)
    structured = normalized_response.structured_sections or {}

    primary_cards = _build_primary_cards(normalized_response, intent)
    secondary_cards = _build_secondary_cards(normalized_response, intent)
    risk_cards = _build_risk_cards(normalized_response)
    financial_cards = _build_financial_cards(normalized_response, intent)
    headline, subheadline = _build_headlines(normalized_response, intent, ctx)
    sections = _build_sections(structured, intent, normalized_response)
    ui_flags = _build_ui_flags(normalized_response, intent, risk_cards, financial_cards)
    render_hints = _build_render_hints(intent)

    return UIRenderContract(
        ui_intent=intent,
        layout_type=layout,
        primary_cards=primary_cards,
        secondary_cards=secondary_cards,
        risk_cards=risk_cards,
        financial_cards=financial_cards,
        headline=headline,
        subheadline=subheadline,
        sections=sections,
        ui_flags=ui_flags,
        render_hints=render_hints,
    )


def apply_ui_render_contract_to_response(
    response: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach ui_render_contract when normalized_response is present."""
    if not isinstance(response, dict):
        return response
    du = dict(response.get("data_used") or {})
    norm_raw = du.get("normalized_response")
    if not isinstance(norm_raw, dict):
        return response

    normalized = _normalized_from_dict(norm_raw, answer=str(response.get("answer") or ""))
    contract = build_ui_render_contract(normalized, context)
    contract_dict = contract.to_dict()
    if du.get("truth_compression_applied"):
        try:
            from services.truth_compression.truth_compression_layer import compress_ui_contract_sections
            from services.truth_compression.truth_synthesizer import synthesize_truth_state

            truth = synthesize_truth_state(du)
            contract_dict = compress_ui_contract_sections(
                contract_dict,
                truth,
                pathways=du.get("redundant_truth_pathways"),
            )
        except Exception:
            pass
    out = dict(response)
    du["ui_render_contract"] = contract_dict
    du["ui_render_contract_applied"] = 1
    out["data_used"] = du
    return out


def _normalized_from_dict(raw: Dict[str, Any], *, answer: str = "") -> NormalizedConsultantResponse:
    sections = raw.get("structured_sections") or {}
    if not isinstance(sections, dict):
        sections = {}
    return NormalizedConsultantResponse(
        intent_type=str(raw.get("intent_type") or "other"),
        aircraft=list(raw.get("aircraft") or []),
        primary_recommendation=dict(raw.get("primary_recommendation") or {}),
        alternatives=list(raw.get("alternatives") or []),
        financial_summary=dict(raw.get("financial_summary") or {}),
        mission_fit=dict(raw.get("mission_fit") or {}),
        verdict=str(raw.get("verdict") or ""),
        confidence=float(raw.get("confidence") or 0.0),
        notes=list(raw.get("notes") or []),
        structured_sections={
            "overview": str(sections.get("overview") or ""),
            "analysis": str(sections.get("analysis") or ""),
            "recommendation": str(sections.get("recommendation") or ""),
            "risks": list(sections.get("risks") or []),
        },
        data_sources=dict(raw.get("data_sources") or {}),
        answer_text=answer,
    )


def _canonical_ui_intent(intent_type: str) -> str:
    key = str(intent_type or "other").strip().lower()
    return key if key in _UI_INTENTS else "other"


def _layout_for_intent(intent: str) -> str:
    mapping = {
        "comparison": "side_by_side",
        "alternative": "ranked_list",
        "buy_decision": "deal_card",
        "mission": "mission_brief",
        "fact": "info_card",
        "other": "info_card",
    }
    layout = mapping.get(intent, "info_card")
    return layout if layout in _LAYOUT_TYPES else "info_card"


def _build_primary_cards(
    normalized: NormalizedConsultantResponse,
    intent: str,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    primary = normalized.primary_recommendation or {}

    if intent == "comparison":
        models = list(primary.get("models") or [])
        if not models:
            models = normalized.aircraft[:2]
        for idx, model in enumerate(models[:2]):
            cards.append(
                _aircraft_card(
                    card_id=f"compare_primary_{idx}",
                    title=str(model),
                    role="compare_column",
                    column_index=idx,
                )
            )
        return cards

    if intent == "alternative":
        target = str(primary.get("model") or (normalized.aircraft[0] if normalized.aircraft else "")).strip()
        if target:
            cards.append(
                _aircraft_card(
                    card_id="alternative_target",
                    title=target,
                    role="replacement_target",
                    accent="primary",
                )
            )
        return cards

    if intent == "buy_decision":
        model = str(primary.get("model") or (normalized.aircraft[0] if normalized.aircraft else "Aircraft")).strip()
        year = primary.get("year")
        cards.append(
            {
                "card_id": "deal_primary",
                "card_type": "aircraft",
                "title": model or "Aircraft",
                "subtitle": f"Year {year}" if year else "",
                "role": "deal_subject",
                "accent": "primary",
                "metadata": {"year": year},
            }
        )
        return cards

    if intent == "mission":
        model = str(primary.get("model") or (normalized.aircraft[0] if normalized.aircraft else "")).strip()
        if model:
            cards.append(
                _aircraft_card(
                    card_id="mission_lead",
                    title=model,
                    role="mission_lead",
                    accent="primary",
                    metadata={"rank": 1},
                )
            )
        return cards

    if normalized.aircraft:
        cards.append(
            _aircraft_card(
                card_id="info_primary_0",
                title=str(normalized.aircraft[0]),
                role="info_subject",
            )
        )
    return cards


def _build_secondary_cards(
    normalized: NormalizedConsultantResponse,
    intent: str,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    if intent == "alternative":
        for idx, alt in enumerate(normalized.alternatives[:6]):
            cards.append(
                _aircraft_card(
                    card_id=f"alternative_peer_{idx}",
                    title=str(alt),
                    role="tier_peer",
                    accent="secondary",
                    metadata={"tier_rank": idx + 1},
                )
            )
        return cards

    if intent == "mission":
        for idx, alt in enumerate(normalized.alternatives[:4]):
            cards.append(
                _aircraft_card(
                    card_id=f"mission_alternate_{idx}",
                    title=str(alt),
                    role="mission_alternate",
                    accent="secondary",
                    metadata={"rank": idx + 2},
                )
            )
        return cards

    if intent == "comparison" and len(normalized.aircraft) > 2:
        for idx, model in enumerate(normalized.aircraft[2:4]):
            cards.append(
                _aircraft_card(
                    card_id=f"compare_secondary_{idx}",
                    title=str(model),
                    role="compare_context",
                    accent="secondary",
                )
            )
    return cards


def _build_risk_cards(normalized: NormalizedConsultantResponse) -> List[Dict[str, Any]]:
    risks = list((normalized.structured_sections or {}).get("risks") or [])
    cards: List[Dict[str, Any]] = []
    for idx, risk in enumerate(risks[:8]):
        text = str(risk or "").strip()
        if not text:
            continue
        cards.append(
            {
                "card_id": f"risk_{idx}",
                "card_type": "risk",
                "title": "Risk",
                "body": text,
                "accent": "warning",
                "role": "risk_item",
            }
        )
    return cards


def _build_financial_cards(
    normalized: NormalizedConsultantResponse,
    intent: str,
) -> List[Dict[str, Any]]:
    if intent not in ("buy_decision", "mission"):
        return []

    fin = normalized.financial_summary or {}
    cards: List[Dict[str, Any]] = []

    ask = fin.get("ask_usd")
    if ask is not None:
        cards.append(
            {
                "card_id": "financial_ask",
                "card_type": "financial",
                "title": "Ask Price",
                "body": _format_usd(float(ask)),
                "role": "ask_price",
                "accent": "primary",
            }
        )

    for key, label in (
        ("market_avg_usd", "Market Average"),
        ("market_low_usd", "Market Low"),
        ("market_high_usd", "Market High"),
    ):
        val = fin.get(key)
        if val is not None:
            cards.append(
                {
                    "card_id": f"financial_{key}",
                    "card_type": "financial",
                    "title": label,
                    "body": _format_usd(float(val)),
                    "role": key,
                    "accent": "secondary",
                }
            )

    price_position = str(fin.get("price_position") or "").strip()
    if price_position and price_position.lower() != "unknown":
        cards.append(
            {
                "card_id": "financial_price_position",
                "card_type": "financial",
                "title": "Price Position",
                "body": price_position.replace("_", " ").title(),
                "role": "price_position",
                "accent": "secondary",
            }
        )

    if intent == "buy_decision" and fin.get("price_score") is not None:
        cards.append(
            {
                "card_id": "financial_price_score",
                "card_type": "financial",
                "title": "Price Score",
                "body": f"{float(fin['price_score']):.2f}",
                "role": "price_score",
                "accent": "secondary",
            }
        )

    return cards


def _build_headlines(
    normalized: NormalizedConsultantResponse,
    intent: str,
    ctx: Dict[str, Any],
) -> tuple[str, str]:
    overview = str((normalized.structured_sections or {}).get("overview") or "").strip()
    primary = normalized.primary_recommendation or {}
    query = str(ctx.get("query") or "").strip()

    if intent == "comparison":
        models = list(primary.get("models") or normalized.aircraft[:2])
        if len(models) >= 2:
            headline = f"{models[0]} vs {models[1]}"
        elif models:
            headline = f"{models[0]} Comparison"
        else:
            headline = "Aircraft Comparison"
        sub = overview.splitlines()[0] if overview else "Side-by-side comparison on range, cabin, and operating economics."
        return headline, sub

    if intent == "alternative":
        target = str(primary.get("model") or (normalized.aircraft[0] if normalized.aircraft else "Aircraft"))
        headline = f"Alternatives to {target}"
        sub = overview.splitlines()[0] if overview else "Comparable replacements in the same class."
        return headline, sub

    if intent == "buy_decision":
        model = str(primary.get("model") or (normalized.aircraft[0] if normalized.aircraft else "Aircraft"))
        headline = f"Deal Assessment: {model}"
        sub = overview.splitlines()[0] if overview else "Structured market and risk review."
        return headline, sub

    if intent == "mission":
        headline = "Mission Recommendation"
        lead = str(primary.get("model") or "").strip()
        sub = overview.splitlines()[0] if overview else (
            f"Lead candidate: {lead}." if lead else "Constraint-based mission brief."
        )
        return headline, sub

    if intent == "fact":
        headline = "Aircraft Information"
        sub = overview.splitlines()[0] if overview else (query[:120] or "Fact response.")
        return headline, sub

    headline = "Consultant Brief"
    sub = overview.splitlines()[0] if overview else (query[:120] or "")
    return headline, sub


def _build_sections(
    structured: Dict[str, Any],
    intent: str,
    normalized: NormalizedConsultantResponse,
) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for section_type in ("overview", "analysis", "recommendation", "risks"):
        if section_type not in _SECTION_TYPES:
            continue
        raw = structured.get(section_type)
        if section_type == "risks":
            items = [str(r).strip() for r in (raw or []) if str(r).strip()]
            if not items:
                continue
            sections.append(
                {
                    "type": section_type,
                    "content": "\n".join(f"- {item}" for item in items),
                    "render_mode": "bullet",
                    "items": items,
                }
            )
            continue

        content = str(raw or "").strip()
        if not content:
            continue
        render_mode = _infer_render_mode(content, intent=intent, section_type=section_type)
        entry: Dict[str, Any] = {
            "type": section_type,
            "content": content,
            "render_mode": render_mode,
        }
        if render_mode == "table" and intent == "comparison":
            entry["table_rows"] = _comparison_table_rows(content, normalized.aircraft[:2])
        elif render_mode == "bullet":
            entry["items"] = _bullet_items(content)
        sections.append(entry)
    return sections


def _build_ui_flags(
    normalized: NormalizedConsultantResponse,
    intent: str,
    risk_cards: Sequence[Dict[str, Any]],
    financial_cards: Sequence[Dict[str, Any]],
) -> Dict[str, bool]:
    verdict = str(normalized.verdict or "").strip()
    mission_score = (normalized.mission_fit or {}).get("mission_fit_score")
    return {
        "show_verdict_badge": bool(verdict) and (
            intent in _VERDICT_BADGE_INTENTS or verdict in ("GOOD DEAL", "OVERPRICED", "RISKY")
        ),
        "show_price_comparison": intent == "buy_decision" and len(financial_cards) >= 2,
        "show_risk_panel": len(risk_cards) > 0,
        "show_mission_fit_meter": intent == "mission" and mission_score is not None,
    }


def _build_render_hints(intent: str) -> Dict[str, str]:
    return {
        "comparison_mode": "strict_side_by_side" if intent == "comparison" else "none",
        "alternative_mode": "tier_clustered" if intent == "alternative" else "none",
        "buy_mode": "market_delta_emphasis" if intent == "buy_decision" else "none",
        "mission_mode": "constraint_first" if intent == "mission" else "none",
    }


def _aircraft_card(
    *,
    card_id: str,
    title: str,
    role: str,
    accent: str = "primary",
    column_index: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    card: Dict[str, Any] = {
        "card_id": card_id,
        "card_type": "aircraft",
        "title": title,
        "role": role,
        "accent": accent,
        "metadata": dict(metadata or {}),
    }
    if column_index is not None:
        card["metadata"]["column_index"] = column_index
    return card


def _infer_render_mode(content: str, *, intent: str, section_type: str) -> str:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    bullet_lines = sum(1 for ln in lines if re.match(r"^[-*•]\s+", ln))
    if bullet_lines >= 2 or (bullet_lines >= 1 and len(lines) <= 8):
        return "bullet"
    if intent == "comparison" and section_type == "analysis" and len(lines) >= 2:
        if any(re.search(r"\bclass\b|\bnm\b|\bseats\b", ln, re.I) for ln in lines):
            return "table"
    return "text"


def _bullet_items(content: str) -> List[str]:
    items: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(re.sub(r"^[-*•]\s*", "", stripped))
    return items


def _comparison_table_rows(content: str, models: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in content.splitlines():
        stripped = re.sub(r"^[-*•]\s*", "", line.strip())
        if not stripped:
            continue
        label = "detail"
        value = stripped
        if ":" in stripped:
            label, value = stripped.split(":", 1)
            label = label.strip()
            value = value.strip()
        row: Dict[str, str] = {"label": label, "value": value}
        for model in models:
            if model.lower() in value.lower():
                row["aircraft"] = model
        rows.append(row)
    return rows[:12]


def _format_usd(amount: float) -> str:
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${amount:,.0f}"
    return f"${amount:.0f}"


__all__ = [
    "UIRenderContract",
    "apply_ui_render_contract_to_response",
    "build_ui_render_contract",
]
