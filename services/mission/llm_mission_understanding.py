"""
LLM mission understanding — structured inference only (no aircraft, no feasibility claims).

Returns a partial understanding payload merged by ``mission_understanding_merge`` into
the deterministic :class:`MissionUnderstandingPacket` from rules + memory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile

logger = logging.getLogger(__name__)

_LLM_SYSTEM = """You are an aviation acquisition consultant doing MISSION UNDERSTANDING only.

Your job: infer operational posture, latent constraints, and travel pattern from the user text.
You do NOT recommend aircraft models, tail numbers, or feasibility outcomes.

Return JSON only with this shape:
{
  "confidence": 0.0-1.0,
  "inferred_constraints": { "snake_case_key": "value or bool or number" },
  "operational_environment": ["short operational fact", "..."],
  "ownership_profile": "unknown|corporate_shuttle_candidate|family_office|owner_operator|...",
  "travel_pattern": "unknown|executive_shuttle|transatlantic_executive|multi_leg|regional_shuttle|...",
  "corridor_type": "unknown|transatlantic_ulr|caribbean_regional|mountain_field|continental_super_mid|...",
  "runway_complexity": "standard|high",
  "dispatch_priority": "standard|high",
  "comfort_priority": "standard|high|secondary",
  "operating_cost_priority": "standard|high",
  "nonstop_priority": "standard|high",
  "utilization_style": "unknown|executive_shuttle|board_transport|family_leisure|owner_flown|mixed_corporate|...",
  "operational_synthesis": "2-3 sentences of broker-style mission synthesis",
  "understanding_notes": ["brief note", "..."],
  "clarifying_question": "optional single focused question if mission is materially incomplete, else null"
}

Rules:
- Infer hidden priorities (executives -> dispatch/comfort, island ops -> runway, cost-sensitive enterprise -> operating cost).
- Detect multi-role or incompatible legs (e.g. TEB-London + Aspen) and note portfolio structure.
- Do NOT invent city pairs or passenger counts not supported by the text or extracted facts.
- Do NOT name aircraft models or classes as recommendations.
- Do NOT claim nonstop is guaranteed or impossible.
- confidence reflects how complete the mission definition is, not marketing certainty.
"""


@dataclass
class LLMMissionUnderstandingResult:
    confidence: float = 0.0
    inferred_constraints: Dict[str, Any] = field(default_factory=dict)
    operational_environment: List[str] = field(default_factory=list)
    ownership_profile: str = ""
    travel_pattern: str = ""
    corridor_type: str = ""
    runway_complexity: str = ""
    dispatch_priority: str = ""
    comfort_priority: str = ""
    operating_cost_priority: str = ""
    nonstop_priority: str = ""
    utilization_style: str = ""
    operational_synthesis: str = ""
    understanding_notes: List[str] = field(default_factory=list)
    clarifying_question: Optional[str] = None
    model: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (
            bool(self.operational_synthesis)
            or bool(self.inferred_constraints)
            or bool(self.operational_environment)
            or self.confidence > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 3),
            "inferred_constraints": dict(self.inferred_constraints),
            "operational_environment": list(self.operational_environment),
            "ownership_profile": self.ownership_profile,
            "travel_pattern": self.travel_pattern,
            "corridor_type": self.corridor_type,
            "runway_complexity": self.runway_complexity,
            "dispatch_priority": self.dispatch_priority,
            "comfort_priority": self.comfort_priority,
            "operating_cost_priority": self.operating_cost_priority,
            "nonstop_priority": self.nonstop_priority,
            "utilization_style": self.utilization_style,
            "operational_synthesis": self.operational_synthesis,
            "understanding_notes": list(self.understanding_notes),
            "clarifying_question": self.clarifying_question,
            "model": self.model,
            "error": self.error,
        }


def mission_understanding_llm_enabled() -> bool:
    raw = os.getenv("MISSION_UNDERSTANDING_LLM")
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "no", "off", "disabled")
    # Default: hybrid only when an API key is configured (safe for CI/tests).
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _llm_model() -> str:
    return (os.getenv("MISSION_UNDERSTANDING_LLM_MODEL") or "gpt-4o-mini").strip()


def _llm_timeout_s() -> float:
    try:
        return max(4.0, min(30.0, float(os.getenv("MISSION_UNDERSTANDING_LLM_TIMEOUT") or "14")))
    except (TypeError, ValueError):
        return 14.0


def _priority_level(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in ("high", "secondary"):
        return s
    return "standard"


def _strip_json_fences(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _coerce_str_list(value: Any, *, limit: int = 6) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item or "").strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _coerce_constraints(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in value.items():
        key = re.sub(r"\s+", "_", str(k or "").strip().lower())
        if not key or key.startswith("aircraft"):
            continue
        if isinstance(v, str) and re.search(r"\b(?:citation|gulfstream|challenger|falcon|global|phenom)\b", v, re.I):
            continue
        out[key] = v
    return out


def _history_blob(history: Optional[Sequence[Dict[str, str]]], limit: int = 8) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for turn in list(history)[-limit:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user").strip().lower()
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_user_payload(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    history: Optional[Sequence[Dict[str, str]]] = None,
    rule_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "latest_user_message": (query or "").strip(),
        "conversation_history": _history_blob(history),
        "extracted_facts_do_not_contradict": {
            "passengers": profile.passengers,
            "routes": profile.route_labels(),
            "nonstop_required": bool(profile.nonstop_required or mission.nonstop_requirement),
            "regions": list(profile.regions or []),
            "budget_usd_mid": profile.budget_usd_mid,
            "mountain_airports": bool(profile.mountain_airports or profile.mountain_airport_priority),
        },
        "deterministic_understanding_snapshot": rule_snapshot or {},
    }


def parse_llm_mission_understanding_payload(data: Any) -> LLMMissionUnderstandingResult:
    if not isinstance(data, dict):
        return LLMMissionUnderstandingResult(error="invalid_json_shape")
    try:
        conf = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    cq = data.get("clarifying_question")
    clarifying = str(cq).strip() if cq not in (None, "", "null") else None
    return LLMMissionUnderstandingResult(
        confidence=conf,
        inferred_constraints=_coerce_constraints(data.get("inferred_constraints")),
        operational_environment=_coerce_str_list(data.get("operational_environment")),
        ownership_profile=str(data.get("ownership_profile") or "").strip(),
        travel_pattern=str(data.get("travel_pattern") or "").strip(),
        corridor_type=str(data.get("corridor_type") or "").strip(),
        runway_complexity=_priority_level(data.get("runway_complexity")),
        dispatch_priority=_priority_level(data.get("dispatch_priority")),
        comfort_priority=_priority_level(data.get("comfort_priority")),
        operating_cost_priority=_priority_level(data.get("operating_cost_priority")),
        nonstop_priority=_priority_level(data.get("nonstop_priority")),
        utilization_style=str(data.get("utilization_style") or "").strip(),
        operational_synthesis=str(data.get("operational_synthesis") or "").strip(),
        understanding_notes=_coerce_str_list(data.get("understanding_notes"), limit=8),
        clarifying_question=clarifying,
    )


def infer_mission_understanding_llm(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    history: Optional[Sequence[Dict[str, str]]] = None,
    rule_snapshot: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LLMMissionUnderstandingResult:
    """
    Call OpenAI for structured mission understanding. Returns empty/error result on failure.
    """
    if not mission_understanding_llm_enabled():
        return LLMMissionUnderstandingResult(error="llm_disabled")

    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return LLMMissionUnderstandingResult(error="missing_api_key")

    chosen_model = (model or _llm_model()).strip()
    payload = _build_user_payload(
        query, profile, mission, history=history, rule_snapshot=rule_snapshot
    )

    try:
        import openai

        client = openai.OpenAI(api_key=key, timeout=timeout or _llm_timeout_s())
        resp = client.chat.completions.create(
            model=chosen_model,
            temperature=0.1,
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {
                    "role": "user",
                    "content": "Mission context JSON:\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
        )
        raw = _strip_json_fences(resp.choices[0].message.content or "")
        data = json.loads(raw)
        result = parse_llm_mission_understanding_payload(data)
        result.model = chosen_model
        if not result.ok:
            result.error = result.error or "empty_llm_payload"
        return result
    except Exception as exc:
        logger.warning("mission_understanding_llm failed: %s", exc)
        return LLMMissionUnderstandingResult(error=str(exc)[:400], model=chosen_model)


__all__ = [
    "LLMMissionUnderstandingResult",
    "infer_mission_understanding_llm",
    "mission_understanding_llm_enabled",
    "parse_llm_mission_understanding_payload",
]
