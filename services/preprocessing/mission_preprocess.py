"""
Mission pre-processing — structured JSON from every user query before recommendations.

Rules:
  - Infer carefully when obvious (explicit text or validated route extraction).
  - Otherwise mark ``UNKNOWN``.
  - Never fabricate origin/destination (no guessing city pairs from regions alone).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from services.mission.route_extractor import extract_routes, routes_from_extractions
from services.mission_extraction.extractor import (
    _extract_from_to_and,
    _extract_origin_destinations,
    extract_mission_requirements,
)
from services.mission_extraction.schema import MissionExtractionResult
from services.preprocessing.schema import UNKNOWN, PreprocessedMission

_PRIORITY_FIELDS = ("runway_priority", "operating_cost_priority", "luxury_priority")


def _bool_field(value: Optional[bool]) -> bool | str:
    if value is True:
        return True
    if value is False:
        return False
    return UNKNOWN


def _priority_field(value: Optional[str]) -> str:
    if value in ("low", "medium", "high"):
        return value
    return UNKNOWN


def _route_evidence(query: str, route_labels: List[str]) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    """
  Returns (evidence_kind, origin, destination, notes).

  Origin/destination are only returned when evidence is validated_route or explicit_from_to.
    """
    notes: List[str] = []
    if route_labels:
        origin_r, dests_r = _extract_origin_destinations(query, route_labels)
        if origin_r or dests_r:
            dest = _single_destination(
                dests_r, multi_city=bool(dests_r and len(dests_r) > 1)
            )
            return "validated_route", origin_r, dest, notes

    origin_ft, dests_ft = _extract_from_to_and(query)
    if origin_ft or dests_ft:
        dest = _single_destination(
            dests_ft, multi_city=bool(dests_ft and len(dests_ft) > 1)
        )
        return "explicit_from_to", origin_ft, dest, notes

    if re.search(r"\bto\b|\bfrom\b", query, re.I) and not route_labels:
        notes.append("route_language_without_validated_places")

    return "none", None, None, notes


def _single_destination(
    destinations: Optional[List[str]],
    *,
    multi_city: bool,
) -> Optional[str]:
    if not destinations:
        return None
    if multi_city and len(destinations) > 1:
        return None
    return destinations[-1]


def _ownership_field(value: Optional[str]) -> str:
    if value in ("fractional", "full_ownership", "charter", "undecided"):
        return value
    return UNKNOWN


def preprocess_mission_from_query(user_message: str) -> PreprocessedMission:
    """
    Extract structured mission JSON for the current user turn.

    Runs before recommendation generation and before LLM advisory narration.
    """
    raw = (user_message or "").strip()
    if not raw:
        return PreprocessedMission(extraction_notes=["empty_query"])

    route_labels = [r.label() for r in routes_from_extractions(extract_routes(raw))]
    evidence, origin_ev, dest_ev, notes = _route_evidence(raw, route_labels)

    extracted: MissionExtractionResult = extract_mission_requirements(raw)

    # Origin / destination: never copy from extracted model unless route evidence exists
    if evidence in ("validated_route", "explicit_from_to"):
        origin: str = origin_ev if origin_ev else UNKNOWN
        destination: str = dest_ev if dest_ev else UNKNOWN
    else:
        origin = UNKNOWN
        destination = UNKNOWN
        if extracted.origin or extracted.destination:
            notes.append("suppressed_unvalidated_route_fields")

    passengers: int | str = (
        extracted.passengers if extracted.passengers is not None else UNKNOWN
    )

    return PreprocessedMission(
        passengers=passengers,
        origin=origin,
        destination=destination,
        nonstop_required=_bool_field(extracted.nonstop_required),
        westbound=_bool_field(extracted.westbound_sensitive),
        winter_operation=_bool_field(extracted.winter_ops),
        runway_priority=_priority_field(extracted.runway_priority),
        operating_cost_priority=_priority_field(extracted.operating_cost_priority),
        luxury_priority=_priority_field(extracted.cabin_priority),
        budget=extracted.budget if extracted.budget is not None else UNKNOWN,
        annual_hours=extracted.annual_hours if extracted.annual_hours is not None else UNKNOWN,
        ownership_interest=_ownership_field(extracted.ownership_interest),
        mountain_airport=_bool_field(extracted.mountain_airports),
        international=_bool_field(extracted.international_ops),
        transatlantic=_bool_field(extracted.transatlantic),
        transpacific=_bool_field(extracted.transpacific),
        route_evidence=evidence,
        extraction_notes=notes,
    )


def preprocess_mission_json(user_message: str) -> str:
    """Strict JSON string for logging, prompts, and ``data_used``."""
    result = preprocess_mission_from_query(user_message)
    return json.dumps(
        result.to_public_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def attach_mission_preprocessing(
    data_used: Optional[Dict[str, Any]],
    query: str,
) -> PreprocessedMission:
    """
    Run pre-processing and merge into ``data_used`` (if provided).

    Returns the structured mission object.
    """
    pre = preprocess_mission_from_query(query)
    if isinstance(data_used, dict):
        data_used["mission_preprocessing"] = pre.to_public_dict()
        data_used["mission_preprocessing_json"] = preprocess_mission_json(query)
        data_used["mission_preprocessing_meta"] = {
            "route_evidence": pre.route_evidence,
            "extraction_notes": list(pre.extraction_notes),
        }
    return pre
