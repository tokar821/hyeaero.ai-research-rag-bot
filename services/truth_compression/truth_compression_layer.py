"""
Phase 46 — single truth pipeline compression (presentation only).

Runs after executive broker, before conversation layer.
Does not alter routing, IntentLock, valuation, adversarial, or decision logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.truth_compression.decision_deduplicator import deduplicate_decisions_in_answer
from services.truth_compression.redundancy_detector import detect_redundant_pathways
from services.truth_compression.response_simplifier import simplify_response
from services.truth_compression.truth_synthesizer import BrokerTruthState, synthesize_truth_state

logger = logging.getLogger(__name__)


def apply_truth_compression(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Collapse redundant multi-layer expression into a single broker voice.
    """
    del query
    raw = (answer or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not raw:
        return raw

    truth = synthesize_truth_state(du)
    pathways = detect_redundant_pathways(raw, truth, data_used=du)

    compressed = raw
    if pathways or truth.has_executive_recommendation:
        compressed = deduplicate_decisions_in_answer(compressed, truth)
        compressed = simplify_response(compressed, truth, pathways=pathways)

    du["broker_truth_state"] = truth.to_dict()
    du["redundant_truth_pathways"] = pathways
    du["truth_compression_applied"] = 1

    if pathways:
        logger.debug("truth compression pathways: %s", pathways[:4])

    return compressed or raw


def compress_ui_contract_sections(
    contract: Dict[str, Any],
    truth: BrokerTruthState,
    *,
    pathways: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Remove mirrored UI sections when truth compression has a single recommendation owner.
    """
    if not isinstance(contract, dict):
        return contract
    out = dict(contract)
    paths = pathways or []

    sections = [s for s in (out.get("sections") or []) if isinstance(s, dict)]
    if not sections:
        return out

    if truth.has_executive_recommendation:
        primary = (truth.primary_model or "").lower()
        rec_indices = [i for i, s in enumerate(sections) if s.get("type") == "recommendation"]
        if len(rec_indices) > 1:
            keep_i = rec_indices[0]
            if primary:
                for i in rec_indices:
                    if primary in str(sections[i].get("content") or "").lower():
                        keep_i = i
                        break
            sections = [
                s for i, s in enumerate(sections)
                if s.get("type") != "recommendation" or i == keep_i
            ]

        if "REDUNDANT_TEMPLATE_HEADERS" in paths:
            seen: set[str] = set()
            deduped: List[Dict[str, Any]] = []
            for sec in sections:
                st = str(sec.get("type") or "")
                if st in seen and st in ("overview", "analysis", "recommendation"):
                    continue
                if st:
                    seen.add(st)
                deduped.append(sec)
            sections = deduped

    out["sections"] = sections
    out.setdefault("render_hints", {})
    if isinstance(out["render_hints"], dict):
        out["render_hints"]["single_authority"] = "executive_broker"

    return out


__all__ = [
    "apply_truth_compression",
    "compress_ui_contract_sections",
]
