"""
Fact Pack Builder — structured verified facts only (no broker prose).

Feeds the LLM context layer; never emits client-facing answers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def build_fact_pack(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect authoritative structured facts for the current turn."""
    du = data_used if isinstance(data_used, dict) else {}
    pack: Dict[str, Any] = {
        "query": (query or "").strip(),
        "facts": [],
        "sources": {},
    }

    tail_ctx = str(du.get("tail_registry_llm_context") or "").strip()
    if tail_ctx:
        pack["sources"]["tail_registry"] = True
        pack["facts"].append({"kind": "tail_registry_block", "value": tail_ctx[:4000]})

    for key in ("tail_facts", "tail_selected_facts"):
        rows = du.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("value"):
                    pack["facts"].append(
                        {
                            "kind": str(row.get("kind") or "fact"),
                            "label": str(row.get("label") or ""),
                            "value": str(row.get("value") or ""),
                        }
                    )
            if rows:
                pack["sources"]["tail_facts"] = True
                break

    try:
        from services.broker_execution.tail_acquisition_dossier import build_tail_acquisition_dossier_block

        dossier = build_tail_acquisition_dossier_block(query, du)
        if dossier:
            pack["sources"]["tail_acquisition_dossier"] = True
            pack["facts"].append({"kind": "tail_acquisition_dossier", "value": dossier[:6000]})
    except Exception:
        pass

    try:
        from services.broker_execution.tail_aircraft_profile import build_tail_aircraft_profile_block

        profile = build_tail_aircraft_profile_block(query, du)
        if profile:
            pack["sources"]["tail_profile"] = True
            pack["facts"].append({"kind": "tail_profile", "value": profile[:5000]})
    except Exception:
        pass

    cv2 = du.get("comparison_v2")
    if isinstance(cv2, dict) and str(cv2.get("status") or "").upper() == "OK":
        pack["sources"]["comparison_v2"] = True
        pack["facts"].append(
            {
                "kind": "comparison",
                "models": list(cv2.get("models") or []),
                "rows": cv2.get("rows") or cv2.get("comparison_rows"),
            }
        )

    try:
        from services.broker_execution.tail_market_comparison import build_tail_market_comparison_block

        tmc = build_tail_market_comparison_block(query, du)
        if tmc:
            pack["sources"]["tail_market_comparison"] = True
            pack["facts"].append({"kind": "tail_market_comparison", "value": tmc[:4000]})
    except Exception:
        pass

    try:
        from services.broker_execution.comparison_broker_facts import build_comparison_broker_facts_block

        cblock = build_comparison_broker_facts_block(query, du)
        if cblock:
            pack["sources"]["comparison_broker"] = True
            pack["facts"].append({"kind": "comparison_broker", "value": cblock[:4000]})
    except Exception:
        pass

    audit = du.get("listing_parse_audit")
    if isinstance(audit, dict) and audit.get("parse_success"):
        pack["sources"]["listing_parse"] = True
        pack["facts"].append({"kind": "listing", "audit": audit})

    mr = du.get("market_reality")
    if isinstance(mr, dict) and mr:
        pack["sources"]["market_reality"] = True
        pack["facts"].append({"kind": "market_reality", "brief": mr})

    if du.get("authority_dispatch_deferred_llm"):
        pack["sources"]["authority_dispatch"] = str(du.get("authority_dispatch_kind") or "")

    pipeline_facts = str(du.get("pipeline_llm_facts") or "").strip()
    if pipeline_facts:
        pack["sources"]["recommendation_pipeline"] = True
        pack["facts"].append({"kind": "recommendation_pipeline", "value": pipeline_facts[:6000]})
    else:
        pipe = du.get("deterministic_recommendation_pipeline")
        if isinstance(pipe, dict) and pipe.get("recommendations"):
            pack["sources"]["recommendation_pipeline"] = True
            pack["facts"].append(
                {
                    "kind": "recommendation_pipeline",
                    "mission_category": pipe.get("mission_category"),
                    "recommendations": pipe.get("recommendations"),
                    "feasible_models": pipe.get("feasible_models"),
                }
            )

    structured = str(du.get("structured_dispatch_llm_facts") or "").strip()
    if structured:
        pack["sources"]["structured_dispatch"] = str(du.get("authority_dispatch_kind") or "")
        pack["facts"].append({"kind": "structured_dispatch", "value": structured[:4000]})

    try:
        from services.broker_execution.mission_feasibility_broker import (
            build_mission_feasibility_broker_note,
        )

        mf_note = build_mission_feasibility_broker_note(str(pack.get("query") or du.get("consultant_query") or ""))
        if mf_note:
            pack["sources"]["mission_feasibility"] = True
            pack["facts"].append({"kind": "mission_feasibility", "value": mf_note[:2500]})
    except Exception:
        pass

    return pack


def render_fact_pack_for_llm_context(pack: Dict[str, Any]) -> str:
    """Render fact pack as an LLM-only context block (not user-facing)."""
    facts = pack.get("facts") or []
    if not facts:
        return ""
    lines = [
        "[VERIFIED FACT PACK — structured only; narrate in natural language; do not copy template phrases]",
    ]
    for f in facts:
        if not isinstance(f, dict):
            continue
        kind = str(f.get("kind") or "fact")
        if kind == "tail_registry_block":
            lines.append(str(f.get("value") or ""))
            continue
        if kind == "mission_feasibility":
            lines.append(str(f.get("value") or ""))
            continue
        if kind in ("tail_profile", "tail_acquisition_dossier", "comparison_broker"):
            lines.append(str(f.get("value") or ""))
            continue
        if kind in ("recommendation_pipeline", "structured_dispatch"):
            val = str(f.get("value") or "")
            if val:
                lines.append(val)
            elif kind == "recommendation_pipeline":
                try:
                    lines.append(json.dumps(f, default=str)[:3000])
                except Exception:
                    pass
            continue
        if kind == "comparison":
            models = ", ".join(str(m) for m in (f.get("models") or []))
            lines.append(f"Comparison models: {models}")
            rows = f.get("rows")
            if rows:
                try:
                    lines.append(json.dumps(rows, default=str)[:2000])
                except Exception:
                    pass
            continue
        label = str(f.get("label") or kind)
        val = str(f.get("value") or "")
        if val:
            lines.append(f"• {label}: {val}")
    body = "\n".join(lines).strip()
    return body if len(body) > 80 else ""


def attach_fact_pack_to_data_used(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Build pack, store observability on data_used, return context block."""
    du = data_used if isinstance(data_used, dict) else {}
    pack = build_fact_pack(query, du)
    du["fact_pack"] = pack
    du["fact_pack_fact_count"] = len(pack.get("facts") or [])
    block = render_fact_pack_for_llm_context(pack)
    if block:
        du["fact_pack_context_applied"] = 1
    return block


__all__ = [
    "attach_fact_pack_to_data_used",
    "build_fact_pack",
    "render_fact_pack_for_llm_context",
]
