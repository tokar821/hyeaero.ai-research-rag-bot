"""
Single post-layer renderer for deterministic (non-LLM-primary) client answers.

Applies tail shaping, scaffold stripping, and dedupe — one pass at the end.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.broker_execution.client_answer_renderer import collapse_duplicate_registry_blocks
from services.broker_execution.output_governance import is_llm_primary_output


def render_deterministic_client_answer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Final hygiene for template/dispatch paths when the LLM did not author the turn."""
    du = data_used if isinstance(data_used, dict) else {}
    if is_llm_primary_output(du):
        return (answer or "").strip()

    body = (answer or "").strip()
    profile = str(du.get("execution_profile") or "").strip().lower()

    try:
        from services.broker_execution.final_render_stripper import strip_report_scaffolds
        from services.broker_execution.response_mode_classifier import ResponseMode, classify_response_mode

        mode = classify_response_mode(query, data_used=du)
        body = strip_report_scaffolds(body, fact_only=(mode == ResponseMode.FACT_ONLY))
    except Exception:
        pass

    if profile.startswith("tail_"):
        try:
            from services.broker_execution.tail_answer_shaper import shape_tail_client_answer
            from services.broker_execution.tail_depth_mode import TailDepthMode, registry_template_depths

            depth_name = str(du.get("tail_depth_mode") or "").strip().lower()
            try:
                depth_enum = TailDepthMode(depth_name) if depth_name else None
            except ValueError:
                depth_enum = None
            if depth_enum in registry_template_depths() or profile in (
                "tail_owner",
                "tail_sale_status",
            ):
                body = shape_tail_client_answer(body, query=query, data_used=du)
            elif depth_name == "engine_program" or profile == "tail_engine_program":
                from services.broker_execution.tail_acquisition_dossier import render_engine_program_answer

                short = render_engine_program_answer(query, du)
                if short:
                    body = short
                else:
                    shaped = shape_tail_client_answer(body, query=query, data_used=du)
                    if shaped and ("engine program" in shaped.lower() or "apu program" in shaped.lower()):
                        body = shaped
            elif depth_name in ("acquisition_risks", "acquisition") or profile == "tail_acquisition":
                from services.broker_execution.tail_acquisition_dossier import render_acquisition_risks_answer

                risks = render_acquisition_risks_answer(query, du)
                if risks:
                    body = risks
            elif depth_name == "detail" or profile == "tail_detail":
                from services.broker_execution.tail_acquisition_dossier import render_tail_detail_answer

                detail = render_tail_detail_answer(query, du)
                if detail:
                    body = detail
        except Exception:
            pass

    if profile == "comparison":
        try:
            from services.broker_execution.comparison_broker_facts import render_comparison_client_answer

            alt = render_comparison_client_answer(query, du)
            if alt and ("wins on" in alt.lower() or "buy" in alt.lower() or "tradeoff" in alt.lower()):
                body = alt
            elif alt and (not body or "see verified spec" in body.lower() or len(body) < 120):
                body = alt
        except Exception:
            pass

    if profile == "mission":
        try:
            from services.broker_execution.mission_broker_answer import build_deterministic_mission_answer

            mission = build_deterministic_mission_answer(query, du)
            if mission and (
                not body
                or "I would not lead with" in body
                or "INSUFFICIENT_DATA" in body.upper()
                or len(body) < 80
            ):
                body = mission
        except Exception:
            pass

    body = collapse_duplicate_registry_blocks(body)

    try:
        from services.broker_execution.response_compression_formatters import _strip_forbidden_narrative

        body = _strip_forbidden_narrative(body)
    except Exception:
        pass

    du["deterministic_answer_renderer_applied"] = 1
    return body.strip()


__all__ = ["render_deterministic_client_answer"]
