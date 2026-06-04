"""
Output governance — single client-answer post-pipeline.

When the LLM is the primary author, template rewriters are skipped; only hygiene,
deduplication, and the final render contract run. Observability keys are unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.broker_execution.response_mode_classifier import (
    ResponseMode,
    classify_response_mode,
)

logger = logging.getLogger(__name__)

_REGISTRY_BLOCK_RE = re.compile(
    r"(?is)\[(?:registry\s+facts|tail\s+registry|faa\s+registry)\][\s\S]*?(?:\n\n|\Z)"
)
_DEBUG_LINE_RE = re.compile(
    r"(?im)^\s*(?:execution_path|tier_source|market_source|pre_llm_executed|"
    r"authority_dispatch_kind)\s*[:=].*$"
)
_INTERNAL_SECTION_RE = re.compile(
    r"(?im)^\s*(?:operational\s+synthesis|approved\s+shortlist|mission\s+authority\s+kernel)\s*:?\s*$"
)


def is_llm_primary_output(data_used: Optional[Dict[str, Any]]) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    return bool(du.get("llm_executed") or du.get("consultant_llm_draft"))


@dataclass(frozen=True)
class OutputGovernancePlan:
    """Which post-layers may mutate client-visible text for this turn."""

    response_mode: ResponseMode
    llm_primary: bool
    broker_decision: bool = False
    personalize: bool = False
    acquisition_budget: bool = False
    market_reality: bool = False
    data_first: bool = False
    executive: bool = False
    budget_opening: bool = False
    truth_compression: bool = True
    comparison_guard: bool = False
    conversation_full: bool = False
    conversation_hygiene: bool = True
    compression: bool = True
    final_contract: bool = True


def resolve_output_governance(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> OutputGovernancePlan:
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    mode = classify_response_mode(q, data_used=du)
    llm = is_llm_primary_output(du)
    profile = str(du.get("execution_profile") or "").strip().lower()

    if llm:
        return OutputGovernancePlan(
            response_mode=mode,
            llm_primary=True,
            truth_compression=False,
            conversation_hygiene=True,
            compression=True,
            final_contract=True,
        )

    if profile in (
        "mission",
        "comparison",
        "tail_owner",
        "tail_sale_status",
        "tail_summary",
        "tail_detail",
    ):
        return OutputGovernancePlan(
            response_mode=mode,
            llm_primary=False,
            broker_decision=False,
            personalize=False,
            acquisition_budget=False,
            market_reality=False,
            data_first=False,
            executive=False,
            budget_opening=False,
            truth_compression=False,
            comparison_guard=False,
            conversation_full=False,
            conversation_hygiene=True,
            compression=False,
            final_contract=True,
        )

    from services.broker_execution.broker_execution_category import (
        classify_broker_execution_category,
        data_first_required,
        executive_layer_allowed,
    )

    cat = classify_broker_execution_category(q, data_used=du)
    allow_executive = executive_layer_allowed(cat, q)
    allow_data_first = data_first_required(cat)

    return OutputGovernancePlan(
        response_mode=mode,
        llm_primary=False,
        broker_decision=True,
        personalize=True,
        acquisition_budget=True,
        market_reality=mode not in (ResponseMode.FACT_ONLY,),
        data_first=allow_data_first,
        executive=allow_executive,
        budget_opening=allow_executive and mode not in (ResponseMode.FACT_ONLY,),
        truth_compression=True,
        comparison_guard=mode == ResponseMode.COMPARISON,
        conversation_full=True,
        conversation_hygiene=True,
        compression=True,
        final_contract=True,
    )


def _apply_conversation_hygiene(
    answer: str,
    *,
    query: str,
    data_used: dict,
) -> str:
    from services.broker.broker_language import sanitize_broker_language
    from services.conversation.output_cleaner import clean_broker_output

    raw = (answer or "").strip()
    if not raw:
        return raw
    cleaned = clean_broker_output(raw, use_unicode_bullets=True)
    return sanitize_broker_language(cleaned)


def enforce_final_render_contract(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Last pass: strip internal blocks and enforce mode verbosity."""
    from services.broker_execution.final_render_stripper import strip_report_scaffolds
    from services.broker_execution.response_compression_formatters import (
        _FORBIDDEN_ALL_RE,
        _BROKER_TEMPLATE_RE,
        _compact_fact_lines,
        _strip_forbidden_narrative,
    )
    from services.broker_execution.response_compression_layer import truncate_to_token_budget
    from services.broker_execution.response_deduplication import deduplicate_lines
    from services.broker_execution.response_mode_classifier import MAX_TOKENS_BY_MODE

    raw = (answer or "").strip()
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not raw:
        return raw

    body = _REGISTRY_BLOCK_RE.sub("", raw)
    lines = []
    for line in body.splitlines():
        if _DEBUG_LINE_RE.match(line.strip()):
            continue
        if _INTERNAL_SECTION_RE.match(line.strip()):
            continue
        lines.append(line)
    body = deduplicate_lines("\n".join(lines).strip())

    mode = classify_response_mode(q, data_used=du)
    llm = is_llm_primary_output(du)
    fact_only = mode == ResponseMode.FACT_ONLY

    if llm:
        body = strip_report_scaffolds(body, fact_only=fact_only)
    body = _strip_forbidden_narrative(body)

    if fact_only:
        facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
        bloated = (
            _FORBIDDEN_ALL_RE.search(raw)
            or _BROKER_TEMPLATE_RE.search(raw)
            or len(body) > 400
            or body.lower().count("owner") > 2
            or "FAA-registered address" in body
        )
        if facts and (bloated or llm):
            compact = _compact_fact_lines(
                [f for f in facts if isinstance(f, dict)],
                max_lines=5,
            )
            if compact:
                body = compact
        max_tok = MAX_TOKENS_BY_MODE.get(mode, 90)
        body = truncate_to_token_budget(body, max_tok)
    elif llm and mode in (ResponseMode.COMPARISON, ResponseMode.LISTING):
        body = strip_report_scaffolds(body, fact_only=False)
        max_tok = MAX_TOKENS_BY_MODE.get(mode, 450)
        body = truncate_to_token_budget(body, max_tok)
    elif llm and mode == ResponseMode.MISSION:
        body = strip_report_scaffolds(body, fact_only=False)
        max_tok = MAX_TOKENS_BY_MODE.get(mode, 420)
        body = truncate_to_token_budget(body, max_tok)

    du["final_render_contract_applied"] = 1
    du["output_governance_mode"] = mode.value
    du["final_render_llm_primary"] = int(llm)
    return body.strip()


def apply_governed_client_answer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Single post-pipeline entry: recovery, optional template layers, hygiene, final contract.

    LLM-primary turns use ``render_client_answer`` only (one writer after the model).
    """
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    try:
        from services.broker_execution.tail_acquisition_dossier import resolve_query_with_active_tail

        q = resolve_query_with_active_tail(q, du)
    except Exception:
        pass
    try:
        from services.broker_execution.visualization_intent import (
            detect_visualization_intent,
            render_visualization_fallback,
        )

        wants_viz, vkind = detect_visualization_intent(q)
        if wants_viz and not (answer or "").strip():
            fb = render_visualization_fallback(q, kind=vkind)
            if fb:
                return fb
    except Exception:
        pass
    if is_llm_primary_output(du):
        from services.broker_execution.client_answer_renderer import render_client_answer

        return render_client_answer(answer, query=q, data_used=du)

    plan = resolve_output_governance(q, du)
    du["output_governance_llm_primary"] = int(plan.llm_primary)
    du["output_governance_plan"] = {
        k: v
        for k, v in plan.__dict__.items()
        if k != "response_mode" and isinstance(v, (bool, int))
    }
    du["output_governance_plan"]["response_mode"] = plan.response_mode.value

    try:
        from services.broker_execution.broker_execution_category import attach_broker_execution_context

        attach_broker_execution_context(du, query=q)
    except Exception as exc:
        logger.debug("broker execution context skipped: %s", exc)

    try:
        from services.broker_execution.listing_market_reasoning import render_listing_price_reasoning

        listing_body = render_listing_price_reasoning(q, du)
        if listing_body and (not (answer or "").strip() or "median" in (answer or "").lower()[:200]):
            du["listing_market_reasoning_applied"] = 1
            body = listing_body
        else:
            body = (answer or "").strip()
    except Exception:
        body = (answer or "").strip()
    skip_authority = bool(plan.llm_primary)
    try:
        from services.consultant.answer_recovery import recover_client_answer
        from services.consultant.model_authority_guard import enforce_model_authority

        if not body or not plan.llm_primary:
            body = recover_client_answer(query=q, data_used=du, answer=body)
        if not skip_authority:
            body = enforce_model_authority(body, du, query=q)
        else:
            du["model_authority_skipped_llm_primary"] = 1
    except Exception as exc:
        logger.debug("answer recovery skipped: %s", exc)

    if plan.broker_decision:
        try:
            from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis

            body = apply_broker_decision_synthesis(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("broker decision synthesis skipped: %s", exc)

    if plan.personalize:
        try:
            from services.client_context.client_context_layer import personalize_client_answer

            body = personalize_client_answer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("client context personalization skipped: %s", exc)

    if plan.acquisition_budget:
        try:
            from services.executive_broker.acquisition_budget_reality import apply_acquisition_budget_reality

            body = apply_acquisition_budget_reality(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("acquisition budget reality skipped: %s", exc)

    if plan.market_reality:
        try:
            from services.market_reality.market_reality_layer import apply_market_reality_layer

            body = apply_market_reality_layer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("market reality layer skipped: %s", exc)

    if plan.data_first:
        try:
            from services.broker_execution.data_first_layer import apply_data_first_layer

            body = apply_data_first_layer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("data-first layer skipped: %s", exc)

    if plan.executive:
        try:
            from services.executive_broker.executive_broker_layer import apply_executive_broker_layer

            body = apply_executive_broker_layer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("executive broker layer skipped: %s", exc)

    if plan.budget_opening:
        try:
            from services.executive_broker.acquisition_budget_reality import prepend_budget_reality_opening

            body = prepend_budget_reality_opening(body, data_used=du)
        except Exception as exc:
            logger.debug("budget reality opening prepend skipped: %s", exc)

    if plan.truth_compression:
        try:
            from services.truth_compression.truth_compression_layer import apply_truth_compression

            body = apply_truth_compression(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("truth compression skipped: %s", exc)

    if plan.comparison_guard:
        try:
            from services.broker_execution.comparison_presentation_guard import (
                apply_comparison_presentation_guard,
            )

            body = apply_comparison_presentation_guard(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("comparison presentation guard skipped: %s", exc)

    if plan.conversation_full:
        try:
            from services.conversation.broker_conversation_layer import apply_broker_conversation_layer

            body = apply_broker_conversation_layer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("broker conversation layer skipped: %s", exc)
    elif plan.conversation_hygiene:
        body = _apply_conversation_hygiene(body, query=q, data_used=du)

    if plan.compression:
        try:
            from services.broker_execution.response_compression_layer import (
                apply_response_compression_layer,
            )

            body = apply_response_compression_layer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("response compression layer skipped: %s", exc)

    if plan.final_contract:
        body = enforce_final_render_contract(body, query=q, data_used=du)

    if not plan.llm_primary:
        try:
            from services.broker_execution.deterministic_answer_renderer import (
                render_deterministic_client_answer,
            )

            body = render_deterministic_client_answer(body, query=q, data_used=du)
        except Exception as exc:
            logger.debug("deterministic answer renderer skipped: %s", exc)

    try:
        from services.broker_execution.fact_flow import attach_fact_flow
        from services.broker_execution.retrieval_utilization import attach_retrieval_utilization

        attach_fact_flow(q, body, du)
        attach_retrieval_utilization(body, du)
    except Exception as exc:
        logger.debug("retrieval utilization metrics skipped: %s", exc)

    du["output_governance_applied"] = 1
    return body.strip()


def refresh_cached_consultant_payload(
    payload: Dict[str, Any],
    *,
    query: str = "",
) -> Dict[str, Any]:
    """
    Re-apply output governance on cache hits so stale template-heavy answers cannot replay.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    answer = str(out.get("answer") or "").strip()
    if not answer:
        return out
    du = dict(out.get("data_used") or {})
    trace = du.get("execution_trace") if isinstance(du.get("execution_trace"), dict) else {}
    if trace.get("llm_executed"):
        du.setdefault("llm_executed", True)
        du.setdefault("consultant_llm_draft", 1)
    q = (query or du.get("query") or "").strip()
    if is_llm_primary_output(du):
        answer = apply_governed_client_answer(answer, query=q, data_used=du)
    else:
        answer = enforce_final_render_contract(answer, query=q, data_used=du)
    du["cached_answer_governance_refresh"] = 1
    out["answer"] = answer
    out["data_used"] = du
    return out


def mark_llm_primary_data_used(data_used: Optional[Dict[str, Any]]) -> None:
    """Call when query_service completes an LLM draft (authoritative contract)."""
    if not isinstance(data_used, dict):
        return
    data_used["llm_executed"] = True
    data_used["consultant_llm_draft"] = 1
    trace = data_used.get("execution_trace")
    if isinstance(trace, dict):
        trace["llm_executed"] = True
    else:
        data_used["execution_trace"] = {"llm_executed": True}


__all__ = [
    "OutputGovernancePlan",
    "apply_governed_client_answer",
    "enforce_final_render_contract",
    "is_llm_primary_output",
    "mark_llm_primary_data_used",
    "refresh_cached_consultant_payload",
    "resolve_output_governance",
]
