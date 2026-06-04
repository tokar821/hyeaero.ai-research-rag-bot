"""
Phase 42 — client context orchestrator.

Runs at turn start (memory update + consistency hints) and at answer finalize (personalization).
Does not alter routing, market math, or IntentLock.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.client_context.acquisition_stage_detector import detect_acquisition_stage, merge_stage
from services.client_context.answer_personalizer import enrich_query_context_for_reasoning, personalize_answer
from services.client_context.broker_context_builder import BrokerConversationContext, build_broker_context
from services.client_context.client_profile import ClientProfile
from services.client_context.conversation_memory import (
    ConversationMemory,
    memory_to_profile,
    update_memory_from_turn,
)
from services.client_context.recommendation_consistency import (
    apply_consistency_to_broker_reasoning,
)

logger = logging.getLogger(__name__)

_CLIENT_CONTEXT_KEY = "client_context_state"


def _load_state(client_conversation_state: Optional[Dict[str, Any]]) -> tuple[ClientProfile, ConversationMemory]:
    raw = {}
    if isinstance(client_conversation_state, dict):
        raw = client_conversation_state.get(_CLIENT_CONTEXT_KEY) or {}
    if not isinstance(raw, dict):
        raw = {}

    profile = ClientProfile.from_dict(raw.get("profile"))
    memory = ConversationMemory.from_dict(raw.get("memory"))
    return profile, memory


def apply_client_context_turn(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    client_conversation_state: Optional[Dict[str, Any]] = None,
) -> BrokerConversationContext:
    """
    Update rolling memory from this turn and stamp ``data_used`` for downstream layers.
    Call before broker_reasoning / adversarial preprocess.
    """
    du = data_used if isinstance(data_used, dict) else {}
    q = (query or "").strip()

    from services.broker_execution.broker_execution_category import (
        classify_broker_execution_category,
        tail_memory_isolated,
    )

    exec_cat = classify_broker_execution_category(q, data_used=du)
    if tail_memory_isolated(exec_cat):
        du["broker_execution_category"] = exec_cat.value
        du["tail_memory_isolated"] = True
        du["broker_memory_isolated"] = True
        du["executive_layer_allowed"] = False
        du["client_context_layer_applied"] = 1
        du["broker_conversation_context"] = {
            "isolated": True,
            "reason": exec_cat.value,
        }
        return BrokerConversationContext()

    profile, memory = _load_state(client_conversation_state)

    # Merge intent persistence budget/aircraft if present.
    ip = None
    if isinstance(client_conversation_state, dict):
        ip = client_conversation_state.get("intent_persistence")
    if isinstance(du.get("intent_persistence"), dict):
        ip = du.get("intent_persistence")
    if isinstance(ip, dict):
        if ip.get("active_budget_usd"):
            try:
                profile.preferred_budget_musd = float(ip["active_budget_usd"]) / 1_000_000.0
            except (TypeError, ValueError):
                pass
        if ip.get("active_aircraft"):
            ac = str(ip["active_aircraft"]).strip()
            if ac and ac not in profile.preferred_aircraft:
                profile.preferred_aircraft.insert(0, ac)

    memory = update_memory_from_turn(memory, q, history=history)
    profile = memory_to_profile(memory, profile)

    detected = detect_acquisition_stage(q, prior_stage=profile.acquisition_stage)
    profile.acquisition_stage = merge_stage(profile.acquisition_stage, detected)

    intent_ip = ip if isinstance(ip, dict) else None
    ctx = build_broker_context(profile, memory, query=q, intent_persistence=intent_ip)

    enrich_query_context_for_reasoning(du, ctx)
    apply_consistency_to_broker_reasoning(du, ctx)

    du["client_profile"] = profile.to_dict()
    du["client_context"] = ctx.to_dict()
    du["client_context_layer_applied"] = 1

    logger.debug(
        "client context: budget=%s stage=%s targets=%s",
        ctx.remembered_budget_musd,
        ctx.stage,
        ctx.remembered_targets[:3],
    )
    return ctx


def personalize_client_answer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply memory-aware framing to final answer."""
    du = data_used if isinstance(data_used, dict) else {}
    if du.get("broker_memory_isolated") or du.get("tail_memory_isolated"):
        return (answer or "").strip()
    ctx_raw = du.get("broker_conversation_context") or du.get("client_context")
    ctx = BrokerConversationContext.from_dict(ctx_raw if isinstance(ctx_raw, dict) else None)
    if not ctx.remembered_budget_musd and not ctx.remembered_targets and not ctx.preferred_manufacturers:
        return (answer or "").strip()

    out = personalize_answer(answer, query=query, context=ctx, profile_dict=du.get("client_profile"))
    if out != answer:
        du["client_context_personalized"] = 1
    return out


def finalize_client_context(
    data_used: Dict[str, Any],
    client_conversation_state: Optional[Dict[str, Any]],
    *,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Persist client context for client echo on next turn.

    Returns state blob to nest under ``consultant_conversation_state``.
    """
    profile = ClientProfile.from_dict(data_used.get("client_profile"))
    memory = ConversationMemory.from_dict(
        (client_conversation_state or {}).get(_CLIENT_CONTEXT_KEY, {}).get("memory")
        if isinstance(client_conversation_state, dict)
        else None
    )
    if not memory.turn_count:
        memory = ConversationMemory.from_dict(
            (data_used.get("client_profile") and {}) or {}
        )

    memory = update_memory_from_turn(memory, query or "", history=history)
    profile = memory_to_profile(memory, profile)
    if isinstance(data_used.get("client_profile"), dict):
        profile = ClientProfile.from_dict(data_used["client_profile"])

    ctx = build_broker_context(
        profile,
        memory,
        query=query,
        intent_persistence=data_used.get("intent_persistence"),
    )

    blob = {
        "profile": profile.to_dict(),
        "memory": memory.to_dict(),
        "context": ctx.to_dict(),
    }

    if isinstance(client_conversation_state, dict):
        client_conversation_state[_CLIENT_CONTEXT_KEY] = blob
    data_used["client_context_state"] = blob
    return blob


__all__ = [
    "apply_client_context_turn",
    "finalize_client_context",
    "personalize_client_answer",
]
