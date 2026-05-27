"""
Hook: apply Consultant Intelligence Layer before containment (response_safety).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def apply_consultant_intelligence_before_containment(
    answer: str,
    *,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer, data_used) with intelligence metadata merged into data_used.
    """
    du = dict(data_used) if isinstance(data_used, dict) else {}
    try:
        from services.consultant.intelligence_engine import run_consultant_intelligence_layer

        result = run_consultant_intelligence_layer(
            answer=answer,
            query=query,
            history=history,
            data_used=du,
            conversation_state=conversation_state,
        )
        du.update(result.data_used_patch)
        return result.answer, du
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "consultant intelligence layer skipped (non-fatal): %s", exc
        )
        du["consultant_intelligence_error"] = str(exc)[:200]
        return answer, du
