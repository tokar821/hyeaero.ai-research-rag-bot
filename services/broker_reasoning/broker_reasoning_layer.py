"""
Phase 40 orchestrator — broker reasoning interpretation layer.

Runs after adversarial preprocess, before IntentLock. Metadata only; does not alter
IntentLock, dispatch ordering, market/adversarial/consistency math, or UBS core fields.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_reasoning.category_resolver import (
    resolve_category,
    resolve_reference_alternatives,
)
from services.broker_reasoning.comparison_soft_resolution import (
    comparison_models_for_dispatch,
    soft_resolve_comparison,
)
from services.broker_reasoning.intent_expander import ExpandedIntent, IntentCategory, expand_intent
from services.broker_reasoning.mission_interpreter import MissionInterpretation, interpret_mission
from services.broker_reasoning.multi_intent_planner import plan_multi_intent
from services.intent_collapse.canonical_intent_frame import CanonicalIntentFrame, PrimaryIntent

_BUY_UNDER_RE = re.compile(
    r"(?is)\b(?:buy|get|find|looking\s+for)\s+(?:a|an|the\s+)?(?P<model>[A-Za-z][\w\s+\-]{2,40}?)\s+"
    r"(?:under|below|within)\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b",
)
_CAN_GET_FOR_RE = re.compile(
    r"(?is)\bcan\s+i\s+get\s+(?:a|an)\s+(?P<model>[A-Za-z][\w\s+\-]{2,40}?)\s+"
    r"(?:for|at)\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b",
)
_ACQUISITION_SHAPE_RE = re.compile(
    r"(?is)\b(?:buy|get|find|can\s+i\s+get|looking\s+for|best\s+(?:jet|aircraft)|what\s+(?:can|should)\s+i\s+buy)\b",
)


def _price_to_usd(amount: str, unit: str) -> Optional[float]:
    try:
        val = float(str(amount).replace(",", ""))
    except ValueError:
        return None
    u = (unit or "").lower()
    if u in ("m", "mm", "million", "mil"):
        return val * 1_000_000.0
    if u == "k":
        return val * 1_000.0
    if val < 1000:
        return val * 1_000_000.0
    return val


def _resolve_model_name(raw: str) -> Optional[str]:
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    model = (raw or "").strip().rstrip("?.!,")
    if not model:
        return None
    alias = resolve_aircraft_alias(model) or model
    lock = lock_comparison_aircraft([alias])
    if lock.canonical:
        return lock.canonical[0]
    return alias if alias else None


def infer_buy_fields(query: str) -> Optional[Dict[str, Any]]:
    """Infer buy/acquisition fields when explicit year+ask regex does not match."""
    q = (query or "").strip()
    for pat in (_BUY_UNDER_RE, _CAN_GET_FOR_RE):
        m = pat.search(q)
        if not m:
            continue
        model = _resolve_model_name(m.group("model") or "")
        budget_usd = _price_to_usd(m.group("amt"), m.group("unit") or "m")
        if model and budget_usd:
            return {
                "model": model,
                "year": None,
                "ask_usd": None,
                "budget_usd": budget_usd,
                "budget_musd": budget_usd / 1_000_000.0,
                "kind": "acquisition_budget",
            }
    return None


def _frame_from_data_used(du: Dict[str, Any]) -> Optional[CanonicalIntentFrame]:
    if not du.get("intent_collapse_applied"):
        return None
    return CanonicalIntentFrame.from_dict(du.get("canonical_intent_frame"))


def _expanded_from_frame(frame: CanonicalIntentFrame) -> ExpandedIntent:
    """Map collapsed intent to legacy expansion shape — execution only, no re-interpretation."""
    scope = frame.aircraft_scope
    mfr = scope.manufacturer
    price_high = scope.price_sensitive or frame.budget.tier_hint == "LOW"
    primary = frame.primary_intent

    if primary == PrimaryIntent.COMPARE.value:
        return ExpandedIntent(
            category=IntentCategory.COMPARISON,
            manufacturer=mfr,
            acquisition_focus=False,
            price_sensitivity="high" if price_high else "normal",
            raw_signals=["canonical_frame"],
        )

    if scope.scope_type == "ENTRY_LEVEL_GULFSTREAM_SCOPE":
        return ExpandedIntent(
            category=IntentCategory.MANUFACTURER_FAMILY,
            manufacturer="Gulfstream",
            acquisition_focus=True,
            price_sensitivity="high",
            budget_sensitive=frame.budget.unknown,
            intent_hint="manufacturer_discovery",
            raw_signals=["canonical_frame", "entry_level_gulfstream"],
        )

    if scope.models and len(scope.models) == 1:
        return ExpandedIntent(
            category=IntentCategory.EXPLICIT_MODEL,
            reference_model=scope.models[0],
            manufacturer=mfr,
            acquisition_focus=primary in (PrimaryIntent.BUY.value, PrimaryIntent.DISCOVERY.value),
            price_sensitivity="high" if price_high else "normal",
            budget_sensitive=not frame.budget.unknown,
            raw_signals=["canonical_frame"],
        )

    if mfr:
        return ExpandedIntent(
            category=IntentCategory.MANUFACTURER_FAMILY,
            manufacturer=mfr,
            acquisition_focus=primary in (PrimaryIntent.BUY.value, PrimaryIntent.DISCOVERY.value),
            price_sensitivity="high" if price_high else "normal",
            budget_sensitive=not frame.budget.unknown,
            intent_hint="manufacturer_discovery",
            raw_signals=["canonical_frame"],
        )

    return ExpandedIntent(
        category=IntentCategory.UNKNOWN,
        acquisition_focus=primary == PrimaryIntent.BUY.value,
        price_sensitivity="high" if price_high else "normal",
        raw_signals=["canonical_frame"],
    )


def _mission_from_frame(frame: CanonicalIntentFrame) -> MissionInterpretation:
    mf = frame.mission
    budget_m = frame.budget.cap_musd
    return MissionInterpretation(
        acquisition_budget_musd=budget_m,
        acquisition_budget_usd=budget_m * 1_000_000.0 if budget_m is not None else None,
        passengers=mf.pax,
        route=mf.route,
        range_nm=mf.range_nm,
        missing_fields=list(mf.missing_fields),
        follow_up_questions=[],
    )


def _apply_reasoning_from_canonical_frame(
    query: str,
    du: Dict[str, Any],
    frame: CanonicalIntentFrame,
) -> Dict[str, Any]:
    """Execute canonical frame — downstream modules must not re-classify intent."""
    q = (query or "").strip()
    expanded = _expanded_from_frame(frame)
    mission = _mission_from_frame(frame)
    multi = plan_multi_intent(q)

    scope = frame.aircraft_scope
    if scope.models and scope.scope_type in (
        "EXPLICIT_MODELS",
        "COMPARISON_PAIR",
        "ENTRY_LEVEL_GULFSTREAM_SCOPE",
    ):
        candidates = list(scope.models)
        ranking_basis = "canonical_frame"
        notes = ["Resolved by intent collapse layer."]
    else:
        category = resolve_category(
            q,
            manufacturer=scope.manufacturer or expanded.manufacturer,
            budget_musd=frame.budget.cap_musd,
            price_sensitive=scope.price_sensitive,
        )
        candidates = list(category.candidates)
        ranking_basis = category.ranking_basis
        notes = list(category.notes) + ["category_from_canonical_execution"]

    alternatives = None
    if expanded.alternative_search and expanded.reference_model:
        alternatives = resolve_reference_alternatives(
            expanded.reference_model,
            budget_musd=frame.budget.cap_musd,
            lower_cost=expanded.constraint == "lower_acquisition_cost",
        )

    buy_hint = None
    if frame.budget.cap_musd is not None and scope.models:
        buy_hint = {
            "model": scope.models[0],
            "budget_musd": frame.budget.cap_musd,
            "budget_usd": frame.budget.cap_musd * 1_000_000.0,
            "kind": "canonical_frame",
        }
    elif frame.budget.cap_musd is not None:
        buy_hint = infer_buy_fields(q) or {
            "budget_musd": frame.budget.cap_musd,
            "budget_usd": frame.budget.cap_musd * 1_000_000.0,
            "kind": "canonical_frame",
        }
    else:
        buy_hint = infer_buy_fields(q)

    compare_models: Optional[List[str]] = None
    soft = None
    if frame.primary_intent == PrimaryIntent.COMPARE.value and len(scope.models) >= 2:
        compare_models = scope.models[:2]
        soft = soft_resolve_comparison(q)
    elif frame.primary_intent == PrimaryIntent.COMPARE.value:
        soft = soft_resolve_comparison(q)
        compare_models = comparison_models_for_dispatch(soft)

    patch: Dict[str, Any] = {
        "intent_expansion": expanded.to_dict(),
        "mission": mission.to_dict(),
        "category": {
            "phrase": scope.manufacturer or "canonical",
            "manufacturer": scope.manufacturer,
            "candidates": candidates,
            "ranking_basis": ranking_basis,
            "notes": notes,
        },
        "multi_intent": multi.to_dict(),
        "buy_parse_hint": buy_hint,
        "canonical_execution": True,
    }

    if soft is not None:
        patch["comparison_soft"] = soft.to_dict()
        if compare_models:
            patch["compare_models"] = compare_models

    if alternatives is not None:
        patch["alternatives"] = {
            "reference_model": expanded.reference_model,
            "candidates": list(alternatives.candidates),
            "notes": list(alternatives.notes),
        }

    du["broker_reasoning"] = patch
    du["broker_reasoning_layer_applied"] = 1
    du["broker_reasoning_from_canonical_frame"] = 1
    return patch


def apply_broker_reasoning_layer(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full broker reasoning interpretation and stamp ``data_used`` (additive only)."""
    du = data_used if isinstance(data_used, dict) else {}
    q = (query or "").strip()

    try:
        from services.broker_execution.execution_intent_lock import (
            attach_execution_intent_lock,
            should_skip_broker_reasoning_layer,
        )

        attach_execution_intent_lock(du, q)
        if should_skip_broker_reasoning_layer(du):
            du["broker_reasoning_layer_skipped"] = 1
            return {}
    except Exception:
        pass

    frame = _frame_from_data_used(du)
    if frame is not None:
        return _apply_reasoning_from_canonical_frame(q, du, frame)

    expanded = expand_intent(q, data_used=du)
    mission = interpret_mission(q)
    multi = plan_multi_intent(q)
    soft = soft_resolve_comparison(q)

    category = resolve_category(
        q,
        manufacturer=expanded.manufacturer,
        budget_musd=mission.acquisition_budget_musd,
        price_sensitive=expanded.price_sensitivity == "high" or expanded.budget_sensitive,
    )

    alternatives = None
    if expanded.alternative_search and expanded.reference_model:
        alternatives = resolve_reference_alternatives(
            expanded.reference_model,
            budget_musd=mission.acquisition_budget_musd,
            lower_cost=expanded.constraint == "lower_acquisition_cost",
        )

    buy_hint = infer_buy_fields(q)
    compare_models = comparison_models_for_dispatch(soft)

    patch: Dict[str, Any] = {
        "intent_expansion": expanded.to_dict(),
        "mission": mission.to_dict(),
        "category": {
            "phrase": category.phrase,
            "manufacturer": category.manufacturer,
            "candidates": list(category.candidates),
            "ranking_basis": category.ranking_basis,
            "notes": list(category.notes),
        },
        "multi_intent": multi.to_dict(),
        "buy_parse_hint": buy_hint,
    }

    if soft is not None:
        patch["comparison_soft"] = soft.to_dict()
        if compare_models:
            patch["compare_models"] = compare_models

    if alternatives is not None:
        patch["alternatives"] = {
            "reference_model": expanded.reference_model,
            "candidates": list(alternatives.candidates),
            "notes": list(alternatives.notes),
        }

    du["broker_reasoning"] = patch
    du["broker_reasoning_layer_applied"] = 1
    return patch


def _reasoning(du: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(du, dict):
        return {}
    br = du.get("broker_reasoning")
    return br if isinstance(br, dict) else {}


def get_broker_reasoning_compare_models(data_used: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    br = _reasoning(data_used)
    models = br.get("compare_models")
    if isinstance(models, list) and len(models) >= 2:
        return [str(m) for m in models[:2]]
    soft = br.get("comparison_soft") or {}
    if isinstance(soft, dict):
        action = str(soft.get("action") or "")
        raw_models = soft.get("models") or []
        if action in ("auto", "auto_with_note") and len(raw_models) >= 2:
            return [str(m) for m in raw_models[:2]]
    return None


def get_broker_reasoning_buy_parse(data_used: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    br = _reasoning(data_used)
    hint = br.get("buy_parse_hint")
    return hint if isinstance(hint, dict) else None


def is_acquisition_budget_query(query: str) -> bool:
    q = (query or "").strip()
    if not _ACQUISITION_SHAPE_RE.search(q):
        return False
    if infer_buy_fields(q) is not None:
        return True
    mission = interpret_mission(q)
    if mission.acquisition_budget_musd is None:
        return False
    # Require explicit budget language — avoid mission-range false positives.
    return bool(
        re.search(
            r"(?is)\b(?:under|below|budget|around|about|\$\d|\d+\s*m\b|\d+\s*million|what\s+(?:can|should)\s+i\s+buy\s+for)\b",
            q,
        )
    )


def render_acquisition_guidance(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Deterministic broker guidance for budget/manufacturer discovery queries."""
    du = data_used or {}
    br = _reasoning(du)
    if not br:
        apply_broker_reasoning_layer(query, data_used=du)
        br = _reasoning(du)

    expansion = br.get("intent_expansion") or {}
    mission = br.get("mission") or {}
    category = br.get("category") or {}
    alternatives = br.get("alternatives") or {}
    buy_hint = br.get("buy_parse_hint") or {}

    lines: List[str] = []
    budget_m = mission.get("acquisition_budget_musd")
    follow_ups = list(mission.get("follow_up_questions") or [])[:1]

    if buy_hint.get("model") and buy_hint.get("budget_musd"):
        model = str(buy_hint["model"])
        cap = float(buy_hint["budget_musd"])
        lines.append(
            f"At ${cap:.1f}M, a {model} is "
            + ("a realistic target in today's market." if cap >= 12 else "likely tight — expect older hours, deferred maintenance, or a long search.")
        )
        lines.append(
            "I would verify total acquisition cost (paint, engines, avionics) before treating the cap as all-in."
        )
        return "\n\n".join(lines)

    if alternatives.get("candidates"):
        ref = alternatives.get("reference_model") or expansion.get("reference_model") or "that model"
        alts = ", ".join(str(c) for c in alternatives["candidates"][:4])
        lines.append(f"If you want something like a {ref} but cheaper, start with: {alts}.")
        for note in alternatives.get("notes") or []:
            lines.append(str(note))
        if follow_ups:
            lines.append(follow_ups[0])
        return "\n\n".join(lines)

    candidates = category.get("candidates") or []
    mfr = category.get("manufacturer") or expansion.get("manufacturer")
    super_mid = bool(re.search(r"(?is)\bsuper-?\s*midsize\b", query or ""))
    if candidates and (mfr or super_mid):
        ranked = ", ".join(str(c) for c in candidates[:4])
        label = mfr or "super-midsize"
        if expansion.get("price_sensitivity") == "high":
            lines.append(f"For a budget-conscious {label} search, I would start with: {ranked}.")
        elif budget_m is not None:
            lines.append(f"With roughly ${budget_m:.0f}M, credible {label} options include: {ranked}.")
        else:
            lines.append(f"Within {label}, the credible starting points are: {ranked}.")
        for note in category.get("notes") or []:
            lines.append(str(note))
        if follow_ups:
            lines.append(follow_ups[0])
        return "\n\n".join(lines)

    if budget_m is not None:
        lines.append(
            f"With about ${budget_m:.0f}M, you are in large-cabin to entry-ultra-long territory depending on age and hours."
        )
        lines.append(
            "Tell me whether range, cabin size, or operating cost matters most and I will narrow the list."
        )
        return "\n\n".join(lines)

    if follow_ups:
        return follow_ups[0]

    return ""


def append_multi_intent_overlays(
    answer: str,
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    db: Any = None,
) -> str:
    """Append secondary overlays (temporal, buy read) without changing primary dispatch."""
    du = data_used if isinstance(data_used, dict) else {}
    br = _reasoning(du)
    multi = br.get("multi_intent") or {}
    overlays = list(multi.get("overlays") or [])
    if not overlays or not (answer or "").strip():
        return answer

    parts = [answer.strip()]
    compare_models = get_broker_reasoning_compare_models(du)
    if not compare_models:
        from services.routing.authority_dispatch import _comparison_models

        compare_models = _comparison_models(query)[:2]

    if "temporal" in overlays and len(compare_models or []) >= 2:
        try:
            from services.temporal_market.temporal_market_intelligence import (
                format_comparison_temporal_overlay,
            )

            overlay = format_comparison_temporal_overlay(compare_models[0], compare_models[1], db=db)
            if overlay:
                parts.append("")
                parts.append("Market trend read:")
                parts.extend(overlay)
        except Exception:
            pass

    if "buy_read" in overlays and len(compare_models or []) >= 2:
        parts.append("")
        parts.append("Acquisition read:")
        for model in compare_models[:2]:
            tier = _ACQUISITION_TIER_LINE(model)
            parts.append(f"• {model}: {tier}")

    if "valuation_snapshot" in overlays and compare_models:
        parts.append("")
        parts.append("Valuation snapshot:")
        for model in compare_models[:2]:
            parts.append(f"• {model}: confirm year and total time for a band — I can value a specific tail on request.")

    return "\n".join(parts).strip()


def _ACQUISITION_TIER_LINE(model: str) -> str:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    tier = _ACQUISITION_TIER_MUSD.get(model)
    if tier is None:
        return "confirm year and hours for a market band."
    return f"typical acquisition tier roughly ${tier:.0f}M+ for late-model examples (varies by year and program status)."


__all__ = [
    "apply_broker_reasoning_layer",
    "append_multi_intent_overlays",
    "get_broker_reasoning_buy_parse",
    "get_broker_reasoning_compare_models",
    "infer_buy_fields",
    "is_acquisition_budget_query",
    "render_acquisition_guidance",
]
