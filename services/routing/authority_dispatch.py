"""
Consultant authority dispatch — deterministic responder priority override.

Runs before pre-LLM mission pipeline. Does not depend on unified enforce flags or rollout %.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute
from services.recommendation.query_recommendation_intent import (
    QueryRecommendationIntent,
    QueryRecommendationIntentResult,
)


@dataclass(frozen=True)
class AuthorityDispatchResult:
    """When set, caller must return this answer and skip pre-LLM pipeline + legacy LLM."""

    answer: str
    dispatch_kind: str
    progress_step: str
    data_used: Dict[str, Any] = field(default_factory=dict)


_SAFETY_FALLBACK_ANSWERS: Dict[str, str] = {
    "comparison": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Verified catalog comparison requires two recognized aircraft models."
    ),
    "alternative": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Tier-peer alternatives require a verified catalog target aircraft."
    ),
    "buy_decision": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Structured buy-decision analysis requires a resolved model and ask price."
    ),
    "valuation": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Structured valuation requires a resolved aircraft model and verified market context."
    ),
    "fleet": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Fleet portfolio analysis requires at least two verified aircraft in fleet input."
    ),
    "optimization": (
        "Insufficient verified data for deterministic execution.\n\n"
        "Multi-criteria optimization requires at least two verified candidate aircraft."
    ),
}


def _comparison_responder_succeeded(data_used: Dict[str, Any], answer: str) -> bool:
    """
    Structured acceptance for comparison dispatch — do not substring-match answer prose.

    ``respond_aircraft_comparison`` may include ``Insufficient verified`` inside VERDICT
    while still producing a valid catalog comparison (``comparison_v2.status == OK``).
    """
    if not str(answer or "").strip():
        return False
    cv2 = data_used.get("comparison_v2")
    if isinstance(cv2, dict) and str(cv2.get("status") or "").upper() == "OK":
        return True
    engine = data_used.get("comparison_structured_engine")
    if isinstance(engine, dict) and engine.get("type") == "comparison_v2_json":
        return True
    return False


def _build_dispatch_safety_fallback(
    dispatch_kind: str,
    data_used: Dict[str, Any],
) -> AuthorityDispatchResult:
    """Fail-closed deterministic response when hard intent cannot resolve (mirrors Phase 15 guard)."""
    kind = str(dispatch_kind or "comparison").strip().lower()
    if kind not in _SAFETY_FALLBACK_ANSWERS:
        kind = "comparison"
    du = dict(data_used)
    du["deterministic_execution"] = {
        "bypassed_llm": True,
        "trigger_reason": "hard_intent_insufficient_resolution",
        "final_responder": "deterministic_safety_fallback",
        "deterministic_intent": kind,
    }
    du["authority_dispatch_safety_fallback"] = kind
    du["authority_dispatch_kind"] = kind
    return AuthorityDispatchResult(
        answer=_SAFETY_FALLBACK_ANSWERS[kind],
        dispatch_kind=kind,
        progress_step=f"path_authority_dispatch_{kind}_safety_fallback",
        data_used=du,
    )


_ALTERNATIVES_TO_RE = re.compile(
    r"(?is)\balternatives?\s+to\s+(?:a|an|the\s+)?(.+?)\??\s*$"
)

_BUY_DEAL_RE = re.compile(
    r"(?is)(?:"
    r"(?P<year>(?:19|20)\d{2})(?!\s*nm)\s+(?P<model>[A-Za-z][\w\s+\-]{1,40}?)\s+"
    r"(?:\$|usd\s*)(?P<price>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
    r"(?:\s+(?:good\s+deal|fair\s+deal|worth\s+it|overpriced|good\s+buy))?"
    r"|"
    r"(?P<model2>[A-Za-z][\w\s+\-]{1,40}?)\s+(?:\$|usd\s*)(?P<price2>\d+(?:\.\d+)?)\s*(?P<unit2>m|mm|million|mil|k)\b"
    r"\s+(?:good\s+deal|fair\s+deal|worth\s+it|overpriced|good\s+buy|fair\s+price)"
    r")"
)

_DISPATCH_MODEL_ALIASES: Dict[str, str] = {
    "longitude": "Citation Longitude",
    "latitude": "Citation Latitude",
    "cj3+": "Citation CJ3+",
    "cj3 plus": "Citation CJ3+",
    "cj4": "Citation CJ4",
    "phenom 300": "Phenom 300",
    "phenom 300e": "Phenom 300E",
    "g650": "Gulfstream G650",
    "g700": "Gulfstream G700",
    "g550": "Gulfstream G550",
    "g280": "Gulfstream G280",
    "falcon 8x": "Falcon 8X",
    "falcon 7x": "Falcon 7X",
    "praetor 600": "Praetor 600",
}


def _normalize_alias_key(raw: str) -> str:
    return re.sub(r"\s+", " ", (raw or "").strip().lower())


def _resolve_alternative_target_for_dispatch(query: str) -> Optional[str]:
    """Resolve alternative target including shorthand tokens missed by detect_models."""
    from services.catalog.catalog_alias_resolver import resolve_canonical_display_name
    from services.comparison.alternative_pipeline_responder import (
        _resolve_alternative_target,
        is_alternative_execution_query,
    )
    from services.consultant.recommendation_engine import detect_models_from_text

    q = (query or "").strip()
    if not q:
        return None

    target = _resolve_alternative_target(q)
    if target:
        return target

    m = _ALTERNATIVES_TO_RE.search(q)
    if m:
        raw = (m.group(1) or "").strip().rstrip("?.!")
        if raw:
            key = _normalize_alias_key(raw)
            if key in _DISPATCH_MODEL_ALIASES:
                return _DISPATCH_MODEL_ALIASES[key]
            resolved = resolve_canonical_display_name(raw)
            if resolved and resolved.lower() != raw.lower():
                return resolved
            found = detect_models_from_text(raw) or detect_models_from_text(f"Citation {raw}")
            if found:
                return resolve_canonical_display_name(found[0]) or found[0]
            if len(raw) >= 3:
                return resolve_canonical_display_name(raw) or raw

    if is_alternative_execution_query(q):
        found = detect_models_from_text(q)
        if found:
            return resolve_canonical_display_name(found[0]) or found[0]
    return None


def _comparison_models(query: str) -> List[str]:
    from services.catalog.alias_expansion_engine import resolve_comparison_models_from_query

    expanded = resolve_comparison_models_from_query(query or "")
    if len(expanded) >= 2:
        return expanded

    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
    from services.consultant.recommendation_engine import detect_models_from_text

    raw = detect_models_from_text(query or "")
    resolved = [resolve_aircraft_alias(m) or m for m in raw]
    lock = lock_comparison_aircraft(resolved)
    return [m for m in lock.canonical if m]


def _looks_like_aircraft_buy_query(query: str, parsed: Dict[str, Any]) -> bool:
    from services.consultant.recommendation_engine import detect_models_from_text

    model = str(parsed.get("model") or "").strip().rstrip(",")
    if not model:
        return False
    if re.search(r"\b(?:nm|passengers?|runway|legs?|ft)\b", model, re.I):
        return False
    if detect_models_from_text(model) or detect_models_from_text(query):
        return True
    key = _normalize_alias_key(model)
    return key in _DISPATCH_MODEL_ALIASES


def _is_buy_decision_query(query: str, qri: Optional[QueryRecommendationIntentResult]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if qri is not None and getattr(qri.intent, "value", str(qri.intent)) == "listing_valuation":
        return True
    parsed = _parse_buy_decision_fields(q)
    if parsed and _looks_like_aircraft_buy_query(q, parsed):
        return True
    return False


_VALUATION_SHAPE_RE = re.compile(
    r"\b(?:(?:what\s+(?:is|are)\s+(?:a|an|the\s+)?.+?\s+worth)|"
    r"(?:worth\s+of|valuation\s+of|market\s+value\s+of|how\s+much\s+is\s+.+\s+worth))\b",
    re.I,
)


def _is_valuation_query(query: str, qri: Optional[QueryRecommendationIntentResult]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if qri is not None and getattr(qri.intent, "value", str(qri.intent)) == "listing_valuation":
        return True
    if _is_buy_decision_query(q, qri):
        return False
    if not _VALUATION_SHAPE_RE.search(q):
        return False
    from services.consultant.recommendation_engine import detect_models_from_text

    if detect_models_from_text(q):
        return True
    return bool(re.search(r"(?:19|20)\d{2}", q))


def _parse_buy_decision_fields(query: str) -> Optional[Dict[str, Any]]:
    q = (query or "").strip()
    m = _BUY_DEAL_RE.search(q)
    if not m:
        return None

    def _clean_buy_model(raw: str) -> str:
        mdl = re.sub(r"(?is)\s+for\s*$", "", (raw or "").strip()).rstrip("?.!,")
        mdl = re.sub(r"(?is)\s+at\s*$", "", mdl)
        return mdl.strip()

    if m.group("year"):
        model = _clean_buy_model(m.group("model") or "")
        parsed = {
            "model": model,
            "year": int(m.group("year")),
            "ask_usd": _price_to_usd(m.group("price"), m.group("unit")),
        }
        return parsed if _looks_like_aircraft_buy_query(q, parsed) else None

    model = _clean_buy_model(m.group("model2") or "")
    parsed = {
        "model": model,
        "year": None,
        "ask_usd": _price_to_usd(m.group("price2"), m.group("unit2")),
    }
    return parsed if _looks_like_aircraft_buy_query(q, parsed) else None


def _price_to_usd(amount: Optional[str], unit: Optional[str]) -> Optional[float]:
    if not amount:
        return None
    try:
        val = float(str(amount).replace(",", ""))
    except ValueError:
        return None
    u = (unit or "").lower().strip()
    if u in ("m", "mm", "million", "mil"):
        return val * 1_000_000.0
    if u == "k":
        return val * 1_000.0
    if val < 1000:
        return val * 1_000_000.0
    return val


def respond_buy_decision(
    query: str,
    *,
    db: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Structured buy-decision response via unified broker state (single market pass)."""
    from services.adversarial.adversarial_preprocessor import get_pipeline_query, try_adversarial_buy_block
    from services.consistency.consistency_injection_layer import (
        prepare_buy_decision_state,
        render_buy_decision_answer,
    )

    du = data_used if isinstance(data_used, dict) else {}
    q_norm = get_pipeline_query(query, du)
    blocked = try_adversarial_buy_block(q_norm, du)
    if blocked:
        return blocked

    parsed = _parse_buy_decision_fields(q_norm)
    if not parsed:
        return ""

    state = prepare_buy_decision_state(
        query=q_norm,
        parsed=parsed,
        db=db,
        data_used=du,
    )
    return render_buy_decision_answer(state)


def _format_buy_decision_response(
    model: str,
    year: Optional[int],
    ask_usd: Optional[float],
    verdict_payload: Dict[str, Any],
    market_data: Dict[str, Any],
    *,
    mi_bundle: Any = None,
) -> str:
    from services.market_intelligence.market_intelligence_engine import (
        format_buy_decision_market_sections,
        format_deal_assessment,
    )

    lines: List[str] = [f"Aircraft: {model}"]
    if year:
        lines.append(f"Year: {year}")

    lines.append("")
    lines.append("Market Reality:")
    if mi_bundle is not None:
        lines.extend(format_buy_decision_market_sections(mi_bundle, market_data))
    else:
        comp_n = int(market_data.get("comp_row_count") or 0)
        if comp_n > 0:
            low = market_data.get("price_range_low")
            high = market_data.get("price_range_high")
            avg = market_data.get("avg_price")
            if low is not None and high is not None:
                lines.append(
                    f"- Synced comp slice ({comp_n} rows): roughly ${float(low)/1e6:.1f}M–${float(high)/1e6:.1f}M "
                    f"(avg ~${float(avg)/1e6:.1f}M)." if avg else
                    f"- Synced comp slice ({comp_n} rows): ${float(low)/1e6:.1f}M–${float(high)/1e6:.1f}M."
                )
        elif market_data.get("authority_band"):
            low = market_data.get("price_range_low")
            high = market_data.get("price_range_high")
            mid = market_data.get("avg_price")
            if low is not None and high is not None:
                lines.append(
                    f"- Verified catalog band (authority): roughly ${float(low)/1e6:.1f}M–${float(high)/1e6:.1f}M "
                    f"(mid ~${float(mid)/1e6:.1f}M)." if mid else
                    f"- Verified catalog band (authority): ${float(low)/1e6:.1f}M–${float(high)/1e6:.1f}M."
                )
        else:
            lines.append(
                "- Limited synced comp data for this model slice — price verdict uses ask-level heuristics only."
            )

    broker = (verdict_payload.get("broker_comment") or "").strip()
    if broker:
        lines.append(f"- {broker}")

    reasons = list(verdict_payload.get("key_reasons") or [])[:4]
    for r in reasons:
        if r and not str(r).startswith("Assessment:"):
            lines.append(f"- {r}")

    red = list(verdict_payload.get("red_flags") or [])[:6]
    if red:
        lines.append("")
        lines.append("Red Flags:")
        for f in red:
            lines.append(f"- {f}")

    if mi_bundle is not None:
        lines.append("")
        lines.append("Deal Assessment:")
        lines.extend(format_deal_assessment(ask_usd, mi_bundle.deal_quality))

    lines.append("")
    lines.append("Verdict:")
    lines.append(str(verdict_payload.get("verdict") or "FAIR DEAL"))
    return "\n".join(lines)


def consult_authority_dispatch(
    query: str,
    *,
    qri: Optional[QueryRecommendationIntentResult] = None,
    unified_route: Optional[UnifiedIntentRoute] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[AuthorityDispatchResult]:
    """
    Select highest-priority deterministic responder for this turn.

    Returns None when legacy pipeline should continue unchanged.
    """
    ctx = context if isinstance(context, dict) else {}
    from services.adversarial.adversarial_preprocessor import get_pipeline_query

    q = get_pipeline_query(query or "", {"clean_normalized_query": ctx.get("clean_normalized_query")})
    q = (q or "").strip()
    if not q:
        return None

    data_used: Dict[str, Any] = {"consultant_authority_dispatch": 1}
    if isinstance(ctx.get("clean_normalized_query"), dict):
        data_used["clean_normalized_query"] = ctx["clean_normalized_query"]
    if isinstance(ctx.get("broker_reasoning"), dict):
        data_used["broker_reasoning"] = ctx["broker_reasoning"]

    def _flush_dispatch_data() -> None:
        patch = ctx.get("pre_llm_pipeline_patch")
        if isinstance(patch, dict):
            patch.update(data_used)

    from services.core.semantic_intent_lock_engine import IntentLock

    _intent_lock: Optional[IntentLock] = None
    _lock_raw = ctx.get("intent_lock")
    if isinstance(_lock_raw, IntentLock):
        _intent_lock = _lock_raw
    elif _lock_raw is not None:
        _intent_lock = IntentLock.from_dict(_lock_raw)

    def _locked_models() -> List[str]:
        if _intent_lock is not None and _intent_lock.canonical_models:
            return list(_intent_lock.canonical_models)
        return _comparison_models(q)

    def _locked_budget_m() -> Optional[float]:
        if _intent_lock is not None and _intent_lock.constraints.get("budget_m") is not None:
            try:
                return float(_intent_lock.constraints["budget_m"])
            except (TypeError, ValueError):
                pass
        from services.routing.intent_conflict_resolution import _parse_budget_millions

        return _parse_budget_millions(q)

    # --- 1. COMPARISON (highest priority for explicit A vs B) ---
    from services.comparison.alternative_pipeline_responder import is_explicit_comparison_query
    from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison

    comparison_path = (
        unified_route is not None
        and unified_route.execution_path == UnifiedExecutionPath.COMPARISON
    )
    qri_comparison = (
        qri is not None
        and qri.intent == QueryRecommendationIntent.AIRCRAFT_COMPARISON
    )
    models = _locked_models()
    comparison_intent = comparison_path or qri_comparison or is_explicit_comparison_query(q)
    if _intent_lock is not None and _intent_lock.intent_type == "comparison":
        comparison_intent = True
    if comparison_intent:
        compare_models = list(models)
        from services.broker_reasoning.broker_reasoning_layer import get_broker_reasoning_compare_models

        br_models = get_broker_reasoning_compare_models(data_used)
        if br_models and len(br_models) >= 2:
            compare_models = br_models
        from services.routing.intent_conflict_resolution import _apply_budget_filter

        budget_m = _locked_budget_m()
        if budget_m is not None and len(compare_models) >= 2:
            compare_models, constraint_result = _apply_budget_filter(compare_models, budget_m)
            data_used["authority_dispatch_budget_filter"] = {
                "budget_m": budget_m,
                "constraint_result": constraint_result,
            }
        if len(compare_models) >= 2:
            answer = respond_aircraft_comparison(
                q,
                compare_models=compare_models,
                data_used=data_used,
            )
            if _comparison_responder_succeeded(data_used, answer):
                from services.broker_reasoning.broker_reasoning_layer import append_multi_intent_overlays

                answer = append_multi_intent_overlays(
                    answer,
                    q,
                    data_used=data_used,
                    db=ctx.get("db"),
                )
                data_used["authority_dispatch_kind"] = "comparison"
                data_used["authority_dispatch_models"] = compare_models
                from services.consultant.consultant_llm_policy import (
                    consultant_llm_narration_enabled,
                    consultant_narrate_structured_dispatch,
                )

                if consultant_llm_narration_enabled() and consultant_narrate_structured_dispatch():
                    data_used["comparison_deferred_llm"] = 1
                    data_used["comparison_structured_for_llm"] = answer
                    _flush_dispatch_data()
                    return None
                _flush_dispatch_data()
                return AuthorityDispatchResult(
                    answer=answer,
                    dispatch_kind="comparison",
                    progress_step="path_authority_dispatch_comparison",
                    data_used=data_used,
                )
        return _build_dispatch_safety_fallback("comparison", data_used)

    # --- 2. ALTERNATIVE ---
    from services.comparison.alternative_pipeline_responder import (
        is_alternative_execution_query,
        respond_aircraft_alternative,
    )

    alt_target = _resolve_alternative_target_for_dispatch(q)
    if _intent_lock is not None and _intent_lock.intent_type == "alternative" and _intent_lock.canonical_models:
        alt_target = _intent_lock.canonical_models[0]
    br_expansion = (data_used.get("broker_reasoning") or {}).get("intent_expansion") or {}
    if br_expansion.get("alternative_search") and br_expansion.get("reference_model"):
        alt_target = str(br_expansion["reference_model"])
    alt_query = q
    starts_with_alt = bool(re.match(r"(?is)^\s*alternatives?\s+to\b", q))
    alternative_intent = is_alternative_execution_query(q) or starts_with_alt or bool(
        br_expansion.get("alternative_search")
    )
    if alternative_intent:
        if alt_target:
            if alt_target.lower() not in q.lower():
                alt_query = f"alternatives to {alt_target}"
            from services.aircraft_truth.constants import UNVERIFIED_AIRCRAFT_MESSAGE

            answer = respond_aircraft_alternative(alt_query, data_used=data_used)
            if answer and answer.strip() != UNVERIFIED_AIRCRAFT_MESSAGE.strip():
                data_used["authority_dispatch_kind"] = "alternative"
                data_used["authority_dispatch_target"] = alt_target
                from services.consultant.consultant_llm_policy import (
                    consultant_llm_narration_enabled,
                    consultant_narrate_structured_dispatch,
                )

                if consultant_llm_narration_enabled() and consultant_narrate_structured_dispatch():
                    data_used["alternative_deferred_llm"] = 1
                    data_used["alternative_structured_for_llm"] = answer
                    _flush_dispatch_data()
                    return None
                _flush_dispatch_data()
                return AuthorityDispatchResult(
                    answer=answer,
                    dispatch_kind="alternative",
                    progress_step="path_authority_dispatch_alternative",
                    data_used=data_used,
                )
        from services.broker_reasoning.broker_reasoning_layer import render_acquisition_guidance

        alt_guidance = render_acquisition_guidance(q, data_used=data_used)
        if alt_guidance:
            data_used["authority_dispatch_kind"] = "alternative"
            data_used["broker_reasoning_alternative_guidance"] = 1
            return AuthorityDispatchResult(
                answer=alt_guidance,
                dispatch_kind="alternative",
                progress_step="path_authority_dispatch_alternative_guidance",
                data_used=data_used,
            )
        return _build_dispatch_safety_fallback("alternative", data_used)

    # --- 3. BUY DECISION ---
    from services.broker_reasoning.broker_reasoning_layer import is_acquisition_budget_query

    if _is_buy_decision_query(q, qri) or is_acquisition_budget_query(q):
        answer = respond_buy_decision(q, db=ctx.get("db"), data_used=data_used)
        if answer:
            data_used["authority_dispatch_kind"] = "buy_decision"
            return AuthorityDispatchResult(
                answer=answer,
                dispatch_kind="buy_decision",
                progress_step="path_authority_dispatch_buy_decision",
                data_used=data_used,
            )
        from services.broker_reasoning.broker_reasoning_layer import render_acquisition_guidance

        guidance = render_acquisition_guidance(q, data_used=data_used)
        if guidance:
            data_used["authority_dispatch_kind"] = "buy_decision"
            data_used["broker_reasoning_acquisition_guidance"] = 1
            return AuthorityDispatchResult(
                answer=guidance,
                dispatch_kind="buy_decision",
                progress_step="path_authority_dispatch_acquisition_guidance",
                data_used=data_used,
            )
        return _build_dispatch_safety_fallback("buy_decision", data_used)

    # --- 4. VALUATION (fail-closed when model/market cannot resolve) ---
    if _is_valuation_query(q, qri):
        answer = respond_buy_decision(q, db=ctx.get("db"), data_used=data_used)
        if answer:
            from services.broker_reasoning.broker_reasoning_layer import append_multi_intent_overlays

            answer = append_multi_intent_overlays(
                answer,
                q,
                data_used=data_used,
                db=ctx.get("db"),
            )
            data_used["authority_dispatch_kind"] = "buy_decision"
            return AuthorityDispatchResult(
                answer=answer,
                dispatch_kind="buy_decision",
                progress_step="path_authority_dispatch_valuation",
                data_used=data_used,
            )
        return _build_dispatch_safety_fallback("valuation", data_used)

    # --- 5. Category / budget discovery (alternative guidance) ---
    from services.broker_reasoning.broker_reasoning_layer import (
        apply_broker_reasoning_layer,
        is_acquisition_budget_query,
        render_acquisition_guidance,
    )

    _CATEGORY_DISCOVERY_RE = re.compile(
        r"(?is)\b(?:cheap\s+gulfstream|best\s+jet\s+under|best\s+super-?midsize|"
        r"g\d{3}\s+for\s+\d+\s*m\b|g\d{3}\s+under\s+\d+\s*m\b|what\s+should\s+i\s+buy|"
        r"g650\s+for\s+\d+)\b"
    )
    if _CATEGORY_DISCOVERY_RE.search(q) or is_acquisition_budget_query(q):
        if not isinstance(data_used.get("broker_reasoning"), dict):
            apply_broker_reasoning_layer(q, data_used=data_used)
        from services.consultant.consultant_llm_policy import consultant_llm_narration_enabled

        if consultant_llm_narration_enabled():
            data_used["broker_reasoning_acquisition_guidance"] = 1
            data_used["authority_dispatch_kind"] = "alternative_deferred_llm"
            return None
        guidance = render_acquisition_guidance(q, data_used=data_used)
        if guidance:
            data_used["authority_dispatch_kind"] = "alternative"
            data_used["broker_reasoning_acquisition_guidance"] = 1
            return AuthorityDispatchResult(
                answer=guidance,
                dispatch_kind="alternative",
                progress_step="path_authority_dispatch_category_guidance",
                data_used=data_used,
            )

    # --- 6. Buy / wait timing ---
    _BUY_WAIT_AUTH_RE = re.compile(
        r"(?is)\b(?:should\s+i\s+buy\s+now|buy\s+now\s+or\s+wait|wait\s+or\s+buy|good\s+time\s+to\s+buy)\b"
    )
    if _BUY_WAIT_AUTH_RE.search(q):
        from services.broker_decision.broker_decision_builder import build_broker_decision

        decision = build_broker_decision(q, data_used=data_used)
        if decision and decision.direct_answer:
            data_used["broker_decision"] = decision.to_dict()
            data_used["authority_dispatch_kind"] = "buy_decision"
            return AuthorityDispatchResult(
                answer=decision.direct_answer,
                dispatch_kind="buy_decision",
                progress_step="path_authority_dispatch_buy_wait",
                data_used=data_used,
            )

    # --- 7. Tail / registry — load facts; LLM narrates (no template brief as final answer) ---
    from services.market_reality.listing_detector import ListingMode, detect_listing_signal

    listing = detect_listing_signal(q)
    if listing.mode == ListingMode.TAIL_INVESTIGATION and listing.registrations:
        reg = listing.registrations[0]
        try:
            from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
            from services.broker_execution.tail_fact_renderer import (
                render_tail_facts_for_llm_context,
                select_tail_facts,
            )

            ensure_tail_facts_for_query(q, data_used)
            facts = select_tail_facts(data_used, reg)
            ctx_block = render_tail_facts_for_llm_context(facts, registration=reg)
            if ctx_block:
                data_used["tail_registry_llm_context"] = ctx_block
            data_used["tail_investigation_defer_llm"] = True
            data_used["tail_investigation_dispatch"] = reg
            data_used["authority_dispatch_kind"] = "valuation_deferred_llm"
        except Exception:
            pass
        _flush_dispatch_data()
        return None

    # --- 8. MISSION — no override; legacy pre-LLM pipeline continues ---
    return None


__all__ = [
    "AuthorityDispatchResult",
    "consult_authority_dispatch",
    "respond_buy_decision",
]
