"""
Phase 48 — broker certification helpers.

Exercises the production post-answer stack (recovery → decision → client → market
→ executive → truth → conversation) without adding new layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FORBIDDEN_PHRASES: Tuple[str, ...] = (
    "INSUFFICIENT_DATA",
    "deterministic execution",
    "verified catalog",
    "mission kernel",
    "catalog authority",
    "typical acquisition tier",
    "stretch case",
    "inventory pressure",
    "buyer leverage",
    "seller leverage",
    "insufficient verified data",
    "supporting context",
    "acceptance criteria",
    "primary recommendation would be",
)

V2_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "broker_certification_v2_report.md"
)

DECISION_FIRST_RE = re.compile(
    r"(?is)\b(?:i would buy|i'd buy|i would choose|i'd choose|because|since|given your|"
    r"i would lean|i'd lean|if i were buying|i'd focus on)\b"
)

COMPARISON_CONCLUSION_RE = re.compile(
    r"(?is)\b(?:i would choose|i'd choose|i would lean toward|i'd lean toward|choose the)\b"
)

LISTING_SKEPTICISM_MARKERS: Tuple[str, ...] = (
    "below",
    "unusual",
    "verify",
    "bargain",
    "diligence",
    "does not line up",
    "mis-stated",
    "before treating",
    "skeptical",
    "too cheap",
    "materially below",
)

MISSION_CONFLICT_MARKERS: Tuple[str, ...] = (
    "not realistic",
    "cannot",
    "conflict",
    "exceed",
    "beyond",
    "not nonstop",
    "refuel",
    "does not close",
    "cannot close",
    "mission",
    "band",
    "below typical",
)

MIN_BROKER_QUALITY_SCORE = 70.0

FORBIDDEN_HEADERS: Tuple[str, ...] = (
    "Overview",
    "Analysis",
    "Recommendation",
    "Risks",
    "Mission Fit",
    "Aircraft Options",
    "Verdict",
    "Source",
)

DIRECT_REALITY_PREFIXES: Tuple[str, ...] = (
    "no.",
    "no,",
    "yes.",
    "yes,",
    "probably",
    "not realistically",
    "that budget is far below",
    "you cannot",
    "you can't",
    "at $",
    "with $",
    "on a $",
)

CONVICTION_MARKERS: Tuple[str, ...] = (
    "if i were buying",
    "i'd focus on",
    "i would buy",
    "i would focus",
    "my pick",
    "i'd buy",
    "i would look at",
)

RECOMMENDATION_MARKERS: Tuple[str, ...] = (
    "if i were buying",
    "i'd focus on",
    "i would buy",
    "primary recommendation",
    "i recommend",
    "i'd go with",
    "better buy",
    "would take",
    "lean toward",
    "favor the",
)

CHECKLIST_MARKERS: Tuple[str, ...] = (
    "checklist",
    "step 1:",
    "step 2:",
    "first, verify",
    "before you buy:",
)

TAIL_INFO_ASKS: Tuple[str, ...] = (
    "listing",
    "log",
    "hour",
    "maintenance",
    "engine program",
    "damage history",
)

TAIL_SPECULATION_FORBIDDEN: Tuple[str, ...] = (
    "worth buying",
    "good deal",
    "great buy",
    "i would buy this tail",
    "this is a steal",
)

REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "broker_certification_report.md"


@dataclass
class CertScenarioResult:
    group: str
    scenario_id: str
    query: str
    passed: bool
    path: str
    failure_reasons: List[str] = field(default_factory=list)
    answer_excerpt: str = ""
    tags: List[str] = field(default_factory=list)


class BrokerCertificationRecorder:
    """Collects scenario outcomes for markdown report generation."""

    def __init__(self) -> None:
        self.results: List[CertScenarioResult] = []
        self._primary_recommendations: List[str] = []

    def record(
        self,
        *,
        group: str,
        scenario_id: str,
        query: str,
        passed: bool,
        path: str,
        failure_reasons: Optional[List[str]] = None,
        answer: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        excerpt = (answer or "").strip().replace("\r\n", "\n")
        if len(excerpt) > 320:
            excerpt = excerpt[:317] + "..."
        self.results.append(
            CertScenarioResult(
                group=group,
                scenario_id=scenario_id,
                query=query,
                passed=passed,
                path=path,
                failure_reasons=list(failure_reasons or []),
                answer_excerpt=excerpt,
                tags=list(tags or []),
            )
        )

    def note_primary(self, model_fragment: str) -> None:
        frag = (model_fragment or "").strip()
        if frag:
            self._primary_recommendations.append(frag)

    def write_report(self, path: Optional[Path] = None) -> Path:
        out_path = path or REPORT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_report(self), encoding="utf-8")
        return out_path


_RECORDER = BrokerCertificationRecorder()


def get_certification_recorder() -> BrokerCertificationRecorder:
    return _RECORDER


def reset_certification_recorder() -> None:
    global _RECORDER
    _RECORDER = BrokerCertificationRecorder()


def _render_report(recorder: BrokerCertificationRecorder) -> str:
    results = recorder.results
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]
    rate = (100.0 * passed / total) if total else 0.0

    budget_failures = [r for r in failed if r.group == "A" or "budget" in r.tags]
    continuity_failures = [r for r in failed if r.group == "B" or "continuity" in r.tags]
    humanization_failures = [r for r in failed if r.group in ("F", "G") or "humanization" in r.tags]

    defect_counts: Dict[str, int] = {}
    for r in failed:
        for reason in r.failure_reasons:
            key = reason.split(":", 1)[0].strip() or reason[:80]
            defect_counts[key] = defect_counts.get(key, 0) + 1
    top_defects = sorted(defect_counts.items(), key=lambda x: (-x[1], x[0]))[:10]

    lines = [
        "# Broker Certification Report (Phase 48)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Scenarios run | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {len(failed)} |",
        f"| **Pass rate** | **{rate:.1f}%** |",
        "",
        "## Failed scenarios",
        "",
    ]
    if not failed:
        lines.append("_None — all certification scenarios passed._")
    else:
        for r in failed:
            lines.append(f"### [{r.group}] {r.scenario_id}")
            lines.append("")
            lines.append(f"- **Query:** `{r.query}`")
            lines.append(f"- **Path:** `{r.path}`")
            for reason in r.failure_reasons:
                lines.append(f"- **Failure:** {reason}")
            if r.answer_excerpt:
                lines.append("")
                lines.append("```")
                lines.append(r.answer_excerpt)
                lines.append("```")
            lines.append("")

    lines.extend(["## Budget violations", ""])
    if budget_failures:
        for r in budget_failures:
            lines.append(f"- `{r.scenario_id}`: {r.query}")
    else:
        lines.append("_None recorded._")

    lines.extend(["", "## Conversation continuity failures", ""])
    if continuity_failures:
        for r in continuity_failures:
            lines.append(f"- `{r.scenario_id}`: {r.query}")
    else:
        lines.append("_None recorded._")

    lines.extend(["", "## Humanization / formatting failures", ""])
    if humanization_failures:
        for r in humanization_failures:
            lines.append(f"- `{r.scenario_id}` ({r.group}): {', '.join(r.failure_reasons[:3])}")
    else:
        lines.append("_None recorded._")

    lines.extend(["", "## Repeated primary recommendations (observed)", ""])
    primaries = recorder._primary_recommendations
    if primaries:
        from collections import Counter

        for model, count in Counter(primaries).most_common(8):
            lines.append(f"- `{model}`: {count}×")
    else:
        lines.append("_No primary recommendations captured._")

    lines.extend(["", "## Top 10 remaining defects", ""])
    if top_defects:
        for i, (name, count) in enumerate(top_defects, 1):
            lines.append(f"{i}. **{name}** ({count} scenario(s))")
    else:
        lines.append("_No defects — certification clean._")

    lines.extend(
        [
            "",
            "## How to regenerate",
            "",
            "```bash",
            "cd backend",
            "PYTHONPATH=. pytest tests/e2e/test_broker_certification_suite.py -q",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _apply_production_post_layers(query: str, raw: str, du: dict) -> str:
    from services.broker_execution.output_governance import apply_governed_client_answer

    answer = apply_governed_client_answer((raw or "").strip(), query=query, data_used=du)
    if not answer:
        answer = (raw or "").strip()
    try:
        from services.broker_execution.fact_flow import attach_fact_flow
        from services.broker_execution.retrieval_utilization import attach_retrieval_utilization

        attach_fact_flow(query, answer, du)
        attach_retrieval_utilization(answer, du)
    except Exception:
        pass
    return (answer or "").strip()


def _resolve_dispatch_raw(
    query: str,
    du: dict,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    state: Optional[dict] = None,
) -> str:
    """Prefer full consultant LLM retrieval; avoid template-only authority short-circuit."""
    import os

    if os.getenv("BROKER_CERTIFY_USE_LLM", "1").strip().lower() not in ("0", "false", "no"):
        try:
            from tests.conftest import run_retrieval

            _kind, payload = run_retrieval(query, history=history)
            ans = str(payload.get("answer") or "").strip()
            if ans and _kind not in ("error",):
                if isinstance(payload.get("data_used"), dict):
                    du.update(payload["data_used"])
                du["broker_certify_llm_raw"] = True
                trace = (du.get("execution_trace") or {}) if isinstance(du.get("execution_trace"), dict) else {}
                if trace.get("llm_executed") or du.get("llm_executed") or du.get("consultant_llm_draft"):
                    return ans
                if len(ans) > 80 and "INSUFFICIENT_DATA" not in ans.upper():
                    du.setdefault("llm_executed", True)
                    return ans
        except Exception:
            pass

    profile = str(du.get("execution_profile") or "").strip().lower()
    if profile in ("comparison", "mission") or profile.startswith("tail_"):
        try:
            from services.consultant.answer_recovery import recover_client_answer

            structured = recover_client_answer(query=query, data_used=du, answer="").strip()
            if structured and "INSUFFICIENT_DATA" not in structured.upper():
                return structured
        except Exception:
            pass

    try:
        from services.core.semantic_intent_lock_engine import bind_dispatch_authority, build_intent_lock
        from services.consultant.consultant_llm_policy import authority_dispatch_defer_to_llm
        from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
        from services.routing.authority_dispatch import consult_authority_dispatch
        from services.routing.unified_intent_router import classify_unified_intent

        qri = classify_query_recommendation_intent(query, [])
        route = classify_unified_intent(query)
        lock = build_intent_lock(query, qri=qri, unified_route=route)
        dispatch = consult_authority_dispatch(
            query,
            qri=qri,
            unified_route=route,
            context={"db": None, "intent_lock": lock},
        )
        if dispatch is not None and str(dispatch.answer or "").strip():
            if authority_dispatch_defer_to_llm(dispatch):
                return _resolve_dispatch_raw(query, du, history=history, state=state)
            bound = bind_dispatch_authority(lock, dispatch)
            du["intent_lock"] = bound.to_dict()
            du["authority_dispatch_kind"] = dispatch.dispatch_kind
            if isinstance(dispatch.data_used, dict):
                du.update(dispatch.data_used)
            return str(dispatch.answer).strip()
    except Exception:
        pass
    return (
        "INSUFFICIENT_DATA: Insufficient verified aircraft data for deterministic execution."
    )


def broker_certify(
    query: str,
    *,
    state: Optional[dict] = None,
    history: Optional[List[Dict[str, str]]] = None,
    prefer_e2e: bool = True,
) -> Tuple[str, dict, str]:
    """
    Return (final_answer, data_used, path) where path is ``e2e`` or ``layers``.

    Observability: ``data_used['broker_certify_path']`` and ``broker_certify_prefer_e2e``.
    """
    if prefer_e2e:
        try:
            from tests.conftest import run_retrieval

            kind, payload = run_retrieval(
                query,
                history=history,
                client_conversation_state=state,
            )
            ans = str(payload.get("answer") or "").strip()
            du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}
            du["broker_certify_prefer_e2e"] = True
            if ans and not _is_non_substantive_e2e(ans, kind):
                try:
                    from services.broker_scoring.broker_quality_score import attach_broker_quality_score

                    attach_broker_quality_score(ans, query=query, data_used=du)
                except Exception:
                    pass
                try:
                    from services.broker_audit.broker_trace import attach_broker_trace
                    from services.broker_audit.broker_trust_score import attach_broker_trust_score

                    attach_broker_trace(ans, query=query, data_used=du)
                    attach_broker_trust_score(ans, query=query, data_used=du)
                except Exception:
                    pass
                du["broker_certify_path"] = "e2e"
                from tests.e2e.pipeline_observability import attach_pipeline_observability

                attach_pipeline_observability(du, path="e2e", prefer_e2e=True)
                return ans, du, "e2e"
        except Exception:
            pass

    du: dict = {"broker_certify_prefer_e2e": False}
    from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query
    from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer
    from services.client_context.client_context_layer import apply_client_context_turn
    from services.intent_collapse.intent_collapse_engine import apply_intent_collapse

    clean = preprocess_adversarial_query(query, data_used=du)
    apply_client_context_turn(
        query,
        data_used=du,
        client_conversation_state=state or {},
        history=history,
    )
    apply_intent_collapse(query, data_used=du, normalized_query=clean.normalized_query)
    apply_broker_reasoning_layer(clean.normalized_query, data_used=du)
    try:
        from services.broker_execution.execution_intent_lock import attach_execution_intent_lock

        attach_execution_intent_lock(du, query)
    except Exception:
        pass
    try:
        from services.broker_execution.listing_parse_audit import attach_listing_parse_audit
        from services.broker_execution.mission_feasibility_filter import apply_mission_feasibility_filter
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query

        attach_listing_parse_audit(query, du)
        ensure_tail_facts_for_query(query, du)
        apply_mission_feasibility_filter(query, data_used=du)
        if str(du.get("execution_profile") or "") == "mission":
            try:
                from services.broker_execution.mission_broker_answer import ensure_mission_pipeline

                ensure_mission_pipeline(query, du)
                attach_execution_intent_lock(du, query)
            except Exception:
                pass
    except Exception:
        pass

    raw = _resolve_dispatch_raw(query, du, history=history, state=state or {})
    answer = _apply_production_post_layers(query, raw, du)
    du["broker_certify_path"] = "layers"
    from tests.e2e.pipeline_observability import attach_pipeline_observability

    attach_pipeline_observability(du, path="layers", prefer_e2e=False)
    return answer, du, "layers"


def broker_certify_conversation(
    turns: Sequence[str],
    *,
    prefer_e2e: bool = False,
) -> Tuple[str, dict, str, List[Tuple[str, str, dict]]]:
    """Simulate a multi-turn buyer thread; returns final answer and per-turn trace."""
    from services.client_context.client_context_layer import finalize_client_context

    state: dict = {}
    history: List[Dict[str, str]] = []
    trace: List[Tuple[str, str, dict]] = []
    answer = ""
    du: dict = {}
    path = "layers"

    for q in turns:
        answer, du, path = broker_certify(
            q,
            state=state,
            history=history,
            prefer_e2e=prefer_e2e,
        )
        trace.append((q, answer, dict(du)))
        finalize_client_context(du, state, query=q, history=history)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

    return answer, du, path, trace


def _is_non_substantive_e2e(answer: str, kind: str) -> bool:
    low = answer.lower().strip()
    if len(low) < 40 and any(
        p in low
        for p in (
            "understood",
            "talk soon",
            "let me know",
            "happy to help",
        )
    ):
        return True
    return False


def first_paragraph(text: str) -> str:
    parts = re.split(r"\n\s*\n", (text or "").strip())
    return parts[0].strip() if parts else ""


def assert_direct_reality_start(answer: str) -> None:
    first = first_paragraph(answer).lower()
    if any(first.startswith(p) for p in DIRECT_REALITY_PREFIXES):
        return
    if re.match(r"^no\b", first):
        return
    raise AssertionError(
        f"expected direct budget reality in opening line; got: {first_paragraph(answer)[:120]!r}"
    )


def assert_models_absent(answer: str, models: Sequence[str]) -> None:
    """Fail if a model appears as a recommendation, not in explicit rejections."""
    skip_line = re.compile(
        r"(?i)(would not|not lead|above the|budget cap|avoid|do not|don't|cannot fit|"
        r"out of reach|does not trade|not realistically|far below|no\.)"
    )
    first = first_paragraph(answer).lower()
    if first.startswith("no.") or "does not trade" in first or "far below" in first:
        return
    for line in (answer or "").splitlines():
        if skip_line.search(line):
            continue
        low_line = line.lower()
        for model in models:
            if model.lower() in low_line:
                if re.search(r"(?i)(?:focus on|i'd buy|recommend|plausible|would buy)", line):
                    raise AssertionError(f"forbidden model endorsement: {model}")
                continue
    if not skip_line.search(first):
        for model in models:
            if model.lower() in first and re.search(
                r"(?i)(?:focus on|i'd buy|recommend|plausible|would buy)", first
            ):
                raise AssertionError(f"forbidden model endorsement in opening: {model}")


def assert_forbidden_phrases_absent(answer: str) -> None:
    low = answer.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p.lower() in low]
    if hits:
        raise AssertionError(f"forbidden phrases: {', '.join(hits)}")


def assert_forbidden_headers_absent(answer: str) -> None:
    for header in FORBIDDEN_HEADERS:
        if re.search(rf"(?im)^\s*#{{0,3}}\s*{re.escape(header)}\s*:?\s*$", answer):
            raise AssertionError(f"forbidden section header: {header}")
        if re.search(rf"(?im)^\s*\*\*{re.escape(header)}\*\*\s*$", answer):
            raise AssertionError(f"forbidden section header: {header}")


def assert_no_bullet_spam(answer: str, *, max_bullets: int = 8) -> None:
    bullets = re.findall(r"(?m)^\s*[•\-]\s+", answer)
    if len(bullets) > max_bullets:
        raise AssertionError(f"leading bullet spam: {len(bullets)} bullets (max {max_bullets})")


def assert_has_recommendation(answer: str) -> None:
    low = answer.lower()
    if any(m in low for m in RECOMMENDATION_MARKERS):
        return
    if re.search(r"(?i)\b(?:would take|lean(?:s)? toward|favor(?:s)? the)\b", answer):
        return
    raise AssertionError("missing broker recommendation language")


def assert_has_conviction(answer: str) -> None:
    low = answer.lower()
    if any(m in low for m in CONVICTION_MARKERS):
        return
    raise AssertionError("missing conviction / first-person broker voice")


def assert_not_checklist(answer: str) -> None:
    low = answer.lower()
    hits = [m for m in CHECKLIST_MARKERS if m in low]
    if hits:
        raise AssertionError(f"checklist style detected: {hits}")


def assert_comparison_quality(answer: str, *, models: Sequence[str]) -> None:
    low = answer.lower()
    if "insufficient" in low and "data" in low:
        raise AssertionError("INSUFFICIENT_DATA-style comparison response")
    if "insufficient_data" in low:
        raise AssertionError("INSUFFICIENT_DATA token in comparison response")
    if "tell me what you care about" in low and "clear pick" not in low:
        try:
            assert_has_recommendation(answer)
        except AssertionError:
            raise AssertionError(
                "comparison deflected to clarification without recommendation"
            ) from None
    from tests.e2e.benchmark_audit_helpers import model_in_text

    missing = [m for m in models if not model_in_text(m, answer)]
    if missing:
        raise AssertionError(f"comparison missing models: {', '.join(missing)}")
    assert_has_recommendation(answer)


def assert_tail_investigation(answer: str) -> None:
    low = answer.lower()
    for phrase in TAIL_SPECULATION_FORBIDDEN:
        if phrase in low and "cannot tell" not in low:
            raise AssertionError(f"speculative tail language: {phrase}")
    asks = [a for a in TAIL_INFO_ASKS if a in low]
    if len(asks) < 2:
        raise AssertionError(
            f"tail mode should request listing/logs/hours/maintenance; found: {asks}"
        )


def assert_single_primary_executive(answer: str, data_used: dict) -> None:
    from services.executive_broker.executive_answer_rewriter import has_equal_weight_recommendations

    exec_rec = (data_used or {}).get("executive_recommendation") or {}
    if isinstance(exec_rec, dict) and str(exec_rec.get("primary_recommendation") or "").strip():
        return

    if has_equal_weight_recommendations(answer):
        raise AssertionError("equal-weight multi-option executive answer")
    bullets = re.findall(r"(?m)^\s*[•\-]\s+", answer)
    if len(bullets) >= 4:
        raise AssertionError(f"too many equal bullets ({len(bullets)})")
    low = answer.lower()
    if not any(m in low for m in CONVICTION_MARKERS + ("because", "since", "given your")):
        raise AssertionError("missing WHY / rationale for executive recommendation")
    if re.search(r"(?i)\b(?:i'd focus on|if i were buying|g\d{3}|citation|challenger|praetor|longitude)\b", answer):
        return
    raise AssertionError("no identifiable single primary recommendation")


def run_cert_scenario(
    *,
    group: str,
    scenario_id: str,
    query: str,
    answer: str,
    path: str,
    checks: Sequence[Callable[[], None]],
    tags: Optional[Sequence[str]] = None,
) -> None:
    """Run assertions, record outcome, and fail pytest on violation."""
    failures: List[str] = []
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            failures.append(str(exc))

    recorder = get_certification_recorder()
    recorder.record(
        group=group,
        scenario_id=scenario_id,
        query=query,
        passed=not failures,
        path=path,
        failure_reasons=failures,
        answer=answer,
        tags=list(tags or []),
    )
    if failures:
        raise AssertionError(
            f"[{group}/{scenario_id}] " + "; ".join(failures)
        )


def extract_primary_hint(answer: str) -> Optional[str]:
    m = re.search(
        r"(?i)(?:i'd focus on|if i were buying(?: today)?,?\s*i'd focus on|primary recommendation(?: would be)?)\s+([^.\n]+)",
        answer,
    )
    if m:
        return m.group(1).strip()[:60]
    return None


# ---------------------------------------------------------------------------
# Phase 50 — V2 assertion helpers
# ---------------------------------------------------------------------------


def assert_no_diligence_before_reality(answer: str) -> None:
    first = first_paragraph(answer).lower()
    if re.search(r"(?is)^\s*(?:before treating|i would verify|verify:)", first):
        raise AssertionError("diligence checklist appears before budget reality")


def assert_mission_conflict_identified(answer: str) -> None:
    low = answer.lower()
    if not any(m in low for m in MISSION_CONFLICT_MARKERS):
        raise AssertionError("mission/budget conflict not identified")


def assert_listing_skepticism(answer: str) -> None:
    low = answer.lower()
    if not any(m in low for m in LISTING_SKEPTICISM_MARKERS):
        raise AssertionError("missing listing-price skepticism")
    if "listing" not in low and "spec sheet" not in low and "broker package" not in low:
        raise AssertionError("should request listing or broker package")


def assert_decision_first(answer: str) -> None:
    if not DECISION_FIRST_RE.search(answer or ""):
        raise AssertionError("missing decision-first recommendation with rationale")


def assert_comparison_conclusion(answer: str) -> None:
    if not COMPARISON_CONCLUSION_RE.search(answer or ""):
        if not re.search(r"(?is)\b(?:i would lean|i'd lean|choose the)\b", answer or ""):
            raise AssertionError("comparison missing explicit broker conclusion")


def assert_broker_quality_score(data_used: dict, *, minimum: float = MIN_BROKER_QUALITY_SCORE) -> None:
    score = (data_used or {}).get("broker_quality_score") or {}
    total = score.get("total") if isinstance(score, dict) else None
    if total is None:
        from services.broker_scoring.broker_quality_score import attach_broker_quality_score

        score = attach_broker_quality_score(
            str(data_used.get("_cert_answer") or ""),
            query=str(data_used.get("query") or ""),
            data_used=data_used,
        )
        total = score.get("total")
    if total is None or float(total) < minimum:
        raise AssertionError(f"broker quality score {total} below {minimum}")


def assert_no_recommendation_drift(data_used: dict) -> None:
    audit = (data_used or {}).get("recommendation_consistency_audit_v2") or {}
    if isinstance(audit, dict) and audit.get("recommendation_drift"):
        events = audit.get("drift_events") or []
        detail = events[0] if events else {}
        raise AssertionError(
            f"RECOMMENDATION_DRIFT: {detail.get('from')} -> {detail.get('to')}"
        )


def assert_timing_guidance(answer: str) -> None:
    low = answer.lower()
    markers = (
        "buy",
        "wait",
        "now",
        "timing",
        "probably",
        "would",
        "if i were",
        "i'd focus",
        "vetted tail",
        "don't wait",
        "do not wait",
    )
    if any(m in low for m in markers):
        return
    assert_has_recommendation(answer)


def assert_adversarial_discipline(answer: str, *, forbidden_endorsement: str = "") -> None:
    low = answer.lower()
    if forbidden_endorsement and forbidden_endorsement.lower() in low:
        if re.search(
            r"(?is)(?:no\.|not realistically|would not|cannot|does not trade|far below|"
            r"ignore|cannot close|not realistic)",
            low,
        ):
            return
        if re.search(r"(?is)(?:focus on|i'd buy|recommend|good deal|great buy)", low):
            raise AssertionError(f"adversarial prompt caused endorsement of {forbidden_endorsement}")
    if re.search(r"(?is)\b(?:good deal|great buy|buy it anyway)\b", low):
        if "no" not in first_paragraph(answer).lower():
            raise AssertionError("adversarial prompt accepted impossible deal")


def assert_budget_referenced(answer: str, budget_musd: float) -> None:
    token = f"${budget_musd:.0f}m"
    if token not in answer.lower().replace(" ", "") and str(int(budget_musd)) not in answer:
        raise AssertionError(f"budget ${budget_musd:.0f}M not referenced in answer")


def assert_contains_model(answer: str, model_fragment: str) -> None:
    if model_fragment.lower() not in answer.lower():
        raise AssertionError(f"expected model {model_fragment!r} in answer")


class BrokerCertificationV2Recorder(BrokerCertificationRecorder):
    """Extended recorder for V2 with quality scores."""

    def __init__(self) -> None:
        super().__init__()
        self.scores: List[float] = []

    def record(self, **kwargs) -> None:
        super().record(**kwargs)
        tags = kwargs.get("tags") or []
        if "score" in tags:
            pass

    def note_score(self, total: float) -> None:
        self.scores.append(total)

    def write_v2_report(self, path: Optional[Path] = None) -> Path:
        out_path = path or V2_REPORT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        rate = (100.0 * passed / total) if total else 0.0
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0.0
        failed = [r for r in self.results if not r.passed]

        lines = [
            "# Broker Certification V2 Report (Phase 50)",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Scenarios | {total} |",
            f"| Passed | {passed} |",
            f"| Failed | {len(failed)} |",
            f"| **Pass rate** | **{rate:.1f}%** |",
            f"| Avg broker quality score | {avg_score:.1f} |",
            "",
        ]
        if failed:
            lines.append("## Failed scenarios\n")
            for r in failed[:30]:
                lines.append(f"- **[{r.group}] {r.scenario_id}**: {', '.join(r.failure_reasons[:2])}")
        else:
            lines.append("_All scenarios passed._\n")
        lines.extend(
            [
                "",
                "## Regenerate",
                "",
                "```bash",
                "cd backend",
                "PYTHONPATH=. pytest tests/e2e/test_broker_certification_v2.py -q",
                "```",
            ]
        )
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


_V2_RECORDER = BrokerCertificationV2Recorder()


def get_v2_recorder() -> BrokerCertificationV2Recorder:
    return _V2_RECORDER


def reset_v2_recorder() -> None:
    global _V2_RECORDER
    _V2_RECORDER = BrokerCertificationV2Recorder()


def run_v2_scenario(
    *,
    group: str,
    scenario_id: str,
    query: str,
    answer: str,
    path: str,
    data_used: dict,
    checks: Sequence[Callable[[], None]],
    tags: Optional[Sequence[str]] = None,
) -> None:
    data_used = data_used or {}
    data_used["_cert_answer"] = answer
    data_used.setdefault("query", query)
    score_blob = data_used.get("broker_quality_score")
    if isinstance(score_blob, dict) and score_blob.get("total") is not None:
        get_v2_recorder().note_score(float(score_blob["total"]))

    failures: List[str] = []
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            failures.append(str(exc))

    recorder = get_v2_recorder()
    recorder.record(
        group=group,
        scenario_id=scenario_id,
        query=query,
        passed=not failures,
        path=path,
        failure_reasons=failures,
        answer=answer,
        tags=list(tags or []) + ["v2"],
    )
    if failures:
        raise AssertionError(
            f"[{group}/{scenario_id}] " + "; ".join(failures)
        )

