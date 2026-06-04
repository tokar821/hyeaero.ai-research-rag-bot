"""Phase 33 — E2E response audit runner.

Runs queries through full consultant retrieval and audits the *final answer*.
Routing/dispatch correctness is not scored here (Phase 32 covers that).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass
class E2EResponseCase:
    query_id: str
    category: str
    query: str
    return_kind: str
    answer: str
    data_used: Dict[str, Any]

    intent_lock: Dict[str, Any] = field(default_factory=dict)
    authority_models: List[str] = field(default_factory=list)
    dispatch_kind: Optional[str] = None


@dataclass
class AuditFinding:
    code: str
    message: str


@dataclass
class AuditedCase:
    case: E2EResponseCase
    score: float
    findings: List[AuditFinding] = field(default_factory=list)
    stop_condition_hit: bool = False


def load_production_corpus() -> Dict[str, Any]:
    return json.loads((FIXTURES_DIR / "production_queries.json").read_text(encoding="utf-8"))


def load_broker_review_set() -> Dict[str, Any]:
    return json.loads((FIXTURES_DIR / "broker_review_set.json").read_text(encoding="utf-8"))


def _extract_case(kind: str, payload: Dict[str, Any], *, query_id: str, category: str, query: str) -> E2EResponseCase:
    du = payload.get("data_used") or {}
    lock = du.get("intent_lock") or {}
    try:
        from services.consultant.model_authority_guard import resolve_verified_models

        authority_models = resolve_verified_models(du if isinstance(du, dict) else {})
    except Exception:
        authority_models = list(du.get("authority_dispatch_models") or lock.get("canonical_models") or [])
    return E2EResponseCase(
        query_id=query_id,
        category=category,
        query=query,
        return_kind=kind,
        answer=str(payload.get("answer") or ""),
        data_used=dict(du),
        intent_lock=dict(lock) if isinstance(lock, dict) else {},
        authority_models=authority_models,
        dispatch_kind=du.get("authority_dispatch_kind"),
    )


def iter_cases_from_rows(rows: Iterable[Dict[str, str]], *, svc: Any = None) -> List[E2EResponseCase]:
    from tests.conftest import run_retrieval
    from tests.response_quality.response_audit_service import ResponseAuditService

    service = svc or ResponseAuditService()
    out: List[E2EResponseCase] = []
    for row in rows:
        kind, payload = run_retrieval(row["query"], svc=service)
        out.append(_extract_case(kind, payload, query_id=row["id"], category=row["category"], query=row["query"]))
    return out


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def audited_to_json(results: List[AuditedCase]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "query_id": r.case.query_id,
                "category": r.case.category,
                "query": r.case.query,
                "return_kind": r.case.return_kind,
                "authority_models": r.case.authority_models,
                "dispatch_kind": r.case.dispatch_kind,
                "intent_lock": r.case.intent_lock,
                "score": r.score,
                "stop_condition_hit": r.stop_condition_hit,
                "findings": [asdict(f) for f in r.findings],
                "answer_preview": (r.case.answer or "")[:600],
            }
        )
    return out


def generate_broker_review_report(results: List[AuditedCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = sorted(results, key=lambda r: (-r.score, len(r.findings)))[:25]
    risk = sorted(results, key=lambda r: (r.stop_condition_hit, len(r.findings), -r.score), reverse=True)[:25]

    lines: List[str] = [
        "# Broker Review Report (E2E Final Answers)",
        "",
        "## Top 25 Best Answers",
        "",
    ]
    for i, r in enumerate(best, 1):
        lines.extend(
            [
                f"### {i}. `{r.case.query_id}`",
                "",
                f"**Query:** {r.case.query}",
                "",
                f"**Score:** {r.score}",
                "",
                f"**Authority models:** {', '.join(r.case.authority_models)}",
                "",
                "**Answer preview:**",
                "",
                "```",
                (r.case.answer or "")[:800],
                "```",
                "",
            ]
        )

    lines.extend(["---", "", "## Top 25 Highest-Risk Answers", ""])
    for i, r in enumerate(risk, 1):
        lines.extend(
            [
                f"### {i}. `{r.case.query_id}`",
                "",
                f"**Query:** {r.case.query}",
                "",
                f"**Score:** {r.score}",
                "",
                f"**Stop condition:** {r.stop_condition_hit}",
                "",
                f"**Findings:** {', '.join(f.code for f in r.findings) or '(none)'}",
                "",
                f"**Authority models:** {', '.join(r.case.authority_models)}",
                "",
                "**Answer preview:**",
                "",
                "```",
                (r.case.answer or "")[:800],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Appendix",
            "",
            f"- Total audited: {len(results)}",
            f"- Stop-condition hits: {sum(1 for r in results if r.stop_condition_hit)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

