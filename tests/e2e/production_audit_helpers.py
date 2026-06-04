"""
Phase 53 — shared production audit helpers (measurement only).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tests.e2e.benchmark_audit_helpers import (
    REPORTS_DIR,
    attach_audit_metadata,
    model_in_text,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_production_queries(limit: Optional[int] = None) -> List[Dict[str, str]]:
    path = FIXTURES_DIR / "production_queries.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = list(blob.get("queries") or [])
    cap = limit
    if cap is None:
        env = os.environ.get("PHASE53_REPLAY_LIMIT", "")
        if env.isdigit():
            cap = int(env)
    if cap is not None:
        rows = rows[:cap]
    return rows


def load_golden_expectations() -> Dict[str, Any]:
    path = FIXTURES_DIR / "golden_expectations.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob.get("expectations") or {}


@dataclass
class ReplayRecord:
    query_id: str
    category: str
    query: str
    authority: str = ""
    primary: str = ""
    trust_score: Optional[float] = None
    drift: bool = False
    authority_error: bool = False
    path: str = ""
    has_primary: bool = False
    has_deal_quality: bool = False
    semantic_ok: bool = True
    semantic_failure: str = ""
    mission_primary_present: bool = False
    mission_semantic_ok: bool = True
    prefer_e2e: bool = True
    executive_applied: bool = False


def _resolve_primary(du: dict) -> str:
    rec = du.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        return str(rec["primary_recommendation"]).strip()
    trace = du.get("broker_trace") or {}
    return str(trace.get("executive_primary") or "").strip()


def _primary_is_meaningful(primary: str) -> bool:
    p = (primary or "").strip()
    if not p:
        return False
    if p.lower() == "timing guidance":
        return False
    return True


def evaluate_replay_semantics(
    row: Dict[str, str],
    du: dict,
    *,
    primary: str = "",
    path: str = "",
) -> tuple[bool, str]:
    """Return (semantic_ok, failure_reason)."""
    category = str(row.get("category") or "")
    prim = primary or _resolve_primary(du)

    if category == "mission":
        if not _primary_is_meaningful(prim):
            return False, "mission_missing_primary"
        if not du.get("executive_applied") and path == "layers":
            return False, "mission_executive_not_applied"
        return True, ""

    if category == "listing":
        dq = du.get("deal_quality")
        if not isinstance(dq, dict) or not dq.get("verdict"):
            return False, "listing_missing_deal_quality"
        return True, ""

    return True, ""


def replay_query(row: Dict[str, str], golden: Dict[str, Any]) -> ReplayRecord:
    from tests.e2e.broker_certification_helpers import broker_certify
    from tests.e2e.execution_path_config import prefer_e2e_for_replay

    qid = str(row.get("id") or "")
    query = str(row.get("query") or "")
    category = str(row.get("category") or "")
    use_e2e = prefer_e2e_for_replay(category)
    answer, du, path = broker_certify(query, prefer_e2e=use_e2e)
    attach_audit_metadata(answer, query, du)
    du["replay_prefer_e2e"] = use_e2e

    from tests.e2e.pipeline_observability import assert_replay_row_observability

    assert_replay_row_observability(du, path=path, prefer_e2e=use_e2e)

    trace = du.get("broker_trace") or {}
    auth = str(du.get("authority_dispatch_kind") or trace.get("authority_selected") or "")
    primary = _resolve_primary(du)
    trust_blob = du.get("broker_trust_score") or {}
    trust = float(trust_blob["total"]) if isinstance(trust_blob, dict) and trust_blob.get("total") is not None else None

    audit = du.get("recommendation_consistency_audit_v2") or {}
    drift = bool(
        isinstance(audit, dict)
        and (audit.get("unjustified_recommendation_drift") or audit.get("recommendation_drift"))
    )

    exp = golden.get(qid) or {}
    exp_kind = exp.get("expected_dispatch_kind")
    authority_error = bool(exp_kind and auth and exp_kind not in auth and auth != exp_kind)

    has_primary = _primary_is_meaningful(primary)
    has_deal_quality = bool(
        isinstance(du.get("deal_quality"), dict) and du["deal_quality"].get("verdict")
    )
    semantic_ok, semantic_failure = evaluate_replay_semantics(row, du, primary=primary, path=path)
    mission_primary_present = has_primary if category == "mission" else False
    mission_semantic_ok = semantic_ok if category == "mission" else True

    return ReplayRecord(
        query_id=qid,
        category=category,
        query=query,
        authority=auth,
        primary=primary,
        trust_score=trust,
        drift=drift,
        authority_error=authority_error,
        path=path,
        has_primary=has_primary,
        has_deal_quality=has_deal_quality,
        semantic_ok=semantic_ok,
        semantic_failure=semantic_failure,
        mission_primary_present=mission_primary_present,
        mission_semantic_ok=mission_semantic_ok,
        prefer_e2e=use_e2e,
        executive_applied=bool(du.get("executive_applied")),
    )


def summarize_replay(records: Sequence[ReplayRecord]) -> Dict[str, Any]:
    n = len(records) or 1
    trust_vals = [r.trust_score for r in records if r.trust_score is not None]
    mission = [r for r in records if r.category == "mission"]
    mission_n = len(mission) or 1
    buy = [r for r in records if r.category == "buy_decision"]
    buy_n = len(buy) or 1
    return {
        "total": len(records),
        "authority_error_pct": 100.0 * sum(1 for r in records if r.authority_error) / n,
        "drift_pct": 100.0 * sum(1 for r in records if r.drift) / n,
        "semantic_fail_pct": 100.0 * sum(1 for r in records if not r.semantic_ok) / n,
        "avg_trust": sum(trust_vals) / len(trust_vals) if trust_vals else 0.0,
        "trust_above_95_pct": (
            100.0 * sum(1 for t in trust_vals if t >= 95) / len(trust_vals) if trust_vals else 0.0
        ),
        "mission_primary_pct": 100.0 * sum(1 for r in mission if r.has_primary) / mission_n,
        "mission_semantic_ok_pct": 100.0 * sum(1 for r in mission if r.mission_semantic_ok) / mission_n,
        "mission_executive_applied_pct": 100.0 * sum(1 for r in mission if r.executive_applied) / mission_n,
        "buy_primary_pct": 100.0 * sum(1 for r in buy if r.has_primary) / buy_n,
    }


def primary_distribution(records: Sequence[ReplayRecord]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in records:
        if r.primary:
            c[r.primary] += 1
    return c


def detect_selection_bias(dist: Counter[str], *, threshold_pct: float = 35.0) -> List[str]:
    total = sum(dist.values()) or 1
    flags: List[str] = []
    for model, count in dist.most_common(3):
        pct = 100.0 * count / total
        if pct >= threshold_pct:
            flags.append(f"{model} selected {pct:.1f}% of primaries (possible bias)")
    return flags


def write_health_dashboard(sections: Dict[str, str], path: Optional[Path] = None) -> Path:
    out = path or (REPORTS_DIR / "hyeaero_health_dashboard.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HyeAero Health Dashboard",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Daily KPI Summary",
        "",
    ]
    for title, body in sections.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    lines.extend(
        [
            "## Regenerate",
            "",
            "```bash",
            "cd backend",
            "PYTHONPATH=. python runners/run_phase53_audit.py",
            "```",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
