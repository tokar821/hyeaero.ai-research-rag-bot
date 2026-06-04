"""
Phase 54 — collect and report e2e vs layers execution path divergence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tests.e2e.benchmark_audit_helpers import REPORTS_DIR
from tests.e2e.production_audit_helpers import _primary_is_meaningful, _resolve_primary

PARITY_REPORT = REPORTS_DIR / "phase54_execution_path_parity.md"

# Categories where layers path must expose executive primary (certification lens).
_LAYERS_PRIMARY_REQUIRED = frozenset({"mission", "buy", "acquisition"})


@dataclass
class ParityObservation:
    scenario_id: str
    query: str
    path_e2e: str
    path_layers: str
    primary_e2e: str
    primary_layers: str
    same_primary: bool
    e2e_skipped: bool = False
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def classify_query_category(query: str) -> str:
    ql = (query or "").lower()
    if "coast" in ql or "nonstop" in ql or "passenger" in ql or "pax" in ql:
        return "mission"
    if " vs " in ql or " versus " in ql:
        return "comparison"
    if re_search_buy(ql):
        return "buy"
    if re_search_listing(ql):
        return "listing"
    return "general"


def re_search_buy(ql: str) -> bool:
    import re

    return bool(re.search(r"(?is)\b(?:buy|should i buy|what should i buy|best .{0,40} under)\b", ql))


def re_search_listing(ql: str) -> bool:
    import re

    return bool(re.search(r"(?is)\b(?:listed|listing|asking|fair price|good deal|realistic)\b", ql))


def evaluate_parity(
    scenario_id: str,
    query: str,
    *,
    ans_e2e: str,
    du_e2e: dict,
    path_e2e: str,
    ans_layers: str,
    du_layers: dict,
    path_layers: str,
) -> ParityObservation:
    from tests.e2e.benchmark_audit_helpers import model_in_text

    e2e_skipped = path_e2e != "e2e"
    prim_e2e = _resolve_primary(du_e2e) if not e2e_skipped else ""
    prim_layers = _resolve_primary(du_layers)
    same = prim_e2e == prim_layers or (
        bool(prim_e2e and prim_layers and model_in_text(prim_e2e, prim_layers))
    )

    obs = ParityObservation(
        scenario_id=scenario_id,
        query=query,
        path_e2e=path_e2e,
        path_layers=path_layers,
        primary_e2e=prim_e2e,
        primary_layers=prim_layers,
        same_primary=same,
        e2e_skipped=e2e_skipped,
    )

    if du_layers.get("broker_certify_path") != "layers":
        obs.critical_failures.append("layers_path_tag_mismatch")
    if not e2e_skipped and du_e2e.get("broker_certify_path") != "e2e":
        obs.critical_failures.append("e2e_path_tag_mismatch")

    cat = classify_query_category(query)
    if cat in _LAYERS_PRIMARY_REQUIRED and not _primary_is_meaningful(prim_layers):
        obs.critical_failures.append(f"layers_missing_primary_{cat}")

    if not e2e_skipped and not same:
        if _primary_is_meaningful(prim_layers) and not _primary_is_meaningful(prim_e2e):
            obs.warnings.append("expected_divergence:layers_has_primary_e2e_missing")
        elif _primary_is_meaningful(prim_e2e) and not _primary_is_meaningful(prim_layers):
            obs.critical_failures.append("unexpected:layers_missing_primary_e2e_has")

    if not ans_layers.strip():
        obs.critical_failures.append("layers_empty_answer")

    return obs


def write_parity_report(observations: List[ParityObservation]) -> Path:
    PARITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Phase 54 Execution Path Parity",
        "",
        f"Generated: {ts}",
        "",
        "| Scenario | E2E path | Layers path | Same primary | Critical | Warnings |",
        "|----------|----------|-------------|--------------|----------|----------|",
    ]
    for o in observations:
        crit = ", ".join(o.critical_failures) or "—"
        warn = ", ".join(o.warnings) or "—"
        same = "skip" if o.e2e_skipped else ("yes" if o.same_primary else "no")
        lines.append(
            f"| {o.scenario_id} | {o.path_e2e} | {o.path_layers} | {same} | {crit} | {warn} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- **Critical failures** fail CI when `HYEAERO_PARITY_STRICT=1` (default).",
            "- **Warnings** document expected e2e vs layers divergence (executive primary on layers only).",
            "",
        ]
    )
    PARITY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return PARITY_REPORT


def parity_strict_enabled() -> bool:
    from tests.e2e.execution_path_config import PARITY_STRICT

    return PARITY_STRICT


__all__ = [
    "ParityObservation",
    "evaluate_parity",
    "write_parity_report",
    "parity_strict_enabled",
]
