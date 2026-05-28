"""
HACK v3 — Renderer Integrity & Narrative Lock Layer.

Runs after HACK v2. The renderer is a pure display engine: it may only emit
the frozen RankedAircraftList contract with verbatim verdicts and scores.

No re-ranking, no narrative, no enrichment, no external data access.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

HACK_V3_METADATA_KEY = "hack_v3_renderer"
FREEZE_FRAME_KEY = "freeze_frame"
NULL_FIELD = "NULL_FIELD"

# Consultant / broker narrative patterns that must not appear in locked output.
_FORBIDDEN_OUTPUT_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwhy\s+(?:this\s+)?aircraft\s+fits\b",
        r"\bwhy:\s",
        r"\breason:\s",
        r"\bthis\s+is\s+ideal\s+because\b",
        r"\brecommended\s+due\s+to\b",
        r"\bmission\s+interpretation\b",
        r"\bfinal\s+verdict\b",
        r"\bconstraint\s+summary\b",
        r"\bleads\s+on\s+composite\b",
        r"\balternates:\s",
    )
)


class RenderIntegrityError(RuntimeError):
    """Raised when renderer mutates or augments HACK v2 truth."""


@dataclass(frozen=True)
class FormattedAircraftRow:
    """Display contract — 1:1 with HACK v2 rows."""

    aircraft_name: str
    composite_score: str
    eligibility_status: str
    verdict: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_name": self.aircraft_name,
            "composite_score": self.composite_score,
            "eligibility_status": self.eligibility_status,
            "verdict": self.verdict,
            "rank": self.rank,
        }


def load_hack_v2_ranked_list(data_used: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(data_used, dict):
        return []
    raw = data_used.get("hack_v2_ranking")
    if not isinstance(raw, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("aircraft_name"):
            rows.append(
                {
                    "aircraft_name": str(item["aircraft_name"]),
                    "composite_score": item.get("composite_score"),
                    "eligibility_status": str(item.get("eligibility_status") or ""),
                    "verdict": str(item.get("verdict") or ""),
                }
            )
    return rows


def freeze_ranked_list(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Deep-freeze HACK v2 rows — no downstream mutation."""
    frozen = copy.deepcopy(list(rows))
    return frozen, True


def _format_score(value: Any) -> str:
    if value is None:
        return NULL_FIELD
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else NULL_FIELD


def _format_field(value: Any) -> str:
    if value is None:
        return NULL_FIELD
    text = str(value).strip()
    return text if text else NULL_FIELD


def build_formatted_rows(
    frozen_rows: Sequence[Dict[str, Any]],
) -> List[FormattedAircraftRow]:
    """Map frozen HACK v2 rows to display rows without reinterpretation."""
    out: List[FormattedAircraftRow] = []
    for rank, row in enumerate(frozen_rows, start=1):
        out.append(
            FormattedAircraftRow(
                aircraft_name=_format_field(row.get("aircraft_name")),
                composite_score=_format_score(row.get("composite_score")),
                eligibility_status=_format_field(row.get("eligibility_status")),
                verdict=_format_field(row.get("verdict")),
                rank=rank,
            )
        )
    return out


def assert_recommendations_match_contract(
    recommendations: Optional[Sequence[Any]],
    contract_rows: Sequence[Dict[str, Any]],
) -> None:
    """Optional guard: recommendation objects must not disagree with HACK v2 order."""
    if not recommendations or not contract_rows:
        return
    contract_models = [str(r["aircraft_name"]) for r in contract_rows]
    rec_models = [
        str(getattr(r, "model", "") or "")
        for r in recommendations
        if not getattr(r, "avoid", False) and getattr(r, "model", None)
    ]
    if rec_models and rec_models != contract_models[: len(rec_models)]:
        raise RenderIntegrityError(
            "RENDER_INTEGRITY_ERROR: recommendation order diverges from HACK v2 contract"
        )
    for rec in recommendations:
        if getattr(rec, "avoid", False):
            continue
        model = str(getattr(rec, "model", "") or "")
        if not model:
            continue
        contract = next((r for r in contract_rows if r["aircraft_name"] == model), None)
        if contract is None:
            continue
        rec_verdict = (getattr(rec, "fit_verdict", None) or "").strip()
        if rec_verdict and rec_verdict != contract["verdict"]:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: verdict drift on {model} before render"
            )
        rec_score = getattr(rec, "total_score", None)
        if rec_score is not None:
            try:
                if round(float(rec_score), 4) != round(float(contract["composite_score"]), 4):
                    raise RenderIntegrityError(
                        f"RENDER_INTEGRITY_ERROR: score drift on {model} before render"
                    )
            except (TypeError, ValueError):
                pass


def verify_render_integrity(
    input_rows: Sequence[Dict[str, Any]],
    formatted_rows: Sequence[FormattedAircraftRow],
    rendered_text: str,
) -> None:
    """Mandatory consistency check before returning output."""
    if len(input_rows) != len(formatted_rows):
        raise RenderIntegrityError(
            "RENDER_INTEGRITY_ERROR: row count mismatch between input and output"
        )

    for idx, (inp, out) in enumerate(zip(input_rows, formatted_rows), start=1):
        if out.rank != idx:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: rank mismatch for {out.aircraft_name}"
            )
        if _format_field(inp.get("aircraft_name")) != out.aircraft_name:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: aircraft_name mismatch for rank {idx}"
            )
        if _format_score(inp.get("composite_score")) != out.composite_score:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: composite_score mismatch for {out.aircraft_name}"
            )
        if _format_field(inp.get("eligibility_status")) != out.eligibility_status:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: eligibility_status mismatch for {out.aircraft_name}"
            )
        expected_verdict = _format_field(inp.get("verdict"))
        if expected_verdict != out.verdict:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: verdict mismatch for {out.aircraft_name}"
            )
        if expected_verdict not in rendered_text:
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: verdict not present verbatim for {out.aircraft_name}"
            )

    for pattern in _FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.search(rendered_text):
            raise RenderIntegrityError(
                f"RENDER_INTEGRITY_ERROR: forbidden narrative pattern {pattern.pattern}"
            )


def render_locked_table(formatted_rows: Sequence[FormattedAircraftRow]) -> str:
    """
    Pure display — markdown table only.

    No mission interpretation, no why-it-fits, no final verdict commentary.
    """
    lines: List[str] = ["## Ranked Aircraft List", ""]
    if not formatted_rows:
        lines.append(f"| Rank | Aircraft | Composite Score | Eligibility | Verdict |")
        lines.append(f"| --- | --- | --- | --- | --- |")
        lines.append(f"| {NULL_FIELD} | {NULL_FIELD} | {NULL_FIELD} | {NULL_FIELD} | {NULL_FIELD} |")
        return "\n".join(lines)

    lines.append("| Rank | Aircraft | Composite Score | Eligibility | Verdict |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in formatted_rows:
        lines.append(
            f"| {row.rank} | {row.aircraft_name} | {row.composite_score} | "
            f"{row.eligibility_status} | {row.verdict} |"
        )
    return "\n".join(lines)


def attach_hack_v3_metadata(
    data_used: Dict[str, Any],
    *,
    frozen_rows: Sequence[Dict[str, Any]],
    formatted_rows: Sequence[FormattedAircraftRow],
    freeze_frame: bool = True,
) -> None:
    data_used[FREEZE_FRAME_KEY] = bool(freeze_frame)
    data_used["hack_v3_renderer_locked"] = True
    data_used[HACK_V3_METADATA_KEY] = {
        "freeze_frame": bool(freeze_frame),
        "row_count": len(formatted_rows),
        "formatted_table": [r.to_dict() for r in formatted_rows],
        "source": "hack_v2_ranking",
    }


def render_hack_v3_locked_response(
    data_used: Optional[Dict[str, Any]],
    *,
    recommendations: Optional[Sequence[Any]] = None,
) -> str:
    """
    Render frozen HACK v2 contract — the only allowed renderer output path when
    ``hack_v2_ranking`` is present.
    """
    du = data_used if isinstance(data_used, dict) else {}
    contract_rows = load_hack_v2_ranked_list(du)
    if not contract_rows:
        raise RenderIntegrityError(
            "RENDER_INTEGRITY_ERROR: hack_v2_ranking missing or empty for locked render"
        )

    assert_recommendations_match_contract(recommendations, contract_rows)
    frozen_rows, _ = freeze_ranked_list(contract_rows)
    formatted_rows = build_formatted_rows(frozen_rows)
    rendered = render_locked_table(formatted_rows)
    verify_render_integrity(frozen_rows, formatted_rows, rendered)
    attach_hack_v3_metadata(
        du,
        frozen_rows=frozen_rows,
        formatted_rows=formatted_rows,
        freeze_frame=True,
    )
    return rendered


def should_use_hack_v3_renderer(data_used: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(data_used, dict):
        return False
    if data_used.get("hack_v3_renderer_locked"):
        return True
    return bool(load_hack_v2_ranked_list(data_used))


__all__ = [
    "FormattedAircraftRow",
    "FREEZE_FRAME_KEY",
    "HACK_V3_METADATA_KEY",
    "NULL_FIELD",
    "RenderIntegrityError",
    "attach_hack_v3_metadata",
    "build_formatted_rows",
    "freeze_ranked_list",
    "load_hack_v2_ranked_list",
    "render_hack_v3_locked_response",
    "render_locked_table",
    "should_use_hack_v3_renderer",
    "verify_render_integrity",
]
