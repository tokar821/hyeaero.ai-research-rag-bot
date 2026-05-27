"""
Output format validation before final advisor response rendering.

If validation fails and structured recommendations exist, regenerate from
``format_consultant_response`` rather than shipping a broken LLM merge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment

_BULLET_RE = re.compile(r"^\s*[-•]\s+(.+)$", re.M)
_BULLET_MODEL_RE = re.compile(
    r"^\s*[-•]\s+([^:—–\-\n]{2,80}?)\s*(?:[:—–-]\s*|$)",
    re.M,
)
_ORPHAN_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"top\s+options?|short\s+list|alternatives?|also\s+in\s+the\s+mix|"
    r"worth\s+weighing|side-by-side|names?\s+i['']d|aircraft\s+i['']d|"
    r"practical\s+options?|on\s+my\s+list|from\s+an\s+operating|"
    r"how\s+the\s+alternates?\s+stack\s+up|others?\s+in\s+the\s+conversation"
    r")\s*:?\s*$",
    re.I | re.M,
)
_TRANSITION_LABEL_RE = re.compile(
    r"^(?:"
    r"also\s+in\s+the\s+mix|worth\s+weighing|alternates?|alternatives?|"
    r"side-by-side|others?\s+in\s+the\s+conversation|short\s+list|"
    r"names?\s+i['']d(?:\s+pressure-test)?|aircraft\s+(?:i['']d|worth)|"
    r"from\s+an\s+operating\s+standpoint|on\s+my\s+list|practical\s+options?|"
    r"how\s+the\s+alternates?\s+stack\s+up|aircraft\s+worth\s+a\s+hard\s+look"
    r")\s*:?\s*$",
    re.I,
)
_INCOMPLETE_BULLET_RE = re.compile(
    r"^\s*[-•]\s+\S.{2,}\s*[:—–-]\s*$",
    re.M,
)
_BROKER_FIT_FOOTER_RE = re.compile(
    r"^(?:PRIMARY RECOMMENDATION|VIABLE WITH COMPROMISES|MISSION-RISKY|NOT OPERATIONALLY CREDIBLE|"
    r"BEST FIT|GOOD FIT|CONDITIONAL FIT|NOT A FIT)\s*:",
    re.M,
)
_ARCHITECTURE_HEADER_RE = re.compile(
    r"^Mission Fit:\s*$|^Aircraft Options:\s*$|^Verdict:\s*$",
    re.M,
)


def _is_architecture_format(text: str) -> bool:
    return bool(
        _ARCHITECTURE_HEADER_RE.search(text or "")
        and _BROKER_FIT_FOOTER_RE.search(text or "")
    )


def _is_broker_format(text: str) -> bool:
    """Fixed architecture or legacy broker footer — not unstructured bullet lists."""
    return bool(_BROKER_FIT_FOOTER_RE.search(text or "") or _is_architecture_format(text))


@dataclass
class FormatValidationReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    regenerated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": list(self.issues),
            "regenerated": self.regenerated,
        }


def _normalize_model_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _bullet_models(text: str) -> List[str]:
    models: List[str] = []
    for m in _BULLET_MODEL_RE.finditer(text or ""):
        raw = (m.group(1) or "").strip()
        if raw and len(raw) > 2:
            models.append(raw)
    return models


def _known_models_in_text(text: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(detect_models_from_text(text or ""))
    except Exception:
        return []


def _check_duplicated_aircraft(text: str, issues: List[str]) -> None:
    bullets = _bullet_models(text)
    if not bullets:
        return
    seen: Dict[str, int] = {}
    for b in bullets:
        key = _normalize_model_key(b.split("—")[0].split("–")[0].split("-")[0].split(":")[0])
        if len(key) < 4:
            continue
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            issues.append(f"duplicated_aircraft_in_bullets:{key}")

    detected = [_normalize_model_key(m) for m in _known_models_in_text(text)]
    if len(detected) >= 2:
        from collections import Counter

        counts = Counter(detected)
        for model, count in counts.items():
            if count >= 3 and model in {_normalize_model_key(b) for b in bullets}:
                issues.append(f"repeated_aircraft_mention:{model}")


def _check_incomplete_bullets(text: str, issues: List[str]) -> None:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "•")):
            continue
        if len(stripped) <= 2:
            issues.append("empty_bullet_line")
            continue
        if _INCOMPLETE_BULLET_RE.match(line):
            issues.append(f"incomplete_bullet:{stripped[:60]}")
            continue


def _check_truncated_sections(text: str, issues: List[str]) -> None:
    s = (text or "").strip()
    if not s:
        issues.append("empty_response")
        return
    if s.endswith("...") or s.endswith("…"):
        issues.append("truncated_ellipsis_ending")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return
    last = lines[-1]
    if last.endswith((":", "—", "–", "-")) and not last.startswith(("-", "•")):
        issues.append("truncated_section_ending")
    if len(last) < 12 and not last.endswith("?"):
        issues.append("truncated_short_tail")
    if last.count("(") > last.count(")"):
        issues.append("unclosed_parenthesis")


def _check_orphaned_labels(text: str, issues: List[str]) -> None:
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if not _ORPHAN_LABEL_RE.match(line.strip()):
            continue
        rest = [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
        if not rest:
            issues.append(f"orphaned_label_at_eof:{line.strip()[:40]}")
        elif rest[0].startswith(("-", "•")) is False and _ORPHAN_LABEL_RE.match(rest[0]):
            issues.append(f"orphaned_label_stacked:{line.strip()[:40]}")


def _check_empty_alternatives(
    text: str,
    issues: List[str],
    *,
    recommendations: Optional[List[AircraftRecommendation]],
) -> None:
    if _is_broker_format(text):
        return
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if not _TRANSITION_LABEL_RE.match(line.strip()):
            continue
        following = [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
        has_bullet = any(ln.startswith(("-", "•")) for ln in following)
        if not has_bullet:
            issues.append(f"empty_alternatives_after_transition:{line.strip()[:40]}")

    bullets = [ln for ln in lines if ln.strip().startswith(("-", "•"))]
    viable = [r for r in (recommendations or []) if not r.avoid]
    if len(viable) >= 2 and len(bullets) < 2:
        issues.append("missing_alternate_bullets")
    if len(viable) >= 2 and len(bullets) == 1:
        issues.append("single_bullet_with_multiple_recommendations")


def validateResponseFormatting(
    text: str,
    *,
    recommendations: Optional[List[AircraftRecommendation]] = None,
) -> FormatValidationReport:
    """
    Validate advisor prose before it is shown to the user.

    Checks: duplicate aircraft names, incomplete bullets, truncated sections,
    orphaned section labels, and empty alternative blocks.
    """
    issues: List[str] = []
    _check_duplicated_aircraft(text, issues)
    _check_incomplete_bullets(text, issues)
    _check_truncated_sections(text, issues)
    _check_orphaned_labels(text, issues)
    _check_empty_alternatives(text, issues, recommendations=recommendations)
    return FormatValidationReport(ok=len(issues) == 0, issues=issues)


# PEP 8 alias
validate_response_formatting = validateResponseFormatting


def regenerate_from_structured_recommendations(
    *,
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
    route_assessments: List[RouteFeasibilityAssessment],
    comparison: Optional[StructuredComparison] = None,
    query: str = "",
    turn_seed: str = "",
) -> str:
    """Rebuild user-facing copy from ranked recommendations only."""
    from services.consultant.response_formatter import format_consultant_response

    return format_consultant_response(
        mission=mission,
        recommendations=recommendations,
        route_assessments=route_assessments,
        comparison=comparison,
        query=query,
        turn_seed=turn_seed,
    )


def ensure_validated_consultant_response(
    text: str,
    *,
    mission: MissionState,
    recommendations: Optional[List[AircraftRecommendation]] = None,
    route_assessments: Optional[List[RouteFeasibilityAssessment]] = None,
    comparison: Optional[StructuredComparison] = None,
    query: str = "",
    turn_seed: str = "",
) -> Tuple[str, FormatValidationReport]:
    """
    Validate formatting; on failure regenerate from structured recommendations.
    """
    recs = recommendations or []
    ra = route_assessments or []
    report = validateResponseFormatting(text, recommendations=recs if recs else None)
    if report.ok or not recs:
        return text, report

    regenerated = regenerate_from_structured_recommendations(
        mission=mission,
        recommendations=recs,
        route_assessments=ra,
        comparison=comparison,
        query=query,
        turn_seed=turn_seed,
    )
    regen_report = validateResponseFormatting(regenerated, recommendations=recs)
    if regen_report.ok or len(regen_report.issues) <= len(report.issues):
        merged_issues = list(report.issues) + ["regenerated_from_structured_recommendations"]
        return regenerated, FormatValidationReport(
            ok=regen_report.ok,
            issues=merged_issues if regen_report.ok else regen_report.issues,
            regenerated=True,
        )
    return text, report
