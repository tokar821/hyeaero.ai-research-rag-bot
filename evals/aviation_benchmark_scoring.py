"""
Deterministic scoring for aviation mission benchmark cases.

Scores pipeline outputs (mission extraction, ranker, optional answer text) against golden expectations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

SCORE_DIMENSIONS: tuple[str, ...] = (
    "mission_understanding",
    "entity_accuracy",
    "route_accuracy",
    "aircraft_realism",
    "operational_reasoning",
    "hallucination_rate",
    "contamination_rate",
)

_CONTAMINATION_PATTERNS = (
    r"assuming\s+6[\s–-]8\s+passengers",
    r"mission\s+summary\s*:\s*route\(s\)\s*:",
    r"what\s+would\s+you\s+like\s+to\s+work",
    r"best\s+fit\s+aircraft\s*:\s*challenger\s+350\s+for\s+every",
    r"retrieved\s+context",
    r"pinecone",
)

_INVALID_ROUTE_RE = re.compile(
    r"what\s+would\s+you\s+like|work\s+on\s+today|mission\s+summary\s*:\s*route",
    re.I,
)

_FAKE_MODEL_RE = re.compile(
    r"\b(falcon\s*9000|g\s*6500|global\s*10000|gulfstream\s+g750)\b",
    re.I,
)

_ULTRA_LONG = frozenset({"Gulfstream G650", "Falcon 8X", "Global 7500"})
_LIGHT = frozenset({"Citation CJ2", "Citation CJ4", "Learjet 75"})


@dataclass
class BenchmarkCaseResult:
    case_id: str
    category: str
    passed: bool
    scores: Dict[str, float]
    automated_failures: List[str]
    issues: List[str] = field(default_factory=list)
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _route_blob(routes: List[Any]) -> str:
    parts: List[str] = []
    for r in routes or []:
        if isinstance(r, dict):
            parts.append(f"{r.get('origin', '')} -> {r.get('destination', '')}")
            parts.append(str(r.get("origin", "")))
            parts.append(str(r.get("destination", "")))
        else:
            parts.append(str(r))
    return _norm(" ".join(parts))


def _place_present(place: str, routes: List[Any], input_text: str) -> bool:
    p = _norm(place)
    if not p:
        return True
    blob = _route_blob(routes) + " " + _norm(input_text)
    return p in blob


def _model_in_text(model: str, text: str) -> bool:
    """Match model or alias token (e.g. CJ3+ -> cj3)."""
    t = _norm(text)
    m = _norm(model)
    if m in t:
        return True
    tokens = re.split(r"[\s\-\+]+", m)
    if len(tokens) >= 2 and all(tok in t for tok in tokens):
        return True
    if m.replace(" ", "") in t.replace(" ", ""):
        return True
    return False


def _any_model_match(models: List[str], text: str) -> bool:
    return any(_model_in_text(m, text) for m in models)


def _recommendation_models(recs: List[Dict[str, Any]], *, top_n: int = 6) -> List[str]:
    out: List[str] = []
    for r in (recs or [])[:top_n]:
        if isinstance(r, dict) and r.get("model"):
            out.append(str(r["model"]))
    return out


def _check_constraints(
    profile: Dict[str, Any],
    mission_state: Dict[str, Any],
    required: List[str],
) -> tuple[float, List[str]]:
    if not required:
        return 1.0, []
    hits = 0
    issues: List[str] = []
    p = profile or {}
    ms = mission_state or {}
    blob = _norm(str(p) + str(ms))

    for c in required:
        c_l = c.lower()
        ok = False
        if c_l == "nonstop" and (p.get("nonstop_required") or ms.get("nonstop_requirement")):
            ok = True
        elif c_l in ("westbound", "winter") and (
            p.get("westbound_sensitive") or ms.get("westbound") or "winter" in blob
        ):
            ok = True
        elif c_l == "runway" and (
            "runway" in blob or p.get("runway_priority") not in (None, "none", "")
        ):
            ok = True
        elif c_l == "mountain" and (p.get("mountain_airports") or ms.get("mountain_airport_requirement")):
            ok = True
        elif c_l == "baggage" and (p.get("baggage_priority") not in (None, "none", "") or ms.get("baggage_priority")):
            ok = True
        elif c_l == "operating_cost" and (
            p.get("operating_cost_priority") not in (None, "none", "")
            or ms.get("operating_cost_priority")
            or "operating" in blob
        ):
            ok = True
        elif c_l == "range_realism" and True:
            ok = True
        if ok:
            hits += 1
        else:
            issues.append(f"missing_constraint:{c}")
    return hits / max(len(required), 1), issues


def detect_automated_failures(
    *,
    case: Dict[str, Any],
    turn_profile: Dict[str, Any],
    merged_profile: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    answer: str,
    mission_category: Optional[str],
) -> List[str]:
    failures: List[str] = []
    golden = case.get("golden") or {}
    routes = turn_profile.get("routes") or []
    route_labels = _route_blob(routes)

    # Previous turn leak
    leak = case.get("prior_context_must_not_leak") or {}
    for bad_route in leak.get("routes") or []:
        if _norm(bad_route) in route_labels and not _any_model_match([bad_route], case.get("input") or ""):
            failures.append(f"previous_turn_leak:route:{bad_route}")
    for bad_pax in leak.get("passengers") or []:
        if turn_profile.get("passengers") == bad_pax and str(bad_pax) not in (case.get("input") or ""):
            failures.append(f"previous_turn_leak:passengers:{bad_pax}")

    # Invalid routes
    if golden.get("routes_must_be_empty") and routes:
        failures.append("invalid_routes:expected_empty")
    for r in routes:
        label = r if isinstance(r, str) else f"{r.get('origin', '')} -> {r.get('destination', '')}"
        if _INVALID_ROUTE_RE.search(label):
            failures.append(f"invalid_routes:{label}")
    for pat in golden.get("forbidden_route_patterns") or []:
        if _norm(pat) in route_labels:
            failures.append(f"invalid_routes:pattern:{pat}")

    # Impossible aircraft
    rec_models = _recommendation_models(recommendations)
    cat = (mission_category or "").lower()
    max_leg_hint = "regional" in cat or "mountain" in cat
    for m in rec_models[:3]:
        if max_leg_hint and m in _ULTRA_LONG:
            failures.append(f"impossible_aircraft_recommended:{m}")
        if "ultra_long" in cat and m in _LIGHT:
            failures.append(f"impossible_aircraft_recommended:{m}")

    for forbidden in golden.get("forbidden_any_models") or []:
        if any(_model_in_text(forbidden, m) for m in rec_models[:5]):
            failures.append(f"impossible_aircraft_recommended:{forbidden}")

    # Ignores mission constraints
    req = golden.get("constraints_required") or []
    if req:
        score, miss = _check_constraints(turn_profile, {}, req)
        if score < 0.5:
            failures.append("ignores_mission_constraints:" + ",".join(miss[:3]))

    if golden.get("passenger_count") is not None:
        if turn_profile.get("passengers") != golden["passenger_count"]:
            failures.append(
                f"ignores_mission_constraints:passengers_expected_{golden['passenger_count']}_got_{turn_profile.get('passengers')}"
            )

    # Hallucination in answer
    if answer and _FAKE_MODEL_RE.search(answer) and not re.search(
        r"no\s+such|does\s+not\s+exist|isn'?t\s+real|not\s+a\s+production",
        answer,
        re.I,
    ):
        failures.append("hallucination:fake_model_in_answer")

    return failures


def score_benchmark_case(
    *,
    case: Dict[str, Any],
    turn_profile: Dict[str, Any],
    merged_profile: Dict[str, Any],
    mission_state: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    mission_category: Optional[str] = None,
    answer: str = "",
) -> BenchmarkCaseResult:
    """Score one benchmark case against golden expectations and pipeline artifacts."""
    golden = case.get("golden") or {}
    issues: List[str] = []
    scores: Dict[str, float] = {}

    route_labels = _route_blob(turn_profile.get("routes") or [])
    rec_models = _recommendation_models(recommendations)
    rec_blob = " ".join(rec_models)
    answer_l = answer or ""

    # --- mission understanding ---
    mu = 1.0
    exp_cat = golden.get("expected_mission_category")
    exp_cats = golden.get("expected_mission_category_any") or (
        [exp_cat] if exp_cat else []
    )
    if exp_cats and mission_category:
        if not any(_norm(c) == _norm(mission_category) for c in exp_cats):
            mu -= 0.45
            issues.append(f"mission_category_mismatch:{mission_category}!={exp_cats}")
    if golden.get("ownership_posture"):
        own = (merged_profile.get("ownership_interest") or turn_profile.get("ownership_interest") or "")
        if _norm(str(own)) != _norm(golden["ownership_posture"]):
            mu -= 0.35
            issues.append("ownership_posture_mismatch")
    if golden.get("passenger_count") is not None:
        if turn_profile.get("passengers") != golden["passenger_count"]:
            mu -= 0.4
    c_score, c_issues = _check_constraints(turn_profile, mission_state, golden.get("constraints_required") or [])
    mu = _clamp(mu * 0.7 + c_score * 0.3)
    issues.extend(c_issues)
    scores["mission_understanding"] = _clamp(mu)

    # --- entity accuracy ---
    ent = 1.0
    for place in golden.get("route_must_include") or []:
        if not _place_present(place, turn_profile.get("routes") or [], case.get("input") or ""):
            ent -= 0.25
            issues.append(f"entity_missing:{place}")
    if golden.get("ownership_posture") and _norm(golden["ownership_posture"]) in _norm(
        str(merged_profile.get("ownership_interest") or "")
    ):
        ent = min(1.0, ent + 0.05)
    scores["entity_accuracy"] = _clamp(ent)

    # --- route accuracy ---
    ra = 1.0
    for place in golden.get("route_must_include") or []:
        if not _place_present(place, turn_profile.get("routes") or [], case.get("input") or ""):
            ra -= 0.35
            issues.append(f"route_missing:{place}")
    for bad in golden.get("route_must_not_include") or []:
        if _norm(bad) in route_labels:
            ra -= 0.5
            issues.append(f"route_contamination:{bad}")
    if golden.get("routes_must_be_empty") and turn_profile.get("routes"):
        ra = 0.0
        issues.append("routes_should_be_empty")
    scores["route_accuracy"] = _clamp(ra)

    # --- aircraft realism ---
    ar = 1.0
    forbidden = golden.get("forbidden_any_models") or []
    for f in forbidden:
        if any(_model_in_text(f, m) for m in rec_models[:6]):
            ar -= 0.35
            issues.append(f"forbidden_aircraft_ranked:{f}")
        if answer and _model_in_text(f, answer) and "avoid" not in answer_l.lower():
            ar -= 0.15
    expected = golden.get("expected_any_models") or []
    if expected:
        if not (_any_model_match(expected, rec_blob) or (answer and _any_model_match(expected, answer))):
            ar -= 0.4
            issues.append("expected_aircraft_missing")
    scores["aircraft_realism"] = _clamp(ar)

    # --- operational reasoning ---
    op = 0.5
    if answer:
        if golden.get("routes_must_be_empty") and re.search(
            r"\b(?:city\s+pair|passenger|pax|nonstop|aircraft\s+class|origin)\b",
            answer,
            re.I,
        ):
            op = max(op, 0.82)
        if re.search(r"\b(tradeoff|tradeoffs|compromise|margin|practical|NBAA|reserve|runway|payload|operating)\b", answer, re.I):
            op += 0.35
        if golden.get("ownership_posture") and re.search(
            r"\b(fractional|ownership|hours|capital|charter|utilization|fixed\s+cost)\b",
            answer,
            re.I,
        ):
            op = max(op, 0.75)
        must = golden.get("answer_must_mention_any") or []
        if must:
            hits = sum(1 for m in must if m.lower() in answer_l.lower())
            op = _clamp(max(op, 0.4 + 0.6 * (hits / max(len(must), 1))))
        if rec_models and turn_profile.get("routes"):
            op = min(1.0, op + 0.15)
    elif rec_models and any(
        r.get("explanation") for r in (recommendations or []) if isinstance(r, dict)
    ):
        op = 0.75
    scores["operational_reasoning"] = _clamp(op)

    # --- hallucination rate (inverted: 1 = clean) ---
    hall = 1.0
    if answer and _FAKE_MODEL_RE.search(answer):
        if re.search(r"no\s+such|does\s+not\s+exist|isn'?t\s+real", answer, re.I):
            hall = 0.9
        else:
            hall = 0.0
            issues.append("hallucination_detected")
    scores["hallucination_rate"] = hall

    # --- contamination rate (inverted: 1 = clean) ---
    cont = 1.0
    for pat in _CONTAMINATION_PATTERNS:
        if answer and re.search(pat, answer, re.I):
            cont -= 0.35
            issues.append(f"contamination:{pat[:40]}")
    for bad in golden.get("route_must_not_include") or []:
        if _norm(bad) in route_labels:
            cont -= 0.4
    leak = case.get("prior_context_must_not_leak") or {}
    for bad_route in leak.get("routes") or []:
        if _norm(bad_route) in route_labels:
            cont = 0.0
            issues.append(f"turn_leak:{bad_route}")
    scores["contamination_rate"] = _clamp(cont)

    failures = detect_automated_failures(
        case=case,
        turn_profile=turn_profile,
        merged_profile=merged_profile,
        recommendations=recommendations,
        answer=answer,
        mission_category=mission_category,
    )

    critical = bool(
        failures
        or scores["route_accuracy"] < 0.4
        or scores["aircraft_realism"] < 0.4
        or scores["contamination_rate"] < 0.5
    )
    passed = not failures and all(scores[d] >= 0.55 for d in SCORE_DIMENSIONS)

    return BenchmarkCaseResult(
        case_id=str(case.get("id") or ""),
        category=str(case.get("category") or ""),
        passed=passed,
        scores=scores,
        automated_failures=failures,
        issues=sorted(set(issues)),
        critical=critical,
        metadata={
            "routes": turn_profile.get("routes"),
            "passengers": turn_profile.get("passengers"),
            "mission_category": mission_category,
            "top_recommendations": rec_models[:6],
        },
    )


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def aggregate_scores(results: List[BenchmarkCaseResult]) -> Dict[str, float]:
    if not results:
        return {d: 0.0 for d in SCORE_DIMENSIONS}
    agg: Dict[str, float] = {}
    for d in SCORE_DIMENSIONS:
        agg[d] = sum(r.scores.get(d, 0.0) for r in results) / len(results)
    return agg
