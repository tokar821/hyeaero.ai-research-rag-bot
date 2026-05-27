"""Mission Understanding — FINAL semantic-only 10 query runner.

Runs AFTER route extraction but BEFORE pre-ranking/ranking concerns.
Validates:
- domains + explicit non-uniform weights
- arctic/industrial treated as hard constraint-heavy domains
- continuation hubs don't override origin logic (semantic constraints only)
- rejects single-aircraft semantic flattening
- no aircraft-name leakage in semantic outputs
- no route graph output in printed report (we never print routes)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import build_mission_understanding


QUERIES: List[Tuple[str, str, Dict[str, Any]]] = [
    (
        "Q1 Arctic + executive + industrial conflict",
        "We operate Nunavut gravel strips, Calgary oil fields, and fly executives to London twice "
        "a month nonstop, but winter reliability failures keep happening. Is this a single-mission "
        "system or multiple missions?",
        {"must_flags": ["arctic_hard_domain", "industrial_hard_domain"], "must_invalid": ["single_ulr_covers_arctic_gravel_and_transatlantic_executive"]},
    ),
    (
        "Q2 Founder + global prestige vs domestic reality",
        "Our CEO demands Dubai nonstop capability from New York, but 80% of flights are short hops "
        "between New York, Boston, and Chicago. What is the true mission structure?",
        {"must_flags": ["ulr_continuation_requires_mandate_hub_origin"], "must_domains": ["domestic_utilization", "ulr_continuation"]},
    ),
    (
        "Q3 Mining + Asia + Europe tri-domain stress",
        "We run Perth mining operations, move teams to Singapore and Tokyo, and occasionally send "
        "executives to Frankfurt and Zurich. How should this mission be understood?",
        {"must_flags": ["mining_hard_domain"], "must_domains": ["executive_transport", "ulr_continuation"]},
    ),
    (
        "Q4 Arctic + transatlantic + Caribbean mix",
        "We operate in Northern Canada gravel strips, fly Houston–London nonstop, and also run "
        "seasonal leisure flights to the Caribbean from Miami. What are the dominant mission domains?",
        {"must_flags": ["arctic_hard_domain"], "must_domains": ["caribbean_regional", "executive_transport"]},
    ),
    (
        "Q5 Multi-continent engineering + cargo variability",
        "We transport engineering teams (4–14 pax) between Chicago, Frankfurt, São Paulo, and Lagos, "
        "sometimes with heavy equipment. What is the mission classification?",
        {"must_domains": ["executive_transport"], "weight_nonuniform": True},
    ),
    (
        "Q6 Asia-Pacific + ski season conflict",
        "We fly Los Angeles to Tokyo, Seoul, Singapore, but also move ski teams into Aspen, "
        "Jackson Hole, and Banff. How should this mission be structured semantically?",
        {"must_domains": ["mountain_leisure", "ulr_continuation"], "weight_order": ["executive_transport", "mountain_leisure"]},
    ),
    (
        "Q7 Industrial Africa + EU executive overlay",
        "We operate West African offshore rigs, Houston oil logistics, and executive travel between "
        "Zurich, Paris, and Frankfurt. What are the priority domains?",
        {"must_flags": ["industrial_hard_domain"], "must_domains": ["executive_transport"]},
    ),
    (
        "Q8 Continuation hub ambiguity (Dubai/Singapore)",
        "We route executives from New York to Singapore, sometimes via Dubai, while also doing "
        "New York–London domestic executive cycles. What role do continuation hubs play here?",
        {"must_domains": ["ulr_continuation", "executive_transport"], "must_flags": ["continuation_hubs_semantic_only_not_primary_origin"]},
    ),
    (
        "Q9 Extreme mixed-domain pressure system",
        "We operate: Arctic gravel strips (Nunavut) / Houston oil fields / Miami Caribbean operations / "
        "London executive HQ / Los Angeles Asia-Pacific flights. How should mission priorities be structured?",
        {"must_flags": ["multi_hard_domain_mission"], "must_domains": ["arctic_operations", "industrial_field"]},
    ),
    (
        "Q10 One-aircraft constraint stress test",
        "Leadership insists: “We must use ONE aircraft for everything.” But operations span: "
        "Arctic + industrial + transatlantic + Asia + Caribbean. How should this be interpreted at "
        "the mission understanding layer?",
        {"must_invalid": ["single_aircraft_universal_hard_domain_coverage", "single_aircraft_preference_over_hard_domain_conflict"]},
    ),
]


_AIRCRAFT_LEAK_RE = re.compile(
    r"\b(?:gulfstream|global\s+\d+|falcon\s+\d+|citation|learjet|embraer|phenom|hawker)\b",
    re.I,
)


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(_safe_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(_safe_text(v) for v in x.values())
    return str(x)


def _check(title: str, semantic: Dict[str, Any], packet: Dict[str, Any], checks: Dict[str, Any]) -> Dict[str, Any]:
    flags = set()
    inf = (packet.get("inferred_constraints") or {}) if isinstance(packet, dict) else {}
    for k, v in inf.items():
        if v is True:
            flags.add(k)

    domains = list(semantic.get("mission_domains") or [])
    weights = list(semantic.get("domain_weights") or [])
    invalid = list(semantic.get("invalid_interpretations") or [])
    notes = list(semantic.get("reasoning_clarity_notes") or [])

    errors: List[str] = []

    # No aircraft leakage in semantic payload
    leak_blob = " ".join(
        [
            _safe_text(semantic),
            _safe_text(packet.get("understanding_notes")),
            _safe_text(packet.get("operational_synthesis")),
        ]
    )
    if _AIRCRAFT_LEAK_RE.search(leak_blob):
        errors.append("aircraft_name_leak")

    # Explicit weights required and non-uniform
    if not domains or not weights or len(domains) != len(weights):
        errors.append("missing_domains_or_weights")
    elif len(set(float(w) for w in weights)) <= 1:
        errors.append("uniform_weights_for_multi_domain")

    # Must flags
    for f in checks.get("must_flags") or []:
        if f not in flags:
            errors.append(f"missing_flag:{f}")

    # Must domains (presence)
    for d in checks.get("must_domains") or []:
        if d not in domains:
            errors.append(f"missing_domain:{d}")

    # Must invalid interpretations
    for inv in checks.get("must_invalid") or []:
        if inv not in invalid:
            errors.append(f"missing_invalid:{inv}")

    # Weight order: A should outrank B (if both present)
    for a, b in [tuple(checks.get("weight_order") or [])] if checks.get("weight_order") else []:
        if a in domains and b in domains:
            wa = weights[domains.index(a)]
            wb = weights[domains.index(b)]
            if not (wa > wb):
                errors.append(f"bad_weight_order:{a}<= {b}")

    status = "PASS" if not errors else "FAIL"
    return {
        "title": title,
        "status": status,
        "domains": domains[:6],
        "weights": weights[:6],
        "hard_domains": semantic.get("hard_domains") or [],
        "flags_hit": sorted([f for f in (checks.get("must_flags") or []) if f in flags]),
        "invalid": invalid[:6],
        "notes": notes[:2],
        "errors": errors,
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    counts = {"PASS": 0, "FAIL": 0}

    for title, query, checks in QUERIES:
        profile = extract_mission(query)
        mission = MissionState(routes=profile.route_labels(), passenger_count=profile.passengers)
        data_used: Dict[str, Any] = {}
        pkt = build_mission_understanding(query, profile, mission, broker_memory=None, history=None, data_used=data_used)
        semantic = data_used.get("mission_semantic_model") or {}
        r = _check(title, semantic, pkt.to_dict(), checks)
        results.append(r)
        counts[r["status"]] += 1

    print("=" * 92)
    print("MISSION UNDERSTANDING — FINAL 10 SEMANTIC TEST REPORT")
    print(f"SUMMARY: PASS={counts['PASS']} FAIL={counts['FAIL']}")
    print("=" * 92)
    for r in results:
        print()
        print(f"## {r['title']} [{r['status']}]")
        print(f"Domains: {', '.join(r['domains'])}")
        print(f"Weights: {', '.join(str(w) for w in r['weights'])}")
        if r["hard_domains"]:
            print(f"Hard domains: {', '.join(r['hard_domains'])}")
        if r["flags_hit"]:
            print(f"Required flags present: {', '.join(r['flags_hit'])}")
        if r["invalid"]:
            print(f"Invalid interpretations: {', '.join(r['invalid'])}")
        if r["notes"]:
            print(f"Notes: {' '.join(r['notes'])}")
        if r["errors"]:
            print(f"FAIL reasons: {', '.join(r['errors'])}")

    out = _BACKEND / "evals" / "mission_semantics_final_10_results.json"
    out.write_text(json.dumps({"summary": counts, "results": results}, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

