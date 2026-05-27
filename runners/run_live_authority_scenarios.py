"""
Live consultant orchestration — six authority / segmentation scenarios.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

env_path = _ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402

SCENARIOS = [
    {
        "id": "dual_band_ulr_mountain",
        "title": "Dual-band contradiction (ULR + short-field)",
        "query": (
            "We're a hedge fund based in New York. We fly 4–6 executives weekly to London and Frankfurt, "
            "but every month we also send teams into Telluride and Aspen in winter. We previously tried a "
            "Global 7500 but dispatch into ski airports became unreliable. What structure actually makes sense?"
        ),
        "checks": [
            "structural_decomposition",
            "incompatible_or_portfolio",
            "segment_structure",
            "not_single_global7500_only",
            "mountain_not_global_only",
        ],
    },
    {
        "id": "industrial_transatlantic_remote",
        "title": "Industrial + transatlantic + remote access",
        "query": (
            "We move engineers between Houston, Calgary, and remote oil sites in Northern Canada, but "
            "executives also fly quarterly to London and Zurich. Reliability into short and unpaved runways "
            "matters more than cabin comfort."
        ),
        "checks": [
            "multi_segment",
            "runway_priority",
            "no_false_single_aircraft",
        ],
    },
    {
        "id": "heavy_ulr_domestic_imbalance",
        "title": "Heavy ULR + high-frequency domestic imbalance",
        "query": (
            "Our company does NYC–Los Angeles daily with 3–4 executives, but twice a month the CEO flies "
            "nonstop to Riyadh or Dubai. We don't want multiple aircraft unless absolutely necessary."
        ),
        "checks": [
            "continuation_segment",
            "not_over_fleet_trigger",
            "dual_use_without_forced_portfolio",
        ],
    },
    {
        "id": "multi_region_caribbean",
        "title": "Multi-region portfolio conflicting airport classes",
        "query": (
            "We operate between Miami, São Paulo, Madrid, and small Caribbean islands. Some airports are "
            "short runway, others are long-haul international hubs. We care about dispatch reliability "
            "above all else."
        ),
        "checks": [
            "multi_segment",
            "structural_or_portfolio",
            "dispatch_emphasis",
        ],
    },
    {
        "id": "ceo_override_domestic",
        "title": "CEO override + underutilized aircraft history",
        "query": (
            "We previously owned a large jet that was mostly idle because it was too large for domestic trips. "
            "Now the CEO insists on nonstop New York–Dubai capability, but the rest of the company only flies "
            "domestic 2–3 hour legs."
        ),
        "checks": [
            "continuation_or_ulr_segment",
            "governance_asymmetry",
            "not_blind_global_only",
        ],
    },
    {
        "id": "ski_intercontinental_seasonal",
        "title": "High-complexity ski + intercontinental + seasonal",
        "query": (
            "We fly executives from Los Angeles to Tokyo and Singapore, but during winter we also run constant "
            "Aspen and Jackson Hole trips. Our last aircraft struggled badly in ski season and caused dispatch delays."
        ),
        "checks": [
            "structural_decomposition",
            "winter_or_mountain",
            "dual_aircraft_structure",
            "not_single_jet_collapse",
        ],
    },
]


@dataclass
class ScenarioResult:
    id: str
    title: str
    answer: str
    packet: Dict[str, Any]
    kernel: Dict[str, Any]
    fleet: Dict[str, Any]
    structural: Dict[str, Any]
    checks: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "answer_preview": (self.answer or "")[:1200],
            "answer_len": len(self.answer or ""),
            "packet_summary": _packet_summary(self.packet),
            "kernel_summary": _kernel_summary(self.kernel),
            "fleet_summary": _fleet_summary(self.fleet),
            "structural": self.structural,
            "checks": self.checks,
            "notes": self.notes,
        }


def _packet_summary(pkt: Dict[str, Any]) -> Dict[str, Any]:
    if not pkt:
        return {}
    inf = pkt.get("inferred_constraints") or {}
    return {
        "bands": pkt.get("fallback_operational_band") or [],
        "incompatible_mission_bands": inf.get("incompatible_mission_bands"),
        "dual_use_or_multi_leg": inf.get("dual_use_or_multi_leg"),
        "synthesis_preview": (pkt.get("operational_synthesis") or "")[:400],
        "recommend_aircraft": pkt.get("recommend_aircraft"),
    }


def _kernel_summary(k: Dict[str, Any]) -> Dict[str, Any]:
    if not k:
        return {}
    return {
        "structural_decomposition": k.get("structural_decomposition"),
        "authorized_models": k.get("authorized_models") or [],
        "segment_roles": k.get("segment_roles") or [],
        "peak_segment_id": k.get("peak_segment_id"),
        "single_aircraft_forbidden": k.get("single_aircraft_forbidden"),
    }


def _fleet_summary(f: Dict[str, Any]) -> Dict[str, Any]:
    if not f:
        return {}
    return {
        "multi_aircraft_required": f.get("multi_aircraft_required"),
        "single_aircraft_structurally_invalid": f.get("single_aircraft_structurally_invalid"),
        "assignments": f.get("assignments") or [],
        "doctrine_preview": (f.get("doctrine") or "")[:300],
    }


def _run_checks(scenario: Dict[str, Any], answer: str, pkt: Dict, kernel: Dict, fleet: Dict, structural: Dict) -> tuple[Dict[str, bool], List[str]]:
    a = (answer or "").lower()
    inf = (pkt.get("inferred_constraints") or {}) if pkt else {}
    bands = [b.lower() for b in (pkt.get("fallback_operational_band") or [])]
    notes: List[str] = []
    out: Dict[str, bool] = {}

    struct_req = bool(
        structural.get("required")
        or kernel.get("structural_decomposition")
        or inf.get("incompatible_mission_bands")
    )
    has_segments = "operational segments:" in a or bool(kernel.get("segments"))
    has_portfolio = (
        "per-segment" in a
        or "fleet structure" in a
        or "multi-domain" in a
        or "portfolio" in a
        or fleet.get("multi_aircraft_required")
    )
    mentions_g7500_alone = bool(
        re.search(r"global\s*7500.*(?:only|primary|best|start with|recommend)", a)
        or (a.count("global 7500") >= 1 and "per-segment" not in a and not has_portfolio and struct_req)
    )
    mentions_mountain = any(w in a for w in ("aspen", "telluride", "mountain", "ski", "jackson"))
    mentions_ulr = any(w in a for w in ("ulr", "ultra-long", "dubai", "riyadh", "tokyo", "singapore"))
    single_collapse = bool(re.search(r"one aircraft.*(?:everything|all|covers)", a))

    for ck in scenario.get("checks") or []:
        if ck == "structural_decomposition":
            out[ck] = struct_req
        elif ck == "incompatible_or_portfolio":
            out[ck] = struct_req or has_portfolio or inf.get("dual_use_or_multi_leg")
        elif ck == "segment_structure":
            out[ck] = has_segments or "operational synthesis (authoritative)" in a
        elif ck == "not_single_global7500_only":
            out[ck] = not mentions_g7500_alone or has_portfolio or "per-segment" in a
        elif ck == "mountain_not_global_only":
            out[ck] = mentions_mountain and (has_portfolio or not struct_req or "per-segment" in a)
        elif ck == "multi_segment":
            out[ck] = has_segments or len(bands) >= 2
        elif ck == "runway_priority":
            out[ck] = any(w in a for w in ("runway", "unpaved", "field", "short", "industrial"))
        elif ck == "no_false_single_aircraft":
            out[ck] = not (
                fleet.get("single_aircraft_structurally_invalid") is False
                and struct_req
                and "single aircraft" in a
                and "invalid" not in a
            )
        elif ck == "continuation_segment":
            out[ck] = any("continuation" in b for b in bands) or any(
                w in a for w in ("continuation", "dubai", "riyadh", "middle east")
            )
        elif ck == "not_over_fleet_trigger":
            out[ck] = not (struct_req and not inf.get("incompatible_mission_bands") and "don't want multiple" in a)
        elif ck == "dual_use_without_forced_portfolio":
            out[ck] = inf.get("dual_use_or_multi_leg") and (
                not struct_req or "unless absolutely" in a or "single-aircraft" in a
            )
        elif ck == "structural_or_portfolio":
            out[ck] = struct_req or has_portfolio
        elif ck == "dispatch_emphasis":
            out[ck] = "dispatch" in a or (pkt.get("dispatch_priority") == "high")
        elif ck == "continuation_or_ulr_segment":
            out[ck] = mentions_ulr or any("ulr" in b or "continuation" in b for b in bands)
        elif ck == "governance_asymmetry":
            out[ck] = any(w in a for w in ("ceo", "domestic", "dubai", "idle", "underutil"))
        elif ck == "not_blind_global_only":
            out[ck] = not (mentions_g7500_alone and struct_req)
        elif ck == "winter_or_mountain":
            out[ck] = mentions_mountain or "winter" in a
        elif ck == "dual_aircraft_structure":
            out[ck] = has_portfolio or (struct_req and ("per-segment" in a or fleet.get("multi_aircraft_required")))
        elif ck == "not_single_jet_collapse":
            out[ck] = not single_collapse
        else:
            out[ck] = False

    if struct_req:
        notes.append(f"structural_proof={structural.get('proof_kind') or 'incompatible'}")
    notes.append(f"bands={len(bands)} authorized={kernel.get('authorized_models')}")
    if fleet.get("assignments"):
        notes.append(f"fleet_roles={[a.get('primary_model') for a in fleet.get('assignments') if isinstance(a, dict)]}")
    return out, notes


def run_scenario(sc: Dict[str, Any]) -> ScenarioResult:
    data_used: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}
    orch = run_consultant_orchestration(
        sc["query"],
        conversation_state={"history": []},
        data_used=data_used,
        query_intent="mission_feasibility",
    )
    if isinstance(orch.data_used_patch, dict):
        data_used.update(orch.data_used_patch)

    pkt = data_used.get("mission_understanding_packet") or {}
    if not isinstance(pkt, dict):
        pkt = {}
    kernel = data_used.get("mission_authority_kernel") or {}
    if not isinstance(kernel, dict):
        kernel = {}
    fleet = data_used.get("fleet_composition_plan") or {}
    if not isinstance(fleet, dict):
        fleet = {}
    structural = data_used.get("structural_decomposition") or {}
    if not isinstance(structural, dict):
        structural = {}

    checks, notes = _run_checks(sc, orch.answer or "", pkt, kernel, fleet, structural)
    return ScenarioResult(
        id=sc["id"],
        title=sc["title"],
        answer=orch.answer or "",
        packet=pkt,
        kernel=kernel,
        fleet=fleet,
        structural=structural,
        checks=checks,
        notes=notes,
    )


def main() -> int:
    results: List[ScenarioResult] = []
    for sc in SCENARIOS:
        print(f"\n{'='*60}\nRunning: {sc['title']}\n{'='*60}")
        try:
            r = run_scenario(sc)
            results.append(r)
            passed = sum(1 for v in r.checks.values() if v)
            total = len(r.checks)
            print(f"Checks: {passed}/{total} — {r.checks}")
            print(f"Notes: {r.notes}")
            print(f"\n--- Answer (first 800 chars) ---\n{(r.answer or '')[:800]}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append(
                ScenarioResult(
                    id=sc["id"],
                    title=sc["title"],
                    answer="",
                    packet={},
                    kernel={},
                    fleet={},
                    structural={},
                    notes=[f"error:{exc}"],
                )
            )

    out = _ROOT / "evals" / "live_authority_scenario_results.json"
    payload = {
        "results": [r.to_dict() for r in results],
        "full_answers": {r.id: r.answer for r in results},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
