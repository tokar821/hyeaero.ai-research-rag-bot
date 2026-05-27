"""
P3 — Normalize fleet composition plans into immutable audit traces.

Each operational domain segment carries authoritative corridor, payload, and
elimination lineage so downstream LLM/formatter prose cannot collapse domains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _assignment_by_domain(assignments: List[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for a in assignments or []:
        if not isinstance(a, dict):
            continue
        role = str(a.get("role") or "")
        if role:
            out[role] = a
    return out


def _segment_routes_by_role(segments: List[Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in segments or []:
        if not isinstance(s, dict):
            continue
        role = str(s.get("role") or "")
        if role:
            out[role] = list(s.get("route_labels") or [])
    return out


def build_fleet_audit_trace(fleet_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structured fleet trace for ``hye_reasoning_packet.fleet_audit``.

    Merges domain_traces (authoritative gates) with assignments (presentation).
    """
    if not isinstance(fleet_plan, dict):
        return {}

    assignments = _assignment_by_domain(fleet_plan.get("assignments") or [])
    routes_by_role = _segment_routes_by_role(fleet_plan.get("segments") or [])

    segment_audits: List[Dict[str, Any]] = []
    for tr in fleet_plan.get("domain_traces") or []:
        if not isinstance(tr, dict):
            continue
        domain = str(tr.get("domain") or "")
        asn = assignments.get(domain) or {}
        segment_audits.append(
            {
                "domain": domain,
                "segment_label": str(asn.get("segment_label") or domain),
                "route_labels": routes_by_role.get(domain) or [],
                "corridor_classification": tr.get("corridor_classification"),
                "corridor_decision": tr.get("corridor_decision"),
                "payload_assumptions": dict(tr.get("payload_assumptions") or {}),
                "constraint_triggers": list(tr.get("constraint_triggers") or []),
                "elimination_lineage": list(tr.get("elimination_lineage") or []),
                "feasible_models": list(tr.get("feasible_models") or []),
                "eliminated_models": list(tr.get("eliminated_models") or []),
                "airport_profiles": list(tr.get("airport_profiles") or []),
                "primary_model": str(asn.get("primary_model") or ""),
                "fit_verdict": str(asn.get("fit_verdict") or ""),
                "domain_feasible": bool(asn.get("domain_feasible", True)),
                "alternates": list(asn.get("alternates") or []),
            }
        )

    invariant = fleet_plan.get("fleet_invariant")
    if invariant is None:
        invariant_ok = all(
            not s.get("primary_model")
            or (
                s.get("domain_feasible")
                and s.get("primary_model") in (s.get("feasible_models") or [])
            )
            for s in segment_audits
        )
        invariant = {"ok": invariant_ok, "violations": [], "stripped_models": []}

    return {
        "multi_domain_required": bool(
            fleet_plan.get("multi_domain_required") or fleet_plan.get("multi_aircraft_required")
        ),
        "trigger": str(fleet_plan.get("trigger") or "none"),
        "single_aircraft_structurally_invalid": bool(
            fleet_plan.get("single_aircraft_structurally_invalid")
        ),
        "doctrine": str(fleet_plan.get("doctrine") or ""),
        "universal_survivors": list(fleet_plan.get("universal_survivors") or []),
        "presented_models": list(fleet_plan.get("presented_models") or []),
        "segments": segment_audits,
        "fleet_invariant": (
            dict(invariant) if isinstance(invariant, dict) else {"ok": True}
        ),
    }


def validate_fleet_audit_trace(audit: Dict[str, Any]) -> List[str]:
    """Return violation messages — empty if fleet audit is internally consistent."""
    issues: List[str] = []
    if not audit.get("multi_domain_required"):
        return issues

    if audit.get("single_aircraft_structurally_invalid") and audit.get("universal_survivors"):
        issues.append("structural_invalid_but_universal_survivors_nonempty")

    inv = audit.get("fleet_invariant") or {}
    if isinstance(inv, dict) and inv.get("ok") is False:
        for v in inv.get("violations") or []:
            issues.append(f"fleet_invariant:{v}")

    for seg in audit.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        primary = (seg.get("primary_model") or "").strip()
        if not primary:
            continue
        domain = seg.get("domain", "?")
        feasible = {str(m).lower() for m in seg.get("feasible_models") or []}
        eliminated = {str(m).lower() for m in seg.get("eliminated_models") or []}
        if primary.lower() in eliminated:
            issues.append(f"{primary} in domain eliminated set ({domain})")
        if feasible and primary.lower() not in feasible:
            issues.append(f"{primary} not in domain feasible set ({domain})")
        if not seg.get("domain_feasible", True):
            issues.append(f"{primary} assigned with domain_feasible=false ({domain})")

    return issues


def merge_fleet_eliminations_into_packet(
    eliminations: List[Any],
    fleet_audit: Dict[str, Any],
) -> None:
    """Append per-domain lineage records to packet eliminations list (in-place)."""
    for seg in fleet_audit.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        domain = seg.get("domain", "unknown")
        for entry in seg.get("elimination_lineage") or []:
            if not isinstance(entry, dict) or not entry.get("model"):
                continue
            eliminations.append(
                {
                    "stage": f"fleet_domain_{domain}",
                    "model": str(entry["model"]),
                    "reason": str(entry.get("reason") or ""),
                }
            )
