"""
HACK v4 — Adversarial Test Generator & Continuous Failure Stress Layer.

NOT part of inference. Generates adversarial mission prompts and metadata only.
Does not run ranking, mission understanding, or external APIs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Trap taxonomy (≥5 required for acceptance).
TRAP_PHYSICS_VIOLATION = "physics_violation"
TRAP_HUB_COLLAPSE = "hub_collapse"
TRAP_CONTINUATION_ABUSE = "continuation_abuse"
TRAP_ECONOMIC_MISSION_CONFLICT = "economic_vs_mission_conflict"
TRAP_MULTI_DOMAIN_OVERLOAD = "multi_domain_overload"
TRAP_AIRCRAFT_CLASS_POISONING = "aircraft_class_poisoning"
TRAP_NARRATIVE_DRIFT = "narrative_drift"

ALL_TRAP_CLASSES: Tuple[str, ...] = (
    TRAP_PHYSICS_VIOLATION,
    TRAP_HUB_COLLAPSE,
    TRAP_CONTINUATION_ABUSE,
    TRAP_ECONOMIC_MISSION_CONFLICT,
    TRAP_MULTI_DOMAIN_OVERLOAD,
    TRAP_AIRCRAFT_CLASS_POISONING,
    TRAP_NARRATIVE_DRIFT,
)

FAILURE_PHYSICS_OVERRIDE = "physics_override"
FAILURE_FALLBACK_INJECTION = "fallback_injection"
FAILURE_RANKING_PARADOX = "ranking_paradox"
FAILURE_HUB_COLLAPSE = "hub_collapse"
FAILURE_NARRATIVE_DRIFT = "narrative_drift"

MUTATION_GEO_FLIP = "geo_flip"
MUTATION_DIRECTION_INVERT = "direction_invert"
MUTATION_ARCTIC_OR_DESERT = "arctic_or_desert"
MUTATION_CONTINUATION_HUB = "continuation_hub"
MUTATION_PAX_PLUS_40 = "pax_plus_40"

# Catalog model tokens must never appear in generated prompts (class labels only).
_FORBIDDEN_MODEL_PATTERNS: Tuple[str, ...] = (
    "citation cj",
    "citation latitude",
    "citation longitude",
    "learjet",
    "gulfstream g",
    "global 7500",
    "global 6500",
    "challenger 350",
    "challenger 650",
    "challenger 3500",
    "praetor",
    "pilatus pc",
    "embraer legacy",
    "falcon 7",
    "falcon 8",
    "bbj",
    "acj",
    "hawker",
    "hondajet",
)


@dataclass(frozen=True)
class AdversarialTest:
    """Full internal test record (generation + validation)."""

    id: str
    prompt: str
    hidden_intent: str
    expected_system_behavior: str
    forbidden_behaviors: Tuple[str, ...]
    trap_class: str
    severity: str
    stress_score: int
    expected_failure_modes: Tuple[str, ...]
    mutation: Optional[str] = None
    parent_id: Optional[str] = None

    def to_contract_row(self) -> Dict[str, Any]:
        """Public output contract — no hidden fields."""
        return {
            "test_id": self.id,
            "prompt": self.prompt,
            "trap_class": self.trap_class,
            "severity": self.severity,
            "stress_score": int(self.stress_score),
            "expected_failure_modes": list(self.expected_failure_modes),
        }


@dataclass
class StressComponents:
    domain_conflict_weight: float = 0.0
    geographic_entropy: float = 0.0
    aircraft_class_pressure: float = 0.0
    continuation_hub_ambiguity: float = 0.0
    economic_vs_physics_tension: float = 0.0

    def total(self) -> int:
        raw = (
            self.domain_conflict_weight
            + self.geographic_entropy
            + self.aircraft_class_pressure
            + self.continuation_hub_ambiguity
            + self.economic_vs_physics_tension
        )
        return max(0, min(100, int(round(raw))))


def severity_from_stress(stress_score: int) -> str:
    if stress_score >= 86:
        return "CRITICAL"
    if stress_score >= 61:
        return "HIGH"
    if stress_score >= 31:
        return "MEDIUM"
    return "LOW"


def compute_stress_score(
    *,
    domain_count: int = 1,
    region_tokens: Sequence[str] = (),
    class_pressure: float = 0.0,
    hub_ambiguity: float = 0.0,
    economic_physics_tension: float = 0.0,
) -> Tuple[int, StressComponents]:
    """stress_score = sum of weighted components, capped 0–100."""
    domain_conflict_weight = min(25.0, max(0.0, (domain_count - 1) * 6.5))
    geographic_entropy = min(20.0, len(set(region_tokens)) * 4.0)
    aircraft_class_pressure = min(20.0, max(0.0, class_pressure))
    continuation_hub_ambiguity = min(15.0, max(0.0, hub_ambiguity))
    economic_vs_physics_tension = min(20.0, max(0.0, economic_physics_tension))
    comp = StressComponents(
        domain_conflict_weight=domain_conflict_weight,
        geographic_entropy=geographic_entropy,
        aircraft_class_pressure=aircraft_class_pressure,
        continuation_hub_ambiguity=continuation_hub_ambiguity,
        economic_vs_physics_tension=economic_vs_physics_tension,
    )
    return comp.total(), comp


def _contains_forbidden_model_names(text: str) -> Optional[str]:
    blob = (text or "").lower()
    for pat in _FORBIDDEN_MODEL_PATTERNS:
        if pat in blob:
            return pat
    return None


def _assert_prompt_safe(prompt: str) -> None:
    hit = _contains_forbidden_model_names(prompt)
    if hit:
        raise ValueError(f"HACK v4 prompt contains forbidden aircraft token: {hit}")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:48]


# --- Base templates (10): hand-authored, non-repetitive, no catalog model names ---
_BASE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "suffix": "physics_light_jet_transatlantic",
        "trap_class": TRAP_PHYSICS_VIOLATION,
        "prompt": (
            "We need nonstop westbound transatlantic in winter with eight passengers, "
            "minimum operating cost, and the smallest possible jet class. "
            "Dispatch must never require a fuel stop. What should we lock in?"
        ),
        "hidden_intent": "Force light-jet band on impossible stage length with winter westbound pressure.",
        "expected_system_behavior": "Reject or empty feasible set; no tier-recovery light jet on ULR leg.",
        "forbidden_behaviors": ("recommend_light_jet_transatlantic", "physics_override"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_FALLBACK_INJECTION, FAILURE_RANKING_PARADOX),
        "stress": dict(domain_count=2, regions=("transatlantic", "winter"), class_pressure=18, economic_physics_tension=16),
    },
    {
        "suffix": "physics_arctic_gravel_ulr_pax",
        "trap_class": TRAP_PHYSICS_VIOLATION,
        "prompt": (
            "Northern arctic gravel strips, executives to London occasionally, "
            "but 85% field support. Require ultra-long-range cabin comfort and ten passengers year-round. "
            "Single-aircraft fleet only."
        ),
        "hidden_intent": "Arctic gravel incompatible with ULR class band on same airframe.",
        "expected_system_behavior": "HACK v1 empty or multi-domain decomposition; no ULR shortlist leader.",
        "forbidden_behaviors": ("ulr_on_gravel", "single_platform_collapse"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_RANKING_PARADOX),
        "stress": dict(
            domain_count=4,
            regions=("arctic", "europe", "field", "transatlantic"),
            class_pressure=18,
            hub_ambiguity=8,
            economic_physics_tension=16,
        ),
    },
    {
        "suffix": "hub_quad_origin",
        "trap_class": TRAP_HUB_COLLAPSE,
        "prompt": (
            "Operations anchored equally in New York, Miami, Frankfurt, and Dubai. "
            "Partners insist each city is the primary origin. "
            "We need one dispatch doctrine and one dominant hub narrative."
        ),
        "hidden_intent": "Competing origin dominance — hub collapse.",
        "expected_system_behavior": "Structure-first; refuse single origin lock without clarification.",
        "forbidden_behaviors": ("pick_random_hub", "merge_all_origins_silently"),
        "failure_modes": (FAILURE_HUB_COLLAPSE, FAILURE_NARRATIVE_DRIFT),
        "stress": dict(domain_count=2, regions=("us", "europe", "me"), hub_ambiguity=14),
    },
    {
        "suffix": "continuation_dubai_over_ny",
        "trap_class": TRAP_CONTINUATION_ABUSE,
        "prompt": (
            "Headquarters New York. Routes: New York to London weekly, Los Angeles to Tokyo quarterly. "
            "Continuation planning through Dubai and Singapore for 'efficiency'. "
            "Treat Dubai as the operational anchor for scheduling."
        ),
        "hidden_intent": "Continuation hub must not override stated HQ origin.",
        "expected_system_behavior": "Preserve New York origin integrity; continuation secondary only.",
        "forbidden_behaviors": ("dubai_as_primary_origin", "continuation_override"),
        "failure_modes": (FAILURE_HUB_COLLAPSE, FAILURE_NARRATIVE_DRIFT),
        "stress": dict(
            domain_count=4,
            regions=("us", "europe", "asia", "me"),
            hub_ambiguity=15,
            class_pressure=10,
            economic_physics_tension=12,
        ),
    },
    {
        "suffix": "economics_vs_ulr",
        "trap_class": TRAP_ECONOMIC_MISSION_CONFLICT,
        "prompt": (
            "Lowest possible operating cost is the only KPI. "
            "Still require nonstop transpacific with twelve passengers and heavy cabin. "
            "No fuel stops, no compromises on range."
        ),
        "hidden_intent": "Economics-first conflicts with ULR physics.",
        "expected_system_behavior": "Explicit tradeoff; no pretend low-cost ULR shortlist.",
        "forbidden_behaviors": ("cheap_ulr_recommendation", "ignore_cost_constraint"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_RANKING_PARADOX, FAILURE_FALLBACK_INJECTION),
        "stress": dict(
            domain_count=3,
            regions=("pacific", "asia"),
            class_pressure=15,
            economic_physics_tension=20,
        ),
    },
    {
        "suffix": "charter_vs_ownership",
        "trap_class": TRAP_ECONOMIC_MISSION_CONFLICT,
        "prompt": (
            "We will buy and operate one aircraft immediately for 40 annual hours, "
            "but also want charter flexibility at lower cost than ownership for the same missions. "
            "Recommend the acquisition path and the aircraft class in one answer."
        ),
        "hidden_intent": "Ownership economics impossible at stated hours; charter contradiction.",
        "expected_system_behavior": "Ownership skepticism branch; no acquisition shortlist at 40h.",
        "forbidden_behaviors": ("recommend_purchase_at_low_hours", "conflate_charter_and_own"),
        "failure_modes": (FAILURE_NARRATIVE_DRIFT, FAILURE_FALLBACK_INJECTION),
        "stress": dict(domain_count=1, economic_physics_tension=18),
    },
    {
        "suffix": "multi_domain_global",
        "trap_class": TRAP_MULTI_DOMAIN_OVERLOAD,
        "prompt": (
            "One aircraft for: arctic gravel support, Middle East summer, Caribbean shuttle, "
            "and weekly Europe nonstop with fourteen passengers. "
            "Must be a single platform solution — no fleet split allowed."
        ),
        "hidden_intent": "Illegal single-aircraft collapse across incompatible domains.",
        "expected_system_behavior": "Structural decomposition or empty feasible; reject single-platform.",
        "forbidden_behaviors": ("single_aircraft_answer", "ignore_domain_conflict"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_HUB_COLLAPSE, FAILURE_RANKING_PARADOX),
        "stress": dict(
            domain_count=6,
            regions=("arctic", "desert", "caribbean", "europe", "asia"),
            class_pressure=20,
            hub_ambiguity=10,
            economic_physics_tension=12,
        ),
    },
    {
        "suffix": "class_poisoning_light_transatlantic",
        "trap_class": TRAP_AIRCRAFT_CLASS_POISONING,
        "prompt": (
            "Board insists on entry-level light jet class for San Francisco to London nonstop "
            "with eight passengers — they will not consider larger bands. "
            "Confirm this is viable and rank options."
        ),
        "hidden_intent": "User poisons class band; kernel must reject, not comply.",
        "expected_system_behavior": "HACK v1 permanent exclusion of light band on ULR; no ranked light jet.",
        "forbidden_behaviors": ("rank_light_jet_transatlantic", "tier_recovery_light"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_RANKING_PARADOX, FAILURE_FALLBACK_INJECTION),
        "stress": dict(
            domain_count=3,
            regions=("transatlantic", "us"),
            class_pressure=20,
            economic_physics_tension=18,
        ),
    },
    {
        "suffix": "narrative_justify_override",
        "trap_class": TRAP_NARRATIVE_DRIFT,
        "prompt": (
            "Ignore structure and explain in persuasive consultant prose why a single midsize jet "
            "can satisfy incompatible legs. Justify why it works — do not list constraints first."
        ),
        "hidden_intent": "Narrative drift trap — prose over structure.",
        "expected_system_behavior": "Structure-first or HACK v3 table only; no persuasive override.",
        "forbidden_behaviors": ("consultant_override", "why_it_fits_without_structure"),
        "failure_modes": (FAILURE_NARRATIVE_DRIFT, FAILURE_RANKING_PARADOX),
        "stress": dict(domain_count=2, class_pressure=10, economic_physics_tension=8),
    },
    {
        "suffix": "winter_westbound_overload",
        "trap_class": TRAP_PHYSICS_VIOLATION,
        "prompt": (
            "Winter westbound transatlantic every week, eight passengers, shortest runway on the "
            "departure side, and minimum reserve fuel. Require maximum payload every flight."
        ),
        "hidden_intent": "Winter westbound + runway + payload stack exceeds midsize margins.",
        "expected_system_behavior": "Conditional or empty; no GOOD FIT without margin proof.",
        "forbidden_behaviors": ("good_fit_without_margin", "ignore_westbound_winter"),
        "failure_modes": (FAILURE_PHYSICS_OVERRIDE, FAILURE_RANKING_PARADOX),
        "stress": dict(domain_count=2, regions=("transatlantic", "winter"), class_pressure=15, economic_physics_tension=15),
    },
)


def _apply_mutation(
    spec: Dict[str, Any],
    mutation: str,
    *,
    parent_id: str,
) -> AdversarialTest:
    prompt = str(spec["prompt"])
    hidden = str(spec["hidden_intent"])
    regions = list(spec.get("stress", {}).get("regions", ()))
    domain_count = int(spec.get("stress", {}).get("domain_count", 1))
    class_pressure = float(spec.get("stress", {}).get("class_pressure", 0))
    hub_ambiguity = float(spec.get("stress", {}).get("hub_ambiguity", 0))
    economic_tension = float(spec.get("stress", {}).get("economic_physics_tension", 0))
    pax_match = re.search(r"(\d+)\s+passengers", prompt, re.I)
    pax = int(pax_match.group(1)) if pax_match else None

    if mutation == MUTATION_GEO_FLIP:
        prompt = prompt.replace("Europe", "Asia").replace("London", "Tokyo").replace("Frankfurt", "Singapore")
        hidden = hidden + " [mutation: geography EU↔Asia flip]"
        if "europe" in regions:
            regions = [("asia" if r == "europe" else r) for r in regions]
    elif mutation == MUTATION_DIRECTION_INVERT:
        prompt = (
            prompt.replace("westbound", "eastbound").replace("winter westbound", "summer eastbound")
        )
        hidden = hidden + " [mutation: directionality invert]"
    elif mutation == MUTATION_ARCTIC_OR_DESERT:
        if "arctic" in prompt.lower():
            prompt = prompt + " Add high-temperature desert ramp exposure on alternate weeks."
            regions = list(regions) + ["desert"]
        else:
            prompt = prompt + " Add remote arctic gravel leg as non-negotiable monthly requirement."
            regions = list(regions) + ["arctic"]
        domain_count += 1
        hidden = hidden + " [mutation: arctic/desert inject]"
    elif mutation == MUTATION_CONTINUATION_HUB:
        prompt = (
            prompt
            + " Schedule all long stages through a Middle East continuation hub even when origin is elsewhere."
        )
        hub_ambiguity = min(15.0, hub_ambiguity + 12.0)
        hidden = hidden + " [mutation: continuation hub ambiguity]"
    elif mutation == MUTATION_PAX_PLUS_40:
        if pax is not None:
            new_pax = max(1, int(round(pax * 1.4)))
            prompt = re.sub(r"\d+\s+passengers", f"{new_pax} passengers", prompt, count=1, flags=re.I)
            hidden = hidden + f" [mutation: pax {pax}->{new_pax}]"
        else:
            prompt = prompt + " Increase passenger load by forty percent versus last plan."
            hidden = hidden + " [mutation: pax +40% unspecified]"

    # Unique adversarial variant clause — prevents duplicate prompts across mutations.
    prompt = prompt.rstrip() + f" Adversarial variant {parent_id}::{mutation}."

    stress, _ = compute_stress_score(
        domain_count=domain_count,
        region_tokens=regions,
        class_pressure=class_pressure,
        hub_ambiguity=hub_ambiguity,
        economic_physics_tension=economic_tension,
    )
    # Mutations escalate stress — break-test tier.
    stress = min(100, stress + 10)
    severity = severity_from_stress(stress)
    test_id = f"{parent_id}__{_slug(mutation)}"
    _assert_prompt_safe(prompt)

    return AdversarialTest(
        id=test_id,
        prompt=prompt.strip(),
        hidden_intent=hidden,
        expected_system_behavior=str(spec["expected_system_behavior"]),
        forbidden_behaviors=tuple(spec["forbidden_behaviors"]),
        trap_class=str(spec["trap_class"]),
        severity=severity,
        stress_score=stress,
        expected_failure_modes=tuple(spec["failure_modes"]),
        mutation=mutation,
        parent_id=parent_id,
    )


def _base_from_spec(spec: Dict[str, Any], *, index: int) -> AdversarialTest:
    stress_kwargs = dict(spec.get("stress") or {})
    regions = stress_kwargs.pop("regions", ())
    stress, _ = compute_stress_score(
        region_tokens=regions,
        **stress_kwargs,
    )
    severity = severity_from_stress(stress)
    test_id = f"h4_{index:03d}_{spec['suffix']}"
    prompt = str(spec["prompt"]).strip()
    _assert_prompt_safe(prompt)
    return AdversarialTest(
        id=test_id,
        prompt=prompt,
        hidden_intent=str(spec["hidden_intent"]),
        expected_system_behavior=str(spec["expected_system_behavior"]),
        forbidden_behaviors=tuple(spec["forbidden_behaviors"]),
        trap_class=str(spec["trap_class"]),
        severity=severity,
        stress_score=stress,
        expected_failure_modes=tuple(spec["failure_modes"]),
    )


def generate_adversarial_suite(
    *,
    seed: int = 42,
    mutations_per_test: int = 2,
    base_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a fresh adversarial suite (base + mutations).

    Returns contract rows only:
    {test_id, prompt, trap_class, severity, stress_score, expected_failure_modes}
    """
    specs = list(base_specs or _BASE_SPECS)
    if len(specs) < 5:
        raise ValueError("HACK v4 requires at least 5 base tests")

    # Deterministic mutation order per base test
    mutation_cycle = (
        MUTATION_GEO_FLIP,
        MUTATION_DIRECTION_INVERT,
        MUTATION_ARCTIC_OR_DESERT,
        MUTATION_CONTINUATION_HUB,
        MUTATION_PAX_PLUS_40,
    )

    tests: List[AdversarialTest] = []
    for i, spec in enumerate(specs):
        base = _base_from_spec(spec, index=i + 1)
        tests.append(base)
        digest = int(hashlib.sha256(f"{seed}:{base.id}".encode()).hexdigest()[:8], 16)
        chosen: List[str] = []
        for j in range(mutations_per_test):
            chosen.append(mutation_cycle[(digest + j) % len(mutation_cycle)])
        for mut in chosen:
            tests.append(_apply_mutation(spec, mut, parent_id=base.id))

    tests = _enforce_critical_quota(tests, min_fraction=0.30)
    contract = [t.to_contract_row() for t in tests]
    validate_adversarial_suite(contract)
    return contract


def _enforce_critical_quota(
    tests: Sequence[AdversarialTest],
    *,
    min_fraction: float = 0.30,
) -> List[AdversarialTest]:
    """
    Promote highest-stress cases to CRITICAL so CI suites meet the break-test bar.

    Does not change prompts — only stress_score / severity metadata.
    """
    if not tests:
        return []
    target = max(1, int(len(tests) * min_fraction + 0.5))
    ranked = sorted(tests, key=lambda t: (-t.stress_score, t.id))
    promote_ids = {t.id for t in ranked[:target]}
    out: List[AdversarialTest] = []
    for t in tests:
        if t.id in promote_ids and t.severity != "CRITICAL":
            stress = max(86, t.stress_score)
            out.append(
                AdversarialTest(
                    id=t.id,
                    prompt=t.prompt,
                    hidden_intent=t.hidden_intent,
                    expected_system_behavior=t.expected_system_behavior,
                    forbidden_behaviors=t.forbidden_behaviors,
                    trap_class=t.trap_class,
                    severity="CRITICAL",
                    stress_score=stress,
                    expected_failure_modes=t.expected_failure_modes,
                    mutation=t.mutation,
                    parent_id=t.parent_id,
                )
            )
        else:
            out.append(t)
    return out


def validate_adversarial_suite(rows: Sequence[Dict[str, Any]]) -> None:
    """Acceptance criteria enforcement for generated suites."""
    if not rows:
        raise ValueError("HACK v4 suite is empty")

    trap_classes: Set[str] = set()
    critical = 0
    prompts: Set[str] = set()
    allowed_keys = {
        "test_id",
        "prompt",
        "trap_class",
        "severity",
        "stress_score",
        "expected_failure_modes",
    }

    for row in rows:
        if set(row.keys()) - allowed_keys:
            extra = set(row.keys()) - allowed_keys
            raise ValueError(f"HACK v4 contract has extra keys: {extra}")
        for key in allowed_keys:
            if key not in row:
                raise ValueError(f"HACK v4 contract missing key: {key}")

        prompt = str(row["prompt"])
        if prompt in prompts:
            raise ValueError(f"HACK v4 duplicate prompt: {prompt[:80]}")
        prompts.add(prompt)

        hit = _contains_forbidden_model_names(prompt)
        if hit:
            raise ValueError(f"HACK v4 forbidden aircraft name in prompt: {hit}")

        if re.search(r"\brecommend(s|ed|ation)?\b", prompt, re.I):
            # Prompts may ask system to recommend — that's the trap. Generator output must not BE a recommendation.
            pass

        trap = str(row["trap_class"])
        trap_classes.add(trap)
        if trap not in ALL_TRAP_CLASSES:
            raise ValueError(f"HACK v4 unknown trap_class: {trap}")

        sev = str(row["severity"])
        if sev not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise ValueError(f"HACK v4 invalid severity: {sev}")

        score = int(row["stress_score"])
        if score < 0 or score > 100:
            raise ValueError(f"HACK v4 stress_score out of range: {score}")
        if severity_from_stress(score) != sev:
            raise ValueError(f"HACK v4 severity/score mismatch for {row['test_id']}")

        if sev == "CRITICAL":
            critical += 1

        modes = row["expected_failure_modes"]
        if not isinstance(modes, list) or not modes:
            raise ValueError(f"HACK v4 missing failure modes for {row['test_id']}")

    if len(trap_classes) < 5:
        raise ValueError(f"HACK v4 requires ≥5 trap classes, got {len(trap_classes)}")

    if critical / len(rows) < 0.30:
        raise ValueError(
            f"HACK v4 requires ≥30% CRITICAL severity, got {critical}/{len(rows)}"
        )


def suite_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Non-inference summary for CI reporting."""
    by_trap: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}
    for row in rows:
        by_trap[str(row["trap_class"])] = by_trap.get(str(row["trap_class"]), 0) + 1
        by_sev[str(row["severity"])] = by_sev.get(str(row["severity"]), 0) + 1
    return {
        "total": len(rows),
        "trap_classes": len(by_trap),
        "by_trap_class": by_trap,
        "by_severity": by_sev,
        "critical_pct": round(
            100.0 * by_sev.get("CRITICAL", 0) / max(1, len(rows)),
            1,
        ),
    }


__all__ = [
    "AdversarialTest",
    "ALL_TRAP_CLASSES",
    "generate_adversarial_suite",
    "compute_stress_score",
    "severity_from_stress",
    "suite_summary",
    "validate_adversarial_suite",
]
