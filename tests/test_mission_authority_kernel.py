"""Hard authority chain — kernel law, segment binding, LLM rejection."""

from __future__ import annotations

from services.consultant.broker_advisory_layer import format_broker_advisory_response
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.recommendation_authority import reconcile_answer_with_pipeline
from services.mission.mission_graph import SegmentKind, build_mission_graph
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    attach_mission_understanding,
    bands_are_incompatible,
    needs_portfolio_synthesis,
)
from services.mission.mission_authority_kernel import (
    KERNEL_BLOCK_MARKER,
    build_mission_authority_kernel,
    enforce_kernel_authority,
    filter_recommendations_by_kernel,
    project_kernel_advisory,
)
from services.mission.structural_decomposition import needs_structural_decomposition


def _rec(model: str, category: str = "ultra-long") -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category=category,
        total_score=0.8,
        confidence=0.7,
        rank=1,
        fit="Strong fit",
        avoid=False,
    )


def _profile(mission: MissionState):
    from services.mission.models import MissionProfile, Route

    p = MissionProfile(passengers=mission.passenger_count)
    for lbl in mission.routes or []:
        r = Route.from_label(lbl)
        if r:
            p.routes.append(r)
    return p


def test_aspen_dubai_forces_structural_decomposition():
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic ultra-long-range executive band",
            "Mountain field-flexible short-strip band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
        operational_synthesis="Aspen field performance and Dubai nonstop cannot share one platform.",
    )
    proof = needs_structural_decomposition(pkt)
    assert proof.required
    assert needs_portfolio_synthesis("", pkt)


def test_domestic_ulr_continuation_not_structural_fleet():
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    assert not bands_are_incompatible(pkt.fallback_operational_band)
    assert not needs_structural_decomposition(pkt).required


def test_ulr_continuation_local_to_segment():
    mission = MissionState(
        routes=["Dallas -> New York", "New York -> London", "London -> Dubai"],
        passenger_count=8,
    )
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Transatlantic super-mid / heavy-cabin executive band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    graph = build_mission_graph(pkt, _profile(mission), mission)
    cont = next((s for s in graph.segments if s.kind == SegmentKind.ULR_CONTINUATION), None)
    assert cont is not None
    assert "continuation" in (cont.operational_band or "").lower()
    dom = next((s for s in graph.segments if s.kind == SegmentKind.DOMESTIC_EXECUTIVE), None)
    if dom:
        assert "Middle East" not in (dom.operational_band or "")


def test_industrial_london_blocks_unauthorized_ulr_without_segment():
    mission = MissionState(routes=["TEB -> London"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Transatlantic executive — super-mid band unless ULR nonstop is mandatory.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
        inferred_constraints={"planning_band_ceiling": "super_midsize"},
    )
    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        recommendations=[_rec("Global 7500", "ultra-long")],
    )
    filtered = filter_recommendations_by_kernel([_rec("Global 7500")], kernel)
    assert not filtered or "Global 7500" not in [r.model for r in filtered]


def test_caribbean_riyadh_incompatible_bands():
    pkt = MissionUnderstandingPacket(
        fallback_operational_band=[
            "Caribbean executive regional jet band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    assert needs_structural_decomposition(pkt).required


def test_exactly_one_synthesis_block():
    mission = MissionState(routes=["NYC -> London"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Executive transatlantic.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
    )
    kernel = build_mission_authority_kernel(mission, pkt)
    text = project_kernel_advisory(kernel, [])
    assert text.count(KERNEL_BLOCK_MARKER) == 1


def test_llm_merge_rejected_when_unauthorized_aircraft():
    mission = MissionState(routes=["TEB -> Aspen", "TEB -> Dubai"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Mountain and ULR domains conflict.",
        fallback_operational_band=[
            "Mountain field-flexible short-strip band",
            "Middle East ULR continuation band",
        ],
        inferred_constraints={"incompatible_mission_bands": True},
    )
    du = attach_mission_understanding({}, pkt)
    du["fleet_composition_plan"] = {
        "multi_aircraft_required": True,
        "single_aircraft_structurally_invalid": True,
        "doctrine": "Structural portfolio required.",
        "assignments": [
            {
                "segment_label": "Mountain field access",
                "primary_model": "Pilatus PC-24",
                "fit_verdict": "VIABLE WITH COMPROMISES",
            },
            {
                "segment_label": "Ulr continuation",
                "primary_model": "Gulfstream G650ER",
                "fit_verdict": "VIABLE WITH COMPROMISES",
            },
        ],
    }
    kernel = build_mission_authority_kernel(mission, pkt, data_used=du, recommendations=[])
    llm = (
        "Aircraft Options:\n\n* Global 7500 — best overall jet for everything.\n\n"
        "Verdict:\n\n* PRIMARY: Global 7500"
    )
    enforced, report = enforce_kernel_authority(llm, kernel, [])
    assert report.reject_merge
    assert "Global 7500" not in enforced or "Per-segment" in enforced


def test_reconcile_rejects_llm_flagship_injection():
    mission = MissionState(routes=["TEB -> London"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Super-mid transatlantic band.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
        inferred_constraints={"planning_band_ceiling": "super_midsize"},
        recommend_aircraft=True,
    )
    du = attach_mission_understanding({}, pkt)
    recs = [_rec("Gulfstream G280", "super-midsize")]
    build_mission_authority_kernel(mission, pkt, recommendations=recs, data_used=du)
    llm = "Citation CJ4 is perfect. Gulfstream G650ER too. Global 7500 wins."
    final, regen = reconcile_answer_with_pipeline(
        llm,
        mission=mission,
        recommendations=recs,
        data_used=du,
        query="advise",
    )
    assert regen
    assert KERNEL_BLOCK_MARKER in final


def test_ranked_path_uses_kernel_not_duplicate_synthesis():
    mission = MissionState(routes=["NYC -> London"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        operational_synthesis="Transatlantic executive read.",
        fallback_operational_band=["Transatlantic super-mid / heavy-cabin executive band"],
        recommend_aircraft=True,
    )
    du = attach_mission_understanding({}, pkt)
    body = format_broker_advisory_response(
        mission,
        [_rec("Gulfstream G280", "super-midsize")],
        data_used=du,
    )
    assert body.count(KERNEL_BLOCK_MARKER) == 1
    assert du.get("mission_authority_bound") == 1


def test_no_aircraft_without_segment_justification_when_structural():
    mission = MissionState(routes=["Aspen -> TEB", "TEB -> Dubai"], passenger_count=8)
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"incompatible_mission_bands": True},
        fallback_operational_band=[
            "Mountain field-flexible short-strip band",
            "Middle East ULR continuation band",
        ],
    )
    du = attach_mission_understanding({}, pkt)
    du["fleet_composition_plan"] = {
        "multi_aircraft_required": True,
        "single_aircraft_structurally_invalid": True,
        "assignments": [
            {"segment_label": "Mountain", "primary_model": "PC-24", "fit_verdict": "VIABLE"},
        ],
    }
    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        data_used=du,
        recommendations=[_rec("Global 7500")],
    )
    text = project_kernel_advisory(kernel, [_rec("Global 7500")])
    assert "Per-segment" in text
    assert "Global 7500" not in text or "invalid" in text.lower()


def test_miami_sao_paulo_madrid_corridor_not_hub_collapse():
    from services.mission.mission_corridor_routes import extract_between_corridor
    from services.mission.mission_extractor import extract_mission

    q = (
        "We operate between Miami, São Paulo, Madrid, and small Caribbean islands. "
        "Some airports are short runway, others are long-haul international hubs."
    )
    extractions = extract_between_corridor(q)
    labels = {e.route.label() for e in extractions}
    assert any("Miami" in lbl and "São Paulo" in lbl or "Sao Paulo" in lbl for lbl in labels)
    assert any("Madrid" in lbl for lbl in labels)
    profile = extract_mission(q)
    assert len(profile.routes) >= 2
    assert not (
        len(profile.routes) == 1 and profile.routes[0].destination == "Caribbean"
    )


def test_multi_corridor_segment_bound_suppresses_global_shortlist():
    from services.mission.models import Route

    mission = MissionState(
        routes=[
            "Miami -> Sao Paulo",
            "Miami -> Madrid",
            "Miami -> Caribbean",
        ],
        passenger_count=8,
    )
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"dual_use_or_multi_leg": True},
    )
    prof = _profile(mission)
    prof.routes = [
        Route.from_label(lbl)
        for lbl in mission.routes
        if Route.from_label(lbl)
    ]
    du = attach_mission_understanding({}, pkt)
    kernel = build_mission_authority_kernel(
        mission,
        pkt,
        prof,
        recommendations=[_rec("Citation Latitude", "midsize"), _rec("CJ4", "light")],
        data_used=du,
    )
    assert kernel.segment_bound_presentation or kernel.structural_decomposition
    text = project_kernel_advisory(
        kernel, [_rec("Citation Latitude", "midsize"), _rec("CJ4", "light")]
    )
    assert "Per-segment" in text
    assert "Aircraft Options" not in text
    if kernel.segment_bound_presentation:
        assert "Global aircraft shortlist suppressed" in text
