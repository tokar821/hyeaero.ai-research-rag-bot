"""Structural representation proofs — engine layer before ranking."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.mission_extractor import extract_mission
from services.mission.pre_ranking_representation import apply_pre_ranking_representation
from services.mission.structural_representation import evaluate_structural_representation
from services.mission.field_access_routes import infer_field_access_spokes


def test_field_access_spokes_drilling_sites():
    q = (
        "We transport high-value equipment between Calgary, Houston, remote drilling sites, and Madrid."
    )
    profile = extract_mission(q)
    labels = profile.route_labels()
    assert any("Remote Drilling" in lbl for lbl in labels)
    spokes = infer_field_access_spokes(q, profile)
    assert spokes == [] or any("Remote Drilling" in r.label() for r in spokes)


def test_founder_company_asymmetry_proof():
    q = (
        "We are based in NYC. Daily flights are NYC–Chicago–SF with 4 executives. "
        "But the founder also flies nonstop to Abu Dhabi and sometimes Singapore. "
        "The rest of the company never leaves North America."
    )
    profile = extract_mission(q)
    pkt = MissionUnderstandingPacket()
    proof = evaluate_structural_representation(q, profile, pkt)
    assert proof.required
    assert "founder_company" in proof.proof_kind or "founder_company_asymmetry" in proof.triggers


def test_pre_ranking_populates_routes_for_ski_asia():
    q = (
        "We previously owned a large long-range jet but dispatch reliability into Aspen, "
        "Jackson Hole, and European winter airports caused repeated failures. "
        "Now we need both ski access and Asia capability."
    )
    profile = extract_mission(q)
    mission = MissionState()
    pkt = MissionUnderstandingPacket()
    profile, mission, pkt = apply_pre_ranking_representation(q, profile, mission, pkt, {})
    assert len(profile.routes) >= 2
    assert mission.routes
