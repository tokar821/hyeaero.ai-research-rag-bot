from rag.intent.aircraft_matching_engine import run_aircraft_matching_engine, validate_ulr_peer_list


def test_ulr_anchor_returns_peer_shortlist():
    out = run_aircraft_matching_engine("Something like a Gulfstream G650 but cheaper", history=None)
    assert out["hard_fail"] is False
    names = " ".join(out["aircraft_candidates"]).lower()
    assert "challenger 650" in names
    assert "falcon 7x" in names
    assert "global 5000" in names
    assert "eclipse" not in names
    assert "cj2" not in names


def test_proposed_cj2_hard_fail():
    out = run_aircraft_matching_engine(
        "G650 alternatives",
        proposed_candidates=["Citation CJ2", "Challenger 650"],
    )
    assert out["hard_fail"] is True
    assert out["aircraft_candidates"] == []


def test_validate_ulr_peers_rejects_g500():
    out = validate_ulr_peer_list(["Gulfstream G500", "Falcon 7X"])
    assert out["hard_fail"] is True


def test_validate_ulr_peers_accepts_allowlist():
    out = validate_ulr_peer_list(["Dassault Falcon 7X", "Bombardier Global 6000"])
    assert out["hard_fail"] is False
    assert len(out["aircraft_candidates"]) == 2


def test_history_cj2_does_not_false_fail_current_g650_query():
    hist = [{"role": "user", "content": "I used to fly a Citation CJ2"}]
    out = run_aircraft_matching_engine("Show me interiors like a Global 7500 but cheaper", history=hist)
    assert out["hard_fail"] is False
    assert len(out["aircraft_candidates"]) >= 4
