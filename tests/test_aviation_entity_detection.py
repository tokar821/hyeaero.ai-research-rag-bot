from rag.entities import detect_aviation_entities, detect_aviation_entities_json


def test_models_never_serial_challenger_boeing_airbus():
    q = "Pricing for Challenger 601-3ER vs Boeing 737-800 and Airbus A320-200"
    out = detect_aviation_entities(q, None)
    assert "737-800" in " ".join(out["aircraft_models"]).lower() or any(
        "737-800" in m for m in out["aircraft_models"]
    )
    assert any("601" in m for m in out["aircraft_models"])
    assert any("320-200" in m.replace(" ", "").lower() for m in out["aircraft_models"])
    assert "601-3ER" not in out["serial_numbers"]
    assert "737-800" not in out["serial_numbers"]
    assert "320-200" not in " ".join(out["serial_numbers"])


def test_citation_serial_not_model_525():
    q = "MSN 525A-0349 for N535JF"
    out = detect_aviation_entities(q, None)
    assert any("525A-0349" in s or "525A-0349" == s for s in out["serial_numbers"])
    assert "N535JF" in out["registrations"]


def test_json_wrapper_matches_keys():
    j = detect_aviation_entities_json("Tell me about N1AB and Falcon 8X", None)
    assert set(j.keys()) == {"aircraft_models", "registrations", "serial_numbers"}
    assert "N1AB" in j["registrations"]


def test_tail_and_serial_separate():
    out = detect_aviation_entities("Tail N277G serial 560-5252", None)
    assert "N277G" in out["registrations"]
    assert any("560-5252" in s for s in out["serial_numbers"])


def test_history_merge():
    hist = [{"role": "user", "content": "still on N999XX?"}]
    out = detect_aviation_entities("who operates it", hist)
    assert "N999XX" in out["registrations"]
