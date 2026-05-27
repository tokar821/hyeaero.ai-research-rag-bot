"""Final response cleanup pass — cleanResponseText."""

from services.consultant.response_cleanup import cleanResponseText, clean_response_text


def test_dedupes_duplicate_sentences():
    raw = (
        "Challenger 350 fits this leg well. "
        "Challenger 350 fits this leg well. "
        "Gulfstream G280 adds more cabin if you want a step up."
    )
    out = cleanResponseText(raw)
    assert out.lower().count("challenger 350 fits this leg well") == 1
    assert "g280" in out.lower()


def test_dedupes_repeated_operational_warning():
    raw = (
        "Challenger 350 is the lead. "
        "Watch westbound headwinds on Pacific legs in winter. "
        "Plan for westbound headwinds on Pacific legs in winter."
    )
    out = clean_response_text(raw)
    assert out.lower().count("westbound headwind") == 1


def test_dedupes_duplicate_bullets_and_fixes_malformed_markers():
    raw = (
        "- Challenger 350 — strong cabin and range for the mission\n"
        "- - Challenger 350 — strong cabin and range for the mission\n"
        "- Gulfstream G280 — quieter cabin, similar stage length\n"
        "- \n"
        "- Gulfstream G280 — quieter cabin, similar stage length\n"
    )
    out = cleanResponseText(raw)
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("-")]
    assert len(lines) == 2
    assert "- -" not in out


def test_dedupes_repeated_clause_across_sentences():
    raw = (
        "For Los Angeles to Miami nonstop, Challenger 350 is the practical default. "
        "On Los Angeles to Miami nonstop, Challenger 350 remains the practical default."
    )
    out = clean_response_text(raw)
    assert out.lower().count("los angeles to miami") == 1


def test_preserves_distinct_aircraft_bullets():
    raw = (
        "- Challenger 350 — best balance of cabin and operating cost\n"
        "- Gulfstream G280 — more cabin volume with similar range\n"
        "- Embraer Praetor 600 — strong value with modern avionics\n"
    )
    out = cleanResponseText(raw)
    assert "challenger 350" in out.lower()
    assert "g280" in out.lower()
    assert "praetor 600" in out.lower()
    assert len([ln for ln in out.splitlines() if ln.strip().startswith("-")]) == 3


def test_strips_empty_section_headers():
    raw = (
        "Operational posture\n\n"
        "Distance Considerations:\n\n"
        "Aircraft Options:\n"
        "- Dual-aircraft portfolio for incompatible bands."
    )
    out = clean_response_text(raw)
    assert "Distance Considerations:" not in out
    assert "Aircraft Options:" in out
