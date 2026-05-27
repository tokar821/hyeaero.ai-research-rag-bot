"""Template suppression tests."""

from services.consultant.template_suppression import fallback_contamination_score, suppress_templates


def test_removes_assuming_6_8_passengers_block():
    raw = (
        "Strong opening insight.\n\n"
        "Assuming 6–8 passengers and typical business-use constraints (no extreme hot/high), "
        "here are a few realistic fits:\n"
        "- Challenger 350: balanced.\n\n"
        "Consultant Insight: buyer's remorse is about dispatch."
    )
    out = suppress_templates(raw)
    assert "Assuming 6–8 passengers" not in out.text
    assert "Consultant Insight" not in out.text
    assert "Strong opening insight" in out.text
    assert out.removed_blocks >= 1


def test_dedupes_duplicate_paragraphs():
    para = "The Praetor 600 offers excellent range for transcontinental missions."
    raw = f"{para}\n\n{para}"
    out = suppress_templates(raw)
    assert out.duplicate_paragraphs_removed >= 1
    assert out.text.count("Praetor 600") == 1


def test_contamination_score_high_on_stock_fallback():
    raw = "Assuming 6-8 passengers here are a few realistic fits Consultant Insight: range"
    assert fallback_contamination_score(raw) >= 0.4


def test_repeated_aircraft_bullet_deduped():
    raw = "- Challenger 350: great\n- Challenger 350: great again\n- Praetor 600: alt"
    out = suppress_templates(raw)
    assert out.text.lower().count("challenger 350") == 1
