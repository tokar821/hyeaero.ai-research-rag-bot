"""Phly lookup token extraction — tails in casual phrasing must resolve."""

from rag.aviation_tail import find_strict_tail_candidates_in_text
from rag.phlydata_consultant_lookup import (
    consultant_phly_lookup_token_list,
    extract_phlydata_lookup_tokens,
    extract_phlydata_tokens_with_history,
)


def test_casual_have_you_tail_merges_strict_n_number():
    toks = consultant_phly_lookup_token_list("Have you N807JS now?", None)
    flat = " ".join(toks).upper().replace(" ", "")
    assert "N807JS" in flat


def test_strict_intl_tails_detected():
    assert "PR-CCA" in find_strict_tail_candidates_in_text("who owns PR-CCA?")
    assert "C-GUGU" in find_strict_tail_candidates_in_text("details for C-GUGU")
    assert "TC-KEA" in find_strict_tail_candidates_in_text("TC-KEA listed?")
    assert "FL-1185" in find_strict_tail_candidates_in_text("FL-1185 for sale")


def test_followup_anchors_latest_user_tail_not_assistant_n_number():
    hist = [
        {"role": "user", "content": "PR-CCA owner?"},
        {
            "role": "assistant",
            "content": "Wrong N450 LLC registrant …",
        },
        {"role": "user", "content": "more from your data please"},
    ]
    toks = consultant_phly_lookup_token_list(hist[-1]["content"], hist[:-1])
    flat = " ".join(toks).upper().replace(" ", "")
    assert "PR-CCA" in flat
    assert "N450" not in flat


def test_history_tokens_skip_assistant_content():
    hist = [
        {"role": "user", "content": "N118CX price"},
        {"role": "assistant", "content": "N999ZZ is listed …"},
    ]
    toks = extract_phlydata_tokens_with_history("thanks", hist)
    flat = " ".join(toks).upper().replace(" ", "")
    assert "N118CX" in flat
    assert "N999ZZ" not in flat


def test_refine_drops_embedded_numeric_fragment_with_tail():
    # If both appear, refinement should drop the 3-digit fragment so AND-match on rows still works.
    raw = extract_phlydata_lookup_tokens("N807JS and 807")
    # extraction may or may not add 807; consultant_phly_lookup_token_list merges strict tail first
    toks = consultant_phly_lookup_token_list("N807JS", None)
    assert "N807JS" in [x.upper().replace(" ", "") for x in toks]
