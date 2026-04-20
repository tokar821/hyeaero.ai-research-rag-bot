from rag.consultant_query_anchor import (
    effective_history_for_gallery_tail,
    latest_message_anchors_aircraft_identity,
)


def test_g650_cheaper_interior_is_latest_anchored():
    q = "I want something like a G650 but cheaper. Show interior."
    assert latest_message_anchors_aircraft_identity(q) is True


def test_show_me_that_without_inline_aircraft_not_anchored():
    q = "Show me that."
    assert latest_message_anchors_aircraft_identity(q) is False


def test_so_can_i_see_that_is_not_latest_line_identity_anchor():
    assert latest_message_anchors_aircraft_identity("so can I see that?") is False


def test_tail_only_message_anchored():
    assert latest_message_anchors_aircraft_identity("Photos of N807JS please") is True


def test_effective_history_none_when_anchored():
    hist = [{"role": "user", "content": "N999AB earlier"}]
    q = "G650 cabin interior photos"
    assert effective_history_for_gallery_tail(q, hist) is None


def test_effective_history_preserved_for_deictic():
    hist = [{"role": "user", "content": "Challenger 350"}]
    assert effective_history_for_gallery_tail("Show me that cabin", hist) == hist
