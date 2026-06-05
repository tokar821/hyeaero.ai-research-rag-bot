"""Tail context memory — most-recent user tail and visual follow-ups."""

from rag.aviation_tail import find_most_recent_user_tail_in_history, find_visual_gallery_tail_candidates
from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer
from rag.consultant_query_anchor import gallery_user_query_for_image_pipeline


def test_most_recent_user_tail_not_first_in_thread():
    hist = [
        {"role": "user", "content": "N51ND cabin"},
        {"role": "assistant", "content": "Gallery for N51ND."},
        {"role": "user", "content": "N5616 cabin image"},
        {"role": "assistant", "content": "Gallery for N5616."},
        {"role": "user", "content": "interior too"},
    ]
    assert find_most_recent_user_tail_in_history(hist) == "N5616"
    assert find_visual_gallery_tail_candidates("interior too", hist) == ["N5616"]


def test_deictic_show_me_resolves_recent_tail():
    hist = [
        {"role": "user", "content": "show me N51ND cabine"},
        {"role": "assistant", "content": "Gallery for N51ND."},
        {"role": "user", "content": "can you show me ?"},
    ]
    assert wants_consultant_aircraft_images_in_answer("can you show me ?", hist) is True
    assert find_visual_gallery_tail_candidates("can you show me ?", hist) == ["N51ND"]


def test_interior_too_triggers_gallery_intent():
    assert wants_consultant_aircraft_images_in_answer("interior too") is True
    assert gallery_user_query_for_image_pipeline("interior too", resolved_tail="N5616") == "N5616 cabin interior"
