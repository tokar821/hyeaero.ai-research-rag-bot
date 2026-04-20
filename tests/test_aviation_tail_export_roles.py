"""Tail discovery must work with chat-export roles (You / Consultant), not only user/assistant."""

from rag.aviation_tail import (
    find_strict_tail_candidates,
    find_visual_gallery_tail_candidates,
    history_role_contributes_to_thread,
)


def test_strict_tail_from_you_role_history():
    hist = [
        {"role": "You", "content": "Have you N878BW?"},
        {
            "role": "Consultant",
            "content": "The aircraft **N878BW** is an Eclipse EA500.",
        },
    ]
    assert find_strict_tail_candidates("So, can I see that?", hist) == ["N878BW"]


def test_visual_gallery_tail_prefers_thread_mark():
    hist = [{"role": "You", "content": "Have you N878BW?"}]
    assert find_visual_gallery_tail_candidates("So, can I see that?", hist) == ["N878BW"]


def test_single_letter_u_role_user_carries_tail():
    """Custom role labels (e.g. ``U``) still participate in thread scans — not renamed to ``user``."""
    assert history_role_contributes_to_thread("U") is True
    hist = [{"role": "U", "content": "Have you N878BW?"}]
    assert find_strict_tail_candidates("so can I see that?", hist) == ["N878BW"]
    assert find_visual_gallery_tail_candidates("so can I see that?", hist) == ["N878BW"]


def test_system_role_skipped_for_tail_scan():
    hist = [
        {"role": "system", "content": "N000ZZ bogus"},
        {"role": "user", "content": "Have you N878BW?"},
    ]
    assert find_strict_tail_candidates("see it?", hist) == ["N878BW"]


def test_consultant_line_can_carry_tail_when_user_line_has_none():
    hist = [
        {"role": "You", "content": "Do you have any info?"},
        {"role": "Consultant", "content": "Tail **N878BW** is on file."},
    ]
    assert find_visual_gallery_tail_candidates("let me see that interior", hist) == ["N878BW"]
