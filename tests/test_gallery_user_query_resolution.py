from rag.consultant_query_anchor import gallery_user_query_for_image_pipeline


def test_appends_resolved_tail_for_deictic():
    q = "So, can I see that?"
    out = gallery_user_query_for_image_pipeline(q, resolved_tail="N878BW")
    assert "N878BW" in out.upper()
    assert "that" in out.lower()


def test_no_duplicate_when_tail_already_in_line():
    q = "Show me N878BW cabin"
    assert gallery_user_query_for_image_pipeline(q, resolved_tail="N878BW") == q
