from rag.consultant_query_anchor import gallery_user_query_for_image_pipeline


def test_appends_resolved_tail_for_deictic():
    q = "So, can I see that?"
    out = gallery_user_query_for_image_pipeline(q, resolved_tail="N878BW")
    assert "N878BW" in out.upper()
    assert "that" in out.lower()


def test_tail_in_line_becomes_tail_led_gallery_query():
    q = "Show me N878BW cabin"
    out = gallery_user_query_for_image_pipeline(q, resolved_tail="N878BW")
    assert "N878BW" in out.upper()
    assert "cabin" in out.lower() or "interior" in out.lower()
