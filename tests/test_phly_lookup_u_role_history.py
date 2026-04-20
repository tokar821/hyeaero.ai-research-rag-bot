"""Phly token resolution must see tails from history when the client uses ``U`` for user."""

from rag.phlydata_consultant_lookup import consultant_phly_lookup_token_list


def test_phly_tokens_resolve_tail_from_u_role_history():
    hist = [{"role": "U", "content": "Have you N878BW?"}]
    toks = consultant_phly_lookup_token_list("so can I see that?", hist)
    assert any("N878BW" in (t or "").upper().replace(" ", "") for t in toks)
