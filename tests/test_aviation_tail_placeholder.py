from rag.aviation_tail import is_invalid_placeholder_us_n_tail, registration_format_kind


def test_placeholder_all_zeros():
    assert is_invalid_placeholder_us_n_tail("N00000") is True
    assert is_invalid_placeholder_us_n_tail("n0000") is True
    assert is_invalid_placeholder_us_n_tail("N000") is True


def test_registration_format_kind():
    assert registration_format_kind("N807JS") == "US_N_STRICT"
    assert registration_format_kind("N00000") == "US_N_PLACEHOLDER"


def test_realistic_tail_not_placeholder():
    assert is_invalid_placeholder_us_n_tail("N807JS") is False
    assert is_invalid_placeholder_us_n_tail("N10000") is False
    assert is_invalid_placeholder_us_n_tail("N102345") is False
