"""Regression: N-number post-filter must match serial_number or registration_number."""

from rag.phlydata_consultant_lookup import _token_matches_phly_row


def test_n_tail_matches_when_only_in_serial_column():
    row = {
        "registration_number": "",
        "serial_number": "N628TS",
        "manufacturer": "",
        "model": "",
        "category": "",
        "features": "",
    }
    assert _token_matches_phly_row(row, "N628TS") is True


def test_n_tail_matches_when_only_in_registration_column():
    row = {
        "registration_number": "N628TS",
        "serial_number": "61033",
        "manufacturer": "",
        "model": "",
        "category": "",
        "features": "",
    }
    assert _token_matches_phly_row(row, "N628TS") is True
