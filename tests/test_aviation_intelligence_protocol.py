"""Aviation intelligence protocol — structured envelope."""

from __future__ import annotations

from unittest.mock import patch

from services.aviation_intelligence_protocol import build_aviation_intelligence_envelope


def test_envelope_invalid():
    env = build_aviation_intelligence_envelope(
        user_query="Falcon 9000 cockpit",
        user_wants_gallery=True,
        phly_rows=[],
        aircraft_images=[],
    )
    assert env["status"] == "INVALID"
    assert env["image_pipeline_executed"] is False
    assert env["blocks"][0]["type"] == "status"


def test_envelope_no_visual():
    env = build_aviation_intelligence_envelope(
        user_query="N807JS cabin",
        user_wants_gallery=False,
        phly_rows=[],
        aircraft_images=[],
    )
    assert env["status"] == "NO_VISUAL"


@patch("services.aviation_intelligence_protocol._optional_db", return_value=None)
def test_envelope_ok_with_images(_db):
    env = build_aviation_intelligence_envelope(
        user_query="N807JS cabin",
        user_wants_gallery=True,
        phly_rows=[],
        aircraft_images=[
            {"url": "https://cdn.jetphotos.com/a.jpg", "source": "JetPhotos", "description": "N807JS"},
        ],
    )
    assert env["status"] == "OK"
    assert env["blocks"][1]["type"] == "verified_images"
    assert env["blocks"][1]["count"] == 1


@patch("services.aviation_intelligence_protocol._optional_db", return_value=None)
def test_envelope_retrieval_failed_empty_gallery(_db):
    env = build_aviation_intelligence_envelope(
        user_query="N807JS cabin",
        user_wants_gallery=True,
        phly_rows=[],
        aircraft_images=[],
    )
    assert env["status"] == "RETRIEVAL_FAILED"
    assert env["image_pipeline_executed"] is True
