"""Graceful degradation — bounded guidance, no orchestration collapse."""

from services.broker.graceful_degradation import (
    apply_graceful_degradation_to_answer,
    broker_degraded_message,
    ensure_non_empty_answer,
    safe_broker_fallback_response,
    transform_refusal_prose,
)
from services.orchestration.fail_safe import safe_stage_fallback
from services.orchestration.image_trust_policy import should_activate_image_trust


def test_never_empty_safe_fallback():
    text = safe_broker_fallback_response("TEB to London, 8 pax")
    assert text.strip()
    assert "couldn't get a response" not in text.lower()
    assert "don't have enough verified field-performance" not in text.lower()


def test_safe_stage_fallback_degrades_not_refuses():
    text = safe_stage_fallback("final_response_formatting", query="Aspen to London")
    assert text.strip()
    assert "couldn't complete the full advisory pipeline" not in text.lower()


def test_transform_refusal_prose():
    raw = "I don't have enough verified field-performance data to position this as a reliable recommendation."
    out = transform_refusal_prose(raw)
    assert "don't have enough verified field-performance" not in out.lower()


def test_ensure_non_empty_answer():
    assert ensure_non_empty_answer("", query="Miami to Aspen").strip()


def test_broker_degraded_nonstop_tone():
    msg = broker_degraded_message(context="nonstop_not_credible", route="Aspen–London")
    assert "nonstop" in msg.lower()
    assert "reliable data for this" not in msg.lower()


def test_image_trust_advisory_suppressed():
    assert not should_activate_image_trust("Recommend aircraft for TEB to London, 8 pax, February")
    assert should_activate_image_trust("Show me the cabin layout on the G650")
    assert should_activate_image_trust("range map for G650 from TEB")


def test_low_confidence_prefix_not_refusal():
    out = apply_graceful_degradation_to_answer("Mission fit: super-mid band.", confidence=0.4)
    assert "directional rather than catalog-verified" in out.lower()
