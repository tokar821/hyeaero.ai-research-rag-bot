"""Deterministic fact response services."""

from services.fact.aircraft_fact_responder import respond_aircraft_fact
from services.fact.named_aircraft_capability_responder import respond_aircraft_capability

__all__ = ["respond_aircraft_capability", "respond_aircraft_fact"]
