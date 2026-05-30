"""Comparison v2 structured output layer."""

from services.comparison.alternative_pipeline_responder import respond_aircraft_alternative
from services.comparison.comparison_pipeline_v2 import run_comparison_v2
from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison

__all__ = [
    "respond_aircraft_alternative",
    "respond_aircraft_comparison",
    "run_comparison_v2",
]
