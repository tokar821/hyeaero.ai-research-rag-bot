"""
Ownership economics reasoning path — not aircraft recommendation ranking.

Fractional vs full ownership, utilization bands, capital efficiency, dispatch posture.
"""

from __future__ import annotations

from typing import Optional

from services.consultant.mission_state import MissionState


def format_ownership_economics_response(
    query: str,
    *,
    mission: Optional[MissionState] = None,
    anchor_model: str = "",
) -> str:
    """Dedicated economics advisory — never empty; uses ownership simulator."""
    from services.orchestration.ownership_simulator import simulate_ownership_economics
    from services.consultant.response_formatter import sanitize_advisor_output

    sim = simulate_ownership_economics(query, mission=mission, anchor_model=anchor_model)
    body = "\n".join(sim.lines)
    return sanitize_advisor_output(body)
