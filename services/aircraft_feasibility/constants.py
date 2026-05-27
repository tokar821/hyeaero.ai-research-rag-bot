"""Conservative operational constants — feasibility engine (pre-LLM hard gate)."""

from __future__ import annotations

# NBAA IFR alternate + contingency (nm) — always applied on mission-required side
NBAA_IFR_RESERVE_NM = 200.0

# Nonstop dispatch margin multiplier on required distance
NONSTOP_MARGIN_SHORT = 1.05  # < 2500 nm
NONSTOP_MARGIN_LONG = 1.08  # 2500–4500 nm
NONSTOP_MARGIN_ULR = 1.03  # >= 4500 nm (already at edge)

# Westbound headwind / fuel burn as fraction of stage length → added to required nm
WESTBOUND_REQUIRED_FACTOR = 0.08
WINTER_WESTBOUND_REQUIRED_FACTOR = 0.12
WINTER_AVAILABLE_DEDUCTION_FACTOR = 0.08  # fraction of practical_nm

# Payload — nm per passenger above typical cruise load
PAX_NM_PENALTY_PER_SEAT = 40.0
PAX_NM_PENALTY_CAP = 520.0
BAGGAGE_NM_PENALTY = 90.0

# Mission-side payload additions to required nm (conservative dispatch planning)
MISSION_PAX_REQUIRED_8_PLUS = 120.0
MISSION_PAX_REQUIRED_10_PLUS = 200.0
MISSION_BAGGAGE_REQUIRED = 90.0
MISSION_MOUNTAIN_REQUIRED = 320.0

# Mountain / hot-high available range deduction
MOUNTAIN_AVAILABLE_PENALTY_NM = 300.0
HOT_HIGH_AVAILABLE_PENALTY_NM = 180.0

# Minimum positive margin (nm) — hard reject if below (non-ULR only; ULR may operate tighter)
MIN_DISPATCH_MARGIN_NM = 150.0
MIN_DISPATCH_MARGIN_ULR_NM = 40.0

# Runway limits (ft) by mission environment
RUNWAY_LIMIT_DEFAULT_FT = 5500.0
RUNWAY_LIMIT_INTERNATIONAL_FT = 6000.0
RUNWAY_LIMIT_SHORT_FIELD_FT = 4000.0
RUNWAY_LIMIT_MOUNTAIN_FT = 5200.0

# Category practical range floors for oceanic nonstop (nm) — hard reject below
TRANSATLANTIC_NONSTOP_MIN_PRACTICAL_NM = 4000.0
TRANSPACIFIC_NONSTOP_MIN_PRACTICAL_NM = 5200.0
TRANSPACIFIC_WINTER_WESTBOUND_MIN_PRACTICAL_NM = 5600.0

# Stage length thresholds (nm)
TRANSPACIFIC_STAGE_NM = 4200.0
TRANSATLANTIC_STAGE_NM = 2600.0
