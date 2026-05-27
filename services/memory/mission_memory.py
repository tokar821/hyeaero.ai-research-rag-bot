"""
Safe mission memory — merge stable preferences only; never contaminate routes/pax.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile, OwnershipMode, PriorityLevel

# Fields that may NEVER be stored in cross-turn memory
_FORBIDDEN_MEMORY_KEYS = frozenset(
    {
        "routes",
        "passengers",
        "passenger_count",
        "regions",
        "nonstop_required",
        "westbound_sensitive",
        "eastbound_sensitive",
        "cabin_priority",
        "operating_cost_priority",
        "runway_priority",
        "baggage_priority",
        "seasonal_note",
        "mountain_airports",
        "reserves_requirement",
        "budget_range",
        "budget_usd_mid",
        "preferred_airports",
        "mission_category",
    }
)

# Stable fields allowed to persist across turns
_STABLE_KEYS = frozenset({"home_base", "ownership_posture", "fleet_preferences"})

_DEFAULT_OWNERSHIP_TTL = 8
_DEFAULT_HOME_BASE_TTL = 10
_DEFAULT_FLEET_TTL = 6

_HOME_BASE_RE = re.compile(
    r"\b(?:based in|home (?:base|airport|field)|usually (?:flies?|fly|departs?|depart)(?:ing)? out of|"
    r"primarily (?:from|out of))\s+([a-z][a-z\s\-]{2,24}?)(?:\s+area)?\b",
    re.I,
)


def mission_memory_enabled() -> bool:
    return (os.getenv("CONSULTANT_MISSION_MEMORY") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class MemoryField:
    value: Any
    confidence: float
    ttl_turns: int
    field_key: str = ""

    def is_alive(self) -> bool:
        return int(self.ttl_turns) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(float(self.confidence), 3),
            "ttl_turns": int(self.ttl_turns),
            "field_key": self.field_key,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional[MemoryField]:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                value=raw.get("value"),
                confidence=float(raw.get("confidence") or 0),
                ttl_turns=int(raw.get("ttl_turns") or 0),
                field_key=str(raw.get("field_key") or ""),
            )
        except (TypeError, ValueError):
            return None


@dataclass
class MissionMemory:
    """Client-echoed stable mission preferences (not turn-specific mission facts)."""

    turn_index: int = 0
    home_base: Optional[MemoryField] = None
    ownership_posture: Optional[MemoryField] = None
    fleet_preferences: List[MemoryField] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "turn_index": self.turn_index,
            "home_base": self.home_base.to_dict() if self.home_base else None,
            "ownership_posture": (
                self.ownership_posture.to_dict() if self.ownership_posture else None
            ),
            "fleet_preferences": [f.to_dict() for f in self.fleet_preferences if f.is_alive()],
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> MissionMemory:
        if not isinstance(raw, dict):
            return cls()
        raw = strip_forbidden_from_memory_dict(raw)
        mem = cls(turn_index=int(raw.get("turn_index") or 0))
        mem.home_base = MemoryField.from_dict(raw.get("home_base"))
        mem.ownership_posture = MemoryField.from_dict(raw.get("ownership_posture"))
        fps = raw.get("fleet_preferences")
        if isinstance(fps, list):
            for item in fps:
                f = MemoryField.from_dict(item)
                if f and f.is_alive():
                    mem.fleet_preferences.append(f)
        return mem


def load_mission_memory(
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionMemory:
    """Load persisted memory from client state or last response patch."""
    for src in (conversation_state, data_used):
        if not isinstance(src, dict):
            continue
        raw = src.get("mission_memory") or src.get("consultant_mission_memory")
        if isinstance(raw, dict):
            return expire_stale_fields(MissionMemory.from_dict(raw))
    return MissionMemory()


def expire_stale_fields(memory: MissionMemory) -> MissionMemory:
    """
    Drop expired TTL fields and purge any forbidden turn-specific keys.

    Routes and passenger counts expire every turn (never stored; strip if legacy leak).
    """
    mem = MissionMemory.from_dict(memory.to_dict())
    mem.turn_index = int(mem.turn_index or 0)

    for attr in ("home_base", "ownership_posture"):
        fld: Optional[MemoryField] = getattr(mem, attr)
        if fld and not fld.is_alive():
            setattr(mem, attr, None)

    mem.fleet_preferences = [f for f in mem.fleet_preferences if f.is_alive()]

    # Legacy pollution guard — wipe unknown/forbidden keys if present on dict round-trip
    return mem


def _copy_profile(profile: MissionProfile) -> MissionProfile:
    return MissionProfile(
        passengers=profile.passengers,
        routes=list(profile.routes),
        regions=list(profile.regions),
        nonstop_required=profile.nonstop_required,
        westbound_sensitive=profile.westbound_sensitive,
        eastbound_sensitive=profile.eastbound_sensitive,
        cabin_priority=profile.cabin_priority,
        operating_cost_priority=profile.operating_cost_priority,
        runway_priority=profile.runway_priority,
        baggage_priority=profile.baggage_priority,
        ownership_interest=profile.ownership_interest,
        mission_category=profile.mission_category,
        budget_range=profile.budget_range,
        budget_usd_mid=profile.budget_usd_mid,
        preferred_airports=list(profile.preferred_airports),
        seasonal_note=profile.seasonal_note,
        mountain_airports=profile.mountain_airports,
        reserves_requirement=profile.reserves_requirement,
        home_base=profile.home_base,
        fleet_preferences=list(profile.fleet_preferences),
    )


def merge_memory(
    current_turn: MissionProfile,
    memory: Optional[MissionMemory],
) -> MissionProfile:
    """
    Merge optional stable memory into the current turn profile.

    **Current turn always wins** — memory only fills allowed gaps.
    """
    merged = _copy_profile(current_turn)

    if not mission_memory_enabled() or not memory:
        return merged

    mem = expire_stale_fields(memory)

    # Ownership posture (never override explicit current-turn ownership)
    if merged.ownership_interest is None and mem.ownership_posture and mem.ownership_posture.is_alive():
        raw = mem.ownership_posture.value
        try:
            if isinstance(raw, str):
                merged.ownership_interest = OwnershipMode(raw)
            elif isinstance(raw, OwnershipMode):
                merged.ownership_interest = raw
        except ValueError:
            pass

    # Home base — only when current turn did not state a new home
    if not (merged.home_base or "").strip() and mem.home_base and mem.home_base.is_alive():
        merged.home_base = str(mem.home_base.value).strip()

    # Fleet preferences — union without overriding explicit current fleet hints
    current_fleet = {f.lower() for f in merged.fleet_preferences}
    for fld in mem.fleet_preferences:
        if not fld.is_alive():
            continue
        val = str(fld.value).strip()
        if val and val.lower() not in current_fleet:
            merged.fleet_preferences.append(val)

    return merged


def detect_home_base(user_message: str) -> Optional[str]:
    m = _HOME_BASE_RE.search(user_message or "")
    if not m:
        return None
    place = re.sub(r"\s+", " ", m.group(1).strip()).title()
    if len(place) < 3:
        return None
    return place


def _detect_fleet_preferences(user_message: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(dict.fromkeys(detect_models_from_text(user_message or "")))[:6]
    except Exception:
        return []


def advance_memory(
    memory: Optional[MissionMemory],
    current_turn: MissionProfile,
    *,
    user_message: str = "",
) -> MissionMemory:
    """
    Build memory snapshot for the *next* turn from current turn facts only.

    Only stable preferences are stored; turn-specific fields are never written.
    """
    mem = expire_stale_fields(memory or MissionMemory())
    mem.turn_index = int(mem.turn_index or 0) + 1

    # Decrement TTL on existing stable fields (one turn elapsed)
    for attr in ("home_base", "ownership_posture"):
        fld: Optional[MemoryField] = getattr(mem, attr)
        if fld and fld.is_alive():
            fld.ttl_turns = max(0, int(fld.ttl_turns) - 1)
            if fld.ttl_turns <= 0:
                setattr(mem, attr, None)

    alive_fleet: List[MemoryField] = []
    for fld in mem.fleet_preferences:
        if not fld.is_alive():
            continue
        fld.ttl_turns = max(0, int(fld.ttl_turns) - 1)
        if fld.is_alive():
            alive_fleet.append(fld)
    mem.fleet_preferences = alive_fleet

    msg = (user_message or "").strip()

    # Persist ownership when stated this turn (stable preference)
    if current_turn.ownership_interest is not None:
        mem.ownership_posture = MemoryField(
            value=current_turn.ownership_interest.value,
            confidence=0.92,
            ttl_turns=_DEFAULT_OWNERSHIP_TTL,
            field_key="ownership_posture",
        )

    # Home base from current message or explicit profile field
    home = (current_turn.home_base or "").strip() or (detect_home_base(msg) or "")
    if home:
        mem.home_base = MemoryField(
            value=home,
            confidence=0.88,
            ttl_turns=_DEFAULT_HOME_BASE_TTL,
            field_key="home_base",
        )

    # Fleet preferences from current message only
    fleet_models = list(current_turn.fleet_preferences) or _detect_fleet_preferences(msg)
    if fleet_models:
        mem.fleet_preferences = [
            MemoryField(
                value=model,
                confidence=0.85,
                ttl_turns=_DEFAULT_FLEET_TTL,
                field_key="fleet_preferences",
            )
            for model in fleet_models[:6]
        ]

    return mem


def strip_forbidden_from_memory_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Remove any legacy forbidden keys from a serialized memory blob."""
    cleaned = dict(raw)
    for key in list(cleaned.keys()):
        if key in _FORBIDDEN_MEMORY_KEYS:
            cleaned.pop(key, None)
    return cleaned
