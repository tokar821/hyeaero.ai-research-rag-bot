"""Phase 10 — entity scope isolation for retrieval and memory."""

from .scope import (
    EntityScope,
    aircraft_identities_conflict,
    history_allowed_for_tail_resolution,
    is_deictic_tail_followup,
    resolve_entity_scope,
    should_release_tail_on_model_switch,
)
from .validation import (
    attach_entity_scope_observability,
    filter_phly_rows_by_entity_scope,
    phly_row_matches_scope,
    tail_conflicts_with_aircraft,
)

__all__ = [
    "EntityScope",
    "aircraft_identities_conflict",
    "attach_entity_scope_observability",
    "filter_phly_rows_by_entity_scope",
    "history_allowed_for_tail_resolution",
    "is_deictic_tail_followup",
    "phly_row_matches_scope",
    "resolve_entity_scope",
    "should_release_tail_on_model_switch",
    "tail_conflicts_with_aircraft",
]
