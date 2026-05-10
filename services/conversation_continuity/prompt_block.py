"""State-aware hidden prompt block appended to consultant system prompts."""

from __future__ import annotations

from .drift_prevention import DRIFT_CONTRACT
from .schemas import ConversationContinuityState


def format_continuity_prompt_block(state: ConversationContinuityState) -> str:
    parts: list[str] = []

    locked = state.locked_entity.value if state.locked_entity else None
    lt = locked and f"type={state.locked_entity.type.value}" if state.locked_entity else None

    parts.append("\n\n**CONVERSATION CONTINUITY ENGINE (hidden — obey, do not recite verbatim):**\n")
    bullet: list[str] = []
    if state.current_aircraft:
        bullet.append(f"- focal aircraft anchor: `{state.current_aircraft}`")
    if state.current_tail:
        bullet.append(f"- inferred tail cue: `{state.current_tail}`")
    if lt and locked:
        bullet.append(f"- locked entity ({lt}): `{locked}`")
    if state.current_category.value != "unknown":
        bullet.append(f"- cabin class trajectory: `{state.current_category.value}`")
    if state.aircraft_evolution:
        bullet.append("- evolution trace: `" + " → ".join(state.aircraft_evolution[-8:]) + "`")
    if state.style_preferences:
        bullet.append("- positive style memory: " + "; ".join(f"`{p}`" for p in state.style_preferences[-12:]))
    if state.negative_preferences:
        bullet.append("- avoid / negative cues: " + "; ".join(f"`{p}`" for p in state.negative_preferences[-8:]))
    if state.buyer_direction.size or state.buyer_direction.luxury or state.buyer_direction.budget_usd_approx:
        bd = []
        if state.buyer_direction.size:
            bd.append(str(state.buyer_direction.size))
        if state.buyer_direction.luxury:
            bd.append(f"luxury:{state.buyer_direction.luxury}")
        if state.buyer_direction.budget_usd_approx:
            bd.append(f"budget_usd≈{int(state.buyer_direction.budget_usd_approx)}")
        bullet.append("- buyer direction hints: `" + "; ".join(bd) + "`")
    if state.last_requested_view:
        bullet.append(f"- last explicit view facet: `{state.last_requested_view}`")
    bullet.append(f"- continuity response posture: **`{state.response_mode.value}`**")
    if state.contextual_intent_tags:
        bullet.append("- latent lifestyle tags: " + ", ".join(f"`{t}`" for t in state.contextual_intent_tags[-10:]))
    if state.last_refinement:
        lf = state.last_refinement
        bullet.append(f"- refinement signal: `{lf.type}` inherit_entity={lf.inherit_entity}")
        if lf.requested_view:
            bullet.append(f"  - requested facet: `{lf.requested_view}`")
    parts.append("\n".join(bullet) if bullet else "- (minimal continuity metadata)")
    if state.drift_flags:
        parts.append("\nflags: " + ", ".join(f"`{f}`" for f in state.drift_flags[-12:]))

    parts.append("\n" + DRIFT_CONTRACT)

    if state.response_mode.value in ("visual_only", "short_caption"):
        parts.append(
            "\nIf images are supplied: **Premium gallery mode** — at most **one premium sentence**, "
            "**no pasted URLs**, then let the carousel carry the UX."
        )
    return "".join(parts).strip()
