"""Luxury aviation advisory prompt block from structured memory."""

from __future__ import annotations

from .schemas import ConversationMemoryState, ResponseMode


def format_memory_prompt_block(state: ConversationMemoryState) -> str:
    if not any(
        (
            state.active_aircraft,
            state.active_tail,
            state.aesthetic_preferences,
            state.last_visual_context,
            state.active_budget_usd,
            state.active_mission,
        )
    ):
        return ""

    lines = [
        "\n\n**CONVERSATION STATE ENGINE (internal — luxury advisory memory; do not recite verbatim):**\n",
        f"- Turn index: `{state.turn_index}`",
    ]
    if state.active_aircraft:
        lines.append(f"- **Active aircraft (anchor):** `{state.active_aircraft}`")
    if state.active_tail:
        lines.append(f"- **Active tail (anchor):** `{state.active_tail}`")
    if state.active_category.value != "unknown":
        lines.append(f"- Cabin class trajectory: `{state.active_category.value}`")
    if state.response_mode != ResponseMode.CONSULTANT:
        lines.append(f"- **Response mode:** `{state.response_mode.value}`")
    if state.conversation_goal.value != "unknown":
        lines.append(f"- Conversation goal: `{state.conversation_goal.value}`")
    if state.last_visual_context:
        lines.append(f"- Last visual context: `{state.last_visual_context}`")
    if state.aesthetic_preferences:
        lines.append(
            "- Aesthetic direction: " + "; ".join(f"`{p}`" for p in state.aesthetic_preferences[-10:])
        )
    if state.negative_preferences:
        lines.append("- Avoid: " + "; ".join(f"`{p}`" for p in state.negative_preferences[-8:]))
    if state.active_budget_usd:
        lines.append(f"- Budget anchor: ~USD {int(state.active_budget_usd):,}")
    elif state.active_budget_label:
        lines.append(f"- Budget anchor: `{state.active_budget_label}`")
    if state.active_mission:
        lines.append(f"- Mission: `{state.active_mission}`")
    if state.comparison_target:
        lines.append(f"- Comparison target: `{state.comparison_target}`")
    if state.aircraft_evolution:
        lines.append("- Evolution: `" + " → ".join(state.aircraft_evolution[-6:]) + "`")
    if state.memory_stack:
        lines.append("- Memory priority stack (active): " + ", ".join(f"`{k}`" for k in state.memory_stack[:8]))

    lines.append(
        "\nResolve deictic follow-ups (*bigger*, *more modern*, *cockpit too*, *that one*) using this memory. "
        "Do **not** reset the thread or re-ask which aircraft unless the user clearly changes subject."
    )

    if state.response_mode in (ResponseMode.IMAGE_SHOWCASE, ResponseMode.SHORT_CAPTION, ResponseMode.VISUAL_ONLY):
        lines.append(
            "\n**Gallery-forward mode:** one premium caption line max; let images carry the experience."
        )

    return "\n".join(lines)
