"""
Render visualization bundles into user-visible prose and lightweight SVG charts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.consultant.visual_models import VisualIntelligenceBundle
from services.consultant.visualization_handler import VisualizationKind, VisualizationTurnResult


def _svg_range_map(
    origin: str,
    dest: str,
    practical_nm: float,
    distance_nm: float,
) -> str:
    """Inline SVG range envelope — generated locally, not a verified aircraft photo."""
    pr = max(400.0, min(practical_nm, 8000.0))
    dist = max(200.0, min(distance_nm or pr * 0.65, pr * 1.15))
    reach = min(100.0, (pr / max(dist, 1)) * 72.0)
    cx, cy, r = 120, 100, 72
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="200" '
        f'viewBox="0 0 240 200" role="img" aria-label="Range map">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#3b82f6" stroke-width="2" '
        f'stroke-dasharray="6 4"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{reach:.1f}" fill="rgba(59,130,246,0.15)" '
        f'stroke="#2563eb" stroke-width="2"/>'
        f'<text x="{cx}" y="24" text-anchor="middle" font-size="11" fill="#334155">'
        f"{origin} → {dest}</text>"
        f'<text x="{cx}" y="{cy + r + 18}" text-anchor="middle" font-size="10" fill="#64748b">'
        f"Practical ~{int(pr)} nm</text></svg>"
    )


def summarize_visual_bundle(
    bundle: VisualIntelligenceBundle,
    kind: VisualizationKind,
) -> str:
    lines: List[str] = []

    if bundle.range_maps:
        lines.append("**Range envelope (generated)**")
        for rm in bundle.range_maps[:3]:
            lines.append(
                f"- {rm.aircraft_model}: {rm.origin_label} → {rm.destination_label} — "
                f"practical ~{int(rm.practical_radius_nm)} nm ({rm.classification})"
            )

    if bundle.mission_reachability:
        lines.append("**Mission reachability**")
        for mr in bundle.mission_reachability[:2]:
            lines.append(
                f"- {mr.route_label}: {int(mr.distance_nm)} nm stage — "
                + ", ".join(
                    f"{a.get('model')}: {a.get('classification')}"
                    for a in (mr.aircraft_models or [])[:4]
                )
            )

    if bundle.payload_range_charts:
        lines.append("**Payload–range (directional)**")
        for chart in bundle.payload_range_charts[:2]:
            if chart.points:
                pax_pts = ", ".join(
                    f"{p.passengers} pax → ~{int(p.practical_nm)} nm"
                    for p in chart.points[-3:]
                )
                lines.append(f"- {chart.aircraft_model}: {pax_pts}")

    if bundle.cabin_layouts and kind in (
        VisualizationKind.CABIN_LAYOUT,
        VisualizationKind.CABIN_GRAPHIC,
        VisualizationKind.COMPARE_LAYOUTS,
    ):
        lines.append("**Cabin profile (spec-based layout card)**")
        for cab in bundle.cabin_layouts[:3]:
            stand = "stand-up" if cab.stand_up_cabin else "standard"
            lines.append(
                f"- {cab.aircraft_model}: {cab.category}, {stand}, "
                f"typical {cab.typical_pax} pax"
            )

    if bundle.comparison_cards:
        lines.append("**Comparison**")
        for card in bundle.comparison_cards[:2]:
            if isinstance(card, dict):
                lines.append(f"- {card.get('title') or card.get('model') or 'Aircraft'}")

    return "\n".join(lines)


def attach_visualization_assets(
    result: VisualizationTurnResult,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build user-visible answer text and ``data_used`` patch with SVG + structured charts.
    """
    patch: Dict[str, Any] = {
        "consultant_visual_models": result.bundle.to_dict(),
        "visualization_kind": result.kind.value,
    }

    if result.followup_needed:
        return result.followup_message, patch

    parts: List[str] = []
    if result.caption:
        parts.append(result.caption)

    summary = summarize_visual_bundle(result.bundle, result.kind)
    if summary:
        parts.append(summary)

    svg_blocks: List[str] = []
    origin = result.entities.origin_label or "Origin"
    dest = result.entities.destination_label or "Destination"

    if result.kind in (VisualizationKind.RANGE_MAP, VisualizationKind.REACHABLE_CITIES):
        rm = result.bundle.range_maps[0] if result.bundle.range_maps else None
        pr = float(rm.practical_radius_nm) if rm else 3000.0
        dist = 0.0
        if result.bundle.mission_reachability:
            dist = float(result.bundle.mission_reachability[0].distance_nm or 0)
        if not dist and result.entities.routes:
            try:
                from services.mission.route_extractor import extract_routes

                ex = extract_routes(result.entities.routes[0])
                if ex:
                    dist = float(getattr(ex[0].route, "distance_nm", 0) or 0)
            except Exception:
                dist = pr * 0.7
        svg_blocks.append(_svg_range_map(origin, dest, pr, dist or pr * 0.7))
        patch["consultant_visualization_svg"] = svg_blocks

    if svg_blocks:
        parts.append("\n".join(svg_blocks))

    text = "\n\n".join(p for p in parts if p).strip()
    if not text:
        text = result.caption or "Visualization data is attached for this turn."
    patch["consultant_visualization_rendered"] = 1
    return text, patch


def format_visualization_user_response(
    result: VisualizationTurnResult,
) -> Tuple[str, Dict[str, Any]]:
    """Public entry — caption + summary + generated SVG."""
    return attach_visualization_assets(result)
