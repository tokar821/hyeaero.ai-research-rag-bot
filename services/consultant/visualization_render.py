"""
Render visualization bundles into user-visible prose and lightweight SVG charts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.visual_models import VisualIntelligenceBundle
from services.consultant.visualization_handler import VisualizationKind, VisualizationTurnResult


def _svg_itinerary_map(
    stops: List[str],
    legs: List[str],
    practical_nm: float,
    *,
    limiting_leg: str = "",
) -> str:
    """Schematic multi-leg route — not a geographic chart, but shows every stop in order."""
    n = max(2, len(stops))
    w, h = 280, 200
    margin_x = 28
    step = (w - 2 * margin_x) / max(n - 1, 1)
    cy = 88
    pr = max(400.0, min(practical_nm, 8000.0))
    reach = min(72.0, pr / 45.0)
    limit_idx = 0
    if limiting_leg and legs:
        norm_limit = re.sub(r"\s+", " ", limiting_leg.strip().lower())
        for i, leg in enumerate(legs):
            if re.sub(r"\s+", " ", leg.strip().lower()) == norm_limit:
                limit_idx = i
                break
        else:
            from services.consultant.route_feasibility import estimate_route_distance_nm

            best_dist = 0.0
            for i, leg in enumerate(legs):
                d = float(estimate_route_distance_nm(leg) or 0)
                if d >= best_dist:
                    best_dist = d
                    limit_idx = i
    if n >= 2:
        x1 = margin_x + limit_idx * step
        x2 = margin_x + min(limit_idx + 1, n - 1) * step
        cx = (x1 + x2) / 2.0
    else:
        cx = 140.0
    dots: List[str] = []
    labels: List[str] = []
    for i, stop in enumerate(stops):
        x = margin_x + i * step
        dots.append(
            f'<circle cx="{x:.1f}" cy="{cy}" r="5" fill="#2563eb" stroke="#1d4ed8" stroke-width="1"/>'
        )
        short = stop[:14] + ("…" if len(stop) > 14 else "")
        labels.append(
            f'<text x="{x:.1f}" y="{cy + 22}" text-anchor="middle" font-size="9" fill="#334155">{short}</text>'
        )
    if len(stops) >= 2:
        path_d = f"M{margin_x:.1f},{cy} " + " ".join(
            f"L{margin_x + i * step:.1f},{cy}" for i in range(1, n)
        )
        route_line = (
            f'<path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2" '
            f'stroke-dasharray="5 3"/>'
        )
    else:
        route_line = ""
    title = " → ".join(stops[:5])
    if len(stops) > 5:
        title += " …"
    limit_txt = f"Limit: {limiting_leg}" if limiting_leg else f"Practical ~{int(pr)} nm"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="Itinerary map">'
        f'<circle cx="{cx}" cy="{cy}" r="{reach:.1f}" fill="rgba(59,130,246,0.12)" '
        f'stroke="#2563eb" stroke-width="1.5" stroke-dasharray="6 4"/>'
        f"{route_line}{''.join(dots)}{''.join(labels)}"
        f'<text x="{cx}" y="18" text-anchor="middle" font-size="10" fill="#334155">{title}</text>'
        f'<text x="{cx}" y="{h - 12}" text-anchor="middle" font-size="9" fill="#64748b">{limit_txt}</text>'
        f"</svg>"
    )


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
        for mr in bundle.mission_reachability[:4]:
            dist = float(mr.distance_nm or 0)
            if dist <= 0 and mr.route_label:
                from services.consultant.route_feasibility import estimate_route_distance_nm

                dist = float(estimate_route_distance_nm(mr.route_label) or 0)
            lines.append(
                f"- {mr.route_label}: {int(dist)} nm stage — "
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
        limiting_leg = ""
        legs = list(result.entities.routes or [])
        stops: List[str] = []
        if legs:
            for leg in legs:
                leg_parts = re.split(r"\s*(?:->|→)\s*", leg, maxsplit=1)
                if leg_parts and leg_parts[0].strip():
                    if not stops or stops[-1] != leg_parts[0].strip():
                        stops.append(leg_parts[0].strip())
                if len(leg_parts) > 1 and leg_parts[1].strip():
                    stops.append(leg_parts[1].strip())
        if result.bundle.mission_reachability:
            max_leg_dist = 0.0
            infeasible_leg = ""
            for mr in result.bundle.mission_reachability:
                dist_nm = float(mr.distance_nm or 0)
                if dist_nm <= 0 and mr.route_label:
                    from services.consultant.route_feasibility import estimate_route_distance_nm

                    dist_nm = float(estimate_route_distance_nm(mr.route_label) or 0)
                cls = ""
                if mr.aircraft_models:
                    cls = str(mr.aircraft_models[0].get("classification") or "")
                if dist_nm >= max_leg_dist:
                    max_leg_dist = dist_nm
                    limiting_leg = mr.route_label or limiting_leg
                if dist_nm > pr * 0.88 or "not_feasible" in cls:
                    infeasible_leg = mr.route_label or infeasible_leg
            if not limiting_leg and legs:
                from services.consultant.route_feasibility import estimate_route_distance_nm

                best = 0.0
                for leg in legs:
                    d = float(estimate_route_distance_nm(leg) or 0)
                    if d >= best:
                        best = d
                        limiting_leg = leg
            limiting_leg = infeasible_leg or limiting_leg
            if not limiting_leg and result.bundle.mission_reachability:
                limiting_leg = result.bundle.mission_reachability[0].route_label or ""
            dist = max_leg_dist
            if dist <= 0 and legs:
                from services.consultant.route_feasibility import estimate_route_distance_nm

                dist = float(estimate_route_distance_nm(legs[0]) or 0)
        if not dist and legs:
            try:
                from services.mission.route_extractor import extract_routes

                ex = extract_routes(legs[0])
                if ex:
                    dist = float(getattr(ex[0].route, "distance_nm", 0) or 0)
            except Exception:
                dist = pr * 0.7
        if len(stops) >= 3:
            svg_blocks.append(_svg_itinerary_map(stops, legs, pr, limiting_leg=limiting_leg))
        else:
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
