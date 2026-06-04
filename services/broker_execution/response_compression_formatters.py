"""
Phase 56.5 — mode-specific response formatters (presentation only).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.broker_execution.response_deduplication import deduplicate_lines
from services.broker_execution.response_mode_classifier import ResponseMode

# Display-only spec hints for comparison tables (not ranking / tiers).
try:
    from services.broker_execution.comparison_broker_facts import _COMPARISON_SPEC_ROWS as _COMPARISON_ROWS
except Exception:
    _COMPARISON_ROWS: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {
        ("gulfstream g280", "citation longitude"): [
            ("Range (nm)", "~3,600", "~3,500"),
            ("Cabin", "Large", "Super-mid"),
            ("Operating cost", "Higher", "Mid"),
            ("Market liquidity", "Moderate", "Strong"),
        ],
    }

_FORBIDDEN_ALL_RE = re.compile(
    r"(?is)\b(?:if\s+i\s+were\s+buying\s+today|i\s+would\s+focus\s+on|i\s+would\s+buy|"
    r"before\s+treating\s+(?:it\s+as\s+a\s+)?bargain|before\s+treating\s+this\s+tail|"
    r"operational\s+synthesis|send\s+me\s+(?:the\s+)?listing\s+package|"
    r"i\s+can\s+verify\s+ownership\s+and\s+basic\s+registry|"
    r"what\s+i\s+would\s+do|key\s+risk|inventory:\s*limited|get\s+a\s+spec\s+sheet)\b"
)
_BROKER_TEMPLATE_RE = re.compile(
    r"(?is)^(?:on\s+N[A-Z0-9]{3,6},|before\s+treating|for\s+an\s+acquisition\s+view|"
    r"to\s+assess\s+acquisition\s+merit|engine\s+program\s+status|maintenance\s+records\s+when)"
)

_FACT_LABEL_PRIORITY = (
    ("aircraft_model", "Aircraft"),
    ("ownership", "Owner"),
    ("registry_status", "Status"),
    ("year", "Year"),
    ("serial_number", "Serial"),
    ("registration", "Registration"),
)


def _strip_forbidden_narrative(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    kept = [p for p in paragraphs if not _FORBIDDEN_ALL_RE.search(p)]
    kept = [p for p in kept if not _BROKER_TEMPLATE_RE.search(p.split("\n", 1)[0])]
    return "\n\n".join(kept).strip()


def _compact_fact_lines(facts: List[Dict[str, str]], *, max_lines: int = 5) -> str:
    lines: List[str] = []
    used_kinds: set[str] = set()
    for kind, label in _FACT_LABEL_PRIORITY:
        for f in facts:
            if f.get("kind") != kind and label.lower() not in str(f.get("label") or "").lower():
                continue
            val = str(f.get("value") or "").strip()
            if not val or kind in used_kinds:
                continue
            # Prefer Phly owner over duplicate FAA registrant label
            if kind == "ownership" and any("owner" in ln.lower() for ln in lines):
                continue
            if kind == "aircraft_model" and label == "FAA aircraft type" and any(
                ln.lower().startswith("• aircraft:") for ln in lines
            ):
                continue
            lines.append(f"• {label}: {val}")
            used_kinds.add(kind)
            break
        if len(lines) >= max_lines:
            break
    return "\n".join(lines[:max_lines]).strip()


def format_fact_only(answer: str, *, query: str, data_used: dict) -> str:
    reg = str(data_used.get("tail_registration") or "").upper()
    if not reg:
        m = re.search(r"\b(N[A-Z0-9]{3,6})\b", (query or "").upper())
        reg = m.group(1) if m else ""

    facts = data_used.get("tail_selected_facts") or data_used.get("tail_facts") or []
    if not facts and reg:
        try:
            from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
            from services.broker_execution.tail_fact_renderer import select_tail_facts

            ensure_tail_facts_for_query(query, data_used)
            facts = select_tail_facts(data_used, reg)
        except Exception:
            facts = []

    if facts:
        compact = _compact_fact_lines(facts, max_lines=5)
        if compact:
            return compact

    # Fallback: extract bullets from answer, cap at 5
    bullets = []
    for line in (answer or "").splitlines():
        s = line.strip()
        if s.startswith(("•", "-", "*")) and not _FORBIDDEN_ALL_RE.search(s):
            bullets.append(s if s.startswith("•") else f"• {s.lstrip('-* ')}")
        if len(bullets) >= 5:
            break
    return "\n".join(bullets[:5]).strip() if bullets else _strip_forbidden_narrative(answer)[:400]


def _comparison_models(query: str, data_used: dict) -> Tuple[str, str]:
    try:
        from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison

        res = soft_resolve_comparison(query)
        if res and res.models:
            return res.models[0], res.models[1]
    except Exception:
        pass
    parts = re.split(r"(?is)\bvs\.?\b|\bversus\b", query or "", maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip().title(), parts[1].strip().title()
    br = data_used.get("broker_reasoning") or {}
    comp = br.get("comparison") if isinstance(br, dict) else {}
    models = list((comp or {}).get("models") or [])[:2]
    if len(models) >= 2:
        return str(models[0]), str(models[1])
    return "Aircraft A", "Aircraft B"


def _comparison_table_rows(a: str, b: str) -> List[Tuple[str, str, str]]:
    key = (a.lower(), b.lower())
    rev = (b.lower(), a.lower())
    if key in _COMPARISON_ROWS:
        return _COMPARISON_ROWS[key]
    if rev in _COMPARISON_ROWS:
        return [(r[0], r[2], r[1]) for r in _COMPARISON_ROWS[rev]]
    return [
        ("Range (nm)", "See spec sheet", "See spec sheet"),
        ("Cabin", "Class varies", "Class varies"),
        ("Operating cost", "Compare OPEX", "Compare OPEX"),
        ("Market liquidity", "Varies", "Varies"),
    ]


def format_comparison(answer: str, *, query: str, data_used: dict) -> str:
    a, b = _comparison_models(query, data_used)
    rows = _comparison_table_rows(a, b)
    header = f"| Feature | {a} | {b} |\n| ------- | --- | --- |"
    body_rows = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows)
    table = f"{header}\n{body_rows}"

    verdict_lines = ["Verdict:"]
    low = _strip_forbidden_narrative(answer).lower()
    if "range" in low or "nm" in low:
        verdict_lines.append(f"• Range: compare {a} vs {b} against your mission length.")
    else:
        verdict_lines.append(f"• Range: {rows[0][1]} vs {rows[0][2]} (indicative).")
    verdict_lines.append(f"• Cabin: {rows[1][1]} vs {rows[1][2]}.")
    verdict_lines.append("• Overall: depends on mission and budget.")

    return f"{table}\n\n" + "\n".join(verdict_lines[:4]).strip()


def format_listing(answer: str, *, query: str, data_used: dict) -> str:
    audit = data_used.get("listing_parse_audit") or {}
    if not isinstance(audit, dict) or not audit.get("parse_success"):
        body = _strip_forbidden_narrative(answer)
        return deduplicate_lines(body)

    model = audit.get("detected_model") or "Aircraft"
    year = audit.get("detected_year")
    price = audit.get("detected_price")
    lines = [f"• Price: ${price:.1f}M asking" + (f" ({year} {model})" if year else f" ({model})")]

    mr = data_used.get("market_reality") or {}
    band = mr.get("band_mid_usd") if isinstance(mr, dict) else None
    if band:
        try:
            lines.append(f"• Market context: mid-band near ${float(band)/1e6:.1f}M for {model}.")
        except (TypeError, ValueError):
            lines.append(f"• Market context: see market band for {model}.")
    else:
        ctx = _extract_line_matching(answer, r"(?is)market|band|priced")
        lines.append(f"• Market context: {ctx or 'Review recent comps for this model/year.'}")

    risks: List[str] = []
    for pat in (
        r"(?is)above\s+market",
        r"(?is)below\s+market",
        r"(?is)liquidity",
        r"(?is)hours",
        r"(?is)risk",
    ):
        m = _extract_line_matching(answer, pat)
        if m and m not in risks:
            risks.append(m)
        if len(risks) >= 2:
            break
    if not risks:
        risks = ["Verify hours, programs, and damage history on the spec sheet."]
    for r in risks[:2]:
        lines.append(f"• Risk: {r.lstrip('• ').strip()}")

    dq = data_used.get("deal_quality") or {}
    verdict = dq.get("verdict") if isinstance(dq, dict) else None
    if not verdict:
        verdict = _extract_line_matching(answer, r"(?is)verdict|deal|fair|aggressive")
    lines.append(f"• Verdict: {verdict or 'Review price vs market band and aircraft condition.'}")

    return "\n".join(lines[:8]).strip()


def _extract_line_matching(answer: str, pattern: str) -> Optional[str]:
    for line in (answer or "").splitlines():
        if re.search(pattern, line):
            s = line.strip().lstrip("•-* ")
            if len(s) > 12:
                return s[:200]
    return None


def format_mission(answer: str, *, query: str, data_used: dict) -> str:
    infeasible = data_used.get("mission_infeasible_models") or []
    required = data_used.get("mission_required_range_nm")
    profile = data_used.get("mission_profile") or {}

    if infeasible and required:
        route = profile.get("route") or "this route"
        return (
            f"• Feasibility: FAIL — cannot fly {route} nonstop (~{required} nm) "
            f"with super-midsize jets; not feasible for "
            f"{', '.join(infeasible[:3])}."
        ).strip()

    lines = ["• Feasibility: PASS for aircraft class review."]
    candidates: List[str] = []
    br = data_used.get("broker_reasoning") or {}
    for block in (br.get("category"), br.get("mission"), br.get("alternatives")):
        if isinstance(block, dict):
            candidates.extend(block.get("candidates") or block.get("models") or [])
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        candidates.insert(0, rec["primary_recommendation"])
    seen: set[str] = set()
    for c in candidates:
        cs = str(c).strip()
        if not cs or cs.lower() in seen:
            continue
        seen.add(cs.lower())
        lines.append(f"• Aircraft: {cs}")
        if len(lines) >= 4:
            break

    if len(lines) == 1:
        primary = _extract_line_matching(answer, r"(?is)\b(?:longitude|g280|challenger|g650)\b")
        if primary:
            lines.append(f"• Aircraft: {primary[:80]}")
        else:
            snippet = _strip_forbidden_narrative(answer).split("\n\n", 1)[0][:300]
            if snippet:
                lines.append(f"• Note: {snippet}")

    return "\n".join(lines[:4]).strip()


def format_analysis(answer: str, *, query: str, data_used: dict) -> str:
    del query, data_used
    return deduplicate_lines(_strip_forbidden_narrative(answer))


__all__ = [
    "format_analysis",
    "format_comparison",
    "format_fact_only",
    "format_listing",
    "format_mission",
]
