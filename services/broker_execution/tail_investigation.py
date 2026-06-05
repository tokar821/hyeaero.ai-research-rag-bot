"""
Unified tail investigation — registry + web research + acquisition framing as one dossier.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_PRIOR_OWNER_SIGNAL_RE = re.compile(
    r"(?is)\b(?:previous(?:ly)?\s+operat|prior\s+owner|former\s+owner|formerly\s+operat|"
    r"previously\s+owned|previous\s+registrant|sold\s+to|transferred\s+to|"
    r"operated\s+by|managed\s+by|fleet\s+of)\b"
)


def _extract_tail(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    tail = str(du.get("tail_registration") or "").strip().upper()
    if tail:
        return tail
    m = re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", query or "", re.I)
    return m.group(0).upper() if m else ""


def load_tail_investigation(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registry facts + optional web snippets for one tail turn."""
    du = data_used if isinstance(data_used, dict) else {}
    tail = _extract_tail(query, du)
    out: Dict[str, Any] = {"tail": tail, "facts": [], "web_hints": [], "current_owner": ""}

    if not tail:
        return out

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
        from services.broker_execution.tail_fact_renderer import select_tail_facts

        ensure_tail_facts_for_query(query or "", du)
        facts = select_tail_facts(du, tail)
        out["facts"] = facts
        for f in facts:
            if f.get("kind") == "ownership":
                out["current_owner"] = str(f.get("value") or "").strip()
    except Exception:
        pass

    out["web_hints"] = _extract_web_hints(du, tail)
    return out


def _extract_web_hints(data_used: Dict[str, Any], tail: str) -> List[str]:
    hints: List[str] = []
    tail_low = (tail or "").lower()
    payloads: List[Dict[str, Any]] = []
    for key in ("tavily_payload", "tavily_secondary_payload", "tavily_tertiary_payload"):
        p = data_used.get(key)
        if isinstance(p, dict):
            payloads.append(p)
    block = data_used.get("tavily_block") or data_used.get("tavily_context")
    if isinstance(block, str) and block.strip():
        for line in block.splitlines():
            low = line.lower()
            if tail_low in low and _PRIOR_OWNER_SIGNAL_RE.search(low):
                hints.append(line.strip()[:280])

    seen = set()
    for payload in payloads:
        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            blob = " ".join(
                str(row.get(k) or "")
                for k in ("content", "snippet", "title", "url")
            ).strip()
            if not blob or tail_low not in blob.lower():
                if tail and not _PRIOR_OWNER_SIGNAL_RE.search(blob):
                    continue
            if _PRIOR_OWNER_SIGNAL_RE.search(blob) or (
                tail_low in blob.lower() and any(
                    w in blob.lower()
                    for w in ("owner", "operat", "registrant", "fleet", "spacex", "llc", "inc")
                )
            ):
                snippet = blob[:300].strip()
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    hints.append(snippet)
    return hints[:6]


def render_tail_acquisition_concerns(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    *,
    body: str = "",
) -> str:
    """Buying concerns for a named tail — registry + risks, not generic ChatGPT."""
    inv = load_tail_investigation(query, data_used)
    tail = inv["tail"] or "this tail"
    lines: List[str] = []

    has_registration_line = False
    for f in inv.get("facts") or []:
        if not isinstance(f, dict):
            continue
        label = str(f.get("label") or "").strip()
        val = str(f.get("value") or "").strip()
        if label and val:
            lines.append(f"- **{label}:** {val}")
            if label.lower() == "registration":
                has_registration_line = True

    if not lines and body:
        from services.broker_execution.broker_query_guards import _parse_registry_lines_from_body

        parsed = _parse_registry_lines_from_body(body)
        if parsed.get("aircraft"):
            lines.append(f"- **Aircraft:** {parsed['aircraft']}")
        if parsed.get("owner"):
            lines.append(f"- **Registrant:** {parsed['owner']}")
        if parsed.get("year"):
            lines.append(f"- **Year:** {parsed['year']}")
        if parsed.get("serial"):
            lines.append(f"- **Serial:** {parsed['serial']}")
    if lines:
        if not has_registration_line and tail:
            lines.append(f"- **Registration:** {tail}")
        lines.append("")

    lines.extend(
        [
            f"**{tail} — what would concern me if I were buying today:**",
            "- **Records before cosmetics** — demand logbook abstract, program statement, and "
            "maintenance status before treating listing narrative as clean.",
            "- **Damage & incident chain** — NTSB/FAA/logbook cross-check; fresh paint/interior "
            "without records is a yellow flag, not a plus.",
            "- **Program & upcoming events** — enrollment transferability, engine true-up, and "
            "calendar inspections vs. airframe hours.",
            "- **Title & liens** — FAA title search, UCC, export/lease encumbrances; trust "
            "registrants need seller reps on beneficial owner.",
            "- **Market position** — same-model/year comps, DOM, and why the ask is where it is.",
        ]
    )
    return "\n".join(lines)


def render_prior_ownership_with_research(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Prior ownership — FAA first, then web research hints, then broker inference."""
    inv = load_tail_investigation(query, data_used)
    tail = inv["tail"] or "this tail"
    owner = inv.get("current_owner") or ""
    hints = inv.get("web_hints") or []

    lines: List[str] = [f"**{tail} — ownership history read:**"]

    if owner:
        lines.append(f"- **Current FAA registrant:** {owner}")

    for f in inv.get("facts") or []:
        if isinstance(f, dict) and f.get("kind") == "aircraft_model":
            lines.append(f"- **Type:** {f.get('value')}")

    if hints:
        lines.append("")
        lines.append("**Web research signals (verify before relying):**")
        for h in hints[:4]:
            lines.append(f"- {h}")
        lines.append("")
        lines.append(
            "**Broker inference:** High-profile or corporate tails often show prior operators in "
            "press, planespotter captions, or management-company pages even when FAA only shows "
            "the current trust/LLC. Treat web lines as leads — confirm with title search and "
            "maintenance chain."
        )
    else:
        lines.append("")
        lines.append(
            "I do not have a verified prior-operator chain in synced FAA/Phly feeds alone. "
            "Next sources:"
        )
        lines.append("- **FAA registry chain** — title transfers (often incomplete for trusts).")
        lines.append("- **Title search / escrow** — recorded assignments and releases.")
        lines.append("- **Maintenance & program letters** — operator names in work orders.")
        lines.append(
            "- **Web research** — planespotter/press/listing history for this tail (run a targeted "
            "search on registration + serial + prior owner)."
        )

    if re.search(r"(?is)\bwhat\s+does\s+that\s+(?:history|ownership)\s+suggest\b", query or ""):
        lines.append("")
        lines.append("**What it suggests:**")
        if owner and re.search(r"(?is)trust|bank\s+of\s+utah|trustee", owner):
            lines.append(
                "- Trust/estate registrant → likely tax/estate or privacy structuring; dig for "
                "beneficial owner and seller motivation before LOI."
            )
        elif owner and re.search(r"(?is)llc|inc|corp|aviation|jets|landing", owner):
            lines.append(
                "- Corporate LLC registrant → often flight-department or management-company "
                "structure; maintenance cadence usually professional if records match."
            )
        else:
            lines.append(
                "- Without a verified chain, assume nothing about seller urgency — underwrite to "
                "records, program state, and comps."
            )

    return "\n".join(lines)


__all__ = [
    "load_tail_investigation",
    "render_prior_ownership_with_research",
    "render_tail_acquisition_concerns",
]
