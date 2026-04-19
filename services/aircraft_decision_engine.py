"""
HyeAero.AI — Aircraft Decision Engine (buy / conditional / pass).

Deterministic scoring from parsed mission hints + optional DB market/valuation signals.
Does not invent listing prices or tail-specific history; missing data lowers scores and is named in ``insight``.

**risk_score:** higher = **lower buyer risk** (cleaner comp set, fewer blind spots)—same direction as fit/deal (higher is better).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rag.aviation_tail import normalize_tail_token
from rag.consultant_query_expand import _detect_manufacturers, _detect_models
from services.market_comparison import run_comparison
from services.price_estimate import estimate_value_hybrid
from services.searchapi_aircraft_images import compose_manufacturer_model_phrase, normalize_aircraft_name

logger = logging.getLogger(__name__)

_CLASS_BAND_NM = {
    "light": 1900.0,
    "midsize": 2800.0,
    "super_midsize": 3600.0,
    "large": 5200.0,
    "ultra": 7500.0,
}

# Illustrative direct operating cost — wide bands; always framed as uncertain in prose.
_OPCOST_USD_PER_HR = {
    "light": (2200, 3800),
    "midsize": (3500, 5500),
    "super_midsize": (4500, 7500),
    "large": (7000, 13000),
    "ultra": (9500, 16000),
    "unknown": (4000, 9000),
}

_MODEL_CLASS_HINTS: Tuple[Tuple[str, str], ...] = (
    (r"\bg650\b", "ultra"),
    (r"\bg700\b", "ultra"),
    (r"\bg800\b", "ultra"),
    (r"\bg500\b", "large"),
    (r"\bg600\b", "large"),
    (r"\bg280\b", "super_midsize"),
    (r"\bg200\b", "midsize"),
    (r"\bglobal\s*7500\b", "ultra"),
    (r"\bglobal\b", "large"),
    (r"\bfalcon\s*10x\b", "ultra"),
    (r"\bfalcon\s*8x\b", "large"),
    (r"\bfalcon\s*7x\b", "large"),
    (r"\bfalcon\s*900\b", "large"),
    (r"\bfalcon\s*2000\b", "super_midsize"),
    (r"\bchallenger\s*350\b", "super_midsize"),
    (r"\bchallenger\s*650\b", "large"),
    (r"\blongitude\b", "super_midsize"),
    (r"\bpraetor\b", "super_midsize"),
    (r"\bcitation\s*x\b", "super_midsize"),
    (r"\blatitude\b", "super_midsize"),
    (r"\bsovereign\b", "midsize"),
    (r"\bxls\b", "midsize"),
    (r"\bexcel\b", "midsize"),
    (r"\bcj4\b", "light"),
    (r"\bcj3\b", "light"),
    (r"\bm2\b", "light"),
    (r"\bphenom\s*300\b", "light"),
    (r"\bphenom\s*100\b", "light"),
    (r"\bpc-24\b", "light"),
    (r"\blegacy\b", "midsize"),
)

_ALTERNATIVES: Dict[str, List[str]] = {
    "phenom 300": ["Citation CJ3+", "PC-24", "Lear 75"],
    "citation cj4": ["Phenom 300E", "PC-24"],
    "challenger 350": ["Citation Longitude", "Praetor 600", "G280"],
    "citation longitude": ["Challenger 350", "Praetor 600"],
    "gulfstream g280": ["Challenger 350", "Citation Longitude"],
    "falcon 2000": ["Challenger 650", "G450-class large cabin (older)"],
}


def _optional_db() -> Any:
    try:
        from api.main import get_db

        return get_db()
    except Exception:
        return None


def _optional_embedding_pinecone() -> Tuple[Any, Any]:
    try:
        from api.main import get_embedding_and_pinecone

        return get_embedding_and_pinecone()
    except Exception:
        return None, None


def infer_jet_class(marketing: str) -> str:
    low = (marketing or "").lower()
    for pat, cls in _MODEL_CLASS_HINTS:
        if re.search(pat, low, re.I):
            return cls
    if any(x in low for x in ("global", "falcon 7", "falcon 8", "g650", "g700")):
        return "large"
    if any(x in low for x in ("challenger", "gulfstream", "falcon")):
        return "super_midsize"
    if any(x in low for x in ("citation", "lear", "hawker")):
        return "midsize"
    if any(x in low for x in ("phenom", "honda", "mustang")):
        return "light"
    return "unknown"


def extract_mission_profile(user_query: str) -> Dict[str, Any]:
    raw = (user_query or "").strip()
    low = raw.lower()
    missing: List[str] = []

    pax: Optional[int] = None
    for m in re.finditer(r"\b(\d{1,2})\s*(?:pax|pax\.|passengers?|seats?)\b", low):
        pax = int(m.group(1))
        if 1 <= pax <= 19:
            break
    if pax is None:
        missing.append("passenger_count")

    leg_nm: Optional[float] = None
    for m in re.finditer(r"\b([\d,]{2,5})\s*nm\b", low):
        try:
            leg_nm = float(m.group(1).replace(",", ""))
            if 100 <= leg_nm <= 9000:
                break
        except ValueError:
            continue
    if leg_nm is None and re.search(r"\btranscon\b|\btransatlantic\b|\batlantic\b", low):
        leg_nm = 3000.0 if "transatlantic" in low or "atlantic" in low else 2400.0
    if leg_nm is None:
        missing.append("longest_leg_nm")

    budget_m: Optional[float] = None
    for m in re.finditer(
        r"\$(\d+(?:\.\d+)?)\s*(m|mm|million)\b|\b(\d+(?:\.\d+)?)\s*million\b",
        low,
    ):
        g = m.groups()
        if g[0]:
            budget_m = float(g[0])
            break
        if g[2]:
            budget_m = float(g[2])
            break
    if budget_m is None:
        missing.append("budget")

    usage = ""
    if "charter" in low or "135" in low:
        usage = "charter"
    elif "private" in low or "personal" in low or "corporate" in low:
        usage = "private"
    else:
        missing.append("usage_private_vs_charter")

    typical_routes = bool(
        re.search(
            r"\b(nyc|lax|mia|ord|dfw|bos|sea|lon|par|dub|dxb|tok|hkg|syd)\b.*\b(to|→|-|–)\b",
            low,
        )
    )

    return {
        "passengers": pax,
        "longest_leg_nm": leg_nm,
        "budget_millions_usd": budget_m,
        "usage": usage,
        "typical_routes_hint": typical_routes,
        "missing_fields": missing,
    }


def resolve_target_aircraft(user_query: str) -> Tuple[str, str, str]:
    """
    Returns ``(marketing_label, manufacturer, model_token)``.
    ``model_token`` is best-effort substring for SQL/valuation (may be composite).
    """
    raw = (user_query or "").strip()
    blob_lc = raw.lower()
    mans = _detect_manufacturers(blob_lc)
    mdls = _detect_models(raw)
    tail = ""
    for m in re.finditer(r"\b(N\d{1,5}[A-Z]{0,2})\b", raw, re.I):
        tail = normalize_tail_token(m.group(1))
        if tail:
            break

    mm = ""
    if mans and mdls:
        mm = compose_manufacturer_model_phrase(mans[0], mdls[0])
    elif mdls:
        mm = mdls[0]
    elif mans:
        mm = mans[0]

    mm = normalize_aircraft_name(mm.strip()) if mm else ""
    mfr = mans[0] if mans else ""
    mdl = mdls[0] if mdls else (mm if mm and not mfr else "")
    return mm, mfr, mdl


def _alternatives_for(marketing: str) -> List[str]:
    key = (marketing or "").strip().lower()
    for k, alts in _ALTERNATIVES.items():
        if k in key:
            return alts[:4]
    cls = infer_jet_class(marketing)
    if cls == "light":
        return ["Citation CJ4", "PC-24", "Phenom 300E"]
    if cls == "midsize":
        return ["Citation Latitude", "Lear 60XR-class", "Hawker 900-class"]
    if cls == "super_midsize":
        return ["Challenger 350", "Citation Longitude", "Praetor 600"]
    if cls in ("large", "ultra"):
        return ["Challenger 650", "Falcon 2000-class", "G500-class (mission dependent)"]
    return ["Define mission (pax, longest leg, budget) before swapping models."]


def _mission_incomplete(mission: Dict[str, Any]) -> bool:
    m = mission.get("missing_fields") or []
    return len(m) >= 2 or ("longest_leg_nm" in m and "budget_millions_usd" in m)


def _fit_score(mission: Dict[str, Any], marketing: str, jet_class: str) -> int:
    score = 48
    if mission.get("passengers"):
        score += 14
    if mission.get("longest_leg_nm"):
        score += 18
    if mission.get("budget_millions_usd"):
        score += 10
    if mission.get("usage"):
        score += 8
    if mission.get("typical_routes_hint"):
        score += 4

    leg = mission.get("longest_leg_nm")
    if leg and jet_class != "unknown":
        band = _CLASS_BAND_NM.get(jet_class, 3000.0)
        if leg > band * 1.08:
            score -= 28
        elif leg > band * 0.92:
            score += 6

    if not marketing.strip():
        score -= 25

    miss = mission.get("missing_fields") or []
    if len(miss) >= 3:
        score -= 14
    elif len(miss) >= 2:
        score -= 8

    return int(max(8, min(100, score)))


def _deal_score(
    *,
    valuation: Dict[str, Any],
    market_rows: List[Dict[str, Any]],
    marketing: str,
) -> int:
    if not marketing.strip():
        return 22
    est = valuation.get("estimated_value_millions")
    asks: List[float] = []
    for r in market_rows or []:
        ap = r.get("ask_price")
        if ap is not None:
            try:
                asks.append(float(ap))
            except (TypeError, ValueError):
                pass
    conf = int(valuation.get("confidence_pct") or 0)

    if est and asks:
        med = sorted(asks)[len(asks) // 2]
        ratio = (med / 1_000_000) / float(est) if est else 1.0
        base = 72
        if ratio < 0.88:
            base = 82
        elif ratio > 1.12:
            base = 48
        base += min(10, conf // 15)
        return int(max(12, min(100, base)))

    if est:
        return int(max(35, min(88, 52 + min(28, conf // 4))))

    if market_rows:
        return 52 + min(20, len(market_rows) // 3)

    return 38


def _risk_score(
    *,
    mission: Dict[str, Any],
    market_rows: List[Dict[str, Any]],
    valuation: Dict[str, Any],
    user_query: str,
) -> int:
    """Higher = lower risk / more confidence in the read (not 'danger level')."""
    score = 50
    low = (user_query or "").lower()
    if re.search(r"\b(high time|damage|incident|corrosion|missing logs|export)\b", low):
        score -= 22

    n = len(market_rows or [])
    if n >= 8:
        score += 16
    elif n >= 3:
        score += 10
    else:
        score -= 12

    conf = int(valuation.get("confidence_pct") or 0)
    if conf >= 75:
        score += 12
    elif conf <= 25 and valuation.get("estimated_value_millions") is None:
        score -= 14

    miss = len(mission.get("missing_fields") or [])
    score -= min(22, miss * 5)

    if valuation.get("error"):
        score -= 8

    return int(max(10, min(100, score)))


def _opcost_line(jet_class: str) -> str:
    lo, hi = _OPCOST_USD_PER_HR.get(jet_class, _OPCOST_USD_PER_HR["unknown"])
    return (
        f"Illustrative DOC band for this class: about USD {lo:,}–{hi:,}/flight hour "
        f"(high variance by cycle, fuel contract, crew, and hangar—treat as a briefing anchor, not a quote)."
    )


def _verdict(fit: int, deal: int, risk: int, mission_weak: bool) -> str:
    if fit < 40 or deal < 35 or risk < 30:
        return "PASS"
    if mission_weak and (fit < 62 or risk < 48):
        return "CONDITIONAL BUY"
    if fit >= 66 and deal >= 58 and risk >= 52:
        return "BUY"
    return "CONDITIONAL BUY"


def _insight_direct(
    *,
    aircraft: str,
    mission: Dict[str, Any],
    jet_class: str,
    fit: int,
    deal: int,
    risk: int,
    verdict: str,
    market_n: int,
    valuation_note: str,
    mission_weak: bool,
) -> str:
    parts: List[str] = []
    if not aircraft:
        parts.append("No recognizable aircraft model in the text—scores are mostly meaningless until you name a type or tail.")
    miss = mission.get("missing_fields") or []
    if miss:
        parts.append(
            f"Mission data is incomplete ({', '.join(miss)}); any buy call here is partially blind."
        )
    if jet_class != "unknown":
        band = _CLASS_BAND_NM.get(jet_class, 0)
        leg = mission.get("longest_leg_nm")
        if leg and band and leg > band * 1.05:
            parts.append(
                f"Stated leg (~{leg:.0f} nm) is tight for typical {jet_class.replace('_', ' ')} practical range—either step up in class or accept operational compromises."
            )

    if market_n == 0:
        parts.append("No synced listing comps returned for this filter—deal scoring leans on weaker evidence.")
    else:
        parts.append(f"Listing slice: {market_n} rows—use for positioning only, not serial-level diligence.")

    parts.append(valuation_note)
    parts.append(
        f"Scores — fit {fit}, deal {deal}, risk-quality {risk} (risk-quality higher = fewer blind spots in the data read). "
        f"Verdict: **{verdict}**."
    )
    if mission_weak:
        parts.append("You are not ready to wire money until longest leg, pax, and budget are pinned down.")
    return " ".join(parts)[:1600]


def _recommendation_line(verdict: str, mission_weak: bool, aircraft: str) -> str:
    if verdict == "PASS":
        return "Do not chase this path on the information given—either upgrade data or change aircraft class."
    if verdict == "BUY":
        return f"If {aircraft or 'the aircraft'} clears pre-buy and log review, proceed toward LOI with a sharp exit if findings drift."
    if mission_weak:
        return "Hold capital: answer the missing mission questions first, then rerun the decision with hard numbers."
    return "Proceed only with a structured pre-buy (engines, CPCP, back-to-birth records); the scores are supportive but not a substitute for inspection."


def run_aircraft_decision_engine(
    user_query: str,
    *,
    db: Any = None,
    region: Optional[str] = None,
    embedding_service: Any = None,
    pinecone_client: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate a buy decision from natural language + optional Hye Aero DB signals.

    Returns the brokerage JSON contract (``verdict``, scores, ``insight``, ``recommendation``, ``alternatives``).
    """
    raw = (user_query or "").strip()
    mission = extract_mission_profile(raw)
    marketing, mfr, mdl = resolve_target_aircraft(raw)
    jet_class = infer_jet_class(marketing)

    db = db if db is not None else _optional_db()
    market_rows: List[Dict[str, Any]] = []
    valuation: Dict[str, Any] = {
        "estimated_value_millions": None,
        "confidence_pct": 0,
        "message": None,
        "error": None,
    }

    if db is not None and mdl:
        try:
            mc = run_comparison(
                db=db,
                models=[mdl or marketing],
                region=region or "Global",
                limit=40,
            )
            if not mc.get("error"):
                market_rows = list(mc.get("rows") or [])
        except Exception as e:
            logger.debug("Market comparison skipped: %s", e)

    if db is not None and (mfr or mdl):
        emb, pine = embedding_service, pinecone_client
        if emb is None and pine is None:
            emb, pine = _optional_embedding_pinecone()
        try:
            valuation = estimate_value_hybrid(
                db=db,
                embedding_service=emb,
                pinecone_client=pine,
                manufacturer=mfr or None,
                model=mdl or marketing or None,
                region=region,
            )
        except Exception as e:
            logger.debug("Valuation skipped: %s", e)
            valuation = {**valuation, "error": str(e)}

    mission_weak = _mission_incomplete(mission)
    fit = _fit_score(mission, marketing, jet_class)
    deal = _deal_score(valuation=valuation, market_rows=market_rows, marketing=marketing)
    risk = _risk_score(
        mission=mission,
        market_rows=market_rows,
        valuation=valuation,
        user_query=raw,
    )

    if mission_weak:
        fit = min(fit, 58)
        risk = max(10, int(risk) - 6)

    v = _verdict(fit, deal, risk, mission_weak)

    est = valuation.get("estimated_value_millions")
    if est:
        vnote = f"Historical-sale anchor (noisy): roughly USD {float(est):.1f}M central estimate where comps exist—never treat as an appraisal."
    else:
        vnote = "No reliable valuation anchor from internal comps—assume ask/transaction numbers are unknown until you supply a serial or broker data."

    aircraft_out = marketing or mdl or "unknown aircraft"
    insight = _insight_direct(
        aircraft=aircraft_out,
        mission=mission,
        jet_class=jet_class,
        fit=fit,
        deal=deal,
        risk=risk,
        verdict=v,
        market_n=len(market_rows),
        valuation_note=vnote,
        mission_weak=mission_weak,
    )
    insight = f"{insight} {_opcost_line(jet_class)}"

    alts = _alternatives_for(aircraft_out)

    return {
        "aircraft": aircraft_out,
        "verdict": v,
        "fit_score": fit,
        "deal_score": deal,
        "risk_score": risk,
        "insight": insight.strip(),
        "recommendation": _recommendation_line(v, mission_weak, aircraft_out),
        "alternatives": alts,
        "_mission_profile": mission,
        "_jet_class": jet_class,
    }


def public_decision_payload(engine_out: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal keys for API responses."""
    return {k: v for k, v in engine_out.items() if not k.startswith("_")}


def consultant_query_requests_aircraft_decision(user_query: str) -> bool:
    """
    True when Ask Consultant should attach structured buy scores (tail/model + acquisition wording).
    """
    raw = (user_query or "").strip()
    if len(raw) < 10:
        return False
    low = raw.lower()
    buyish = any(
        p in low
        for p in (
            "worth buying",
            "worth it",
            "should i buy",
            "is it a good buy",
            "buy or pass",
            "is it worth",
        )
    )
    if not buyish:
        return False
    if re.search(r"\bN[A-Z0-9]{2,6}\b", raw, re.I):
        return True
    if _detect_models(raw) or _detect_manufacturers(low):
        return True
    return any(
        w in low
        for w in (
            "jet",
            "aircraft",
            "gulfstream",
            "citation",
            "falcon",
            "challenger",
            "phenom",
            "lear",
            "global",
            "bombardier",
            "cessna",
            "embraer",
            "pilatus",
        )
    )
