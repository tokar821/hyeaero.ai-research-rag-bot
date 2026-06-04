"""
Generate Phase 32 production validation corpus and golden expectations.

Run: python -m tests.production_validation.generate_corpus
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any, Dict, List

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

COMPARE_MODELS = [
    "G650", "Falcon 8X", "Global 7500", "G700", "Longitude", "Challenger 3500",
    "Citation CJ3+", "Citation CJ4", "Praetor 600", "PC-24", "G550", "G280",
    "Challenger 650", "Global 6500", "Falcon 7X", "Citation Latitude",
]

BUY_MODELS = [
    "Citation Latitude", "Challenger 350", "Challenger 3500", "Citation Longitude",
    "Gulfstream G280", "Praetor 600", "Citation CJ3+", "PC-24", "Falcon 2000",
    "Legacy 650", "Citation CJ4", "Gulfstream G550",
]

MISSION_ROUTES = [
    ("TEB", "LAX", "8"), ("TEB", "MIA", "6"), ("London", "Singapore", "12"),
    ("NYC", "Paris", "8"), ("Van Nuys", "Aspen", "4"), ("Dallas", "New York", "6"),
    ("Chicago", "Miami", "8"), ("Boston", "West Palm Beach", "6"), ("Seattle", "Honolulu", "10"),
    ("Los Angeles", "Tokyo", "12"), ("Geneva", "Dubai", "8"), ("Teterboro", "London", "10"),
]

ALTERNATIVE_TARGETS = [
    "G650", "Falcon 8X", "Global 7500", "Longitude", "Challenger 3500",
    "PC-24", "Praetor 600", "Citation Latitude", "G280", "G550", "CJ3+", "Global 6500",
]

VALUATION_MODELS = [
    ("2019", "Falcon 8X"), ("2017", "Citation Longitude"), ("2018", "Challenger 3500"),
    ("2020", "Praetor 600"), ("2016", "Citation Latitude"), ("2019", "G650"),
    ("2018", "PC-24"), ("2017", "G280"), ("2021", "Global 7500"), ("2015", "Falcon 7X"),
]


def _comparison_queries(n: int = 100) -> List[Dict[str, str]]:
    pairs = list(itertools.combinations(COMPARE_MODELS[:12], 2))
    templates = [
        "{a} vs {b}",
        "Compare {a} and {b}",
        "How does {a} compare to {b}?",
        "{a} versus {b} for charter operations",
        "Which is better, {a} or {b}?",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for a, b in pairs:
        for tpl in templates:
            if len(out) >= n:
                return out
            out.append({
                "id": f"cmp-{idx+1:03d}",
                "category": "comparison",
                "query": tpl.format(a=a, b=b),
            })
            idx += 1
    while len(out) < n:
        a, b = pairs[len(out) % len(pairs)]
        out.append({
            "id": f"cmp-{len(out)+1:03d}",
            "category": "comparison",
            "query": f"{a} vs {b}",
        })
    return out


def _buy_queries(n: int = 100) -> List[Dict[str, str]]:
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    prices = [5, 6, 8, 10, 12, 15, 18, 20, 25, 30]
    templates = [
        "Is a {year} {model} for ${price}M a good deal?",
        "{year} {model} at ${price}M — fair price?",
        "Is this {year} {model} for ${price}M overpriced?",
        "Should I buy a {year} {model} listed at ${price}M?",
        "{year} {model} ${price}M good buy?",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for model in BUY_MODELS:
        for year in years:
            for price in prices:
                if len(out) >= n:
                    return out
                tpl = templates[idx % len(templates)]
                out.append({
                    "id": f"buy-{idx+1:03d}",
                    "category": "buy_decision",
                    "query": tpl.format(year=year, model=model, price=price),
                })
                idx += 1
    return out[:n]


def _mission_queries(n: int = 100) -> List[Dict[str, str]]:
    budgets = ["", " under $15M", " under $25M", " under $10M"]
    templates = [
        "{pax} pax {origin}-{dest}{budget}",
        "Need {pax} passengers {origin} to {dest} nonstop{budget}",
        "Mission: {pax} pax from {origin} to {dest}{budget}",
        "What jet for {pax} pax {origin}-{dest}{budget}?",
        "{pax} passengers {origin} to {dest} nonstop{budget}",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for origin, dest, pax in MISSION_ROUTES:
        for budget in budgets:
            for tpl in templates:
                if len(out) >= n:
                    return out
                out.append({
                    "id": f"msn-{idx+1:03d}",
                    "category": "mission",
                    "query": tpl.format(pax=pax, origin=origin, dest=dest, budget=budget),
                })
                idx += 1
    while len(out) < n:
        r = MISSION_ROUTES[len(out) % len(MISSION_ROUTES)]
        out.append({
            "id": f"msn-{len(out)+1:03d}",
            "category": "mission",
            "query": f"{r[2]} pax {r[0]}-{r[1]}",
        })
    return out[:n]


def _alternative_queries(n: int = 100) -> List[Dict[str, str]]:
    templates = [
        "Alternatives to {target}",
        "Show me alternatives to {target}",
        "What are tier-peer alternatives to {target}?",
        "Replacement options for {target}",
        "Similar aircraft to {target}",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for target in ALTERNATIVE_TARGETS:
        for tpl in templates:
            if len(out) >= n:
                return out
            out.append({
                "id": f"alt-{idx+1:03d}",
                "category": "alternative",
                "query": tpl.format(target=target),
            })
            idx += 1
    while len(out) < n:
        t = ALTERNATIVE_TARGETS[len(out) % len(ALTERNATIVE_TARGETS)]
        out.append({
            "id": f"alt-{len(out)+1:03d}",
            "category": "alternative",
            "query": f"Alternatives to {t}",
        })
    return out[:n]


def _valuation_queries(n: int = 100) -> List[Dict[str, str]]:
    templates = [
        "What is a {year} {model} worth?",
        "Estimate market value of a {year} {model}",
        "What is the value of a {year} {model}?",
        "{year} {model} valuation",
        "How much is a {year} {model} worth today?",
    ]
    out: List[Dict[str, str]] = []
    idx = 0
    for year, model in VALUATION_MODELS:
        for tpl in templates:
            if len(out) >= n:
                return out
            out.append({
                "id": f"val-{idx+1:03d}",
                "category": "valuation",
                "query": tpl.format(year=year, model=model),
            })
            idx += 1
    extra_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for model in BUY_MODELS:
        for year in extra_years:
            if len(out) >= n:
                return out
            out.append({
                "id": f"val-{len(out)+1:03d}",
                "category": "valuation",
                "query": f"What is a {year} {model} worth?",
            })
    return out[:n]


def build_corpus() -> Dict[str, Any]:
    queries = (
        _comparison_queries(100)
        + _buy_queries(100)
        + _mission_queries(100)
        + _alternative_queries(100)
        + _valuation_queries(100)
    )
    return {"version": "1", "total": len(queries), "queries": queries}


def build_golden_expectations(queries: List[Dict[str, str]]) -> Dict[str, Any]:
    from services.core.semantic_intent_lock_engine import build_intent_lock
    from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
    from services.routing.authority_dispatch import consult_authority_dispatch
    from services.routing.unified_intent_router import classify_unified_intent

    expectations: Dict[str, Any] = {}
    intent_map = {
        "comparison": "comparison",
        "buy_decision": "buy_decision",
        "alternative": "alternative",
        "valuation": "valuation",
        "mission": "mission",
    }

    for row in queries:
        qid = row["id"]
        query = row["query"]
        category = row["category"]
        qri = classify_query_recommendation_intent(query, [])
        route = classify_unified_intent(query)
        lock = build_intent_lock(query, qri=qri, unified_route=route)
        dispatch = consult_authority_dispatch(
            query, qri=qri, unified_route=route, context={"db": None, "intent_lock": lock}
        )

        expected_intent = intent_map.get(category, lock.intent_type)
        if category == "valuation":
            expected_intent = "valuation"
        elif category == "buy_decision":
            expected_intent = "buy_decision"

        if dispatch is not None:
            path = "authority_dispatch"
            dispatch_kind = dispatch.dispatch_kind
            models = list((dispatch.data_used or {}).get("authority_dispatch_models") or [])
            fail_closed = bool((dispatch.data_used or {}).get("authority_dispatch_safety_fallback"))
            allow_fail_closed = fail_closed or category == "mission"
        else:
            path = "llm_fallback" if category == "mission" else "hybrid_unified"
            dispatch_kind = None
            models = list(lock.canonical_models)
            allow_fail_closed = category in ("mission",)

        expectations[qid] = {
            "expected_intent": expected_intent,
            "expected_execution_path": path,
            "expected_dispatch_kind": dispatch_kind,
            "expected_models": models if models else list(lock.canonical_models),
            "allow_fail_closed": allow_fail_closed,
            "lock_intent_type": lock.intent_type,
        }
    return {"version": "1", "total": len(expectations), "expectations": expectations}


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    golden = build_golden_expectations(corpus["queries"])

    (FIXTURES_DIR / "production_queries.json").write_text(
        json.dumps(corpus, indent=2), encoding="utf-8"
    )
    (FIXTURES_DIR / "golden_expectations.json").write_text(
        json.dumps(golden, indent=2), encoding="utf-8"
    )
    print(f"Wrote {corpus['total']} queries and {golden['total']} golden expectations to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
