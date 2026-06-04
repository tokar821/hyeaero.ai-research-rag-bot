"""One-off QA eval for Phase 38 — not a permanent test file."""
from __future__ import annotations

import json

from services.adversarial.adversarial_preprocessor import (
    check_comparison_safety,
    preprocess_adversarial_query,
    try_adversarial_buy_block,
)
from services.adversarial.budget_conflict_normalizer import classify_price_signals
from services.adversarial.intent_sanitizer import _detect_intent_tags, sanitize_intents
from services.adversarial.model_adversary_resolver import resolve_adversary_models
from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison
from services.routing.authority_dispatch import consult_authority_dispatch, respond_buy_decision

QUERIES = [
    "cheap G700 under $5M",
    "Is a 2015 Citation Latitude for $5M a good deal?",
    "G650 vs longitude jet",
    "longitude vs phenom 300",
    "buy challenger 350 under $8M",
    "Gulfstream vs Dassault cheapest option",
    "$10M budget what jet should I buy",
    "cheap gulfstream for sale",
    "compare g700 vs g650",
    "I want something like longitude but cheaper",
]


def main() -> None:
    results = []
    for q in QUERIES:
        du: dict = {}
        clean = preprocess_adversarial_query(q, data_used=du)
        budget = clean.budget_state
        tags = sorted(_detect_intent_tags(q))
        intent_ov = sanitize_intents(q)
        buy_block = try_adversarial_buy_block(clean.normalized_query, du)
        cmp_block = check_comparison_safety(clean.normalized_query, du)
        ctx = {"db": None, "clean_normalized_query": clean.to_dict()}
        dispatch = consult_authority_dispatch(clean.normalized_query, context=ctx)
        buy_ans = ""
        if "good deal" in q.lower():
            buy_ans = respond_buy_decision(clean.normalized_query, db=None, data_used=du) or ""
        cmp_ans = ""
        if " vs " in q.lower() or q.lower().startswith("compare"):
            cmp_ans = respond_aircraft_comparison(
                clean.normalized_query, data_used={"db": None, **du}
            ) or ""
        results.append(
            {
                "query": q,
                "normalized": clean.normalized_query,
                "intent_tags": tags,
                "intent_override": intent_ov,
                "conflicts": [c.value for c in clean.conflict_report.conflict_type],
                "severity": clean.conflict_report.severity.value,
                "details": list(clean.conflict_report.details),
                "models": [
                    {
                        "name": m.canonical_model,
                        "conf": m.resolution_confidence,
                        "amb": m.ambiguity_type.value,
                    }
                    for m in clean.resolved_models
                ],
                "budget_feasibility": budget.feasibility.value,
                "acquisition_cap": budget.acquisition_cap_musd,
                "listing_ask": budget.listing_ask_musd,
                "signals": [
                    {"kind": s.kind.value, "amt": s.amount_musd} for s in budget.price_signals
                ],
                "buy_block": buy_block,
                "cmp_block": cmp_block,
                "dispatch_kind": dispatch.dispatch_kind if dispatch else None,
                "dispatch_answer": (dispatch.answer if dispatch else "")[:400],
                "buy_answer": buy_ans[:400],
                "cmp_answer": cmp_ans[:400],
                "adversarial": du.get("adversarial"),
            }
        )
    print(json.dumps(results, indent=2))


def final_responder_probe() -> None:
    from services.adversarial.adversarial_preprocessor import try_adversarial_buy_block

    probes = [
        ("cheap G700 under $5M", "block"),
        ("Is a 2015 Citation Latitude for $5M a good deal?", "buy"),
        ("G650 vs longitude jet", "cmp"),
        ("longitude vs phenom 300", "cmp"),
        ("buy challenger 350 under $8M", "buy"),
        ("compare g700 vs g650", "cmp"),
        ("cheap gulfstream for sale", "none"),
        ("$10M budget what jet should I buy", "none"),
    ]
    for q, kind in probes:
        du: dict = {}
        clean = preprocess_adversarial_query(q, data_used=du)
        nq = clean.normalized_query
        if kind == "block":
            out = try_adversarial_buy_block(nq, du) or "(no block)"
        elif kind == "buy":
            out = respond_buy_decision(nq, db=None, data_used=du) or "(empty)"
        elif kind == "cmp":
            out = respond_aircraft_comparison(nq, data_used={"db": None, **du}) or "(empty)"
        else:
            out = "(no dedicated responder in probe)"
        print(f"=== {q}")
        print(out[:350])
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        final_responder_probe()
    else:
        main()
