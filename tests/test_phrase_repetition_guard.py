"""Global ban on repetitive consultant stock phrases."""

from services.consultant.phrase_repetition_guard import (
    analyze_phrase_repetition,
    apply_phrase_repetition_guard,
    applyPhraseRepetitionGuard,
)


def test_detects_repeat_from_prior_turn():
    prior = "The Praetor 600 is a credible alternate if priorities shift on longer legs."
    current = "I'd also keep the Gulfstream G280 as a credible alternate if priorities shift."
    report = analyze_phrase_repetition(current, prior_answer=prior)
    assert report.violations
    assert report.conversation_counts.get("credible alternate if priorities shift", 0) >= 2


def test_varies_phrase_when_prior_used_stock_language():
    prior = "Worth keeping on the list if the trip profile changes."
    current = (
        "Challenger 350 leads for LA to Miami. "
        "G280 is worth keeping on the list if you add cabin priority."
    )
    out, report = applyPhraseRepetitionGuard(
        current,
        prior_answer=prior,
        turn_seed="la-miami-2",
    )
    assert "worth keeping on the list" not in out.lower()
    assert "phrase_repetition" in " ".join(report.actions_taken) or report.phrases_varied


def test_strips_duplicate_within_same_response():
    text = (
        "G280 is a credible alternate if priorities shift. "
        "The Praetor 600 is also a credible alternate if priorities shift."
    )
    out, report = apply_phrase_repetition_guard(text, turn_seed="dup")
    assert out.lower().count("credible alternate if priorities shift") <= 1
    assert report.phrases_varied or report.phrases_stripped


def test_history_turns_count_toward_threshold():
    history = [
        {"role": "user", "content": "recommend a jet"},
        {
            "role": "assistant",
            "content": "Unless priorities shift, I'd start with the Challenger 350.",
        },
        {"role": "user", "content": "what about cabin?"},
    ]
    current = "Unless priorities shift, the G280 adds more cabin volume."
    report = analyze_phrase_repetition(current, history=history)
    assert any("unless priorities shift" in v for v in report.violations)


def test_regenerate_fn_used_when_still_violating():
    calls = {"n": 0}

    def regen() -> str:
        calls["n"] += 1
        return "On this leg, Challenger 350 leads; G280 is the backup if range tightens."

    current = "Unless priorities shift. Unless priorities shift again."
    out, report = apply_phrase_repetition_guard(
        current,
        turn_seed="regen-test",
        regenerate_fn=regen,
    )
    assert calls["n"] >= 1 or "unless priorities shift" not in out.lower()
    assert "regenerated_after_phrase_violation" in report.actions_taken or "unless priorities shift" not in out.lower()
