"""
Global ban on repetitive consultant stock phrases.

Detects frequency in the current response and recent assistant turns; applies
linguistic variation or strips excess occurrences before user delivery.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Max occurrences of a banned phrase in (current + recent assistant turns)
CONVERSATION_PHRASE_LIMIT = 1
# Max per single response body
WITHIN_RESPONSE_LIMIT = 1

_RECENT_ASSISTANT_TURNS = 3

_BANNED_PHRASES: Tuple[Tuple[re.Pattern[str], str, Tuple[str, ...]], ...] = (
    (
        re.compile(r"\bcredible\s+alternate\s+if\s+priorities\s+shift\b", re.I),
        "credible alternate if priorities shift",
        (
            "solid backup if the mission changes",
            "a useful pivot if range or runway becomes the driver",
            "the alternate I'd pressure-test if legs get longer",
        ),
    ),
    (
        re.compile(r"\bworth\s+keeping\s+on\s+the\s+list\b", re.I),
        "worth keeping on the list",
        (
            "still worth a desk slot if constraints change",
            "one to revisit if the trip profile widens",
            "keep in the mix if passengers or bags grow",
        ),
    ),
    (
        re.compile(r"\bhow\s+you'?re\s+actually\s+using\s+the\s+airplane\b", re.I),
        "how you're actually using the airplane",
        (
            "your typical mission pattern",
            "how you fly the airplane week to week",
            "the way you use the aircraft operationally",
        ),
    ),
    (
        re.compile(r"\bframe\s+the\s+mission\s+before\s+anchoring\b", re.I),
        "frame the mission before anchoring",
        (
            "set the mission profile before you pick a tail number",
            "lock the trip shape before you commit to one airframe",
            "define the mission first, then narrow to a model",
        ),
    ),
    (
        re.compile(r"\bunless\s+(?:your\s+)?priorities\s+shift\b", re.I),
        "unless priorities shift",
        (
            "if the mission changes",
            "when your priorities change",
            "if runway or cabin becomes the deciding factor instead",
        ),
    ),
    (
        re.compile(r"\bnet:\s*.+\s+unless\s+(?:your\s+)?priorities\s+shift\b", re.I),
        "net: unless priorities shift",
        (
            "if the mission changes, revisit the alternates",
            "when priorities change, the alternate deserves another look",
        ),
    ),
)


@dataclass
class PhraseRepetitionReport:
    violations: List[str] = field(default_factory=list)
    phrases_varied: List[str] = field(default_factory=list)
    phrases_stripped: List[str] = field(default_factory=list)
    conversation_counts: Dict[str, int] = field(default_factory=dict)
    needs_regenerate: bool = False
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "violations": list(self.violations),
            "phrases_varied": list(self.phrases_varied),
            "phrases_stripped": list(self.phrases_stripped),
            "conversation_counts": dict(self.conversation_counts),
            "needs_regenerate": self.needs_regenerate,
            "actions_taken": list(self.actions_taken),
        }


def _stable_pick(options: Sequence[str], seed: str) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _assistant_history_blob(
    history: Optional[List[Dict[str, str]]],
    *,
    prior_answer: str = "",
    max_turns: int = _RECENT_ASSISTANT_TURNS,
) -> str:
    parts: List[str] = []
    if history:
        assistants = [
            str(t.get("content") or "")
            for t in history
            if isinstance(t, dict) and str(t.get("role", "")).lower() == "assistant"
        ]
        for content in assistants[-max_turns:]:
            if content.strip():
                parts.append(content)
    if prior_answer.strip() and prior_answer not in parts:
        parts.append(prior_answer)
    return "\n\n".join(parts)


def count_phrase_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text or ""))


def analyze_phrase_repetition(
    text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    prior_answer: str = "",
) -> PhraseRepetitionReport:
    """Count banned phrase usage across current text and recent assistant turns."""
    report = PhraseRepetitionReport()
    current = text or ""
    hist_blob = _assistant_history_blob(history, prior_answer=prior_answer)
    corpus = f"{hist_blob}\n\n{current}".strip()

    for pattern, label, _alts in _BANNED_PHRASES:
        in_current = count_phrase_matches(current, pattern)
        in_history = count_phrase_matches(hist_blob, pattern)
        total = count_phrase_matches(corpus, pattern)
        report.conversation_counts[label] = total

        if in_current > WITHIN_RESPONSE_LIMIT:
            report.violations.append(f"{label}: {in_current}x in current response")
        if total > CONVERSATION_PHRASE_LIMIT and in_current > 0:
            report.violations.append(f"{label}: {total}x in conversation window")
        if in_history > 0 and in_current > 0:
            report.violations.append(f"{label}: repeated from prior assistant turn")

    if len(report.violations) >= 3:
        report.needs_regenerate = True
    return report


def _vary_or_strip_sentence(sentence: str, seed: str) -> str:
    """Replace banned fragments in one sentence; drop sentence if nothing remains."""
    out = sentence
    for pattern, _label, alts in _BANNED_PHRASES:
        if pattern.search(out):
            replacement = _stable_pick(alts, f"{seed}:{_label}")
            out = pattern.sub(replacement, out, count=1)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) < 12:
        return ""
    return out


def _apply_variation_to_text(
    text: str,
    *,
    seed: str,
    force_all: bool,
    history_had: Dict[str, bool],
) -> Tuple[str, List[str], List[str]]:
    varied: List[str] = []
    stripped: List[str] = []
    paragraphs = re.split(r"\n\s*\n", (text or "").strip())
    new_paragraphs: List[str] = []

    for pi, para in enumerate(paragraphs):
        if not para.strip():
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        kept: List[str] = []
        seen_in_para: Dict[str, int] = {}

        for si, sent in enumerate(sentences):
            if not sent.strip():
                continue
            matched_labels: List[str] = []
            for pattern, label, _alts in _BANNED_PHRASES:
                if pattern.search(sent):
                    matched_labels.append(label)

            if not matched_labels:
                kept.append(sent.strip())
                continue

            for label in matched_labels:
                seen_in_para[label] = seen_in_para.get(label, 0) + 1

            replace = force_all or any(history_had.get(lbl) for lbl in matched_labels)
            replace = replace or any(
                seen_in_para.get(lbl, 0) > WITHIN_RESPONSE_LIMIT for lbl in matched_labels
            )

            if replace:
                updated = _vary_or_strip_sentence(sent, f"{seed}:{pi}:{si}")
                if updated:
                    kept.append(updated)
                    varied.extend(matched_labels)
                else:
                    stripped.extend(matched_labels)
            else:
                kept.append(sent.strip())

        if kept:
            new_paragraphs.append(" ".join(kept))

    return "\n\n".join(new_paragraphs).strip(), varied, stripped


def apply_phrase_repetition_guard(
    text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    prior_answer: str = "",
    turn_seed: str = "",
    regenerate_fn: Optional[Callable[[], str]] = None,
) -> Tuple[str, PhraseRepetitionReport]:
    """
    Remove or vary banned phrases when frequency exceeds threshold.

    If ``regenerate_fn`` is provided and violations remain severe, call it once.
    """
    s = (text or "").strip()
    if not s:
        return s, PhraseRepetitionReport()

    report = analyze_phrase_repetition(s, history=history, prior_answer=prior_answer)
    if not report.violations:
        return s, report

    hist_blob = _assistant_history_blob(history, prior_answer=prior_answer)
    history_had: Dict[str, bool] = {}
    for pattern, label, _alts in _BANNED_PHRASES:
        history_had[label] = count_phrase_matches(hist_blob, pattern) > 0

    seed = turn_seed or "phrase_guard"
    s, varied, stripped = _apply_variation_to_text(
        s,
        seed=seed,
        force_all=bool(report.violations),
        history_had=history_had,
    )
    report.phrases_varied = list(dict.fromkeys(varied))
    report.phrases_stripped = list(dict.fromkeys(stripped))
    report.actions_taken.append("applied_linguistic_variation")

    # Re-check; regenerate once if still over threshold
    recheck = analyze_phrase_repetition(s, history=history, prior_answer=prior_answer)
    if recheck.violations and regenerate_fn is not None:
        try:
            regen = (regenerate_fn() or "").strip()
            if regen:
                s = regen
                report.actions_taken.append("regenerated_after_phrase_violation")
                report.needs_regenerate = False
                recheck = analyze_phrase_repetition(s, history=history, prior_answer=prior_answer)
        except Exception:
            pass

    if recheck.violations:
        report.violations = recheck.violations
        report.needs_regenerate = recheck.needs_regenerate or report.needs_regenerate
        # Last resort: strip any sentence still containing banned patterns
        parts = re.split(r"(?<=[.!?])\s+", s)
        filtered: List[str] = []
        for sent in parts:
            if any(pat.search(sent) for pat, _, _ in _BANNED_PHRASES):
                report.phrases_stripped.append("sentence_removed")
                report.actions_taken.append("stripped_sentence_with_banned_phrase")
                continue
            filtered.append(sent)
        s = " ".join(filtered).strip()
        s = re.sub(r"\n{3,}", "\n\n", s)

    return s, report


def applyPhraseRepetitionGuard(
    text: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    prior_answer: str = "",
    turn_seed: str = "",
    regenerate_fn: Optional[Callable[[], str]] = None,
) -> Tuple[str, PhraseRepetitionReport]:
    """Public camelCase alias for ``apply_phrase_repetition_guard``."""
    return apply_phrase_repetition_guard(
        text,
        history=history,
        prior_answer=prior_answer,
        turn_seed=turn_seed,
        regenerate_fn=regenerate_fn,
    )
