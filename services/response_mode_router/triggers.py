"""Deterministic trigger patterns per response mode."""

from __future__ import annotations

import re

IMAGE_SHOWCASE_RE = re.compile(
    r"\b("
    r"show\s+me|show\s+us|let\s+me\s+see|can\s+i\s+see|"
    r"photos?|pictures?|images?|gallery|what\s+does\s+it\s+look\s+like|"
    r"interior|interiors|in\s+the\s+cabin|cabin|cockpit|flight\s+deck|"
    r"bedroom\s+setup|bedroom|berth|motion\b|divan|"
    r"ambient\s+light(?:ing)?|modern\s+cabin|luxury\s+cabin|"
    r"premium\s+aesthetic|hotel\s+vibe|huge\s+windows|white\s+interior"
    r")\b",
    re.I,
)

FOLLOWUP_RE = re.compile(
    r"\b("
    r"actually\b.*\b(bigger|larger)|(?:something\s+)?bigger|larger|step\s*up|"
    r"more\s+modern|less\s+corporate|younger\s+feeling|"
    r"cheaper|less\s+expensive|tighter\s+budget|"
    r"wow\s+factor|more\s+like\s+that|not\s+that\s+old|"
    r"same\s+(?:jet|plane|cabin)|that\s+one"
    r")\b",
    re.I,
)

COMPARISON_RE = re.compile(
    r"\b(compare|compared\s+to|versus|difference\s+between|which\s+(?:is\s+)?better)\b",
    re.I,
)
VS_MODEL_RE = re.compile(
    r"\b(citation|gulfstream|falcon|challenger|global|lear|embraer|phenom|latitude)\b[^.?]{0,50}\bvs\.?\b",
    re.I,
)

EDUCATIONAL_RE = re.compile(
    r"\b(explain|how\s+does|how\s+do|why\s+(?:do|does|is|are)|what\s+is\s+the\s+difference|teach\s+me|walk\s+me\s+through)\b",
    re.I,
)

ADVISORY_RE = re.compile(
    r"\b("
    r"should\s+i\s+buy|worth\s+it|worth\s+buying|best\s+option|best\s+pick|"
    r"which\s+should\s+i\s+(?:buy|choose)|recommend|good\s+fit|"
    r"what\s+would\s+you\s+(?:buy|choose)|help\s+me\s+(?:pick|choose)"
    r")\b",
    re.I,
)

DEAL_RE = re.compile(
    r"\b(good\s+deal|overpriced|fair\s+price|market\s+value|would\s+you\s+buy\s+this)\b",
    re.I,
)

CONVERSATION_ONLY_RE = re.compile(
    r"^\s*(hi\b|hello\b|hey\b|thanks!?|thank\s+you!?)[\s!.,?]*$",
    re.I,
)
