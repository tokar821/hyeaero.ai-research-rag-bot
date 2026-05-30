"""
Golden test cases built from verified catalog aircraft only.

All models in AIRCRAFT_NAMES exist in AIRCRAFT_PROFILES / verified spec repository.
"""

from __future__ import annotations

from typing import List

from evaluation.golden_dataset import GoldenTestCase

# Verified catalog aircraft — do not add names not in AIRCRAFT_PROFILES.
AIRCRAFT_NAMES: List[str] = [
    "Challenger 350",
    "Challenger 650",
    "Challenger Longitude",
    "Citation CJ2",
    "Citation CJ4",
    "Citation Latitude",
    "Falcon 2000",
    "Falcon 7X",
    "Falcon 8X",
    "Global 6500",
    "Global 7500",
    "Gulfstream G280",
    "Gulfstream G650",
    "Gulfstream G650ER",
    "Learjet 75",
    "Pilatus PC-12",
    "Pilatus PC-24",
    "Praetor 600",
]

_FACT_TEMPLATES = [
    ("seats", "How many seats does a {model} have?", ["factual_only", "no_mission_synthesis", "broker_style"]),
    ("baggage", "What is the baggage capacity of a {model}?", ["factual_only", "no_mission_synthesis", "broker_style"]),
    ("range", "What is the range of a {model}?", ["factual_only", "no_mission_synthesis", "broker_style"]),
    ("runway", "What runway length does a {model} need?", ["factual_only", "no_mission_synthesis", "broker_style"]),
    ("speed", "What is the maximum speed of a {model}?", ["factual_only", "no_mission_synthesis", "broker_style"]),
]

_MARKET_TEMPLATES = [
    ("worth", "What is a {model} worth?", ["market_price_band", "no_mission_synthesis", "broker_style"]),
    ("cost", "How much does a used {model} cost?", ["market_price_band", "no_mission_synthesis", "broker_style"]),
    ("value", "What is the market value of a {model}?", ["market_price_band", "no_mission_synthesis", "broker_style"]),
    ("sell", "What does a {model} sell for?", ["market_price_band", "no_mission_synthesis", "broker_style"]),
    ("price", "What is the average price of a {model}?", ["market_price_band", "no_mission_synthesis", "broker_style"]),
]

_CAPABILITY_ROUTES = [
    ("Can a {model} fly New York to London nonstop?", ["capability_yes_no", "no_mission_synthesis", "broker_style"]),
    ("Can a {model} fly Los Angeles to Hawaii?", ["capability_yes_no", "no_mission_synthesis", "broker_style"]),
    ("Can a {model} fly Miami to Paris nonstop?", ["capability_yes_no", "no_mission_synthesis", "broker_style"]),
    ("Can a {model} fly SFO to Tokyo?", ["capability_yes_no", "no_mission_synthesis", "broker_style"]),
    ("Is a {model} capable of NYC to LA nonstop?", ["capability_yes_no", "no_mission_synthesis", "broker_style"]),
]

_COMPARISON_PAIRS = [
    ("Falcon 8X", "Gulfstream G650"),
    ("Praetor 600", "Citation Latitude"),
    ("Challenger 650", "Gulfstream G280"),
    ("Global 7500", "Gulfstream G650ER"),
    ("Citation CJ4", "Learjet 75"),
    ("Falcon 7X", "Falcon 8X"),
    ("Challenger 350", "Praetor 600"),
    ("Global 6500", "Challenger Longitude"),
    ("Pilatus PC-24", "Citation CJ2"),
    ("Gulfstream G280", "Citation Latitude"),
    ("Falcon 2000", "Challenger 350"),
    ("Global 7500", "Falcon 8X"),
    ("Gulfstream G650", "Global 6500"),
    ("Citation Latitude", "Praetor 600"),
    ("Learjet 75", "Citation CJ4"),
    ("Pilatus PC-12", "Pilatus PC-24"),
    ("Challenger Longitude", "Citation Latitude"),
    ("Gulfstream G650ER", "Global 7500"),
    ("Falcon 7X", "Gulfstream G650"),
    ("Challenger 650", "Challenger 350"),
    ("Citation CJ2", "Citation CJ4"),
    ("Global 6500", "Gulfstream G650"),
    ("Falcon 2000", "Learjet 75"),
    ("Praetor 600", "Gulfstream G280"),
    ("Challenger 350", "Citation CJ4"),
]

_MISSION_TEMPLATES = [
    "Recommend the best jet for New York to London weekly",
    "What aircraft fits a transatlantic executive mission from Teterboro to Paris?",
    "I need a jet for Miami to Geneva nonstop — what should I consider?",
    "Size a fleet for NYC to Los Angeles with 8 passengers",
    "Which jet is best for Chicago to Aspen winter operations?",
    "Recommend aircraft for Dubai to London regular service",
    "What jet works for Boston to San Francisco with full cabin?",
    "Help me plan acquisition for Tokyo to Singapore routes",
    "Best options for Dallas to New York daily shuttle mission",
    "What aircraft category fits Seattle to Miami nonstop?",
    "Recommend a jet for cross-country US missions with 6 pax",
    "Which platform for Hong Kong to Sydney executive travel?",
    "Mission plan: Geneva to New York, 10 passengers, monthly",
    "What jet for Aspen to Teterboro with short runway constraints?",
    "Recommend for multi-city US tour: LA, NYC, Miami",
    "Best jet for London to Dubai year-round dispatch",
    "Size aircraft for Paris to Riyadh executive corridor",
    "What fits São Paulo to Miami nonstop mission profile?",
    "Recommend jet for West Coast to Hawaii weekly service",
    "Which aircraft for transcontinental US with baggage priority?",
    "Best jet for executive team NYC to London and Paris",
    "Mission: Singapore to Tokyo, need dependable dispatch",
    "Recommend platform for Chicago to London nonstop",
    "What jet for Mexico City to New York mission?",
    "Size jet for Toronto to Palm Beach winter missions",
]

_BUY_TEMPLATES = [
    "Should I buy a Gulfstream G650 or lease for my mission?",
    "What jet should I acquire under $25M for US coast-to-coast?",
    "Is a Challenger 650 a good acquisition at current prices?",
    "Buy vs charter for Falcon 8X transatlantic use",
    "Acquisition advice: Praetor 600 vs Citation Latitude purchase",
    "Should we purchase a Global 7500 for our flight department?",
    "What is the best jet to buy for NYC to London?",
    "Budget $18M — which pre-owned jet should we acquire?",
    "Is now a good time to buy a Citation Latitude?",
    "Acquisition summary for Learjet 75 under $15M",
    "Should I buy new or pre-owned Challenger 350?",
    "Purchase decision: Gulfstream G280 vs Praetor 600",
    "What jet acquisition makes sense for 400 hours/year?",
    "Buy a Falcon 7X or step up to Falcon 8X?",
    "Acquisition budget $30M — recommend aircraft to purchase",
    "Should our company buy a Global 6500?",
    "Pre-owned vs new Citation CJ4 acquisition tradeoffs",
    "Is a Pilatus PC-24 worth buying for regional missions?",
    "Acquisition guidance for first-time jet buyer, $12M budget",
    "Should I acquire a Gulfstream G650ER for ultra-long missions?",
    "Buy decision: Challenger Longitude vs Citation Latitude",
    "What aircraft should we procure for executive travel?",
    "Purchase analysis for Falcon 2000 in today's market",
    "Should I buy fractional or full ownership of a Praetor 600?",
    "Acquisition recommendation under $20M for US missions",
]


def _build_fact_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    idx = 0
    for model in AIRCRAFT_NAMES:
        for field, template, tags in _FACT_TEMPLATES:
            idx += 1
            if idx > 50:
                return cases
            cases.append(
                GoldenTestCase(
                    id=f"fact-{idx:03d}",
                    category="FACT",
                    query=template.format(model=model),
                    expected_execution_path="aircraft_fact",
                    expected_models=[model],
                    expected_behavior_tags=list(tags),
                )
            )
    return cases


def _build_market_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    idx = 0
    for model in AIRCRAFT_NAMES:
        for _field, template, tags in _MARKET_TEMPLATES:
            idx += 1
            if idx > 25:
                return cases
            cases.append(
                GoldenTestCase(
                    id=f"market-{idx:03d}",
                    category="MARKET",
                    query=template.format(model=model),
                    expected_execution_path="aircraft_market_fact",
                    expected_models=[model],
                    expected_behavior_tags=list(tags),
                )
            )
    return cases


def _build_capability_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    idx = 0
    for model in AIRCRAFT_NAMES:
        for template, tags in _CAPABILITY_ROUTES:
            idx += 1
            if idx > 25:
                return cases
            cases.append(
                GoldenTestCase(
                    id=f"capability-{idx:03d}",
                    category="CAPABILITY",
                    query=template.format(model=model),
                    expected_execution_path="capability",
                    expected_models=[model],
                    expected_behavior_tags=list(tags),
                )
            )
    return cases


def _build_comparison_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    queries = [
        "{a} vs {b} — which has more range?",
        "Compare {a} and {b} on cabin size and range",
        "{a} versus {b} — which is more efficient?",
        "How does {a} compare to {b} on range?",
        "Which has longer range, {a} or {b}?",
    ]
    for i, (a, b) in enumerate(_COMPARISON_PAIRS):
        q = queries[i % len(queries)].format(a=a, b=b)
        cases.append(
            GoldenTestCase(
                id=f"comparison-{i + 1:03d}",
                category="COMPARISON",
                query=q,
                expected_execution_path="comparison",
                expected_models=[a, b],
                expected_behavior_tags=["comparison_only", "no_mission_synthesis", "broker_style"],
            )
        )
    return cases


def _build_alternative_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    templates = [
        "What are alternatives to a {model}?",
        "What aircraft should I consider instead of a {model}?",
        "Credible replacements for a {model}",
        "Alternatives to the {model} in the same tier",
        "What are lower-cost alternatives to a {model}?",
    ]
    for i in range(25):
        model = AIRCRAFT_NAMES[i % len(AIRCRAFT_NAMES)]
        cases.append(
            GoldenTestCase(
                id=f"alternative-{i + 1:03d}",
                category="ALTERNATIVE",
                query=templates[i % len(templates)].format(model=model),
                expected_execution_path="alternative",
                expected_models=[model],
                expected_behavior_tags=["alternative_only", "no_mission_synthesis", "broker_style"],
            )
        )
    return cases


def _build_mission_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    for i, query in enumerate(_MISSION_TEMPLATES):
        cases.append(
            GoldenTestCase(
                id=f"mission-{i + 1:03d}",
                category="MISSION",
                query=query,
                expected_execution_path="none",
                expected_models=[],
                expected_behavior_tags=["no_mission_synthesis"],
            )
        )
    return cases


def _build_buy_decision_cases() -> List[GoldenTestCase]:
    cases: List[GoldenTestCase] = []
    for i, query in enumerate(_BUY_TEMPLATES):
        cases.append(
            GoldenTestCase(
                id=f"buy-{i + 1:03d}",
                category="BUY_DECISION",
                query=query,
                expected_execution_path="none",
                expected_models=[],
                expected_behavior_tags=["no_mission_synthesis", "broker_style"],
            )
        )
    return cases


def build_golden_cases() -> List[GoldenTestCase]:
    """Build full golden dataset (175+ cases) from verified catalog aircraft."""
    cases: List[GoldenTestCase] = []
    cases.extend(_build_fact_cases())
    cases.extend(_build_market_cases())
    cases.extend(_build_capability_cases())
    cases.extend(_build_comparison_cases())
    cases.extend(_build_alternative_cases())
    cases.extend(_build_mission_cases())
    cases.extend(_build_buy_decision_cases())
    return cases
