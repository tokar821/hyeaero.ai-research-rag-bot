#!/usr/bin/env python3
"""
HACK v4 — generate adversarial test suite JSON for CI / stress harness.

Does NOT run inference, ranking, or orchestration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from evals.hack_v4_adversarial_generator import (  # noqa: E402
    generate_adversarial_suite,
    suite_summary,
)


def main() -> int:
    seed = 42
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            pass

    rows = generate_adversarial_suite(seed=seed, mutations_per_test=2)
    out_path = _ROOT / "evals" / "hack_v4_adversarial_suite.json"
    payload = {
        "suite": "hack_v4_adversarial",
        "version": 1,
        "description": "HACK v4 — adversarial mission prompts (generation only, no inference).",
        "seed": seed,
        "summary": suite_summary(rows),
        "tests": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} tests -> {out_path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
