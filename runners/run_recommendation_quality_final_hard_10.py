"""Recommendation Quality FINAL HARD 10 — convenience runner."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_SUITE = _BACKEND / "evals" / "recommendation_quality_final_hard_10_suite.json"


def main() -> int:
    argv = ["run_recommendation_quality_final_hard_10.py", "--suite", str(_SUITE)]
    argv.extend(a for a in sys.argv[1:] if not a.startswith("--suite"))
    sys.argv = argv
    from runners.run_recommendation_quality_10 import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
