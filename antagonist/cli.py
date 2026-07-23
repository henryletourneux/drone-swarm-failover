"""Standalone entrypoint: run the full adversarial campaign and print the report.

Usage: python3 -m antagonist.cli [--seed 7]
"""
from __future__ import annotations

import argparse

from .campaign import format_report, run_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the antagonist campaign against a BFT-mode swarm")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    outcomes, meta = run_campaign(seed=args.seed)
    print(format_report(outcomes, meta))


if __name__ == "__main__":
    main()
