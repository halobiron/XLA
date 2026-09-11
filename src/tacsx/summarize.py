from __future__ import annotations
import argparse
from .analysis import build_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate completed TACS-X runs")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args(argv)
    rows = build_summary(args.outputs)
    print(f"wrote comparison artifacts for {len(rows)} runs")


if __name__ == "__main__": main()
