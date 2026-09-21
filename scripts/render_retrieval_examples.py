"""Render deterministic CIFAR-100 query/context examples from saved metadata."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from torchvision import datasets

from tacsx.analysis import render_pairs


def _select_rows(pairs_csv: Path, mode: str, limit: int) -> list[dict[str, str]]:
    with pairs_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    field = "reward" if mode in {"policy_only", "full_tacs"} else "selection_score"
    usable = [row for row in rows if row.get(field, "") != ""]
    # Show both favourable and unfavourable outcomes rather than an arbitrary
    # prefix of the test set. This is deterministic and recorded alongside PNGs.
    usable.sort(key=lambda row: float(row[field]), reverse=True)
    top_count = (limit + 1) // 2
    selected = usable[:top_count] + list(reversed(usable[-(limit - top_count):]))
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args(argv)

    output_root = Path(args.outputs)
    train = datasets.CIFAR100(args.data_root, train=True, download=False)
    test = datasets.CIFAR100(args.data_root, train=False, download=False)
    rendered = 0
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        pairs_csv = run_dir / "selected_pairs.csv"
        metrics_path = run_dir / "metrics.json"
        if not pairs_csv.exists() or not metrics_path.exists():
            continue
        import json
        mode = json.loads(metrics_path.read_text(encoding="utf-8"))["method"]
        rows = _select_rows(pairs_csv, mode, args.limit)
        selection_csv = run_dir / "retrieval_examples.csv"
        with selection_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        render_pairs(test, train, str(selection_csv), str(run_dir / "retrieval_examples"),
                     query_id_offset=len(train), limit=args.limit)
        rendered += 1
    print(f"rendered retrieval examples for {rendered} runs")


if __name__ == "__main__":
    main()
