from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt


def build_summary(outputs: str = "outputs"):
    root = Path(outputs); rows = []
    for metric_path in root.glob("*/metrics.json"):
        data = json.loads(metric_path.read_text()); data["run_name"] = metric_path.parent.name; rows.append(data)
    if rows:
        keys = sorted({k for row in rows for k in row})
        with (root / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
        (root / "ablation_table.md").write_text("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n" + "\n".join("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |" for row in rows))
        (root / "summary.json").write_text(json.dumps(rows, indent=2))
    return rows


def render_pairs(dataset, pairs_csv: str, output_dir: str, limit: int = 16):
    rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8")))[:limit]; out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        q, _ = dataset[int(row["query_id"])]; c, _ = dataset[int(row["candidate_id"])]
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5)); ax[0].imshow(q.permute(1,2,0)); ax[1].imshow(c.permute(1,2,0))
        for a in ax: a.axis("off")
        fig.suptitle(f"q={row['query_label']} c={row['candidate_label']} score={float(row['selection_score']):.3f}")
        fig.savefig(out / f"pair_{i:03d}.png", bbox_inches="tight"); plt.close(fig)


def plot_reward_histogram(pairs_csv: str, output_path: str):
    rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8")))
    values = [float(row["reward"]) for row in rows if row.get("reward", "") != ""]
    if not values:
        return False
    plt.figure(figsize=(5, 3)); plt.hist(values, bins=30); plt.xlabel("detached task-aligned reward"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    return True
