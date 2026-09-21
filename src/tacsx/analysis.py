from __future__ import annotations
import csv, json
from pathlib import Path
import torch
from PIL import Image, ImageDraw


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


def render_pairs(query_dataset, candidate_dataset, pairs_csv: str, output_dir: str,
                 query_id_offset: int = 0, limit: int = 16, sort_by: str | None = None):
    """Render selected query/context pairs using their datasets' global ID mapping.

    `query_id_offset` maps a global query ID in the metadata to the local query
    dataset index. CIFAR-100 test metadata, for example, starts at 50,000 while
    the test dataset itself is indexed from zero.
    """
    with open(pairs_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sort_by:
        rows.sort(key=lambda row: float(row.get(sort_by) or 0.0), reverse=True)
    rows = rows[:limit]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        query_index = int(row["query_id"]) - query_id_offset
        candidate_ids = json.loads(row["topk_candidate_ids"]) if row.get("topk_candidate_ids") else [int(row["candidate_id"])]
        candidate_labels = json.loads(row["topk_candidate_labels"]) if row.get("topk_candidate_labels") else [int(row["candidate_label"])]
        weights = json.loads(row["topk_weights"]) if row.get("topk_weights") else [1.0]
        if not 0 <= query_index < len(query_dataset):
            raise IndexError(f"query_id {row['query_id']} is outside the supplied query dataset")
        if any(not 0 <= candidate_id < len(candidate_dataset) for candidate_id in candidate_ids):
            raise IndexError("a candidate_id is outside the supplied candidate dataset")
        query, _ = query_dataset[query_index]
        contexts = [candidate_dataset[candidate_id][0] for candidate_id in candidate_ids]
        # Raw PIL images and [C,H,W] tensors are both supported.
        if isinstance(query, torch.Tensor):
            query = Image.fromarray((query.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8"))
        images = [query] + contexts
        converted = []
        for image in images:
            if isinstance(image, torch.Tensor):
                image = Image.fromarray((image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8"))
            converted.append(image.convert("RGB"))
        query, contexts = converted[0], converted[1:]
        width = query.width + sum(context.width for context in contexts)
        height = max(image.height for image in converted)
        canvas = Image.new("RGB", (width, height + 40), "white")
        canvas.paste(query, (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"Query (class {row['query_label']})", fill="black")
        x = query.width
        for context, label, weight in zip(contexts, candidate_labels, weights):
            canvas.paste(context, (x, 40))
            draw.text((x + 4, 4), f"Context (class {label}; w={float(weight):.3f})", fill="black")
            x += context.width
        details = f"score={float(row['selection_score']):.3f}"
        if row.get("reward", ""):
            details += f"; reward={float(row['reward']):.3f}"
        draw.text((4, height + 24), details, fill="black")
        canvas.save(out / f"pair_{i:03d}.png")
    return rows


def plot_reward_histogram(pairs_csv: str, output_path: str):
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8")))
    values = [float(row["reward"]) for row in rows if row.get("reward", "") != ""]
    if not values:
        return False
    plt.figure(figsize=(5, 3)); plt.hist(values, bins=30); plt.xlabel("detached task-aligned reward"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    return True
