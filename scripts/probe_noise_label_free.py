"""Fixed, label-free prediction controls on existing noisy validation caches.

Exploratory only; no oracle label enters an inference rule. Report every fixed
rule, including the two-No-Context ensemble as an extra-model control.
"""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_tacs_robustness import paired_ci
from search_native_context import json_save
from run_query_noise_pilot import write_csv


def main():
    out = ROOT / "outputs/noise_label_free_screen_v1"
    out.mkdir(exist_ok=True)
    protocol = {"endpoint": "mean accuracy over sigma .08/.16, all4996 validation queries",
                "scope": "exploratory fixed-rule screen, no training, no test access",
                "rules": ["nc42", "nc43", "nc42_nc43_probability_mean", "staged_selected", "context16_probability_mean",
                          "context16_max_confidence", "nc42_staged_probability_mean", "nc42_context16_probability_mean"],
                "weights": "equal .5 for every two-model mixture, uniform1/16 context average; no parameter sweep",
                "labels_used_by_rules": False}
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Changed screen protocol")
    json_save(protocol, path)
    base = ROOT / "outputs/query_noise_clean_context_3seed_v1"
    saved = {s: torch.load(base / f"seed_{s}/no_context/validation_final.pt", map_location="cpu", weights_only=False) for s in (42, 43)}
    staged = torch.load(ROOT / "outputs/query_noise_staged_warmup_s42_v1/staged_tacs/validation_final.pt", map_location="cpu", weights_only=False)
    for value in (*saved.values(), staged):
        if not torch.equal(value["query_ids"], staged["query_ids"]) or not torch.equal(value["labels"], staged["labels"]):
            raise ValueError("Unpaired queries")
    means, confident = [], []
    for si in range(2):
        chunks = [torch.load(p, map_location="cpu", weights_only=False) for p in
                  sorted((ROOT / "outputs/query_noise_staged_oracle_s42_v1/chunks").glob(f"sigma_{si}_batch_*.pt"))]
        if not torch.equal(torch.cat([c["query_ids"] for c in chunks]), staged["query_ids"]):
            raise ValueError("Context order mismatch")
        logits = torch.cat([c["all16_logits"] for c in chunks])
        probs = logits.softmax(-1)
        means.append(probs.mean(1))
        choice = probs.amax(-1).argmax(-1)
        confident.append(probs[torch.arange(len(probs)), choice])
    mean, conf = torch.stack(means, 1), torch.stack(confident, 1)
    nc42, nc43, full = saved[42]["probabilities"], saved[43]["probabilities"], staged["probabilities"]
    predictions = dict(zip(protocol["rules"], [nc42, nc43, (nc42 + nc43) / 2, full, mean, conf,
                                               (nc42 + full) / 2, (nc42 + mean) / 2]))
    y = staged["labels"]
    correctness = {name: p.argmax(-1).eq(y[:, None]).double() for name, p in predictions.items()}
    rows = []
    for name, correct in correctness.items():
        delta = paired_ci((correct - correctness["nc42"]).mean(1).numpy() * 100)
        delta_ensemble = paired_ci((correct - correctness["nc42_nc43_probability_mean"]).mean(1).numpy() * 100)
        a = correct.mean(0)
        rows.append({"rule": name, "accuracy008": float(a[0]), "accuracy016": float(a[1]), "mean_accuracy": float(a.mean()),
                     "vs_nc42_pp": delta["difference"], "ci95": delta["ci"], "vs_two_nc_pp": delta_ensemble["difference"],
                     "vs_two_nc_ci95": delta_ensemble["ci"]})
    json_save({"protocol": protocol, "rows": rows}, out / "result.json")
    write_csv(rows, out / "comparison.csv")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
