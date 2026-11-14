"""Read-only evidence of actual task fine-tuning and frozen selector encoder."""
import hashlib
import io
import json
import zipfile

import torch

from train_tacs_shortlist import ROOT, json_save


def main():
    torch.set_num_threads(4)
    run = ROOT / "outputs/tacs_shortlist_fullft_clean_v3"
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    entry = next(x for x in archive.namelist() if "cifar100_no_context_c4_e5_unf4v3/" in x and x.endswith("checkpoint_best.pt"))
    common = torch.load(io.BytesIO(archive.read(entry)), map_location="cpu", weights_only=False)["model"]
    official = torch.load(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", map_location="cpu", weights_only=True)
    result = {"common_initialization_crc": archive.getinfo(entry).CRC, "arms": {}}
    epochs = []
    for arm in ("no_context", "full_tacs"):
        content = (run / arm / "checkpoint_best.pt").read_bytes()
        saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        state = saved["model"]
        changed = {}
        for block in range(12):
            prefix = f"task_model.backbone.model.blocks.{block}."
            changed[block] = sum(not torch.equal(value, common[key]) for key, value in state.items() if key.startswith(prefix))
        row = {"best_epoch": saved["best"]["epoch"], "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
               "changed_tensors_by_task_block": changed,
               "all_12_blocks_actually_changed": all(changed.values()),
               "task_head_changed": not torch.equal(state["task_model.head.weight"], common["task_model.head.weight"]),
               "task_patch_embedding_changed": not torch.equal(state["task_model.backbone.model.patch_embed.proj.weight"], common["task_model.backbone.model.patch_embed.proj.weight"])}
        if arm == "full_tacs":
            row["selector_encoder_unchanged_from_official"] = all(torch.equal(state["selector.encoder.model." + key], value) for key, value in official.items())
            weight = state["selector.projector.weight"]
            row["selector_projection_changed"] = not torch.equal(weight, torch.eye(len(weight)))
        result["arms"][arm] = row
        epochs.append(row["best_epoch"])
    folder = run / "audit_snapshots"
    folder.mkdir(exist_ok=True)
    path = folder / f"weights_n{epochs[0]}_t{epochs[1]}.json"
    json_save(result, path)
    print(json.dumps(result, indent=2), flush=True)
    if not all(row["all_12_blocks_actually_changed"] and row["task_head_changed"] for row in result["arms"].values()):
        raise ValueError("Task training mismatch")
    if not result["arms"]["full_tacs"]["selector_encoder_unchanged_from_official"] or not result["arms"]["full_tacs"]["selector_projection_changed"]:
        raise ValueError("Selector training mismatch")


if __name__ == "__main__":
    main()
