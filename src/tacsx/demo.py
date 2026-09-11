from __future__ import annotations

import torch
import gradio as gr

from .models.baselines import dino_similarity_scores


def build_demo(model, candidate_dataset, transform, device="cpu"):
    """Create the inspection UI from an explicit trained model and train pool."""
    model = model.to(device).eval()

    @torch.no_grad()
    def inspect(image):
        query = transform(image).unsqueeze(0).to(device)
        candidates = torch.stack([candidate_dataset[i][0] for i in range(min(64, len(candidate_dataset)))]).unsqueeze(0).to(device)
        dino_scores = dino_similarity_scores(model.task_model.backbone, query, candidates)
        tacs_scores = model.selector(query, candidates)
        dino_i, tacs_i = int(dino_scores.argmax()), int(tacs_scores.argmax())
        dino_context = candidate_dataset[dino_i][0].permute(1, 2, 0).cpu().numpy()
        tacs_context = candidate_dataset[tacs_i][0].permute(1, 2, 0).cpu().numpy()
        no_ctx = int(model.task_model(query, None).argmax(1))
        with_ctx = int(model.task_model(query, candidates[:, tacs_i]).argmax(1))
        info = {"no_context_prediction": no_ctx, "with_tacs_context_prediction": with_ctx,
                "dino_index": dino_i, "tacs_index": tacs_i, "tacs_utility": float(tacs_scores[0, tacs_i])}
        return dino_context, tacs_context, info

    return gr.Interface(inspect, gr.Image(type="pil", label="Query"),
                        [gr.Image(label="DINO nearest neighbor"), gr.Image(label="TACS context"), gr.JSON(label="Predictions and utility")],
                        title="TACS-X retrieval inspector")
