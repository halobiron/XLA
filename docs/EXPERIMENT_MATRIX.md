# Experiment Matrix

| ID | Mode | Selector | Gumbel | Policy | Context |
|---|---|---|---:|---:|---|
| E0 | no_context | none | no | no | none |
| E1 | random_context | random | no | no | random top-1 |
| E2 | dino_similarity | frozen DINO | no | no | nearest top-1 |
| E3 | gumbel_only | learned | yes | no | learned top-1 |
| E4 | policy_only | learned | no | yes | sampled/argmax |
| E5 | full_tacs | learned | yes | yes | learned top-1 |
| E6 | topk_tacs | learned | optional | optional | K=2 |
| E7 | topk_tacs | learned | optional | optional | K=4 |

## Primary result columns

- method
- seed
- best_val_acc
- test_acc
- avg_loss
- cross_class_rate
- mean_query_context_cosine
- mean_selection_entropy
- mean_reward

Use 1 seed during development. Use 3 seeds for final E0/E2/E3/E4/E5 if compute allows.

## Report narrative

1. E0/E1/E2: extra context is not automatically useful.
2. E3/E4 vs E2: learned selection changes retrieval behavior.
3. E5 vs E3/E4: test hybrid optimization.
4. E6/E7: test the paper's single-context limitation.
5. Qualitative pairs: demonstrate complementary selection.

## If E5 underperforms

Check:
- reward variance;
- selector entropy collapse;
- candidate set size;
- frozen vs trainable selector;
- downstream warm-start;
- policy weight;
- gradient norms.

Report negative results honestly.
