# Debug Checklist

## Data
- Candidate pool train-only?
- Can query retrieve itself?
- Label mappings consistent?
- Query/candidate preprocessing consistent?

## Shapes
- query `[B,C,H,W]`
- candidates `[B,N,C,H,W]`
- scores `[B,N]`
- Gumbel mask `[B,N]`

## Gumbel
- `hard=True`
- rows sum to 1
- selector receives gradients
- temperature not collapsing too early

## Policy
- reward sign is `loss_without - loss_with`
- reward detached
- advantage standardized with epsilon
- sampled action matches `log_prob`
- no gradient through reward construction

## Downstream
- query-only path truly context-free
- positional embeddings support concatenated length
- document interpolation/duplication choices
- CLS handling is consistent

## Optimization
- selector parameters included in optimizer
- separate LR groups if backbone unfrozen
- inspect gradient norms
- warm-start task model if needed
- lower `lambda_policy` if policy dominates

## Log
- task loss
- policy loss
- reward mean/std
- selector entropy
- cosine similarity
- cross-class selection rate
