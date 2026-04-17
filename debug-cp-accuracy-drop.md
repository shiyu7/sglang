# [OPEN] CP Accuracy Drop

## Session
- session_id: `cp-accuracy-drop`
- status: `OPEN`
- scope: `Kimi MLA prefill CP`

## Symptom
- CP enabled with `round-robin-split` causes MMLU accuracy to collapse to near zero.
- Current repro uses `tp=8`, `pp=2`, `nnodes=2`, `flashmla`, and online benchmark traffic.

## Expected
- CP should preserve model accuracy relative to non-CP runs.

## Hypotheses
1. CP is still being enabled on multi-request prefill batches, so round-robin metadata is built under a single-sequence assumption and silently corrupts outputs.
2. The current MLA round-robin prefill path computes per-token `kv_len` incorrectly when prefix cache exists, so each token attends to the wrong visible KV range.
3. TP/PP rank layout makes some attention CP groups span nodes or mix unexpected ranks, so CP communication topology differs from the assumed node-local layout.
4. KV cache reorder/all-gather for MLA under generic CP is inconsistent with the local query token order in round-robin mode, so Q and KV observe different global token orders.
5. The FlashInfer MLA round-robin special path is numerically or semantically wrong even for batch size 1, independent of batching and topology.

## Evidence Plan
- Record when CP is enabled at runtime: batch size, seq counts, CP mode, cp rank/size, pp rank, and whether metadata is present.
- Record round-robin token-to-kv mapping inputs: local q length, prefix len, computed `kv_len`, and expected visible token count.
- Record attention CP group membership to verify node-local grouping assumptions.
- Compare one failing CP run against one non-CP run on the same prompt shape.

## Next Step
- Add instrumentation only. No business-logic fix before evidence is collected.

## Evidence Collected
- `Hypothesis 1` rejected for the captured failing cases:
  - Round-robin MLA branch entered with `batch_size=1`, `cp_rank=0..7`, same `req_pool_indices`.
- `Hypothesis 2` largely rejected:
  - `kv_len` follows `prefix_len + global_token_idx + 1` and is self-consistent across all CP ranks.
- `Hypothesis 3` rejected for the captured case:
  - `cp_group_ranks=[0,1,2,3,4,5,6,7]`, matching a node-local TP/CP group.
- `Hypothesis 4` narrowed:
  - MLA KV rebuild and final hidden-state gather both show `local_len * cp_size == full_len` (e.g. `17 -> 136`), so group topology and gather lengths look correct.

## Current Leading Hypothesis
1. CP gather lengths are correct, but the rebuilt full MLA KV may be written back with a local-length `out_cache_loc`, silently truncating the effective KV state stored in the paged pool.
2. If the write-back location length is correct, then the remaining highest-probability root cause is the round-robin per-token paged MLA special path itself (`call_begin_forward(... req_to_token ...)`) producing semantically wrong paged indices despite correct `kv_len`.

## Next Evidence To Collect
- Compare `len(forward_batch.out_cache_loc)` against `local_latent_len/full_latent_len` at MLA KV rebuild/save time.
- If `out_cache_loc` already has full length, log a small suffix of generated `kv_indices` for one round-robin token to verify that paged indices match the expected global token order.
