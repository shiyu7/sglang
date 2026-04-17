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
