# Debug Session: qo-indptr-mismatch

- Status: OPEN
- Goal: collect runtime evidence for `q.shape[0] != qo_indptr[-1]` under MLA + CP + long prompt.
- Scope: instrumentation only, no business-logic fix in this phase.

## Hypotheses

1. `prepare_mlp_sync_batch()` pads token count for CP/DP alignment, so `q.shape[0]` includes padded tokens while `qo_indptr` still tracks real extend tokens.
2. `deepseek_v2.py` trims `hidden_states` by `actual_num_tokens`, but downstream attention metadata is built from a different token-count basis.
3. The mismatch is specific to the `MHA one-shot -> FlashInferMhaChunkKVRunner` path and does not affect the normal CP ragged closure path.
4. `seq_lens`, `extend_seq_lens`, and `extend_prefix_lens` mix padded and unpadded values in CP mode, causing `sum(seq_lens - prefix_lens)` to drift from the actual `q` token count.

## Planned Probes

1. Record pre/post CP split token counts in `deepseek_v2.py`.
2. Record `qo_indptr`, `expected_q_tokens`, and q/k/v shapes in `flashinfer_mla_backend.py`.
3. Record padding-related counters from `ForwardBatch.prepare_mlp_sync_batch()`.

## Handoff

- Reproduce on the remote environment.
- Return the generated `.dbg/trae-debug-log-qo-indptr-mismatch.ndjson` log file contents or key excerpts.
