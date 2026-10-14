# DSV4.1 SM90 decode tuning

These default-off optimizations were evaluated on 8 NVIDIA H20 GPUs, attention
DP=8, EP=8, BF16 residuals, block-FP8 dense projections (Humming), and DSpark
block size 5 (6 target verification rows per request). MegaMoE is unchanged.

The feature branches form a stack based on `f27dd5bdda`:

1. `feat/dsv41-sm90-mhc`
2. `feat/dsv41-sm90-prefix-logits` (includes 1)
3. `feat/dsv41-sm90-wo-a` (includes 1 and 2)

The first branch includes two reproducibility prerequisites from the measured
baseline: honoring the configured dense FP8 runner instead of prematurely
forcing non-128 weight blocks to Triton, and obtaining PP size from
`get_parallel()` during FlashInfer autotune setup. These are baseline fixes,
not separately measured speedups. Earlier experimental C2 and Q-LoRA fusion
changes are not included in this stack.

## mHC

Enable `SGLANG_OPT_DSV41_SM90_FUSED_MHC=1` before graph capture. For the admitted
SM90 V4.1 shapes, use the existing TF32x3 partial projection and fused
reduce/Sinkhorn with one pipeline stage. At measured small row counts use
2 warps; retain 4 warps for aligned M>=128 shapes and outside the measured
96..180 interval. K splits, BLOCK_K=64, dot precision and Sinkhorn math remain
unchanged. Unsupported shapes retain the original path.

At M=120, an 80-call stack with independent inputs/weights (40 layers x 2)
reduced partial-projection time from 3.192 to 2.606 ms. At M=100 it changed
from 2.687 to 1.931 ms. These are operator totals, not whole-model latency.
The small-batch launch change was bitwise checked against the preceding fused
baseline, including graph replay, weight/input updates, strided rows and
cross-batch agreement. This is not a claim of whole-model bitwise equivalence
between enabled and disabled mHC algorithms.

```bash
PYTHONPATH=python python3 test/registered/kernels/ops/layernorm/test_mhc_sm90_launch.py
```
