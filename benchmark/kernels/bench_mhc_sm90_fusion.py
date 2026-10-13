"""Compare V4.1 model-dispatched mHC stats with the SM90 opt-in off/on.

PYTHONPATH=python python benchmark/kernels/bench_mhc_sm90_fusion.py --output result.json

CUDA Graph timing, synthetic BF16 activations/FP32 weights, hot cache. Includes
the split-K projection, reduction and Sinkhorn; excludes combine, RMSNorm,
post-mix and other model work. These are component timings, not serving TPOT.
Use --independent-inputs --calls-per-graph 80 for distinct activations/weights
across 80 stats calls (the call count of 40 layers, not a complete model pass).
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[6, 12, 36, 48, 96, 192, 256]
    )
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls-per-graph", type=int, default=32)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument(
        "--independent-inputs",
        action="store_true",
        help="Use different inputs and weights for every call in the graph",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.calls_per_graph < 1 or any(m < 1 for m in args.rows):
        parser.error("rows, samples and calls-per-graph must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error("requires an SM90 CUDA GPU")
    if args.output and args.output.exists():
        parser.error("output already exists")

    torch.manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    layer = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        hc_pre_from_prev_sublayer=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    results = []
    for m in args.rows:
        inputs = [
            (
                torch.randn(m, 4, 5120, device="cuda", dtype=torch.bfloat16),
                torch.randn(24, 20480, device="cuda", dtype=torch.float32) / 20480**0.5,
                torch.tensor([0.7, 1.1, 0.3], device="cuda"),
                torch.randn(24, device="cuda") * 0.2,
            )
            for _ in range(args.calls_per_graph if args.independent_inputs else 1)
        ]

        def run(i=0):
            return DeepseekV4DecoderLayer._hc_mix_stats(
                layer, *inputs[i if args.independent_inputs else 0]
            )

        graphs, outputs = [], []
        for enabled in (False, True):
            with envs.SGLANG_OPT_DSV41_SM90_FUSED_MHC.override(enabled):
                for _ in range(3):
                    for i in range(len(inputs)):
                        run(i)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = [run(i) for i in range(args.calls_per_graph)]
                graph.replay()
                graphs.append(graph)
                outputs.append(captured)
        for baseline, fused in zip(*outputs):
            for a, b in zip(baseline, fused):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

        timings = [[], []]
        for _ in range(args.samples):
            order = [0, 1]
            order_rng.shuffle(order)
            for i in order:
                graphs[i].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                timings[i].append(start.elapsed_time(end) * 1000 / args.calls_per_graph)
        old, new = (statistics.median(t) for t in timings)
        row = dict(
            rows=m,
            baseline_us=old,
            fused_us=new,
            speedup=old / new,
            reduction_percent=100 * (1 - new / old),
            baseline_graph_ms=old * args.calls_per_graph / 1000,
            fused_graph_ms=new * args.calls_per_graph / 1000,
            samples_us=timings,
            numerical_check="passed",
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        scope="model-dispatched mix statistics + Sinkhorn, CUDA Graph, not E2E",
        independent_inputs=args.independent_inputs,
        seed=args.seed,
        calls_per_graph=args.calls_per_graph,
        results=results,
    )
    if args.output:
        with args.output.open("x") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
