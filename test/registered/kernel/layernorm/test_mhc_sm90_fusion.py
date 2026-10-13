"""Exercise the V4.1 opt-in through the actual model dispatch on Hopper."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90 CUDA",
)
class TestMhcSm90Fusion(CustomTestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.layer = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            hc_pre_from_prev_sublayer=True,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
        )

    def inputs(self, m, hidden=5120, dtype=torch.bfloat16, strided=False):
        hc = self.layer.hc_mult
        k, mix = hc * hidden, (hc + 2) * hc
        x = torch.randn(m * (2 if strided else 1), hc, hidden, device="cuda").to(dtype)
        fn = torch.randn(mix, k + (64 if strided else 0), device="cuda") / k**0.5
        scale = torch.tensor([0.7, 1.1, 0.3], device="cuda")
        base = torch.randn(mix, device="cuda") * 0.2
        if strided:
            x, fn = x[::2], fn[:, :k]
        return x, fn, scale, base

    def run_stats(self, args, enabled, **kwargs):
        with envs.SGLANG_OPT_DSV41_SM90_FUSED_MHC.override(enabled):
            return DeepseekV4DecoderLayer._hc_mix_stats(self.layer, *args, **kwargs)

    def assert_coefficients(self, actual, expected):
        for got, ref in zip(actual, expected):
            self.assertEqual(got.dtype, torch.float32)
            self.assertTrue(torch.isfinite(got).all().item())
            torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-5)

    def test_decode_and_verify_shapes(self):
        # Includes empty/IDLE, draft, uneven per-DP batches, DSPARK block=5,
        # max-running-requests=256 / DP8 -> 192 VERIFY rows, and cap fallback.
        for m in (0, 1, 2, 6, 12, 18, 24, 30, 36, 42, 48, 96, 180, 192, 256, 257):
            with self.subTest(rows=m):
                args = self.inputs(m)
                self.assert_coefficients(
                    self.run_stats(args, True), self.run_stats(args, False)
                )

    def test_projection_schedule_is_bitwise(self):
        # Scheduling must not change split-K partials or coefficient arithmetic.
        for m in (1, 6, 36, 180, 192, 256):
            for strided in (False, True):
                with self.subTest(rows=m, strided=strided):
                    x, fn, scale, base = self.inputs(m, strided=strided)
                    args = (x.flatten(1), fn, scale, base, 4, 20, 1e-6, 1e-6)
                    expected = mhc.hc_mix_stats_sinkhorn(*args)
                    actual = mhc.hc_mix_stats_sinkhorn(*args, sm90_decode=True)
                    for got, ref in zip(actual, expected):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_strides_scales_and_sinkhorn_iterations(self):
        for iters in (1, 2, 20):
            for magnitude in (0.0, 1e-3, 1.0, 1e3):
                with self.subTest(iters=iters, magnitude=magnitude):
                    self.layer.hc_sinkhorn_iters = iters
                    self.layer.rms_norm_eps = 3e-5
                    self.layer.hc_eps = 2e-6
                    args = self.inputs(36, strided=True)
                    args[0].mul_(magnitude)
                    self.assert_coefficients(
                        self.run_stats(args, True), self.run_stats(args, False)
                    )

    def test_opt_in_dispatch_and_fallbacks(self):
        cases = (
            ({}, True, True),
            ({}, False, False),
            ({"m": 1}, False, True),  # Existing default one-row path.
            ({"m": 1}, True, True),  # Opt-in also tunes this projection.
            ({"m": 0}, True, False),
            ({"m": 257}, True, False),
            ({"hidden": 4096}, True, False),
            ({"dtype": torch.float32}, True, False),
            ({"model_type": "deepseek_v4"}, True, False),
            ({"previous_pre": False}, True, False),
            ({"hc": 2}, True, False),
            ({"batch_invariant": True}, True, False),
        )
        for options, enabled, expected in cases:
            with self.subTest(options=options, enabled=enabled):
                self.layer.config.model_type = options.get("model_type", "deepseek_v41")
                self.layer.hc_pre_from_prev_sublayer = options.get("previous_pre", True)
                self.layer.hc_mult = options.get("hc", 4)
                args = self.inputs(
                    options.get("m", 36),
                    options.get("hidden", 5120),
                    options.get("dtype", torch.bfloat16),
                )
                with (
                    patch.object(
                        mhc, "hc_mix_stats_sinkhorn", wraps=mhc.hc_mix_stats_sinkhorn
                    ) as fused,
                    patch(
                        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                        return_value=options.get("batch_invariant", False),
                    ),
                ):
                    self.run_stats(args, enabled)
                    self.assertEqual(fused.call_count, int(expected))
                    if expected:
                        self.assertEqual(fused.call_args.kwargs["sm90_decode"], enabled)

    def test_batch_invariance_inside_supported_range(self):
        args = self.inputs(256)
        row = args[0][:1].clone()
        args[0][0].copy_(row[0])
        expected = self.run_stats((row, *args[1:]), True)
        for m in (2, 6, 12, 36, 96, 180, 192, 256):
            got = self.run_stats((args[0][:m], *args[1:]), True)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a[:1], b, rtol=0, atol=0)

    def test_graph_replay_reads_updated_inputs_and_weights(self):
        args = self.inputs(36)
        for _ in range(3):
            self.run_stats(args, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.run_stats(args, True)
        for _ in range(3):
            for tensor in args:
                tensor.copy_(torch.randn_like(tensor) * 0.05)
            graph.replay()
            self.assert_coefficients(captured, self.run_stats(args, False))

    def test_side_stream_and_previous_pre_dependency(self):
        args = self.inputs(36)
        reference = self.run_stats(args, False)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        got = self.run_stats(args, True, stats_stream=side)
        torch.cuda.current_stream().wait_stream(side)
        self.assert_coefficients(got, reference)

        # V4.1 combines with the PREVIOUS sublayer's pre. New coefficients
        # must not leak into this sublayer's normalized input.
        previous_pre = torch.rand(36, 4, device="cuda")
        norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
        outputs = []
        for enabled in (False, True):
            with envs.SGLANG_OPT_DSV41_SM90_FUSED_MHC.override(enabled):
                outputs.append(
                    DeepseekV4DecoderLayer._hc_mix_and_combine(
                        self.layer, *args, previous_pre, norm
                    )
                )
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0, rtol=0)
        self.assert_coefficients(outputs[1][1:], outputs[0][1:])


if __name__ == "__main__":
    unittest.main()
