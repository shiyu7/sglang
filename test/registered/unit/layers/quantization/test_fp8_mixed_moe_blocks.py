"""Mixed routed/dense FP8 block layouts, without launching a GPU kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.deepseek_common.utils import quant_blocks_shared_experts_fusion
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFp8MixedMoEBlocks(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            patch.object(get_flags().moe, "runner_backend", MoeRunnerBackend.TRITON)
        )
        self.enterContext(get_parallel().override(tp_size=1))

    @staticmethod
    def config(**overrides):
        return Fp8Config.from_config(
            dict(
                {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [32, 32],
                    "moe_weight_block_size": [128, 128],
                    "scale_fmt": "ue8m0",
                },
                **overrides,
            )
        )

    @staticmethod
    def layer(cls, **attrs):
        # Skip distributed initialization; exercise real dispatch and allocation.
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        for name, value in attrs.items():
            setattr(layer, name, value)
        return layer

    def test_routed_and_dense_scale_shapes(self):
        config = self.config()
        routed = self.layer(
            FusedMoE,
            num_fused_shared_experts=0,
            moe_runner_config=MoeRunnerConfig(is_gated=True),
        )
        method = config.get_quant_method(routed, "model.layers.0.mlp.experts")
        method.create_weights(routed, 2, 256, 128, torch.bfloat16)
        self.assertEqual(tuple(routed.w13_weight_scale_inv.shape), (2, 2, 2))
        self.assertEqual(tuple(routed.w2_weight_scale_inv.shape), (2, 2, 1))
        self.assertEqual(method.weight_block_size, [128, 128])
        self.assertEqual(method.quant_config.scale_fmt, "ue8m0")
        self.assertIsNot(method.quant_config, config)

        for prefix in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.shared_experts.gate_up_proj",
        ):
            with self.subTest(prefix=prefix):
                dense = self.layer(LinearBase)
                dense_method = config.get_quant_method(dense, prefix)
                dense_method.create_weights(dense, 256, [128], 256, 128, torch.bfloat16)
                self.assertEqual(tuple(dense.weight_scale_inv.shape), (4, 8))
                self.assertEqual(dense_method.weight_block_size, [32, 32])
                self.assertIs(dense_method.quant_config, config)
        self.assertEqual(config.weight_block_size, [32, 32])
        self.assertEqual(config.moe_weight_block_size, [128, 128])

    def test_fusion_policy_and_guard(self):
        config = self.config()
        self.assertTrue(quant_blocks_shared_experts_fusion(config))
        fused = self.layer(FusedMoE, num_fused_shared_experts=1)
        with self.assertRaisesRegex(ValueError, "disable-shared-experts-fusion"):
            config.get_quant_method(fused, "model.layers.0.mlp.experts")
        for override in (None, [32, 32]):
            with self.subTest(override=override):
                matched = self.config(moe_weight_block_size=override)
                self.assertFalse(quant_blocks_shared_experts_fusion(matched))
                method = matched.get_quant_method(fused, "model.layers.0.mlp.experts")
                self.assertEqual(method.weight_block_size, [32, 32])

    def test_ignored_dense_layer_stays_unquantized(self):
        prefix = "model.layers.0.self_attn.q_proj"
        config = self.config(ignored_layers=[prefix])
        self.assertIsInstance(
            config.get_quant_method(self.layer(LinearBase), prefix),
            UnquantizedLinearMethod,
        )

    def test_v4_vetoes_enforced_fusion_for_mixed_blocks(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        override = get_context().override_server_args(
            enforce_shared_experts_fusion=True
        )
        override.install()
        self.addCleanup(override.restore)
        reason = DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
            SimpleNamespace(n_shared_experts=1), self.config()
        )
        self.assertIn("different block layout", reason)

    def test_rejects_invalid_override(self):
        for size in (
            [],
            [128],
            [128, 128, 128],
            [0, 128],
            [-1, 128],
            [True, 128],
            [1.5, 128],
            "128",
            128,
        ):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "two positive integers"):
                    self.config(moe_weight_block_size=size)
        for overrides in (
            {"weight_block_size": None},
            {"activation_scheme": "static"},
            {"quant_method": "mxfp8"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.config(**overrides)

    def test_rejects_fp4_flag_injected_by_loader(self):
        config = self.config()
        config.is_fp4_experts = True
        with self.assertRaisesRegex(ValueError, "requires FP8 expert weights"):
            config.get_quant_method(
                self.layer(FusedMoE, num_fused_shared_experts=0),
                "model.layers.0.mlp.experts",
            )


if __name__ == "__main__":
    unittest.main()
