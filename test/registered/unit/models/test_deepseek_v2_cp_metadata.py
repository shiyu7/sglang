import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.utils.cp_utils import (
    get_round_robin_paged_prefill_plan,
    prepare_context_parallel_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDeepseekV2CPMetadata(CustomTestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            enable_prefill_context_parallel=True,
            enable_nsa_prefill_context_parallel=False,
            prefill_cp_mode="round-robin-split",
            nsa_prefill_cp_mode="in-seq-split",
        )
        self.args_patcher = patch(
            "sglang.srt.layers.utils.cp_utils.get_global_server_args",
            return_value=self.args,
        )
        self.args_patcher.start()

    def tearDown(self):
        self.args_patcher.stop()

    def test_round_robin_metadata_builds_local_prefill_plan(self):
        metadata = prepare_context_parallel_metadata(
            kv_len=48,
            cp_rank=3,
            cp_size=8,
            seqs_len=[432],
            req_pool_indices=torch.tensor([89], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        self.assertEqual(metadata.rr_qo_indptr_tensor.tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(metadata.rr_seq_lens_tensor.tolist(), [388, 396, 404, 412, 420, 428])
        self.assertEqual(
            metadata.rr_prefix_lens_tensor.tolist(),
            [387, 395, 403, 411, 419, 427],
        )
        self.assertEqual(metadata.rr_req_pool_indices_tensor.tolist(), [89] * 6)

    def test_round_robin_metadata_handles_empty_local_rank(self):
        metadata = prepare_context_parallel_metadata(
            kv_len=3,
            cp_rank=5,
            cp_size=8,
            seqs_len=[13],
            req_pool_indices=torch.tensor([7], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        self.assertEqual(metadata.rr_qo_indptr_tensor.tolist(), [0])
        self.assertEqual(metadata.rr_seq_lens_tensor.numel(), 0)
        self.assertEqual(metadata.rr_prefix_lens_tensor.numel(), 0)
        self.assertEqual(metadata.rr_req_pool_indices_tensor.numel(), 0)

    def test_round_robin_plan_accessor_returns_backend_inputs(self):
        metadata = prepare_context_parallel_metadata(
            kv_len=10,
            cp_rank=1,
            cp_size=4,
            seqs_len=[74],
            req_pool_indices=torch.tensor([23], dtype=torch.int32),
            device=torch.device("cpu"),
        )
        forward_batch = SimpleNamespace(attn_cp_metadata=metadata, nsa_cp_metadata=None)

        req_pool_indices, seq_lens, prefix_lens, qo_indptr = get_round_robin_paged_prefill_plan(
            forward_batch
        )

        self.assertEqual(req_pool_indices.tolist(), [23, 23, 23])
        self.assertEqual(seq_lens.tolist(), [65, 69, 73])
        self.assertEqual(prefix_lens.tolist(), [64, 68, 72])
        self.assertEqual(qo_indptr.tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
