from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    _mask_topk_ids_padded_region,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )

        assert not apply_routed_scaling_factor_on_output, "not implemented"

    def empty_topk_output(self, device: torch.device):
        topk = self.topk - self.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids

    def _debug_fused_hash_topk_inputs(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> None:
        with torch.no_grad():
            num_tokens = input_ids.numel()
            tid2eid_rows, tid2eid_topk = self.tid2eid.shape
            num_routed_experts = router_logits.shape[1]
            msg = (
                "HashTopK debug: "
                f"num_tokens={num_tokens}, "
                f"router_logits(shape={tuple(router_logits.shape)}, "
                f"dtype={router_logits.dtype}, "
                f"device={router_logits.device}, "
                f"is_contiguous={router_logits.is_contiguous()}), "
                f"input_ids(shape={tuple(input_ids.shape)}, "
                f"dtype={input_ids.dtype}, device={input_ids.device}), "
                f"tid2eid(shape={tuple(self.tid2eid.shape)}, "
                f"dtype={self.tid2eid.dtype}, device={self.tid2eid.device}), "
                f"topk={self.topk}, routed_topk={tid2eid_topk}, "
                f"num_experts={self.num_experts}, "
                f"num_routed_experts={num_routed_experts}, "
                f"num_fused_shared_experts={self.num_fused_shared_experts}"
            )

            if num_tokens == 0:
                logger.warning("%s, empty input_ids", msg)
                return

            input_min = int(input_ids.min().item())
            input_max = int(input_ids.max().item())
            bad_token_mask = (input_ids < 0) | (input_ids >= tid2eid_rows)
            bad_token_count = int(bad_token_mask.sum().item())
            msg = (
                f"{msg}, input_ids_min={input_min}, "
                f"input_ids_max={input_max}, "
                f"bad_token_count={bad_token_count}"
            )

            if bad_token_count > 0:
                bad_pos = int(bad_token_mask.nonzero()[0].item())
                bad_token = int(input_ids[bad_pos].item())
                logger.error(
                    "%s, first_bad_token_pos=%d, first_bad_token=%d",
                    msg,
                    bad_pos,
                    bad_token,
                )
                raise RuntimeError(
                    "HashTopK fused kernel would read tid2eid out of bounds: "
                    f"input_ids[{bad_pos}]={bad_token}, tid2eid_rows={tid2eid_rows}"
                )

            selected_tid2eid = self.tid2eid[input_ids]
            eid_min = int(selected_tid2eid.min().item())
            eid_max = int(selected_tid2eid.max().item())
            bad_eid_mask = (selected_tid2eid < 0) | (
                selected_tid2eid >= num_routed_experts
            )
            bad_eid_count = int(bad_eid_mask.sum().item())
            msg = (
                f"{msg}, selected_tid2eid_min={eid_min}, "
                f"selected_tid2eid_max={eid_max}, "
                f"bad_selected_eid_count={bad_eid_count}"
            )

            if bad_eid_count > 0:
                bad_pos = bad_eid_mask.nonzero()[0]
                bad_token_row = int(bad_pos[0].item())
                bad_topk_col = int(bad_pos[1].item())
                bad_token = int(input_ids[bad_token_row].item())
                bad_eid = int(selected_tid2eid[bad_token_row, bad_topk_col].item())
                logger.error(
                    "%s, first_bad_eid_token_row=%d, first_bad_token=%d, "
                    "first_bad_eid_col=%d, first_bad_eid=%d",
                    msg,
                    bad_token_row,
                    bad_token,
                    bad_topk_col,
                    bad_eid,
                )
                raise RuntimeError(
                    "HashTopK fused kernel would read router_logits out of bounds: "
                    f"tid2eid[input_ids[{bad_token_row}]={bad_token}, "
                    f"{bad_topk_col}]={bad_eid}, "
                    f"num_routed_experts={num_routed_experts}"
                )

            logger.warning("%s", msg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        assert (
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.jit_kernel.deepseek_v4 import hash_topk

            if envs.SGLANG_HASH_TOPK_DEBUG.get():
                self._debug_fused_hash_topk_inputs(router_logits, input_ids)

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)

        if is_hip():
            topk_weights = topk_weights.to(torch.float32)

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        return topk_output
