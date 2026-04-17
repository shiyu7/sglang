from __future__ import annotations

"""
Support attention backend for flashinfer MLA.
The flashinfer_mla_disable_ragged flag controls whether to use ragged prefill wrapper and defaults to be false.
When it's set to false, all wrappers are BatchMLAPaged wrapper.
When it's set to true, the backend uses BatchRagged and BatchMLAPaged wrapper for prefilling,
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    is_in_piecewise_cuda_graph,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_cp_rank, get_attention_tp_size
from sglang.srt.layers.utils.cp_utils import (
    cp_attn_forward_extend,
    cp_use_prefill,
    is_prefill_cp_round_robin_split,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashinfer_mla_backend import (
        FlashInferMlaAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

if is_flashinfer_available():
    from flashinfer import (
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )


@dataclass
class DecodeMetadata:
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferMhaChunkKVRunner:
    def __init__(
        self, model_runner: ModelRunner, attn_backend: FlashInferMlaAttnBackend
    ):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype

        # Buffers and wrappers
        self.qo_indptr = attn_backend.qo_indptr
        self.kv_indptr = attn_backend.kv_indptr
        self.workspace_buffer = attn_backend.workspace_buffer
        self.fmha_backend = attn_backend.fmha_backend

        self.chunk_ragged_wrappers = []
        self.ragged_wrapper = attn_backend.prefill_wrapper_ragged

    def update_prefix_chunks(self, num_prefix_chunks: int):
        while num_prefix_chunks > len(self.chunk_ragged_wrappers):
            ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD", backend=self.fmha_backend
            )
            self.chunk_ragged_wrappers.append(ragged_wrapper)

    def update_wrapper(
        self,
        forward_batch: ForwardBatch,
        disable_flashinfer_ragged: bool = False,
    ):
        assert forward_batch.num_prefix_chunks is not None
        num_prefix_chunks = forward_batch.num_prefix_chunks
        self.update_prefix_chunks(num_prefix_chunks)

        prefix_lens = forward_batch.extend_prefix_lens
        seq_lens = forward_batch.seq_lens

        bs = len(seq_lens)
        qo_indptr = self.qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        for chunk_idx in range(forward_batch.num_prefix_chunks):
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert forward_batch.prefix_chunk_max_seq_lens is not None

            kv_indptr = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=False,
            )
        # ragged prefill
        if not disable_flashinfer_ragged:
            kv_indptr = (
                qo_indptr
                if not forward_batch.mha_one_shot
                else self.kv_indptr[: bs + 1]
            )
            self.ragged_wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        logits_soft_cap = layer.logit_cap
        if forward_batch.attn_attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            assert chunk_idx >= 0
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            o = wrapper.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            forward = (
                self.ragged_wrapper.forward_return_lse
                if forward_batch.mha_return_lse
                else self.ragged_wrapper.forward
            )
            o = forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        return o


class FlashInferMLAAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        self.enable_chunk_kv = (
            not skip_prefill
            and get_global_server_args().disaggregation_mode != "decode"
            and not get_global_server_args().disable_chunked_prefix_cache
            and not get_global_server_args().flashinfer_mla_disable_ragged
        )
        self.page_size = model_runner.page_size

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer

        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        if q_indptr_decode_buf is None:
            self.q_indptr_decode = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=model_runner.device
            )
        else:
            self.q_indptr_decode = q_indptr_decode_buf

        if is_sm100_supported():
            self.fmha_backend = "cutlass"
        else:
            self.fmha_backend = "auto"

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=self.fmha_backend
        )

        if not self.skip_prefill:
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

            # FlashinferMLA backend uses mla wrapper for target verify
            self.prefill_wrapper_verify = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.enable_chunk_kv:
                self.mha_chunk_kv_cache = FlashInferMhaChunkKVRunner(model_runner, self)

        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                init_metadata_replay=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_paged, False)
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_verify,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_verify, False)
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            # Whether this extend has any prefix cache to attend.
            # Used to decide if we can use ragged prefill wrapper safely.
            extend_no_prefix = (
                forward_batch.extend_prefix_lens_cpu is not None
                and not any(forward_batch.extend_prefix_lens_cpu)
            )
            use_ragged = (
                not get_global_server_args().flashinfer_mla_disable_ragged
                and extend_no_prefix
                # Piecewise cuda graph should use paged prefill to be compatible with prefix cache
                and not is_in_piecewise_cuda_graph()
            )

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_qo_indptr = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr = self.kv_indptr.clone()
        self.cuda_graph_kv_lens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # For fast decode plan in graph replaying
        self.cuda_graph_qo_indptr_cpu = self.cuda_graph_qo_indptr.to("cpu")
        self.cuda_graph_kv_indptr_cpu = self.cuda_graph_kv_indptr.to("cpu")
        self.fast_decode_kwargs = {
            "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu,
            "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu,
            "kv_indices": self.cuda_graph_kv_indices,
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: num_tokens + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: num_tokens + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:num_tokens],
                backend="auto",
            )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrapper=decode_wrapper,
                init_metadata_replay=False,
                spec_info=spec_info,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = DecodeMetadata(decode_wrapper)
            decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)
        elif forward_mode.is_target_verify():
            verify_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=verify_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = verify_wrapper
            self.forward_metadata = PrefillMetadata(verify_wrapper, False)
        elif forward_mode.is_draft_extend():
            draft_extend_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=draft_extend_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = draft_extend_wrapper
            self.forward_metadata = PrefillMetadata(draft_extend_wrapper, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            kv_len_arr_cpu = seq_lens_cpu[:bs]
            self.cuda_graph_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                kv_len_arr_cpu, dim=0
            )
            self.fast_decode_kwargs.update(
                {
                    "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu[: bs + 1],
                    "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu[: bs + 1],
                    "kv_len_arr_cpu": kv_len_arr_cpu,
                }
            )

            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
                init_metadata_replay=True,
                spec_info=spec_info,
                **self.fast_decode_kwargs,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        """Init the metadata for a forward pass."""
        self.mha_chunk_kv_cache.update_wrapper(forward_batch, disable_flashinfer_ragged)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        # CP contract:
        # When `cp_use_prefill(forward_batch)` is true, callers must provide `k/v` that
        # are already all-gathered and reranked into the global token order expected
        # by the CP metadata. This backend will not re-allgather/rerank the input KV.
        if forward_batch.attn_attend_prefix_cache is not None and any(
            forward_batch.extend_prefix_lens_cpu
        ):  # MHA Chunk
            assert self.enable_chunk_kv
            assert q_rope is None
            assert k_rope is None
            return self.mha_chunk_kv_cache.forward(q, k, v, layer, forward_batch)

        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        prefill_wrapper_paged = self.forward_metadata.prefill_wrapper
        use_prefill_cp = cp_use_prefill(forward_batch)
        cp_size = get_global_server_args().attn_cp_size

        # Save kv cache
        if save_kv_cache and k is not None:
            assert v is not None
            # #region debug-point H:mla-save-kv
            try:
                import json, os, time, urllib.request

                _evt = {
                    "sessionId": "cp-accuracy-drop",
                    "runId": "pre-fix",
                    "hypothesisId": "H",
                    "traceId": str(id(forward_batch)),
                    "location": "flashinfer_mla_backend.forward_extend:save-kv",
                    "msg": "[DEBUG] MLA KV save invoked",
                    "data": {
                        "save_kv_cache": bool(save_kv_cache),
                        "k_len": int(k.shape[0]),
                        "v_len": int(v.shape[0]),
                        "out_cache_loc_len": None
                        if cache_loc is None
                        else int(cache_loc.shape[0]),
                        "use_prefill_cp": bool(use_prefill_cp),
                        "is_rr": bool(is_prefill_cp_round_robin_split()),
                        "batch_size": getattr(forward_batch, "batch_size", None),
                    },
                    "ts": time.time_ns() // 1000000,
                }
                _u = "http://127.0.0.1:7777/event"
                try:
                    with open(".dbg/cp-accuracy-drop.env") as _f:
                        _env = _f.read().splitlines()
                    _u = next(
                        (l.split("=", 1)[1] for l in _env if l.startswith("DEBUG_SERVER_URL=")),
                        _u,
                    )
                except Exception:
                    pass
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            _u,
                            data=json.dumps(_evt).encode(),
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=0.2,
                    ).read()
                except Exception:
                    os.makedirs(".dbg", exist_ok=True)
                    with open(".dbg/trae-debug-log-cp-accuracy-drop.ndjson", "a") as _f:
                        _f.write(json.dumps(_evt) + "\n")
            except Exception:
                pass
            # #endregion
            if k_rope is not None:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer, cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
        if q_rope is not None:
            q = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )

        if self.forward_metadata.use_ragged:
            # ragged prefill
            if use_prefill_cp and q_rope is None and not is_prefill_cp_round_robin_split():
                assert k is not None and v is not None
                k_all = k.to(q.dtype).view(-1, layer.tp_k_head_num, layer.head_dim)
                v_all = v.to(q.dtype).view(-1, layer.tp_k_head_num, layer.v_head_dim)

                def _cp_ragged_mha_attn(
                    q_chunk, _cu_seqlens_q_cp, cache_seqlens_cp, _max_seqlen_q_cp
                ):
                    kv_len = int(cache_seqlens_cp[0].item())
                    qo_indptr = torch.tensor(
                        [0, q_chunk.shape[0]], device=q_chunk.device, dtype=torch.int32
                    )
                    kv_indptr = torch.tensor(
                        [0, kv_len], device=q_chunk.device, dtype=torch.int32
                    )
                    self.prefill_wrapper_ragged.begin_forward(
                        qo_indptr=qo_indptr,
                        kv_indptr=kv_indptr,
                        num_qo_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        head_dim_qk=layer.head_dim,
                        head_dim_vo=layer.v_head_dim,
                        q_data_type=q_chunk.dtype,
                        causal=True,
                    )
                    return self.prefill_wrapper_ragged.forward(
                        q_chunk.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k_all[:kv_len],
                        v_all[:kv_len],
                        causal=True,
                        sm_scale=layer.scaling,
                        logits_soft_cap=logits_soft_cap,
                    )

                o = cp_attn_forward_extend(
                    forward_batch,
                    q,
                    q.device,
                    _cp_ragged_mha_attn,
                )
            elif use_prefill_cp and q_rope is None and is_prefill_cp_round_robin_split():
                assert k is not None and v is not None
                q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                k_all = k.to(q.dtype).view(-1, layer.tp_k_head_num, layer.head_dim)
                v_all = v.to(q.dtype).view(-1, layer.tp_k_head_num, layer.v_head_dim)
                prefix_len = 0
                if (
                    forward_batch.seq_lens_cpu is not None
                    and len(forward_batch.seq_lens_cpu) == 1
                ):
                    prefix_len = max(
                        int(forward_batch.seq_lens_cpu[0]) - int(k_all.shape[0]),
                        0,
                    )
                cp_rank = get_attention_cp_rank()
                outputs = []
                for token_idx in range(q_all.shape[0]):
                    kv_len = prefix_len + cp_rank + token_idx * cp_size + 1
                    qo_indptr = torch.tensor(
                        [0, 1], device=q_all.device, dtype=torch.int32
                    )
                    kv_indptr = torch.tensor(
                        [0, kv_len], device=q_all.device, dtype=torch.int32
                    )
                    self.prefill_wrapper_ragged.begin_forward(
                        qo_indptr=qo_indptr,
                        kv_indptr=kv_indptr,
                        num_qo_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        head_dim_qk=layer.head_dim,
                        head_dim_vo=layer.v_head_dim,
                        q_data_type=q_all.dtype,
                        causal=True,
                    )
                    outputs.append(
                        self.prefill_wrapper_ragged.forward(
                            q_all[token_idx : token_idx + 1],
                            k_all[:kv_len],
                            v_all[:kv_len],
                            causal=True,
                            sm_scale=layer.scaling,
                            logits_soft_cap=logits_soft_cap,
                        )
                    )
                o = torch.cat(outputs, dim=0)
            else:
                if q_rope is not None:
                    q = torch.cat([q, q_rope], dim=-1)
                if k_rope is not None:
                    k = torch.cat([k, k_rope], dim=-1)
                qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                k_all = k.to(q.dtype).view(-1, layer.tp_k_head_num, layer.head_dim)
                v_all = v.to(q.dtype).view(-1, layer.tp_k_head_num, layer.v_head_dim)
                o = self.prefill_wrapper_ragged.forward(
                    qall,
                    k_all,
                    v_all,
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
        else:
            # mla paged prefill
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            if q_rope is None:
                qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                q, q_rope = (
                    qall[:, :, : layer.v_head_dim],
                    qall[:, :, layer.v_head_dim :],
                )
            if use_prefill_cp and not is_prefill_cp_round_robin_split():
                # CP + paged MLA: the wrapper.plan done in init_forward_metadata is
                # for the global qo_indptr. In CP, each rank holds only a q chunk,
                # so we must re-plan per-half using CP-provided kv_len (prefix-aware).
                if not hasattr(self, "_prefill_wrapper_paged_cp"):
                    self._prefill_wrapper_paged_cp = BatchMLAPagedAttentionWrapper(
                        self.workspace_buffer, backend="auto"
                    )

                k_nope_buf = k_buf[:, :, : layer.v_head_dim]
                k_rope_buf = k_buf[:, :, layer.v_head_dim :]
                q_all = torch.cat([q, q_rope], dim=-1)

                def _cp_paged_mla_attn(
                    q_chunk, _cu_seqlens_q_cp, cache_seqlens_cp, _max_seqlen_q_cp
                ):
                    kv_len = int(cache_seqlens_cp[0].item())
                    qo_indptr = torch.tensor(
                        [0, q_chunk.shape[0]],
                        device=q_chunk.device,
                        dtype=torch.int32,
                    )
                    kv_indptr = torch.tensor(
                        [0, kv_len], device=q_chunk.device, dtype=torch.int32
                    )
                    paged_kernel_lens = torch.tensor(
                        [kv_len], device=q_chunk.device, dtype=torch.int32
                    )
                    # Plan a temporary wrapper for this (q_chunk, kv_len) pair.
                    self.indices_updater_decode.call_begin_forward(
                        self._prefill_wrapper_paged_cp,
                        forward_batch.req_pool_indices[:1],
                        paged_kernel_lens,
                        kv_len,
                        qo_indptr,
                        kv_indptr,
                        init_metadata_replay=False,
                        spec_info=None,
                    )
                    q_nope = q_chunk[:, :, : layer.v_head_dim]
                    q_rope = q_chunk[:, :, layer.v_head_dim :]
                    o_chunk = q_nope.new_empty(q_nope.shape)
                    return self._prefill_wrapper_paged_cp.run(
                        q_nope, q_rope, k_nope_buf, k_rope_buf, out=o_chunk
                    )

                o = cp_attn_forward_extend(
                    forward_batch,
                    q_all,
                    q.device,
                    _cp_paged_mla_attn,
                )
            elif use_prefill_cp and is_prefill_cp_round_robin_split():
                # CP + round-robin + paged MLA:
                # each rank owns interleaved query tokens, so a single causal prefill
                # plan with local q is invalid. Re-plan per token using the global
                # causal KV length for that token position.
                if not hasattr(self, "_prefill_wrapper_paged_cp"):
                    self._prefill_wrapper_paged_cp = BatchMLAPagedAttentionWrapper(
                        self.workspace_buffer, backend="auto"
                    )

                k_nope_buf = k_buf[:, :, : layer.v_head_dim]
                k_rope_buf = k_buf[:, :, layer.v_head_dim :]
                prefix_len = 0
                if (
                    forward_batch.extend_prefix_lens_cpu is not None
                    and len(forward_batch.extend_prefix_lens_cpu) == 1
                ):
                    prefix_len = int(forward_batch.extend_prefix_lens_cpu[0])

                cp_rank = get_attention_cp_rank()
                # #region debug-point B:rr-mla-branch
                try:
                    import json, os, time, urllib.request

                    _evt = {
                        "sessionId": "cp-accuracy-drop",
                        "runId": "pre-fix",
                        "hypothesisId": "B",
                        "traceId": str(id(forward_batch)),
                        "location": "flashinfer_mla_backend.forward_extend:rr-enter",
                        "msg": "[DEBUG] entered MLA round-robin paged prefill branch",
                        "data": {
                            "batch_size": getattr(forward_batch, "batch_size", None),
                            "cp_rank": int(cp_rank),
                            "cp_size": int(cp_size),
                            "local_q_len": int(q.shape[0]),
                            "k_buf_len": int(k_buf.shape[0]),
                            "prefix_len": int(prefix_len),
                            "req_pool_indices": forward_batch.req_pool_indices[:1].tolist()
                            if getattr(forward_batch, "req_pool_indices", None) is not None
                            else None,
                        },
                        "ts": time.time_ns() // 1000000,
                    }
                    _u = "http://127.0.0.1:7777/event"
                    _env_path = ".dbg/cp-accuracy-drop.env"
                    try:
                        with open(_env_path) as _f:
                            _env = _f.read().splitlines()
                        _u = next(
                            (
                                l.split("=", 1)[1]
                                for l in _env
                                if l.startswith("DEBUG_SERVER_URL=")
                            ),
                            _u,
                        )
                        _evt["sessionId"] = next(
                            (
                                l.split("=", 1)[1]
                                for l in _env
                                if l.startswith("DEBUG_SESSION_ID=")
                            ),
                            _evt["sessionId"],
                        )
                    except Exception:
                        pass
                    try:
                        urllib.request.urlopen(
                            urllib.request.Request(
                                _u,
                                data=json.dumps(_evt).encode(),
                                headers={"Content-Type": "application/json"},
                            ),
                            timeout=0.2,
                        ).read()
                    except Exception:
                        os.makedirs(".dbg", exist_ok=True)
                        with open(
                            ".dbg/trae-debug-log-cp-accuracy-drop.ndjson", "a"
                        ) as _f:
                            _f.write(json.dumps(_evt) + "\n")
                except Exception:
                    pass
                # #endregion
                outputs = []
                for token_idx in range(q.shape[0]):
                    kv_len = prefix_len + cp_rank + token_idx * cp_size + 1
                    if token_idx < 2 or token_idx == q.shape[0] - 1:
                        # #region debug-point D:rr-mla-kvlen
                        try:
                            import json, os, time, urllib.request

                            _evt = {
                                "sessionId": "cp-accuracy-drop",
                                "runId": "pre-fix",
                                "hypothesisId": "D",
                                "traceId": str(id(forward_batch)),
                                "location": "flashinfer_mla_backend.forward_extend:rr-token",
                                "msg": "[DEBUG] MLA round-robin token kv_len computed",
                                "data": {
                                    "token_idx": int(token_idx),
                                    "global_token_idx": int(cp_rank + token_idx * cp_size),
                                    "kv_len": int(kv_len),
                                    "prefix_len": int(prefix_len),
                                    "cp_rank": int(cp_rank),
                                    "cp_size": int(cp_size),
                                    "local_q_len": int(q.shape[0]),
                                    "k_buf_len": int(k_buf.shape[0]),
                                },
                                "ts": time.time_ns() // 1000000,
                            }
                            _u = "http://127.0.0.1:7777/event"
                            _env_path = ".dbg/cp-accuracy-drop.env"
                            try:
                                with open(_env_path) as _f:
                                    _env = _f.read().splitlines()
                                _u = next(
                                    (
                                        l.split("=", 1)[1]
                                        for l in _env
                                        if l.startswith("DEBUG_SERVER_URL=")
                                    ),
                                    _u,
                                )
                                _evt["sessionId"] = next(
                                    (
                                        l.split("=", 1)[1]
                                        for l in _env
                                        if l.startswith("DEBUG_SESSION_ID=")
                                    ),
                                    _evt["sessionId"],
                                )
                            except Exception:
                                pass
                            try:
                                urllib.request.urlopen(
                                    urllib.request.Request(
                                        _u,
                                        data=json.dumps(_evt).encode(),
                                        headers={"Content-Type": "application/json"},
                                    ),
                                    timeout=0.2,
                                ).read()
                            except Exception:
                                os.makedirs(".dbg", exist_ok=True)
                                with open(
                                    ".dbg/trae-debug-log-cp-accuracy-drop.ndjson", "a"
                                ) as _f:
                                    _f.write(json.dumps(_evt) + "\n")
                        except Exception:
                            pass
                        # #endregion
                    qo_indptr = torch.tensor(
                        [0, 1], device=q.device, dtype=torch.int32
                    )
                    kv_indptr = torch.tensor(
                        [0, kv_len], device=q.device, dtype=torch.int32
                    )
                    paged_kernel_lens = torch.tensor(
                        [kv_len], device=q.device, dtype=torch.int32
                    )
                    self.indices_updater_decode.call_begin_forward(
                        self._prefill_wrapper_paged_cp,
                        forward_batch.req_pool_indices[:1],
                        paged_kernel_lens,
                        kv_len,
                        qo_indptr,
                        kv_indptr,
                        init_metadata_replay=False,
                        spec_info=None,
                    )
                    q_nope = q[token_idx : token_idx + 1]
                    q_rope_i = q_rope[token_idx : token_idx + 1]
                    o_token = q_nope.new_empty(q_nope.shape)
                    outputs.append(
                        self._prefill_wrapper_paged_cp.run(
                            q_nope,
                            q_rope_i,
                            k_nope_buf,
                            k_rope_buf,
                            out=o_token,
                        )
                    )
                o = torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)
            else:
                o = q.new_empty(q.shape)
                planned_q_tokens = int(self._last_prefill_qo_indptr[-1].item())
                actual_q_tokens = q.shape[0]
                assert actual_q_tokens == planned_q_tokens, (
                    "FlashInfer MLA paged prefill q token count mismatch: "
                    f"q.shape[0]={actual_q_tokens}, qo_indptr[-1]={planned_q_tokens}, "
                    f"batch_size={forward_batch.batch_size}, "
                    f"seq_lens={forward_batch.seq_lens.tolist()}"
                )
                o = prefill_wrapper_paged.run(
                    q,
                    q_rope,
                    k_buf[:, :, : layer.v_head_dim],
                    k_buf[:, :, layer.v_head_dim :],
                    out=o,
                )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        decode_wrapper = self.forward_metadata.decode_wrapper
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Reshape inputs
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )

        o = q_nope.new_empty(q_nope.shape)
        # Direct call to run without the wrapper
        o = decode_wrapper.run(
            q_nope,
            q_rope,
            k_buffer[:, :, : layer.v_head_dim],
            k_buffer[:, :, layer.v_head_dim :],
            out=o,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashInferMLAIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.dtype
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.q_indptr = attn_backend.q_indptr_decode

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        **fast_decode_kwargs,
    ):
        decode_wrapper = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.q_indptr,
            self.kv_indptr,
            init_metadata_replay,
            spec_info,
            **fast_decode_kwargs,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        **fast_decode_kwargs,
    ):
        bs = len(req_pool_indices)
        q_indptr = q_indptr[: bs + 1]
        kv_lens = paged_kernel_lens.to(torch.int32)
        sm_scale = self.scaling
        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            device = self.attn_backend.device
            kv_indices = (
                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device=device)
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"]
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        if not init_metadata_replay:
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                self.data_type,
            )
        else:
            wrapper.plan(
                fast_decode_kwargs["qo_indptr_cpu"],
                fast_decode_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_decode_kwargs["kv_len_arr_cpu"],
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                self.data_type,
            )


class FlashInferMLAIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype
        self.attn_backend = attn_backend
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
    ):
        # In the MLA prefill updater, ragged mode is only enabled for no-prefix
        # extends, so both ragged and paged paths use the current extend lengths.
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            spec_info,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # Speculative flows (verify/draft-extend) can pass prefix_lens=None.
            # In that case, the query length equals seq_lens.
            if prefix_lens is None:
                qo_lens = seq_lens
            else:
                qo_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(qo_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            # Piecewise CUDA graph padding:
            # In PCG, input_ids are padded to a static token count, so q.shape[0] can be
            # larger than the real extend tokens (sum(qo_lens)). FlashInfer expects
            # qo_indptr[-1] to match q.shape[0], otherwise it may read invalid KV indices.
            # Append a dummy request for the padding tokens; map its KV to slot 0 (scratch).
            fwd_ctx = get_forward_context()
            pcg_num_tokens = fwd_ctx.num_tokens if fwd_ctx is not None else None
            actual_qo_tokens = (
                fwd_ctx.forward_batch.extend_num_tokens if fwd_ctx is not None else None
            )
            extra_kv = 0
            if (
                pcg_num_tokens is not None
                and actual_qo_tokens is not None
                and pcg_num_tokens > actual_qo_tokens
            ):
                extra_kv = pcg_num_tokens - actual_qo_tokens

            kv_indices = torch.empty(
                paged_kernel_lens_sum + extra_kv,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )

            bs_eff = bs
            if extra_kv > 0:
                kv_start = paged_kernel_lens_sum  # equals kv_indptr[-1] (no .item() needed)
                kv_indices[kv_start : kv_start + extra_kv] = 0
                qo_indptr = torch.cat(
                    [qo_indptr, qo_indptr.new_tensor([pcg_num_tokens])]
                )
                kv_indptr = torch.cat(
                    [kv_indptr, kv_indptr.new_tensor([kv_start + extra_kv])]
                )
                bs_eff = bs + 1
        else:
            # SpecInput provides KV indices/indptr; still compute qo_indptr from seq lengths.
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            kv_indptr = kv_indptr[: bs + 1]
            if prefix_lens is None:
                qo_lens = seq_lens
            else:
                qo_lens = seq_lens - prefix_lens
            qo_indptr = qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(qo_lens, dim=0)

        if use_ragged:
            # ragged prefill
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )
        else:
            # mla paged prefill
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            self.attn_backend._last_prefill_qo_indptr = qo_indptr
            wrapper_paged.plan(
                qo_indptr[: bs_eff + 1] if spec_info is None else qo_indptr[: bs + 1],
                kv_indptr[: bs_eff + 1] if spec_info is None else kv_indptr[: bs + 1],
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


class FlashInferMLAMultiStepDraftBackend:
    """
    Wrap multiple flashinfer mla attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        if topk > 1:
            raise ValueError(
                "Currently Flashinfer MLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferMLAAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    q_indptr_decode_buf=self.q_indptr_decode,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.zeros(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def fast_mla_decode_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """A faster version of BatchMLAPagedAttentionWrapper::plan,
    for skipping the stream synchronization in original plan function during
    cuda graph replaying.
    """
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale

    try:
        # Standard version with just the required arguments (no use_profiler)
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
    except Exception as e:
        raise RuntimeError(f"Error in alternate MLA plan: {e}")
