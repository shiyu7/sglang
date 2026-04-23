# mapping on device memory, host memory and memory allocator

import weakref
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import get_num_new_pages

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseNSATokenToKVPool(NSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: NSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def _debug_hisparse_leak_enabled(self) -> bool:
        import os as _os

        return _os.environ.get("SGLANG_DEBUG_HISPARSE_LEAK") == "1"

    def _debug_page_summary(self, indices: torch.Tensor) -> str:
        valid = indices[indices > 0]
        if valid.numel() == 0:
            return "valid_numel=0 page_numel=0 pages=[]"

        pages = torch.unique(valid // self.page_size)
        max_pages_to_show = 32
        pages_list = pages.tolist()
        if len(pages_list) > max_pages_to_show:
            shown_pages = pages_list[:max_pages_to_show]
            pages_repr = f"{shown_pages}...(total={len(pages_list)})"
        else:
            pages_repr = str(pages_list)

        return (
            f"valid_numel={int(valid.numel())} "
            f"page_numel={int(pages.numel())} "
            f"pages={pages_repr}"
        )

    def _debug_log_mapping_write(
        self,
        caller: str,
        logical_indices: torch.Tensor,
        hisparse_indices: torch.Tensor,
    ) -> None:
        if not self._debug_hisparse_leak_enabled():
            return
        print(
            f"[HiSparseDebug] mapping_write "
            f"caller={caller} "
            f"logical_indices={logical_indices.tolist()} "
            f"hisparse_indices={hisparse_indices.tolist()} "
            f"{self._debug_page_summary(hisparse_indices)}",
            flush=True,
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "Page size = 1 is not supported in HiSparse allocator"
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(
                hisparse_indices[need_size:],
                caller="alloc_device_buffer.trim_old_mapping",
                emit_trace=True,
            )
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])
        return buffer_indices

    def free_hisparse_indices(
        self,
        buffer_indices: torch.Tensor,
        caller: str = "unknown",
        emit_trace: bool = False,
    ):
        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        if self._debug_hisparse_leak_enabled():
            import traceback as _tb

            _before = self.hisparse_attn_allocator.available_size()
            _valid = buffer_indices[buffer_indices > 0]
            _trace = "".join(_tb.format_stack()) if emit_trace else ""
            print(
                f"[HiSparseDebug] free_hisparse_indices.begin "
                f"caller={caller} "
                f"input_numel={int(buffer_indices.numel())} "
                f"{self._debug_page_summary(_valid)} "
                f"hisparse_available_before={_before}"
                + (f"\n{_trace}" if _trace else ""),
                flush=True,
            )
            self.hisparse_attn_allocator.free(_valid)
            _after = self.hisparse_attn_allocator.available_size()
            print(
                f"[HiSparseDebug] free_hisparse_indices.end "
                f"caller={caller} "
                f"hisparse_available_after={_after} delta={_after - _before}",
                flush=True,
            )
        else:
            self.hisparse_attn_allocator.free(buffer_indices[buffer_indices > 0])

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        hisparse_last_locs = self._kvcache._translate_loc_to_hisparse_device(last_locs)
        return hisparse_last_locs

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self._debug_log_mapping_write(
            "allocator.alloc_extend", logical_indices, hisparse_indices
        )
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        import os as _os
        if _os.environ.get("SGLANG_DEBUG_HISPARSE_LEAK") == "1":
            print(
                f"[HiSparseDebug] alloc_extend "
                f"logical_numel={int(logical_indices.numel())} "
                f"hisparse_numel={int(hisparse_indices.numel())} "
                f"num_new_pages={int(num_new_pages)} "
                f"hisparse_available={self.hisparse_attn_allocator.available_size()} "
                f"logical_available={self.logical_attn_allocator.available_size()}",
                flush=True,
            )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        return logical_indices

    def alloc_decode_debug(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_decode(
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
        )

        if logical_indices is None or hisparse_indices is None:
            return None

        self._debug_log_mapping_write(
            "allocator.alloc_decode_debug", logical_indices, hisparse_indices
        )
        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        import os as _os
        if _os.environ.get("SGLANG_DEBUG_HISPARSE_LEAK") == "1":
            print(
                f"[HiSparseDebug] alloc_decode_debug "
                f"logical_numel={int(logical_indices.numel())} "
                f"hisparse_numel={int(hisparse_indices.numel())} "
                f"hisparse_available={self.hisparse_attn_allocator.available_size()} "
                f"logical_available={self.logical_attn_allocator.available_size()}",
                flush=True,
            )

        return logical_indices

    def free_hisparse(self, free_indices: torch.Tensor):
        import os as _os
        _dbg = _os.environ.get("SGLANG_DEBUG_HISPARSE_LEAK") == "1"
        if _dbg:
            _before = self.hisparse_attn_allocator.available_size()
            print(
                f"[HiSparseDebug] free_hisparse.begin "
                f"free_indices_numel={int(free_indices.numel())} "
                f"hisparse_available_before={_before}",
                flush=True,
            )
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(
            hisparse_indices,
            caller="allocator.free_hisparse",
            emit_trace=True,
        )
        self.full_to_hisparse_device_index_mapping[free_indices] = 0
        if _dbg:
            _after = self.hisparse_attn_allocator.available_size()
            print(
                f"[HiSparseDebug] free_hisparse.end "
                f"translated_numel={int(hisparse_indices.numel())} "
                f"hisparse_available_after={_after} delta={_after - _before}",
                flush=True,
            )

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        import traceback as _tb
        import os as _os
        if _os.environ.get("SGLANG_DEBUG_HISPARSE_LEAK") == "1":
            print(
                f"[HiSparseDebug] free called with numel={int(free_index.numel())} "
                f"logical_avail={self.logical_attn_allocator.available_size()} "
                f"hisparse_avail={self.hisparse_attn_allocator.available_size()}\n"
                f"{''.join(_tb.format_stack())}",
                flush=True,
            )

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)

        _logical_avail = self.logical_attn_allocator.available_size()
        _hisparse_avail = self.hisparse_attn_allocator.available_size()
        if _logical_avail > self.logical_attn_allocator.size or _hisparse_avail > self.hisparse_attn_allocator.size:
            print(
                f"[HiSparseDebug] ASSERTION WOULD FAIL  "
                f"logical_avail={_logical_avail}/{self.logical_attn_allocator.size}  "
                f"hisparse_avail={_hisparse_avail}/{self.hisparse_attn_allocator.size}\n"
                f"free_index={free_index.tolist()}\n"
                f"{''.join(_tb.format_stack())}",
                flush=True,
            )
        assert _logical_avail <= self.logical_attn_allocator.size
        assert _hisparse_avail <= self.hisparse_attn_allocator.size
