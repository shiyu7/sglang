import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_DEEPGEMM_SM90_FP8_FP4_CONTIG,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()


# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):

            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def _ensure_cuda(
    pair: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],
        pair[1].cuda() if not pair[1].is_cuda else pair[1],
    )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs, rhs, out, m_indices, **fp4_kwargs
        )


def _get_sm90_fp8_fp4_tile_overrides(m: int, n: int, num_groups: int):
    if m >= num_groups * 256 and n >= 1024:
        return 256, 64
    return None, None


def _check_fp8_fp4_contig_inputs(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    lhs_data, lhs_scale = lhs
    rhs_data, rhs_scale = rhs

    assert lhs_data.dtype == torch.float8_e4m3fn, lhs_data.dtype
    assert rhs_data.dtype == torch.int8, rhs_data.dtype
    assert rhs_scale.dtype == torch.float32, rhs_scale.dtype
    assert not getattr(rhs_scale, "format_ue8m0", False), (
        "SM90 FP8-FP4 fused DeepGEMM path expects canonical FP4 scales, "
        "not pre-transformed UE8M0 scales."
    )
    assert out.dtype == torch.bfloat16, out.dtype
    assert m_indices.dtype == torch.int32, m_indices.dtype

    m, k = lhs_data.shape
    num_groups, n, half_k = rhs_data.shape
    assert half_k * 2 == k, f"{rhs_data.shape=} is not packed FP4 for {k=}"
    assert out.shape == (m, n), f"{out.shape=} != {(m, n)}"
    assert m_indices.shape == (m,), f"{m_indices.shape=} != {(m,)}"
    assert lhs_scale.shape == (
        m,
        (k + 127) // 128,
    ), f"{lhs_scale.shape=} is not canonical FP8 activation scale"
    assert rhs_scale.shape == (
        num_groups,
        n,
        (k + 127) // 128,
    ), f"{rhs_scale.shape=} is not canonical FP4 weight scale"


def grouped_gemm_nt_f8fp4bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    block_m_override: Optional[int] = None,
    block_n_override: Optional[int] = None,
    decode_stub: bool = False,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_CONTIG_SM90

    if m == 0:
        return

    if not ENABLE_DEEPGEMM_SM90_FP8_FP4_CONTIG:
        return grouped_gemm_nt_f8f8bf16_contig(
            lhs,
            rhs,
            out,
            m_indices,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )

    _check_fp8_fp4_contig_inputs(lhs, rhs, out, m_indices)

    if block_m_override is None and block_n_override is None:
        block_m_override, block_n_override = _get_sm90_fp8_fp4_tile_overrides(
            m, n, num_groups
        )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
            lhs,
            rhs,
            out,
            m_indices,
            gran_k=128,
            compiled_dims="nk",
            use_psum_layout=False,
            block_m_override=block_m_override,
            block_n_override=block_n_override,
            decode_stub=decode_stub,
        )


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_deep_gemm_config(gpu_id, server_args)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
