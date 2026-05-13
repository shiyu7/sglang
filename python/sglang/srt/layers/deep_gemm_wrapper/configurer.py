import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_blackwell_supported,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)


def _compute_enable_sm90_fp8_fp4_contig():
    mode = (envs.SGLANG_DEEPGEMM_SM90_FP8_FP4_CONTIG.get() or "auto").lower()
    if mode in ("0", "false", "off", "disable", "disabled"):
        return False
    force_enable = mode in ("1", "true", "on", "enable", "enabled")
    if mode != "auto" and not force_enable:
        logger.warning(
            "Invalid SGLANG_DEEPGEMM_SM90_FP8_FP4_CONTIG=%s; disabling the path.",
            mode,
        )
        return False

    sm_version = get_device_sm()
    supported_platform = _is_cuda and sm_version == 90
    if not ENABLE_JIT_DEEPGEMM or not supported_platform:
        if force_enable:
            logger.warning(
                "SGLANG_DEEPGEMM_SM90_FP8_FP4_CONTIG is enabled but requires "
                "DeepGEMM on CUDA SM90; got sm=%s.",
                sm_version,
            )
        return False

    try:
        import deep_gemm
    except ImportError:
        if force_enable:
            logger.warning(
                "SGLANG_DEEPGEMM_SM90_FP8_FP4_CONTIG is enabled but deep_gemm "
                "cannot be imported."
            )
        return False

    has_kernel = hasattr(
        deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma"
    )
    if force_enable and not has_kernel:
        logger.warning(
            "SGLANG_DEEPGEMM_SM90_FP8_FP4_CONTIG is enabled but the installed "
            "deep_gemm package does not expose "
            "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma."
        )
    return has_kernel


ENABLE_DEEPGEMM_SM90_FP8_FP4_CONTIG = _compute_enable_sm90_fp8_fp4_contig()
