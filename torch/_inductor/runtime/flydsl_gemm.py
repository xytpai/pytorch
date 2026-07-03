from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Callable

import torch


def _add_flydsl_paths() -> None:
    """Allow using a sibling FlyDSL checkout without requiring installation."""
    candidates: list[Path] = []

    for env_name in ("TORCHINDUCTOR_FLYDSL_PATH", "FLYDSL_HOME"):
        if value := os.environ.get(env_name):
            candidates.append(Path(value))

    # Source-tree convenience for the local development layout:
    #   <workspace>/pytorch
    #   <workspace>/FlyDSL
    try:
        candidates.append(Path(__file__).resolve().parents[4] / "FlyDSL")
    except IndexError:
        pass

    for root in candidates:
        if not root.exists():
            continue
        for path in (root, root / "python"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "f16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "f32"
    raise NotImplementedError(f"FlyDSL GEMM does not support dtype {dtype}")


@functools.lru_cache(maxsize=512)
def _rdna_launcher(
    arch: str,
    m: int,
    n: int,
    k: int,
    in_dtype: str,
    out_dtype: str,
) -> Callable[..., object]:
    _add_flydsl_paths()

    if arch.startswith("gfx11"):
        from kernels.rdna3_f16_gemm import create_wmma_gemm_module
    else:
        from kernels.rdna_f16_gemm import create_wmma_gemm_module

    launch_fn, _, _, _ = create_wmma_gemm_module(
        m,
        n,
        k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    return launch_fn


def _run_rdna_gemm(
    out: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    arch: str,
) -> torch.Tensor:
    m, k = mat1.shape
    k2, n = mat2.shape
    if k != k2:
        raise RuntimeError(f"FlyDSL GEMM expected K dimensions to match, got {k} and {k2}")

    in_dtype = _dtype_name(mat1.dtype)
    out_dtype = _dtype_name(out.dtype)
    launcher = _rdna_launcher(arch, m, n, k, in_dtype, out_dtype)
    mat2_t = mat2.t().contiguous()
    launcher(out, mat1.contiguous(), mat2_t, torch.cuda.current_stream())
    return out


def _run_cdna_gemm(
    out: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    _add_flydsl_paths()
    from kernels.hgemm_splitk import hgemm_splitk_

    mat2_t = mat2.t().contiguous()
    hgemm_splitk_(out, mat1.contiguous(), mat2_t, stream=torch.cuda.current_stream())
    return out


def mm(mat1: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """Out-variant wrapper used by the Inductor FlyDSL GEMM extern choice."""
    if mat1.dim() != 2 or mat2.dim() != 2 or out.dim() != 2:
        raise RuntimeError("FlyDSL GEMM only supports 2D tensors")

    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"FlyDSL GEMM expected matching input dtypes, got {mat1.dtype} and {mat2.dtype}"
        )

    if mat1.device.type != "cuda" or mat2.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("FlyDSL GEMM only supports CUDA/ROCm tensors")

    _add_flydsl_paths()
    from flydsl.runtime.device import get_rocm_arch

    arch = str(get_rocm_arch() or "")
    if arch.startswith(("gfx11", "gfx120")):
        return _run_rdna_gemm(out, mat1, mat2, arch)
    if arch.startswith(("gfx9", "gfx94", "gfx95")):
        return _run_cdna_gemm(out, mat1, mat2)

    raise RuntimeError(f"FlyDSL GEMM does not support ROCm arch {arch!r}")
