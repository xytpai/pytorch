# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor import config
from torch._inductor.runtime import flydsl_gemm
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import ensure_flydsl_available


class TestFlyDSLGemm(TestCase):
    def test_dtype_name(self):
        self.assertEqual(flydsl_gemm._dtype_name(torch.float16), "f16")
        self.assertEqual(flydsl_gemm._dtype_name(torch.bfloat16), "bf16")
        self.assertEqual(flydsl_gemm._dtype_name(torch.float32), "f32")
        with self.assertRaises(NotImplementedError):
            flydsl_gemm._dtype_name(torch.float64)

    @unittest.skipIf(
        not (torch.version.hip and torch.cuda.is_available() and ensure_flydsl_available()),
        "FlyDSL GEMM requires a ROCm GPU and FlyDSL",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="FLYDSL",
        autotune_fallback_to_aten=False,
    )
    def test_compile_mm_smoke(self):
        def fn(a, b):
            return torch.mm(a, b)

        opt_fn = torch.compile(fn)
        a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)

        actual = opt_fn(a, b)
        expected = fn(a, b)

        self.assertTrue(torch.allclose(actual, expected, atol=0.1, rtol=0.1))


if __name__ == "__main__":
    run_tests()
