"""Numerical parity tests: our custom ops vs their torch equivalents."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extension.build import load_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires a CUDA GPU")

STAGES = ["naive", "coalesced", "smem", "blocktile_1d", "blocktile_2d",
          "vectorized", "cublas"]

SHAPES = [(256, 256, 256), (512, 384, 768), (1024, 1024, 1024),
          (129, 257, 65), (1, 4096, 4096), (333, 111, 222)]

@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_sgemm_parity(stage, M, N, K):
    ext = load_extension()
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    ref = (A.double() @ B.double())
    got = ext.sgemm(A, B, stage)
    err = (got.double() - ref).abs().max().item()
    assert err <= 3e-6 * K, f"{stage} {M}x{N}x{K}: max abs err {err:.3e}"

def test_sgemm_batched_leading_dims():
    """A (batch, seq, K) activation must work like a (batch*seq, K) matrix."""
    ext = load_extension()
    torch.manual_seed(0)
    A = torch.randn(4, 128, 512, device="cuda")
    B = torch.randn(512, 256, device="cuda")
    got = ext.sgemm(A, B, "vectorized")
    assert got.shape == (4, 128, 256)
    ref = A.double() @ B.double()
    assert (got.double() - ref).abs().max().item() <= 3e-6 * 512

def test_sgemm_alpha_beta():
    ext = load_extension()
    torch.manual_seed(0)
    A = torch.randn(256, 512, device="cuda")
    B = torch.randn(512, 128, device="cuda")
    C = torch.randn(256, 128, device="cuda")
    C0 = C.clone()
    got = ext.sgemm(A, B, "blocktile_2d", alpha=0.5, beta=2.0, out=C)
    ref = 0.5 * (A.double() @ B.double()) + 2.0 * C0.double()
    assert (got.double() - ref).abs().max().item() <= 3e-6 * 512

@pytest.mark.parametrize("rows,cols", [(64, 128), (128, 257), (1024, 1024),
                                       (7, 4096), (4096, 31)])
@pytest.mark.parametrize("fused", [True, False])
def test_softmax_parity(rows, cols, fused):
    ext = load_extension()
    torch.manual_seed(0)
    X = torch.randn(rows, cols, device="cuda")
    got = ext.softmax(X, fused=fused)
    ref = torch.softmax(X, dim=-1)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

def test_softmax_extreme_values():
    """The streaming-max formulation must survive inputs that overflow exp()."""
    ext = load_extension()
    X = torch.tensor([[500.0, 501.0, 499.0, -500.0]], device="cuda")
    got = ext.softmax(X, fused=True)
    ref = torch.softmax(X, dim=-1)
    assert torch.isfinite(got).all(), "softmax produced non-finite values"
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("rows,cols", [(64, 128), (128, 257), (1024, 768)])
def test_layernorm_parity(rows, cols):
    ext = load_extension()
    torch.manual_seed(0)
    X = torch.randn(rows, cols, device="cuda")
    g = torch.randn(cols, device="cuda")
    b = torch.randn(cols, device="cuda")
    got = ext.layernorm(X, g, b, 1e-5)
    ref = torch.nn.functional.layer_norm(X, (cols,), g, b, 1e-5)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)

def test_layernorm_large_offset():
    """Welford must hold up where E[x^2]-E[x]^2 would cancel catastrophically."""
    ext = load_extension()
    torch.manual_seed(0)
    X = torch.randn(64, 1024, device="cuda") + 10000.0
    got = ext.layernorm(X, None, None, 1e-5)
    ref = torch.nn.functional.layer_norm(X, (1024,), None, None, 1e-5)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
