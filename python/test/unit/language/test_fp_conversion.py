
import torch
import triton
import triton.language as tl
import pytest

# Kernels
@triton.jit
def kernel_fp32_to_fp16_roundtrip(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float16).to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_fp16_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_fp32_to_bf16_roundtrip(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_bf16_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)

# Test Cases
@pytest.mark.parametrize("test_name, kernel, x, y_ref, atol, rtol", [
    ("fp32_to_fp16_roundtrip", kernel_fp32_to_fp16_roundtrip,
     torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 65504, -65504], dtype=torch.float32, device='cuda'),
     torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 65504, -65504], dtype=torch.float32, device='cuda').to(torch.float16).to(torch.float32),
     1e-3, 1e-3),
    ("fp16_to_fp32", kernel_fp16_to_fp32,
     torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.float16, device='cuda'),
     torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.float16, device='cuda').to(torch.float32),
     1e-6, 1e-6),
    ("fp32_to_bf16_roundtrip", kernel_fp32_to_bf16_roundtrip,
     torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 300.5, -400.6], dtype=torch.float32, device='cuda'),
     torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 300.5, -400.6], dtype=torch.float32, device='cuda').to(torch.bfloat16).to(torch.float32),
     1e-2, 1e-2),
    ("bf16_to_fp32", kernel_bf16_to_fp32,
     torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.bfloat16, device='cuda'),
     torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.bfloat16, device='cuda').to(torch.float32),
     1e-2, 1e-2),
])
def test_fp_conversion(test_name, kernel, x, y_ref, atol, rtol):
    y = torch.empty_like(y_ref)
    kernel[(1,)](x, y, BLOCK_SIZE=x.shape[0])
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
