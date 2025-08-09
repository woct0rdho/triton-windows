import torch
import triton
import triton.language as tl

# Teste específico para funções de conversão FP8

@triton.jit
def kernel_fp32_to_fp8_e4m3(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float8e4nv)  # float8e4nv é o tipo E4M3 no Triton
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)

@triton.jit
def kernel_fp32_to_fp8_e5m2(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float8e5)  # float8e5 é o tipo E5M2 no Triton
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)

@triton.jit
def kernel_fp8_e4m3_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)

@triton.jit
def kernel_fp8_e5m2_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)

def test_fp32_to_fp8_e4m3():
    """Teste de conversão FP32 para FP8 E4M3"""
    print("Testing FP32 to FP8 E4M3 conversion...")
    try:
        x = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float32, device='cuda')
        y = torch.empty_like(x, dtype=torch.uint8)  # FP8 é armazenado como uint8
        
        # Executar kernel
        kernel_fp32_to_fp8_e4m3[(1,)](x, y, BLOCK_SIZE=x.shape[0])
        
        print("  PASSED - FP32 to FP8 E4M3 conversion executed without errors")
        return True
    except Exception as e:
        print(f"  FAILED - Error: {e}")
        return False

def test_fp32_to_fp8_e5m2():
    """Teste de conversão FP32 para FP8 E5M2"""
    print("Testing FP32 to FP8 E5M2 conversion...")
    try:
        x = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float32, device='cuda')
        y = torch.empty_like(x, dtype=torch.uint8)  # FP8 é armazenado como uint8
        
        # Executar kernel
        kernel_fp32_to_fp8_e5m2[(1,)](x, y, BLOCK_SIZE=x.shape[0])
        
        print("  PASSED - FP32 to FP8 E5M2 conversion executed without errors")
        return True
    except Exception as e:
        print(f"  FAILED - Error: {e}")
        return False

def test_fp8_e4m3_to_fp32():
    """Teste de conversão FP8 E4M3 para FP32"""
    print("Testing FP8 E4M3 to FP32 conversion...")
    try:
        # Criar valores FP8 E4M3
        x = torch.tensor([0x30, 0x38, 0x48, 0x50], dtype=torch.uint8, device='cuda')  # Valores de exemplo
        y = torch.empty_like(x, dtype=torch.float32)
        
        # Executar kernel
        kernel_fp8_e4m3_to_fp32[(1,)](x, y, BLOCK_SIZE=x.shape[0])
        
        print("  PASSED - FP8 E4M3 to FP32 conversion executed without errors")
        return True
    except Exception as e:
        print(f"  FAILED - Error: {e}")
        return False

def test_fp8_e5m2_to_fp32():
    """Teste de conversão FP8 E5M2 para FP32"""
    print("Testing FP8 E5M2 to FP32 conversion...")
    try:
        # Criar valores FP8 E5M2
        x = torch.tensor([0x20, 0x28, 0x50, 0x60], dtype=torch.uint8, device='cuda')  # Valores de exemplo
        y = torch.empty_like(x, dtype=torch.float32)
        
        # Executar kernel
        kernel_fp8_e5m2_to_fp32[(1,)](x, y, BLOCK_SIZE=x.shape[0])
        
        print("  PASSED - FP8 E5M2 to FP32 conversion executed without errors")
        return True
    except Exception as e:
        print(f"  FAILED - Error: {e}")
        return False

if __name__ == "__main__":
    print("Running FP8 Conversion Tests")
    print("=" * 40)
    
    results = []
    results.append(test_fp32_to_fp8_e4m3())
    results.append(test_fp32_to_fp8_e5m2())
    results.append(test_fp8_e4m3_to_fp32())
    results.append(test_fp8_e5m2_to_fp32())
    
    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All FP8 conversion tests PASSED!")
    else:
        print("Some FP8 conversion tests FAILED!")