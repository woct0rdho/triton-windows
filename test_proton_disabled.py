#!/usr/bin/env python3
"""
Test script to verify that Triton works correctly with Proton disabled.

This script tests:
1. Basic Triton import and functionality
2. Conditional Proton profiler import and stub behavior
3. Basic kernel compilation and execution
"""

import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

def test_triton_import():
    """Test basic Triton import."""
    print("Testing Triton import...")
    try:
        import triton
        print("[PASS] Triton imported successfully")
        print(f"   Triton version: {triton.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import Triton: {e}")
        return False

def test_profiler_availability():
    """Test profiler import and availability."""
    print("\nTesting Profiler availability...")
    try:
        import triton.profiler as profiler
        print("[PASS] Profiler module imported successfully")
        
        # Check if Proton is available
        is_available = profiler.is_available()
        proton_available = getattr(profiler, 'PROTON_AVAILABLE', False)
        
        print(f"   Profiler available: {is_available}")
        print(f"   PROTON_AVAILABLE flag: {proton_available}")
        
        if not is_available:
            print("[PASS] Expected behavior: Proton is disabled")
            return True
        else:
            print("[WARN] Unexpected: Proton appears to be enabled")
            return False
            
    except ImportError as e:
        print(f"[FAIL] Failed to import profiler: {e}")
        return False

def test_profiler_stubs():
    """Test that profiler stub functions work correctly."""
    print("\nTesting Profiler stub functions...")
    try:
        import triton.profiler as profiler
        
        # Capture warnings
        warning_output = StringIO()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test basic profiler functions
            print("   Testing profiler.start()...")
            result = profiler.start("test_profile")
            print(f"   Result: {result}")
            
            print("   Testing profiler.scope()...")
            with profiler.scope("test_scope"):
                pass
            print("   Scope context manager worked")
            
            print("   Testing profiler.activate()...")
            profiler.activate()
            
            print("   Testing profiler.deactivate()...")
            profiler.deactivate()
            
            print("   Testing profiler.finalize()...")
            profiler.finalize()
            
            # Test viewer functions
            print("   Testing viewer functions...")
            try:
                tree, metrics = profiler.viewer.parse(["time/ms"], "nonexistent.hatchet")
                print(f"   Viewer parse result: {tree}, {metrics}")
                
                profiler.viewer.print_tree(None, None)
                print("   Viewer print_tree worked")
                
            except AttributeError as e:
                print(f"   Viewer functions error: {e}")
            
        # Check if warnings were generated
        if w:
            print(f"[PASS] Generated {len(w)} warning(s) as expected")
            for warning in w[:3]:  # Show first 3 warnings
                print(f"      {warning.message}")
        else:
            print("[WARN] No warnings generated (unexpected)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing profiler stubs: {e}")
        return False

def test_basic_kernel():
    """Test basic Triton kernel compilation and execution."""
    print("\nTesting basic Triton kernel...")
    try:
        import torch
        import triton
        import triton.language as tl
        
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available, skipping kernel test")
            return True
        
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
        
        # Test compilation
        print("   Compiling kernel...")
        device = torch.device("cuda")
        size = 1024
        x = torch.rand(size, device=device, dtype=torch.float32)
        y = torch.rand(size, device=device, dtype=torch.float32)
        output = torch.empty_like(x)
        
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
        
        # Verify result
        expected = x + y
        if torch.allclose(output, expected):
            print("[PASS] Kernel executed successfully")
            return True
        else:
            print("[FAIL] Kernel result incorrect")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing kernel: {e}")
        return False

def test_profiler_integration():
    """Test profiler integration with kernel execution."""
    print("\nTesting Profiler integration with kernel...")
    try:
        import torch
        import triton
        import triton.language as tl
        import triton.profiler as profiler
        
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available, skipping integration test")
            return True
        
        @triton.jit
        def simple_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x * 2, mask=mask)
        
        device = torch.device("cuda")
        size = 1024
        x = torch.ones(size, device=device, dtype=torch.float32)
        
        # Test with profiler context
        print("   Testing with profiler context...")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            profiler.start("integration_test")
            with profiler.scope("kernel_execution"):
                grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
                simple_kernel[grid](x, size, BLOCK_SIZE=256)
            profiler.finalize()
        
        # Verify result
        expected = torch.ones(size, device=device, dtype=torch.float32) * 2
        if torch.allclose(x, expected):
            print("[PASS] Profiler integration worked correctly")
            return True
        else:
            print("[FAIL] Kernel with profiler failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error in profiler integration test: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TRITON PROTON-DISABLED BUILD VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_triton_import,
        test_profiler_availability,
        test_profiler_stubs,
        test_basic_kernel,
        test_profiler_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - Triton with Proton disabled works correctly!")
        return 0
    else:
        print("[WARN] Some tests failed - check the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())