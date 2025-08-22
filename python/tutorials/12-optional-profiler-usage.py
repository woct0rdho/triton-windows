"""
Optional Profiler Usage Example
===============================

This tutorial demonstrates best practices for using the Triton profiler
when it may or may not be available. The code will work whether Proton
is built or not, and gracefully degrades when profiling is unavailable.

.. note::
   When Proton is not available, profiling calls will be ignored with a warning.
   To enable profiling, build Triton with -DTRITON_BUILD_PROTON=ON.
"""

import torch
import triton
import triton.language as tl

# Import profiler with conditional usage
import triton.profiler as profiler

# Alternative explicit check
if profiler.is_available():
    print("✓ Proton profiler is available")
else:
    print("⚠ Proton profiler is not available - profiling calls will be no-ops")

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor):
    """Vector addition with optional profiling."""
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    
    # Use a context manager for scoped profiling (recommended)
    with profiler.scope("vector_add_triton"):
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def benchmark_with_optional_profiling():
    """Demonstrates benchmarking with optional profiling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU example")
        return
    
    size = 1024 * 1024
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)
    
    # Method 1: Start/stop profiling session (best for complex benchmarks)
    profiler.start("vector_add_benchmark", hook="triton")
    
    # Warm up
    for _ in range(5):
        _ = vector_add(x, y)
    
    # Actual benchmark with scoped profiling
    profiler.activate()
    for i in range(10):
        with profiler.scope(f"vector_add_iteration_{i}"):
            result = vector_add(x, y)
    profiler.deactivate()
    
    # Verify correctness
    expected = x + y
    torch.testing.assert_close(result, expected)
    
    profiler.finalize()
    
    # Method 2: Check availability before detailed profiling
    if profiler.is_available():
        print("Profiling completed successfully")
        
        # You can also import the viewer conditionally
        try:
            import triton.profiler.viewer as viewer
            # Use viewer functions here
            print("Viewer is available for result analysis")
        except (ImportError, AttributeError):
            print("Viewer not available")
    else:
        print("Profiling was not available, but benchmark completed successfully")


def demonstrate_conditional_viewer():
    """Show how to use viewer functions conditionally."""
    # This will work whether profiler is available or not
    if profiler.is_available():
        try:
            import triton.profiler.viewer as viewer
            
            # Example usage (would need actual profile file)
            # tree, metrics = viewer.parse(["time/ms"], "profile.hatchet")
            # viewer.print_tree(tree, metrics)
            
            print("Viewer functions are available")
        except (ImportError, AttributeError) as e:
            print(f"Viewer not available: {e}")
    else:
        print("Profiler not available, skipping viewer example")


if __name__ == "__main__":
    print("Running optional profiler usage example...")
    benchmark_with_optional_profiling()
    demonstrate_conditional_viewer()
    print("Example completed!")