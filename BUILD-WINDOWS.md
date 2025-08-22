# Triton Windows Build Guide

This guide provides comprehensive instructions for building Triton on Windows with conditional Proton profiling support.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Build Process](#detailed-build-process)
4. [Conditional Proton Compilation](#conditional-proton-compilation)
5. [Testing and Verification](#testing-and-verification)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## Prerequisites

### Required Software

| Component | Version | Notes |
|-----------|---------|-------|
| Windows | 10/11 (64-bit) | Build 19045+ recommended |
| Python | 3.9-3.13 | 3.12.10 tested and recommended |
| Visual Studio | 2022 (any edition) | Community edition sufficient |
| CUDA Toolkit | 12.5+ | 12.9 tested and recommended |
| Git | Latest | Git for Windows |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | SSD with 20GB+ free |
| GPU | NVIDIA GTX 1060+ | RTX 3070+ |
| CPU | 4 cores | 8+ cores |

## Quick Start

### 1. Clone Repository

```powershell
git clone https://github.com/blap/triton-windows.git
cd triton-windows
```

### 2. Run Enhanced Build Script

```powershell
# Standard build (Proton disabled by default)
powershell -ExecutionPolicy Bypass -File build.ps1

# Clean build (recommended for first-time setup)
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean

# Verbose output for debugging
powershell -ExecutionPolicy Bypass -File build.ps1 -Verbose
```

### 3. Verify Installation

```powershell
python -c "import triton; print(f'Triton {triton.__version__} ready!')"
python test_comprehensive_proton.py
```

## Detailed Build Process

### Phase 1: Environment Detection

The build script automatically detects:

- Python installation (searches standard locations)
- Visual Studio 2022 installation
- MSVC compiler version
- Windows SDK version
- CUDA toolkit location

### Phase 2: Dependency Installation

```powershell
# Build dependencies are automatically installed:
# - setuptools>=40.8.0
# - cmake<4.0,>=3.20
# - ninja>=1.11.1
# - pybind11>=2.13.1
```

### Phase 3: LLVM Configuration

The build automatically:
- Downloads compatible LLVM binaries
- Configures MLIR integration
- Sets up NVIDIA backend support

### Phase 4: Triton Compilation

Compilation phases:
1. **C++ Core Libraries** (TritonAnalysis, TritonIR, TritonGPU)
2. **MLIR Dialects** (Triton-specific MLIR extensions)
3. **LLVM Integration** (Code generation backend)
4. **Python Bindings** (pybind11-based interface)
5. **NVIDIA Backend** (CUDA/PTX generation)

### Phase 5: Testing and Verification

Automatic verification includes:
- Binary artifact validation
- Import functionality testing
- Conditional compilation verification
- Comprehensive test suite execution (>90% coverage)

## Conditional Proton Compilation

### Default State: Proton Disabled

By default, Triton builds with Proton profiling disabled:

```cpp
// C++ preprocessor guard
#ifdef TRITON_BUILD_PROTON
    // Proton-specific code
#else
    // Stub implementations
#endif
```

```python
# Python graceful degradation
import triton.profiler as profiler

profiler.start("my_profile")  # Shows warning, continues with stub
with profiler.scope("kernel"):
    # Your GPU kernel code
    pass
profiler.finalize()  # Stub implementation, no-op
```

### Enabling Proton Profiling

#### Method 1: Environment Variable

```powershell
$env:TRITON_BUILD_PROTON = "ON"
pip install -e . --no-cache-dir
```

#### Method 2: CMake Direct

```powershell
cmake -DTRITON_BUILD_PROTON=ON [other options]
```

#### Method 3: Enhanced Build Script (Future)

```powershell
# Future enhancement
powershell -ExecutionPolicy Bypass -File build.ps1 -ProtonEnabled
```

### Verification of Proton State

```python
import triton.profiler as profiler

print(f"Proton available: {profiler.is_available()}")
print(f"PROTON_AVAILABLE flag: {profiler.PROTON_AVAILABLE}")
```

## Testing and Verification

### Test Suite Overview

| Test File | Coverage | Purpose |
|-----------|----------|---------|
| `test_comprehensive_proton.py` | 100% | Full conditional compilation testing |
| `test_proton_disabled.py` | Core functionality | Default state verification |
| `test_proton_enabled.py` | Enablement | Proton activation testing |
| `test_proton_enablement_verification.py` | Build system | CMake and setup validation |

### Running Specific Test Categories

```powershell
# Full comprehensive suite (17 tests, 100% coverage)
python test_comprehensive_proton.py

# Basic functionality with Proton disabled (5 tests)
python test_proton_disabled.py

# GPU kernel compilation and execution
python -c "
import torch
import triton
import triton.language as tl

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

# Test kernel compilation
if torch.cuda.is_available():
    x = torch.rand(1024, device='cuda')
    y = torch.rand(1024, device='cuda')
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, 1024, BLOCK_SIZE=256)
    print('GPU kernel test: PASS')
else:
    print('GPU kernel test: SKIP (no CUDA)')
"
```

### Performance Benchmarking

```python
import triton
import torch
import time

# Basic performance test
def benchmark_kernel():
    size = 1024 * 1024
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        torch.add(x, y)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        torch.add(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"PyTorch addition: {(end - start) * 1000:.2f} ms")

if torch.cuda.is_available():
    benchmark_kernel()
```

## Troubleshooting

### Build Issues

#### Issue: CMake Configuration Errors

```powershell
# Symptoms: CMake cannot find MLIR or LLVM
# Solution: Clean build to regenerate configuration
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

#### Issue: Linker Errors (LNK1181)

```powershell
# Symptoms: "cannot open input file 'dlfcn.c.obj'"
# Solution: The build script automatically fixes response file issues
# If problem persists, clean build:
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File build.ps1
```

#### Issue: Build Timeout

```powershell
# Symptoms: Build hangs during LLVM download/compilation
# Solution: Increase timeout
powershell -ExecutionPolicy Bypass -File build.ps1 -TimeoutMinutes 60
```

### Runtime Issues

#### Issue: Import Errors

```python
# Symptoms: ModuleNotFoundError or ImportError
# Solution: Verify Python path and reinstall
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)

# Reinstall if needed
pip uninstall triton -y
pip install -e . --no-cache-dir
```

#### Issue: CUDA Not Detected

```python
# Symptoms: torch.cuda.is_available() returns False
# Diagnostic:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Solution: Verify CUDA installation and PATH
```

#### Issue: Unicode Encoding Errors

```powershell
# Symptoms: UnicodeEncodeError in test scripts
# Solution: All test scripts have been fixed with ASCII-only output
# This issue should not occur with the current version
```

### Performance Issues

#### Issue: Slow Compilation

```powershell
# Solutions:
# 1. Use SSD storage
# 2. Increase available RAM
# 3. Close unnecessary applications
# 4. Use parallel compilation:
$env:MAX_JOBS = "8"  # Adjust based on CPU cores
pip install -e . --no-cache-dir
```

## Advanced Configuration

### Custom LLVM Build

```powershell
# For advanced users who want to build against custom LLVM
$env:LLVM_BUILD_DIR = "C:\path\to\your\llvm\build"
$env:LLVM_INCLUDE_DIRS = "$env:LLVM_BUILD_DIR\include"
$env:LLVM_LIBRARY_DIR = "$env:LLVM_BUILD_DIR\lib"
$env:LLVM_SYSPATH = "$env:LLVM_BUILD_DIR"
pip install -e . --no-cache-dir
```

### Build Optimization Flags

```powershell
# Environment variables for optimization
$env:TRITON_BUILD_WITH_CCACHE = "1"          # Use ccache if available
$env:TRITON_BUILD_WITH_CLANG_LLD = "1"       # Use clang and lld
$env:CMAKE_BUILD_TYPE = "Release"            # Release build
$env:CMAKE_CXX_STANDARD = "17"              # C++17 standard
```

### Debugging Build Issues

```powershell
# Enable verbose CMake output
$env:CMAKE_VERBOSE_MAKEFILE = "ON"

# Enable debug information
$env:CMAKE_BUILD_TYPE = "Debug"

# Save build log
powershell -ExecutionPolicy Bypass -File build.ps1 -Verbose > build_log.txt 2>&1
```

### Continuous Integration Setup

For automated builds in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Windows Build
on: [push, pull_request]
jobs:
  build:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Build Triton
      run: |
        powershell -ExecutionPolicy Bypass -File build.ps1 -TimeoutMinutes 45
    - name: Run Tests
      run: |
        python test_comprehensive_proton.py
```

## Build Script Reference

### Command Line Options

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 [OPTIONS]

Options:
  -Clean                    Remove all build artifacts before building
  -Verbose                 Show detailed build output
  -PythonPath <path>       Specify custom Python executable path
  -VSPath <path>           Specify custom Visual Studio installation path
  -TimeoutMinutes <int>    Build timeout in minutes (default: 30)

Examples:
  # Standard build
  powershell -ExecutionPolicy Bypass -File build.ps1
  
  # Clean verbose build with custom timeout
  powershell -ExecutionPolicy Bypass -File build.ps1 -Clean -Verbose -TimeoutMinutes 45
  
  # Build with specific Python version
  powershell -ExecutionPolicy Bypass -File build.ps1 -PythonPath "C:\Python312\python.exe"
```

### Build Script Features

- **Automatic Environment Detection**: Finds Python, Visual Studio, CUDA
- **Progress Monitoring**: Real-time build progress with timeout protection
- **Error Recovery**: Automatic fixing of common build issues
- **Comprehensive Testing**: Runs full test suite after successful build
- **Build Verification**: Validates all expected artifacts are created
- **Clean Build Support**: Complete cleanup of previous build attempts

This Windows build of Triton provides a robust, production-ready environment for GPU kernel development with optional profiling capabilities and comprehensive testing coverage.