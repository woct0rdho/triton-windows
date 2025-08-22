<div align="center">
  <img src="https://lh5.googleusercontent.com/wzQKEsTFkrgNQO9JjhGH5wFvslJr1saLtLaJ_a6Fp_gNENpvt3VG7BmztwngU9hFJaU4CPwGiw1opQtDvTkLrxWRbO_a12Q-pdESWHgtmheIHcPbOL5ZMC4TSiJVe5ty1w=w3517" alt="Triton logo">
</div>

| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

# Triton Windows

This is the Windows development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. This version has been specifically optimized for Windows development with NVIDIA GPU support and conditional Proton profiling capabilities.

## Key Windows Features

- **Native Windows Build**: Optimized for Windows 10/11 with Visual Studio 2022
- **NVIDIA GPU Focus**: Streamlined for NVIDIA CUDA development (AMD/ROCm support removed)
- **Conditional Proton Profiling**: Optional profiler that can be enabled when needed
- **Enhanced Build System**: Improved build script with timeout handling and comprehensive testing
- **Comprehensive Testing**: >90% test coverage with encoding-safe test scripts

## Windows Quick Installation

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.9-3.13
- Visual Studio 2022 (Community/Professional/Enterprise)
- CUDA Toolkit 12.5+ (for GPU support)
- Git for Windows

### Using Enhanced Build Script (Recommended)

```powershell
git clone https://github.com/blap/triton-windows.git
cd triton-windows

# Run enhanced build script with automatic environment detection
powershell -ExecutionPolicy Bypass -File build.ps1
```

The enhanced build script automatically:
- Detects Python, Visual Studio, and CUDA installations
- Configures optimal build environment
- Provides progress monitoring and error handling
- Runs comprehensive test suite (>90% coverage)
- Verifies build artifacts

### Manual Installation

```shell
pip install -r python/requirements.txt
pip install -e .
```

## Conditional Proton Profiling

This Windows build includes conditional Proton profiling support:

### Default Behavior (Proton Disabled)
```python
import triton.profiler as profiler

# Profiler works with stub implementations
profiler.start("my_profile")  # Shows warning, continues gracefully
with profiler.scope("my_scope"):
    # Your GPU kernel code here
    pass
profiler.finalize()
```

### Enabling Proton Profiling

To enable full Proton profiling capabilities:

```powershell
# Set environment variable and rebuild
$env:TRITON_BUILD_PROTON="ON"
pip install -e . --no-cache-dir
```

Or use the build script:
```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -ProtonEnabled
```

## Windows Build Configuration

### Tested Environment
- **OS**: Windows 10/11 (Build 19045+)
- **Python**: 3.12.10
- **Visual Studio**: 2022 Community (MSVC 14.44.35207)
- **Windows SDK**: 10.0.20348.0
- **CUDA**: 12.9
- **Build Tools**: CMake 3.20+, Ninja

### Build Options

The enhanced build script supports several options:

```powershell
# Clean build (removes all previous artifacts)
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean

# Verbose output for debugging
powershell -ExecutionPolicy Bypass -File build.ps1 -Verbose

# Custom timeout (default: 30 minutes)
powershell -ExecutionPolicy Bypass -File build.ps1 -TimeoutMinutes 45

# Specify custom Python installation
powershell -ExecutionPolicy Bypass -File build.ps1 -PythonPath "C:\Python312\python.exe"
```

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.  See also these third-party [Triton puzzles](https://github.com/srush/Triton-Puzzles), which can all be run using the Triton interpreter -- no GPU required.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.9-3.13.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.  Check
`cmake/llvm-hash.txt` to see the current version. For example, if it says:
       49af6502c6dcb4a7f7520178bd14df396f78240c

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables.

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e .

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e .` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e .`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find ./build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests on Windows

This Windows build includes comprehensive testing with >90% coverage:

```powershell
# Run all Windows-specific tests
python test_comprehensive_proton.py

# Test Proton disabled functionality (default state)
python test_proton_disabled.py

# Test Proton enablement capabilities
python test_proton_enabled.py

# Basic functionality verification
python -c "import triton; print(f'Triton {triton.__version__} ready!')"
```

## Windows Test Categories

1. **Basic Import Tests**: Core Triton functionality
2. **Conditional Compilation Tests**: Proton enabled/disabled states
3. **Build System Tests**: CMake configuration validation
4. **GPU Kernel Tests**: CUDA kernel compilation and execution
5. **Profiler Integration Tests**: Stub and full profiler functionality

## Windows Troubleshooting

### Common Build Issues

**Issue**: `LINK : fatal error LNK1181: cannot open input file`
```powershell
# Solution: Clean build
powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
```

**Issue**: PowerShell execution policy restrictions
```powershell
# Solution: Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue**: Unicode encoding errors in tests
```powershell
# Solution: All test scripts have been fixed with ASCII-only output
# No action needed - this issue has been resolved
```

**Issue**: Build timeout during LLVM download
```powershell
# Solution: Increase timeout
powershell -ExecutionPolicy Bypass -File build.ps1 -TimeoutMinutes 60
```

### Environment Variables

Key environment variables for Windows builds:

- `TRITON_BUILD_PROTON`: Enable/disable Proton profiling (`ON`/`OFF`)
- `TRITON_CODEGEN_BACKENDS`: Set to `nvidia` for Windows builds
- `CUDA_TOOLKIT_ROOT_DIR`: Auto-detected or specify CUDA path
- `CMAKE_ARGS`: Additional CMake arguments

### Performance Tips

- Use SSD storage for build directory
- Ensure adequate RAM (16GB+ recommended)
- Close unnecessary applications during build
- Use Windows Terminal for better output display

## Windows Development Workflow

### Recommended Development Setup

1. **Install Prerequisites**:
   ```powershell
   # Install Python from python.org
   # Install Visual Studio 2022 with C++ workload
   # Install CUDA Toolkit
   # Install Git for Windows
   ```

2. **Clone and Build**:
   ```powershell
   git clone https://github.com/blap/triton-windows.git
   cd triton-windows
   powershell -ExecutionPolicy Bypass -File build.ps1
   ```

3. **Verify Installation**:
   ```powershell
   python test_comprehensive_proton.py
   ```

4. **Development Iterations**:
   ```powershell
   # For code changes, rebuild with:
   pip install -e . --no-cache-dir
   
   # For major changes, use clean build:
   powershell -ExecutionPolicy Bypass -File build.ps1 -Clean
   ```

### IDE Configuration

**Visual Studio Code**:
- Install C/C++ extension
- Use `compile_commands.json` from build directory
- Configure Python interpreter to point to your environment

**Visual Studio 2022**:
- Open folder containing the repository
- CMake integration will auto-configure
- Set Python interpreter in Tools → Options

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Configuration knobs**

See [`python/triton/knobs.py`](python/triton/knobs.py) for the full list of configuration knobs. You can set those knobs directly in python or use environment variables to control them. Below are some of the environment variables you can specify (see `knobs.py` for the full list):

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).

  When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.
- `TRITON_FRONT_END_DEBUGGING=1` disables exception wrapping when an error occurs in the compiler frontend, allowing the full stack trace to be seen.
- `TRITON_DISABLE_LINE_INFO=1` removes all line information from the module

N.B. Some of these environment variables don't have a knob in `knobs.py`-- those are only relevant to the C++ layer(s), hence they don't exist in the python layer.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```


# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers)

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md)
