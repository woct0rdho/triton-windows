# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `v3.1.x-windows` branch for the code.

Based on [andreigh](https://github.com/andreigh/triton/tree/windows), [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), [anmyachev](https://github.com/triton-lang/triton/issues?q=author%3Aanmyachev), and more development in the community. Thank you all!

## Why?

* Free software should run on non-free platforms, as per Richard Stallman
* This is required by `torch.compile`, and used by torchao, SageAttention, ParaAttention, and more packages
* Catgirl matters

## Progress

* Forked from the `release/3.1.x` branch of the official repo
* `triton.jit` and `torch.compile` just work
* When I run Flux or CogVideoX in ComfyUI on Windows, it's almost as fast as on WSL on the same machine
* All unit tests passed
* Only MSVC is supported, because it's much more stable than GCC and Clang when working with CUDA on Windows
* Only Nvidia GPU is supported, help wanted to support other backends
    * For AMD GPU, you may try https://github.com/Repeerc/triton-amdgpu-windows
    * For Intel XPU, you may try https://github.com/intel/intel-xpu-backend-for-triton
* TODO: Set up CI (help wanted)
* TODO: Make a minimal bundle of MSVC, Windows SDK, and CUDA SDK in the wheels (help wanted)

## Install from wheel

~~(From the instructions below you can see what a hell programming on Windows is)~~

### 1. GPU

Check your GPU model. Triton officially supports Nvidia GPUs with sm >= 80 (also known as 'CUDA arch' or 'compute capability'), such as RTX 30xx and newer. RTX 20xx and older may work when running some simple AI models, but not always.

Many AI models use bf16 (also known as bfloat16). In Triton, it only works with sm >= 80. If you really want to run it on older GPUs, you may try changing bf16 to fp16.

Some AI models use fp8 (also known as float8). In Triton, it only works with sm >= 89, such as RTX 40xx and newer. See the [known issue](https://github.com/woct0rdho/triton-windows#fp8-is-not-supported-on-rtx-30xx-and-older-gpus).

### 2. Python environment

Check how your Python is installed. We mainly support either of the following environments:
* **System-wide**: You install Python at a location like `C:\Python310\` and directly use it
* **Embeded**: You use an all-in-one package of ComfyUI (or some other AI software), and there is a folder `python_embeded` in it
    * In this case, don't directly use `pip`, but use `python -m pip` instead. It's because `pip.exe` is not in the folder `python_embeded`, and you may accidentally call a `pip.exe` installed elsewhere
* **conda**: You create a virtual environment using `conda`
* **Python venv**: You create a virtual environment using `venv` or `virtualenv`
    * This still has some issues when importing DLL

For other environment managers like poetry or uv, if you find problems, please open an issue.

Make sure what environment you're using. You can use `Get-Command python` in PowerShell (or `where python` in cmd) to see the installation path of Python, and `python --version` to see its version.

Don't mix two environments, unless you know them very well.
* If you're using ComfyUI with embeded Python, then don't use conda or venv
* If you're already using conda, then always create a new env using conda, and don't use Python venv

### 3. PyTorch

Check your PyTorch version: Triton 3.1.0 works with torch >= 2.4.0 . torch 2.3.x and older versions are not supported.

Triton 3.2.0 works with torch >= 2.6.0 . For now don't install it unless you're using the pre-release version of PyTorch.

### 4. CUDA

CUDA 12 is required. CUDA 11.x and older versions are not supported. The wheels here are built against CUDA 12.5, and they should work with other CUDA 12.x.

Choose either of the following ways to install CUDA:

**a) conda**: Do this only if you're already using conda
<details>
<summary>Expand</summary>

* Install PyTorch with CUDA using conda, according to [PyTorch's guide](https://pytorch.org/get-started/locally/#windows-anaconda)
* You can verify the existance of CUDA in the conda env by running `conda list cuda`
</details>

**b) System-wide**: Recommended for most people
<details>
<summary>Expand</summary>

1. Install PyTorch with CUDA using pip
2. Install CUDA toolkit from [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. When installing, you need to choose both 'CUDA Development' and 'CUDA Runtime'. Make sure these folders exist on your computer: (Change the version number according to your installation)
    ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64
    ```
4. Then you need to add the path of CUDA to the Windows `PATH`:
    * The path is like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin`
    * Make sure this folder exists
5. If you open a new PowerShell, type `ptxas --version`, and it shows your CUDA version like `Cuda compilation tools, release 12.5, V12.5.82`, then you're doing right
</details>

**c) pip**: Recommended if you don't want to install too much boilerplate, and you want to contain everything in a venv, with minimal impact to the system
<details>
<summary>Expand</summary>

1. Install PyTorch with CUDA using pip
2. install the following packages:
    ```sh
    pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
    ```
3. There should be a folder `Lib\site-packages\nvidia\cuda_runtime\` in your Python installation path (or venv), and you need to add a library in it
    * Download it from https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post8/cuda_12.4_lib.zip
    * Put the folder `lib` into `cuda_runtime`

For details about compatibility of various pip packages and CUDA versions, see https://github.com/woct0rdho/triton-windows/issues/43
</details>

### 5. MSVC and Windows SDK

MSVC and Windows SDK are required, because Triton compiles Python functions on your computer.
* You can install them in Visual Studio
    * If you don't want to install the whole Visual Studio, you can just use Visual Studio Build Tools
    * If you don't want any MS boilerplate, you may try [PortableBuildTools](https://github.com/Data-Oriented-House/PortableBuildTools)
* Visual Studio >= 2017 is supported
* Choose the latest version of MSVC and Windows SDK from the list

Then you need to add the path containing `cl.exe` to the Windows `PATH`:
* The path is like `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64`
* Change the version numbers according to your installation, and make sure this folder accually exists on your computer
* If you open a new PowerShell, type `cl`, and it shows `Microsoft (R) C/C++ Optimizing Compiler ...`, then you're doing right

### 6. vcredist

vcredist is required (also known as 'Visual C++ Redistributable for Visual Studio 2015-2022', `msvcp140.dll`, `vcruntime140.dll`). Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe

### 7. Triton

Now you can download the wheel from [releases](https://github.com/woct0rdho/triton-windows/releases), e.g.,
```sh
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.1.0-cp310-cp310-win_amd64.whl
```
* Choose the wheel according to your Python version. If you're using Python 3.12, then you need to change `cp310` to `cp312`

### 8. Special notes for ComfyUI with embeded Python

* There should be a folder `python_embeded` in your ComfyUI installation path
* You need to put two folders `include` and `libs` into `python_embeded` to make Triton work
    * Be careful: It is 'libs', not 'lib'. The folder `Lib` should already exist in `python_embeded`
    * If you're using ComfyUI_windows_portable >= 0.2.4 with Python 3.12.7, you can download the two folders here: https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip
    * If you're using another Python version, you can copy-paste them from a usual installation of Python with the same version

## Test if it works

Run the following script:
```python
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

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

a = torch.rand(3, device="cuda")
b = a + a
b_compiled = add(a, a)
print(b_compiled - b)
print("If you see tensor([0., 0., 0.], device='cuda:0'), then it works")
```

If you see `ImportError: DLL load failed`, and there are `vcruntime140.dll` and `vcruntime140_1.dll` in the folder containing `python.exe`, then you may try:
* Install the latest version of vcredist from https://aka.ms/vs/17/release/vc_redist.x64.exe
* Copy-paste `msvcp140.dll`, `vcruntime140.dll`, and `vcruntime140_1.dll` from `C:\Windows\System32\` to the folder containing `python.exe`, and replace the existing DLLs
* Delete the cache folders:
    ```
    C:\Users\<your username>\.triton\cache\
    C:\Users\<your username>\AppData\Local\Temp\torchinductor_<your username>\
    ```

You may also need to delete the cache folders when you change the Python version, install another version of Triton, or change the version of MSVC, Windows SDK, or CUDA.

### dlltracer

If the above still doesn't work, you may try:
* Install [dlltracer](https://github.com/microsoft/dlltracer-python) in the same Python environment
* In an administrator PowerShell, run the following script:
```python
import sys
import dlltracer
with dlltracer.Trace(out=sys.stdout):
    import triton
```
* Open an issue and paste the results

## Build from source

**(This is for developers)**

Set the binary, include, and library paths of Python, MSVC, Windows SDK, and CUDA in PowerShell (help wanted to automatically find these in CMake, or using something equivalent to `vcvarsall.bat` in PowerShell):
```pwsh
$Env:Path =
"C:\Windows\System32;" +
"C:\Python310;" +
"C:\Python310\Scripts;" +
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" +
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64;" +
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;" +
"C:\Program Files\Git\cmd"
$Env:INCLUDE =
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\include;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;" +
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include;" +
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\extras\CUPTI\include"
$Env:LIB =
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\lib\x64;" +
"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;" +
"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
```
* cibuildwheel needs the binaries in `C:\Windows\System32\`
* If you want to build the C++ unit tests and don't set `TRITON_BUILD_UT=0`, then you need git

Build LLVM using MSVC according to the instructions of the official Triton:
```pwsh
# Check out the commit according to cmake/llvm-hash.txt
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_BUILD_TOOLS=OFF -DLLVM_CCACHE_BUILD=ON llvm
cmake --build build -j 8 --config Release
```
* See https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm
* You may need to add the following compiler options to make MSVC happy, see https://reviews.llvm.org/D90116 and https://github.com/llvm/llvm-project/issues/65255:
```diff
diff --git a/llvm/CMakeLists.txt b/llvm/CMakeLists.txt
index c06e661573ed..80b31843f45d 100644
--- a/llvm/CMakeLists.txt
+++ b/llvm/CMakeLists.txt
@@ -821,6 +821,8 @@ if(MSVC)
   if (BUILD_SHARED_LIBS)
     message(FATAL_ERROR "BUILD_SHARED_LIBS options is not supported on Windows.")
   endif()
+  add_compile_options("/utf-8")
+  add_compile_options("/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
 else()
   option(LLVM_LINK_LLVM_DYLIB "Link tools against the libllvm dynamic library" OFF)
   option(LLVM_BUILD_LLVM_C_DYLIB "Build libllvm-c re-export library (Darwin only)" OFF)
```

Download JSON and pybind11 according to `setup.py`:
* https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
* https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip

Set their paths:
```pwsh
$Env:LLVM_SYSPATH = "C:/llvm-project/build"
$Env:JSON_SYSPATH = "C:/json"
$Env:PYBIND11_SYSPATH = "C:/pybind11"
```

Only offline build is supported:
```pwsh
$Env:TRITON_OFFLINE_BUILD = "1"
```

I recommend to use ccache:
```pwsh
$Env:TRITON_BUILD_WITH_CCACHE = "1"
```

Clone this repo, checkout `v3.1.x-windows` branch, make an editable build using pip:
```pwsh
pip install --no-build-isolation --verbose -e python
```

Build the wheels:
```pwsh
$Env:CIBW_BUILD = "{cp38-win_amd64,cp39-win_amd64,cp310-win_amd64,cp311-win_amd64,cp312-win_amd64}"
cibuildwheel python
```

## Dev notes

* To implement `dlopen`:
    * For building the package, [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32) is added to `thirdparty/` and linked in CMake, so I don't need to rewrite it every time
    * For jitting, in `third_party/nvidia/backend/driver.c` and `driver.py` it's rewritten with `LoadLibrary`
* In `lib/Analysis/Utility.cpp` and `lib/Dialect/TritonGPU/Transforms/Utility.cpp`, explicit namespaces are added to support the resolution behaviors of MSVC
* In `python/src/interpreter.cc` the GCC built-in `__ATOMIC` memory orders are replaced with `std::memory_order`, see https://github.com/triton-lang/triton/pull/4976
* `python/triton/windows_utils.py` contains many ways to find the paths of Python, MSVC, Windows SDK, and CUDA
* In `third_party/nvidia/backend/driver.py`, function `make_launcher`, `int64_t` should map to `L` in `PyArg_ParseTuple`. This fixes the error `Python int too large to convert to C long`. See https://github.com/triton-lang/triton/pull/5351
* How TorchInductor is designed to support Windows: https://github.com/pytorch/pytorch/issues/124245

## Known issues

### Windows file path length limit (260) causes compilation failure

Triton would create file cache for complied modules. With module name in the filename, the cache filename is quite long. In some deep module, the path length would exceed Windows' 260 chars length limit, causing error like:
```
... site-packages\torch\_inductor\runtime\triton_heuristics.py:479] [0/0]
File "C:\Anaconda3\Lib\site-packages\triton\compiler\compiler.py", line 288, in compile
 metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
File "...\triton\runtime\cache.py", line 122, in put
 with open(temp_path, mode) as f:
      ^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\[USERNAME]\\AppData\\Local\\Temp\\...LONG..FILE..NAME..'
```
The solution is to shorten your module name or [enable Windows' long path support](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation). A reboot is required after the modification.

### fp8 is not supported on RTX 30xx and older GPUs

If you see error messages like
```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
CompilationError: at 8:11:
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
           ^
```
and in the full error log you find
```
AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
```
then it's because in Triton, fp8 only works on Nvidia GPUs with sm >= 89, such as RTX 40xx and newer. You may disable fp8 in the node or the code.

This is not Windows-specific. It should be possible to emulate fp8 on older hardware like XLA does, even if without time or memory improvement compared to fp16. Help wanted if anyone has time for this.

### Error with `os.rename`

If you see error messages like
```
FileExistsError: [WinError 183] Cannot create a file when that file already exists: ...
```
then you need: https://github.com/pytorch/pytorch/issues/138211

This will be solved when PyTorch 2.6 is out.
