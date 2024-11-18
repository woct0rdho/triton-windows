# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `v3.1.x-windows` branch for the code.

Based on [andreigh](https://github.com/andreigh/triton/tree/windows), [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), and more development in the community. Thank you all!

## Why?

* Free software should run on non-free platforms, as per Richard Stallman
* This is the basis for torchao, which crucially changes some large models from "can't run" to "can run" on consumer GPUs. That's easier than supporting them in other quantization frameworks, or letting the consumers use Linux or WSL
* Catgirl matters

## Progress

* Forked from the `release/3.1.x` branch of the official repo
* `triton.jit` and `torch.compile` just work
* `torchao.autoquant` just works
    * You can install the prereleased wheel of torchao by `pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124`, and you can choose cu121/cu124 according to your CUDA version
* When I run Flux or CogVideoX in ComfyUI on Windows, it's almost as fast as on WSL on the same machine
* Most tests passed, except some overflows because on Windows the C long has only 4 bytes
* Only MSVC is supported, from my experience it's much more stable than GCC and Clang when working with CUDA on Windows
* Only Nvidia GPU is supported, help wanted to support other backends
    * For AMD GPU, you may try https://github.com/Repeerc/triton-amdgpu-windows
    * For Intel XPU, you may try https://github.com/intel/intel-xpu-backend-for-triton/tree/gregory/windows-support
* TODO: Set up CI (help wanted)
* TODO: Make a minimal bundle of MSVC and Windows SDK in the wheels (help wanted)

## Install from wheel

Triton 3.1.0 works with torch >= 2.4.0, not 2.3.x.

1. CUDA 12 is required. The wheels are built against CUDA 12.5, and they should work with other CUDA 12.x. You can either:
    * If you're using conda, then install PyTorch with CUDA in conda according to [PyTorch's guide](https://pytorch.org/get-started/locally/#windows-anaconda)
        * You can verify the existance of CUDA in the conda env by running `conda list cuda`
    > OR
    * If you're not using conda, then install CUDA in your system using the installer from [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
        1. When installing, you need to choose both 'CUDA Development' and 'CUDA Runtime'
            * Make sure these folders exist on your computer: (Change the version number according to your installation)
                ```
                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include
                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64
                ```
            * Make sure this file exists: `C:\Windows\System32\nvcuda.dll`
        2. Then you need to add the path of CUDA to the Windows `PATH`:
            * The path is like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin`
            * Make sure this folder exists
        3. If you open a new PowerShell, type `ptxas --version`, and it shows your CUDA version like `Cuda compilation tools, release 12.5, V12.5.82`, then you're doing right

2. MSVC and Windows SDK are required, because Triton compiles Python functions on your computer. You can install them in Visual Studio, or just Visual Studio Build Tools. Then you need to add the path containing `cl.exe` to the Windows `PATH`:
    * The path is like `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64`
    * Change the version numbers according to your installation, and make sure this folder accually exists on your computer
    * If you open a new PowerShell, type `cl`, and it shows `Microsoft (R) C/C++ Optimizing Compiler ...`, then you're doing right

3. vcredist is required (also known as 'Visual C++ Redistributable for Visual Studio 2015-2022', `msvcp140.dll`, `vcruntime140.dll`). Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe

4. Now you can download the wheel from [releases](https://github.com/woct0rdho/triton-windows/releases), e.g.,
```sh
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp310-cp310-win_amd64.whl
```

Special notes if you're using ComfyUI with the embeded Python:
* There should be a folder `python_embeded` in your ComfyUI installation path
* You need to put two folders `include` and `libs` in `python_embeded` to make Triton work
    * Be careful: It is 'libs', not 'lib'. The folder `Lib` should already exist in `python_embeded`
* If you're using ComfyUI_windows_portable <= 0.2.3, you can download the two folders for Python 3.11.9 here: https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.11.9_include_libs.zip
* If you're using ComfyUI_windows_portable >= 0.2.4, you can download the two folders for Python 3.12.7 here: https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip
* If you're using another version, you can copy-paste them from a usual installation of Python, with the same version as ComfyUI uses
* If you're not sure, run `path\to\python_embeded\python.exe --version` to see the Python version

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
    assert x.is_cuda and y.is_cuda and output.is_cuda
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
* Copy-paste `msvcp140.dll`, `vcruntime140.dll`, and `vcruntime140_1.dll` from `C:\Windows\System32\` to the folder containing `python.exe`
* Delete the cache folder `C:\Users\<your username>\.triton\cache\`

You may also need to delete the cache folder when you change the Python version, install another version of Triton, or change the version of MSVC, Windows SDK, or CUDA.

If it still doesn't work, you may try:
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

Set the binary, include, and library paths of Python, MSVC, Windows SDK, and CUDA in PowerShell (help wanted to automatically find these in CMake):
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
* https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm
* You may need to add the compiler options `/utf-8 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING` to make MSVC happy, see https://reviews.llvm.org/D90116 and https://github.com/llvm/llvm-project/issues/65255

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
* In `python/src/interpreter.cc` the GCC built-in `__ATOMIC` memory orders are replaced with `std::memory_order`
* `windows_utils.py` contains many ways to find the paths of Python, MSVC, Windows SDK, and CUDA
* On Windows the C long has only 4 bytes, so some tests failed because of overflow, and I marked them xfail
* How TorchInductor is designed to support Windows: https://github.com/pytorch/pytorch/issues/124245

## Known Issues

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
