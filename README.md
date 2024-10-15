# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `v3.0.x-windows` branch for the code.

Based on [andreigh](https://github.com/andreigh/triton/tree/windows), [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), and more development in the community. Thank you all!

## Why?

* Free software should run on non-free platforms, as per Richard Stallman
* This is the basis for torchao, which crucially changes some large models from "can't run" to "can run" on consumer GPUs. That's easier than supporting them in other quantization frameworks, or letting the consumers use Linux or WSL
* Catgirl matters

## Progress

* Forked from the `release/3.0.x` branch of the official repo
* `triton.jit` and `torch.compile` just work
* When I run Flux or CogVideoX in ComfyUI on Windows, it's almost as fast as on WSL on the same machine
* Most tests passed, except some overflows because on Windows the C long has only 4 bytes
* Only MSVC is supported, from my experience it's much more stable than GCC and Clang when working with CUDA on Windows
* Only CUDA is supported, help wanted to support AMD
* TODO: Set up CI

## Install from wheel

The wheels are built against CUDA 12.5, and they should work with other CUDA 12.x.

MSVC and Windows SDK are required, because Triton compiles Python functions on your machine. You can install them in Visual Studio, or just Visual Studio Build Tools.

Then you need to add the path containing `cl.exe`, such as `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64`, to your `PATH`. If you type `cl` in PowerShell and it shows `Microsoft (R) C/C++ Optimizing Compiler ...`, then you're doing right.

Now you can download the wheel from [releases](https://github.com/woct0rdho/triton/releases).

## Test if it works

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
```

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
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
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

If you want to build the C++ unit tests and don't set `TRITON_BUILD_UT=0`, then you also need to add git to the paths.

Build LLVM using MSVC according to the instructions of the official Triton:
* https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm
* You may need to remove the non-ASCII characters in the comments of `mlir/lib/Dialect/ArmSME/Transforms/VectorLegalization.cpp` to make MSVC happy

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

Clone this repo, checkout `v3.0.x-windows` branch, make an editable build using pip:
```pwsh
pip install --no-build-isolation --verbose -e python
```

Build the wheels:
```pwsh
$Env:CIBW_BUILD = "{cp310-win_amd64,cp311-win_amd64,cp312-win_amd64}"
cibuildwheel python
```

## Dev notes

* To implement `dlopen`, [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32) is added to `thirdparty/` and linked in CMake for building the package, and in `third_party/nvidia/backend/driver.c` and `driver.py` it's rewritten with `LoadLibrary` for jitting
* In `lib/Analysis/Utility.cpp` and `lib/Dialect/TritonGPU/Transforms/Utility.cpp`, explicit namespaces are added to support the resolution behaviors of MSVC
* In `python/src/interpreter.cc` the GCC built-in `__ATOMIC` memory orders are replaced with `std::memory_order`
* In `python/triton/runtime/build.py`, the paths of MSVC and Windows SDK are automatically found
* On Windows the C long has only 4 bytes, so some tests failed because of overflow, and I marked them xfail
