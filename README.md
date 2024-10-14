# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `v3.0.x-windows` branch for the code.

Based on [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), and more development in the community. Thank you all!

## Why?

* Free software should run on non-free platforms, as per Richard Stallman
* Catgirl matters

## Progress

* Forked from the `release/3.0.x` branch of the official repo
* Built the package locally when the paths of Python, MSVC, Windows SDK, and CUDA are manually set
* `triton.jit` works when the paths are manually set in `python/triton/runtime/build.py`
* `torch.compile` works
* When I run Flux or CogVideoX in ComfyUI on Windows, it's almost as fast as on WSL on the same machine (although the memory usage is hard to profile in WSL)
* Only MSVC is supported, from my experience it's much more stable than GCC and Clang when working with CUDA on Windows
* Only CUDA is supported, help wanted to support AMD
* TODO: Fix all tests
* TODO: Auto find the paths in `python/triton/runtime/build.py`
* TODO: Build wheels using cibuildwheel

## Build locally

First, build LLVM using MSVC according to the instructions of the official triton:
https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm
You may need to remove the non-ASCII characters in the comments of `mlir/lib/Dialect/ArmSME/Transforms/VectorLegalization.cpp` to make MSVC happy

Download JSON and pybind11 according to `setup.py`:
https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip

Set their paths (in PowerShell):
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

Set the binary, include, and library paths of Python, MSVC, Windows SDK, and CUDA (help wanted to auto find these in CMake):
```pwsh
$Env:Path =
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

Then, make an editable build using pip:
```pwsh
pip install --no-build-isolation --verbose -e python
```

## Dev notes

* To implement `dlopen`, [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32) is added to `thirdparty/` and linked in CMake for building the package, and in `third_party/nvidia/backend/driver.c` and `driver.py` it's rewritten with `LoadLibrary` for jitting
* In `lib/Analysis/Utility.cpp` and `lib/Dialect/TritonGPU/Transforms/Utility.cpp`, explicit namespaces are added to support the resolution behaviors of MSVC
* In `python/src/interpreter.cc` the GCC built-in `__ATOMIC` memory orders are replaced with `std::memory_order`
