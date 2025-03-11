## Build from source

As of the release `post12`, the wheels are built with MSVC v143 (the default version in VS 2022). However, I just learned that PyTorch is built with MSVC v142 (the default version in VS 2019), and a binary built with a newer MSVC may not work with an older vcredist (which is a cause of `ImportError: DLL load failed while importing libtriton`). In future it's better to be a bit more conservative and use MSVC v142. 

Set the binary, include, and library paths of Python, MSVC, Windows SDK, and CUDA in PowerShell (help wanted to automatically find these in CMake, or using something equivalent to `vcvarsall.bat` in PowerShell):
```pwsh
$Env:Path =
"C:\Windows\System32;" +
"C:\Python312;" +
"C:\Python312\Scripts;" +
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" +
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64;" +
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;" +
"C:\Program Files\Git\cmd"
$Env:INCLUDE =
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\include;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;" +
"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;" +
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include;" +
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\CUPTI\include"
$Env:LIB =
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\lib\x64;" +
"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;" +
"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
```
* cibuildwheel needs the binaries in `C:\Windows\System32\`
* If you want to build the C++ unit tests and don't set `TRITON_BUILD_UT=0`, then you need git

Then you can either download some dependencies online, or set up an offline build: (When switching between online/offline build, remember to delete `CMakeCache.txt`)

<details>
<summary>Download dependencies online</summary>

`setup.py` will download LLVM and JSON into the cache folder set by `TRITON_HOME` (by default `C:\Users\<your username>\.triton\`) and link against them.

A minimal CUDA toolchain (`ptxas.exe`, `cuda.h`, `cuda.lib`) will also be downloaded and bundled in the wheel.

If you're in China, make sure to have a good Internet connection.
</details>

<details>
<summary>Offline build</summary>

Enable offline build:
```pwsh
$Env:TRITON_OFFLINE_BUILD = "1"
```

Build LLVM using MSVC according to the instructions of the official Triton:
```pwsh
# Check out the commit according to cmake/llvm-hash.txt (Sadly, you need to rebuild LLVM every week if you want to keep up to date)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_BUILD_TOOLS=OFF -DLLVM_CCACHE_BUILD=ON llvm
cmake --build build -j 8 --config Release
```
* See https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm
* When cloning LLVM, use `git clone --filter=blob:none https://github.com/llvm/llvm-project.git`. You don't want to clone the whole history as it's too large
* The official Triton enables `-DLLVM_ENABLE_ASSERTIONS=ON` when compiling LLVM, and this will increase the binary size of Triton
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

Download JSON according to `setup.py`:
* https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip

Set their paths:
```pwsh
$Env:LLVM_SYSPATH = "C:/llvm-project/build"
$Env:JSON_SYSPATH = "C:/json"
```
(For triton <= 3.1, you also need to download pybind11 and set its path according to `setup.py`)

The CUDA toolchain is not bundled by default in the offline build.
</details>

You can disable these if you don't need them: (`TRITON_BUILD_BINARY` is added in my fork)
```pwsh
$Env:TRITON_BUILD_BINARY = "0"
$Env:TRITON_BUILD_PROTON = "0"
$Env:TRITON_BUILD_UT = "0"
```

I recommend to use ccache if you installed it:
```pwsh
$Env:TRITON_BUILD_WITH_CCACHE = "1"
```

Clone this repo, checkout `v3.2.x-windows` branch, make an editable build using pip:
```pwsh
pip install --no-build-isolation --verbose -e python
```

Build the wheels: (This is for distributing the wheels to others. You don't need this if you only use Triton on your own computer)
```pwsh
git clean -dfX
$Env:CIBW_BUILD = "{cp39-win_amd64,cp310-win_amd64,cp311-win_amd64,cp312-win_amd64,cp313-win_amd64}"
$Env:CIBW_BUILD_VERBOSITY = "1"
$Env:TRITON_WHEEL_VERSION_SUFFIX = "+windows"
cibuildwheel python
```

## Dev notes

* To implement `dlopen`:
    * For building the package, [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32) is added to `thirdparty/` and linked in CMake, so I don't need to rewrite it every time
    * For jitting, in `third_party/nvidia/backend/driver.c` and `driver.py` it's rewritten with `LoadLibrary`
* `python/triton/windows_utils.py` contains many ways to find the paths of Python, MSVC, Windows SDK, and CUDA
* ~~In `lib/Analysis/Utility.cpp` and `lib/Dialect/TritonGPU/Transforms/Utility.cpp`, explicit namespaces are added to support the resolution behaviors of MSVC~~ (This is no longer needed since Triton 3.3)
* ~~In `python/src/interpreter.cc` the GCC built-in `__ATOMIC` memory orders are replaced with `std::memory_order`~~ (Upstreamed, see https://github.com/triton-lang/triton/pull/4976 )
* ~~In `third_party/nvidia/backend/driver.py`, function `make_launcher`, `int64_t` should map to `L` in `PyArg_ParseTuple`. This fixes the error `Python int too large to convert to C long`.~~ (Upstreamed, see https://github.com/triton-lang/triton/pull/5351 )
* How TorchInductor is designed to support Windows: https://github.com/pytorch/pytorch/issues/124245
