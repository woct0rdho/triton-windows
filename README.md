# [Triton](https://github.com/triton-lang/triton) fork for Windows support

**Announcement**: The development has moved to https://github.com/triton-lang/triton-windows . Users are not affected by this change, and you can continue to install triton-windows using pip. Currently the maintainers of the new repo are [@woct0rdho](https://github.com/woct0rdho) and [@jammm](https://github.com/jammm).

Based on [andreigh](https://github.com/andreigh/triton/tree/windows), [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), [anmyachev](https://github.com/triton-lang/triton/issues?q=author%3Aanmyachev), and more development in the community. Thank you all!

## What's supported

* `triton.jit` and `torch.compile` just work
* All unit tests passed
* It's as fast as on WSL on the same machine
* Windows 10 and 11 are supported
* Nvidia GPU is supported
    * For AMD GPU, we're beginning to add support in this repo, see https://github.com/woct0rdho/triton-windows/pull/179
    * For older AMD GPUs that are not supported by TheRock, [ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda) has a lot of information. Despite the name, they have information for both ZLUDA and ROCm. They use https://github.com/lshqqytiger/triton , which is based on https://github.com/Repeerc/triton-amdgpu-windows
* On free-threaded Python (Python 3.13t/3.14t), it seems to work with MSVC/GCC/Clang, but not with the bundled TinyCC
* Proton profiler does not fully work yet. If you want to try it, you can build from source

## Installation

Triton accelerates your AI model by compiling things on your computer. You need to install it in the correct environment.

### 1. GPU

Check your GPU model. Technically they're categorized by 'compute capability' (also known as 'CUDA architecture', 'streaming multiprocessor version', or 'sm'). For example:

<details>
<summary>RTX 50xx (Blackwell architecture, sm120)</summary>

This is officially supported by Triton. It only works with Triton >= 3.3, PyTorch >= 2.7, and CUDA >= 12.8 .
</details>

<details>
<summary>RTX 40xx (Ada architecture, sm89)</summary>

This is officially supported by Triton.
</details>

<details>
<summary>RTX 30xx (Ampere architecture, sm86)</summary>

This is officially supported by Triton. Although fp8 (also known as float8) on Ampere is not supported by the official Triton, it's supported since `triton-windows 3.5.0.post21`.
</details>

<details>
<summary>GTX 16xx/RTX 20xx (Turing architecture, sm75)</summary>

This is officially supported by Triton <= 3.2 . Support for Turing has been dropped since Triton 3.3, see https://github.com/triton-lang/triton/pull/5066

Although fp8 (also known as float8) and bf16 (also known as bfloat16) on Turing are not supported by the official Triton, fp8 is supported since `triton-windows 3.2.0.post21`.
</details>

<details>
<summary>GTX 10xx (Pascal architecture, sm61) and older</summary>

This is not supported. See https://github.com/woct0rdho/triton-windows/issues/133 if you want to help.
</details>

Also, make sure you have the latest GPU driver.

### 2. Python environment

Check how your Python is installed. Either of the following environments is supported:
* **Embeded**: You use an all-in-one AI software package such as ComfyUI
    * There should be a folder `python_embeded` in the ComfyUI installation folder
        * For FramePack, it's `system\python` in the FramePack installation folder
        * Other AI software may put this folder at a different path
    * In this case, don't directly run `python`, but use the full path `C:\path\to\python_embeded\python.exe`
    * Also, don't directly run `pip`, but instead run `C:\path\to\python_embeded\python.exe -m pip`
    * By default there is no `pip.exe` in the folder `python_embeded`. If you directly run `pip`, you're actually running a `pip.exe` installed somewhere else on your computer
    * It's ok to first `cd` to `python_embeded`, then run `.\python.exe`, but remember to add `.\` to run an executable in the current folder. In PowerShell, without `.\`, you're still running a `python.exe` installed somewhere else on your computer
* **System-wide**: You install Python at a location like `C:\Python312\` or `C:\Program Files\Python312\` and directly use it
* **User-wide**: You install Python at a location like `C:\Users\<your username>\AppData\Local\Programs\Python\Python312\` and directly use it
* **conda**: You create a virtual environment using `conda`
* **Python venv**: You create a virtual environment using `venv` or `virtualenv`

I don't recommend installing Python from Windows Store, because it's complicated to interact with a 'packaged' Windows app.

For other environment managers like poetry or uv, if you find problems, please open an issue.

Make sure what environment you're using. You can run `Get-Command -All python` in PowerShell (or `where python` in cmd) to see the installation path of Python, and `python --version` to see its version. If you see multiple Python installations, make sure that you install and run everything from the first one.
* For example, if you think you're using Python 3.12, but pip downloads a wheel with `cp311` in its name, then it means you're not using the Python environment you think

Don't mix two environments, unless you know them very well.
* If you're using ComfyUI with embeded Python, then don't use conda or venv
* If you're already using conda, then always create a new env using conda, and don't use Python venv

### 3. PyTorch

Although technically Triton can be used alone, in the following let's assume you use it with PyTorch. Each PyTorch minor version is only guaranteed to work with a specific Triton minor version:
| PyTorch | Triton |
| --- | --- |
| 2.4 | 3.1 |
| 2.5 | 3.1 |
| 2.6 | 3.2 |
| 2.7 | 3.3 |
| 2.8 | 3.4 |
| 2.9 | 3.5 |
| 2.10 | 3.6 |

PyTorch 2.3 and older are not supported in this repo.

If you have to use Triton 3.2 because you're using an old GPU, then you can try to use Triton 3.2 with PyTorch >= 2.7, but it's not guaranteed to always work.

### 4. CUDA

You can skip this.

<details>
<summary>Details</summary>

Since `triton-windows 3.2.0.post11`, a minimal CUDA toolchain is bundled in the Triton wheels, so you don't need to manually install it.

CUDA toolchain minor version bundled in each Triton minor version:
| Triton | CUDA |
| --- | --- |
| 3.1 .. 3.2 | 12.4 |
| 3.3 .. 3.6 | 12.8 |

See [nvidia-toolchain-version.json](https://github.com/triton-lang/triton/blob/main/cmake/nvidia-toolchain-version.json) for the detailed versions.

If you need to override the CUDA toolchain, you can set the environment variable `CUDA_PATH`.
</details>

<details>
<summary>Instructions for older or custom wheels without bundled CUDA</summary>

CUDA 12 is required. CUDA 11 and older are not supported. Choose either of the following ways to install CUDA:

**a) System-wide**: Recommended for most people
<details>
<summary>Expand</summary>

1. Install PyTorch with CUDA using pip
2. Install CUDA toolkit from [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. When installing, you need to choose both 'CUDA Development' and 'CUDA Runtime'. Make sure these folders exist on your computer: (Change the version number according to your installation)
    ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64
    ```
4. Then you need to add the path of CUDA to the Windows `PATH`:
    * The path is like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`
    * Make sure this folder exists
5. If you open a new PowerShell, type `ptxas --version`, and it shows your CUDA version like `Cuda compilation tools, release 12.8, V12.8.61`, then you're doing right
</details>

**b) conda**: Do this only if you're already using conda
<details>
<summary>Expand</summary>

* Install the following packages:
    ```pwsh
    conda install -c conda-forge cuda-nvcc pytorch-gpu
    ```
* Starting from PyTorch 2.6, PyTorch is no longer released in `pytorch` channel, and it should be installed in `conda-forge` channel
</details>

**c) pip**: Do this if you don't want to install too much boilerplate, and you want to contain everything in a venv, with minimal impact to the system
<details>
<summary>Expand</summary>

1. Install PyTorch with CUDA using pip
2. Install the following packages:
    ```pwsh
    pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
    ```
3. There should be a folder `Lib\site-packages\nvidia\cuda_runtime\` in your Python installation path (or venv), and you need to add a library in it
    * Download it from https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/cuda_12.8_lib.zip
    * Choose 12.4, 12.6, or 12.8 according to your CUDA version
    * Put the folder `lib` into `cuda_runtime`
</details>

For details about version compatibility of various pip packages and CUDA, see https://github.com/woct0rdho/triton-windows/issues/43
</details>

### 5. C compiler

You can skip this.

<details>
<summary>Details</summary>

Since `triton-windows 3.2.0.post13`, TinyCC is bundled in the Triton wheels, so you don't need to manually install a C compiler to use Triton. Packages that directly call `triton.jit`, such as SageAttention, will just work.

You still need to install a C++ compiler if you use `torch.compile` targeting CPU. This may happen when you use nodes like 'CompileModel' in ComfyUI. Triton does not affect how PyTorch configures the C++ compiler in this case.

If you need to override the C compiler, you can set the environment variable `CC`. MSVC, GCC, and Clang are supported for the JIT compilation in Triton.

If you set `CC` in the 'Environment Variables' window, then it should be a string, not a list. A list will implicitly add a semicolon `;` at its end and cause problems.
</details>

<details>
<summary>Instructions for older or custom wheels without bundled TinyCC</summary>

If you don't have a C compiler, I recommend to install MSVC and Windows SDK.
* You can install them in Visual Studio
    * If you don't want to install the whole Visual Studio, you can just install [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)
* Visual Studio >= 2017 is supported
* Choose the latest version of MSVC and Windows SDK from the list

Then you need to add the path containing `cl.exe` to the Windows `PATH`:
* The path is like `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64`
* Change the version numbers according to your installation, and make sure this folder accually exists on your computer
* If you open a new PowerShell, type `cl`, and it shows `Microsoft (R) C/C++ Optimizing Compiler ...`, then you're doing right
</details>

<details>
<summary>Note on automatically adding the path</summary>

Do this if you don't want to permanently modify the Windows `PATH`.

Before running Python, if you use PowerShell, run the following: (Find the ps1 file according to your installation)
```pwsh
&"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```
Or if you use cmd, run the following: (This is equivalent to 'x64 Native Tools Command Prompt' from the Start menu)
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
```
It automatically adds the paths containing `cl.exe` and other relevant VS components, see https://github.com/woct0rdho/triton-windows/issues/79 . Although it does not set the environment variable `CC`, it sets `VCINSTALLDIR, VCToolsVersion, WindowsSdkDir, WindowsSDKVersion`, and Triton will recognize them.
</details>

### 6. vcredist

vcredist is required (also known as 'Visual C++ Redistributable for Visual Studio 2015-2022', `msvcp140.dll`, `vcruntime140.dll`), because `libtriton.pyd` is compiled by MSVC. Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe

### 7. Triton

Since `triton-windows 3.2.0.post11`, the wheels are published to https://pypi.org/project/triton-windows/ , so you don't need to manually download a wheel from GitHub releases, and pip will automatically find it.

If you've installed an old version of `triton`, first uninstall it:
```pwsh
pip uninstall triton
```
Now you can install `triton-windows 3.6`, or upgrade the already installed version. To prevent breaking with your installed PyTorch when a new version of Triton is released in future, you can limit the version to be < 3.7:
```pwsh
pip install -U "triton-windows<3.7"
```
Note again that if you're using the embeded Python, then instead of directly run `pip`, you need:
```pwsh
C:\path\to\python_embeded\python.exe -m pip install -U "triton-windows<3.7"
```
Or if you want `triton-windows 3.2`, then run:
```pwsh
pip install -U "triton-windows<3.3"
```

### 8. Special notes for ComfyUI with embeded Python

* There should be a Python folder
    * For ComfyUI, it's `python_embeded` in the ComfyUI installation folder
    * For FramePack, it's `system\python` in the FramePack installation folder
    * Other AI software may put the Python folder at a different path
    * If you created a venv, depending on how you created it, the Python folder may be just the venv folder or the `venv\Scripts` folder
    * If you're not sure, you can run `os.path.dirname(sysconfig.get_paths()["include"])` to find the Python folder, see [`py_include_dir`](https://github.com/woct0rdho/triton-windows/blob/819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c/python/triton/runtime/build.py#L28)
* You need to put two folders `include` and `libs` into the Python folder to make Triton work
    * Be careful: It is 'libs', not 'lib'. There may already be a folder `Lib` in the Python folder, containing things like `site-packages` or `__future__.py`. You should not modify the `Lib` folder
    * If you're using ComfyUI_windows_portable >= 0.3.50 with Python 3.13, then download the two folders here: [python_3.13.2_include_libs.zip](https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.13.2_include_libs.zip)
    * If you're using FramePack with Python 3.10, then download the two folders here: [python_3.10.11_include_libs.zip](https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.10.11_include_libs.zip)
    * The minor version (3.9/3.10 ...) must be correct, but the patch version (3.10.6/3.10.7 ...) can be different
    * If you're using another Python version, you can find the two folders at https://github.com/woct0rdho/triton-windows/releases/v3.0.0-windows.post1/
* (For developers: This is equivalent to `python-dev` on Linux, and you can obtain the two folders from nuget when bundling Python in your app, see https://github.com/comfyanonymous/ComfyUI/pull/7200 )

## Test if it works

Before using Triton in larger projects like ComfyUI, please run the following script to test if Triton itself works.
* You need to save the code in a file, such as `test_triton.py`, then run `python test_triton.py`
* When you open an issue, please show the command you use to run this test, and the full error log
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

## Troubleshoot the test above

### `ModuleNotFoundError: No module named 'triton.language'; 'triton' is not a package`

Don't name the test script `triton.py`. Also, check if there is a folder named `triton` in your current directory. If so, Python will think it's the 'triton' package and fail to import.

### `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'`

This is because your `setuptools` is outdated. Run the following and try again:
```pwsh
python -m ensurepip -U
python -m pip install -U pip
python -m pip install -U setuptools
```

### `PermissionError: [WinError 5] Access is denied: 'C:\\Users\\<your username>\\.triton'`

This is because of the permission settings of your user folder, see https://github.com/lllyasviel/FramePack/issues/221

### `ImportError: DLL load failed while importing libtriton`

This is usually because your vcredist DLLs are too old.

If you're using conda, then you may try:
<details>
<summary>conda</summary>

```pwsh
conda install -c conda-forge vc14_runtime
```
</details>

If you're not using conda, then you need to find the vcredist DLLs (`vcruntime140.dll`, `vcruntime140_1.dll`) in your Python installation folder:

<details>
<summary>Embeded Python (You use an all-in-one AI software package such as ComfyUI)</summary>

* For ComfyUI, the DLLs should be in the folder `python_embeded`.
* For FramePack, it's `system\python` in the FramePack installation folder
* Other AI software may put this folder at a different path
</details>

<details>
<summary>Other Python installation (system-wide/user-wide/venv)</summary>

If you're not sure, you can run the following in the same Python environment:
```python
import sysconfig
print(sysconfig.get_paths())
```
For example, it may show `{'stdlib': 'C:\\Python312\\Lib', 'platstdlib': 'C:\\tmp\\.venv\\Lib', ...}`, where `stdlib` shows that the 'base' Python installation folder (not the venv folder) is `C:\Python312\` (without the last `Lib`). The DLLs should be in this folder.
</details>

After finding the DLLs in the Python installation folder, you can install the latest [vcredist](https://aka.ms/vs/17/release/vc_redist.x64.exe), then copy the DLLs `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll` from `C:\Windows\System32\` to the Python installation folder, and replace the existing ones.

You can right-click the DLL -> Properties -> Details to see its version. A new enough version, such as 14.42, is required by my Triton wheels.

### `ImportError: DLL load failed while importing cuda_utils`

1. Delete the cache folders:
    ```
    C:\Users\<your username>\.triton\cache\
    C:\Users\<your username>\AppData\Local\Temp\torchinductor_<your username>\
    ```
    * You may also need to delete these cache folders when you change the Python version, install another version of Triton, or change the C compiler or CUDA
    * It's ok if these folders do not exist on your computer. The first folder exists only if you have used `triton.jit` (which is used by packages like SageAttention), and the second folder exists only if you have used `torch.compile`
2. Double check your Python version: You can run `Get-Command -All python` in PowerShell (or `where python` in cmd) to see the installation path of Python, and `python --version` to see its version. If you see multiple Python installations, make sure that you install and run everything from the first one
3. If you're using ComfyUI with embeded Python, make sure that you copy-pasted the folders `include` and `libs` from the correct version of Python

### `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`

You also need to delete the cache folders above.

This should not happen if you upgrade to Python 3.13, see https://github.com/python/cpython/issues/104922 . If the error still exists, you may try to debug following https://github.com/woct0rdho/triton-windows/issues/163

### dlltracer

If the above still doesn't work, you may try:
* Install [dlltracer](https://github.com/microsoft/dlltracer-python) in the same Python environment
* In an administrator PowerShell, run the following script:
```python
import sys
import dlltracer

print("import torch")
with dlltracer.Trace(out=sys.stdout):
    import torch

print("import triton")
with dlltracer.Trace(out=sys.stdout):
    import triton

print("begin definition")
with dlltracer.Trace(out=sys.stdout):
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

print("begin torch add")
with dlltracer.Trace(out=sys.stdout):
    a = torch.rand(3, device="cuda")
    b = a + a

print("begin jit add")
with dlltracer.Trace(out=sys.stdout):
    b_compiled = add(a, a)

print(b_compiled - b)
print("If you see tensor([0., 0., 0.], device='cuda:0'), then it works")
```
* Open an issue. Please show the command you use to run this test, and the full error log

If it shows `PermissionError: [WinError 5] failed to start trace (0x00000005)`, then you need to make sure to run it as administrator.

(**Security reminder**: You don't need the administrator privilege to run Triton and other usual Python code. It's only dlltracer that needs it.)

If it shows `Failed \Device\...\cuda_utils.pyd`, please also:
* Find `cuda_utils.pyd` at this location
* Use [DependenciesGui](https://github.com/lucasg/Dependencies) (or similar tools) to check what DLLs this `cuda_utils.pyd` depends on, and send a screenshot (or other related information) in the issue

## Known issues

### Windows file path length limit (260) causes compilation failure

`torch.compile` may create temp files with very long filenames, causing errors like:
```
  File "C:\...\Lib\site-packages\torch\_inductor\runtime\triton_heuristics.py", line 537, in _precompile_config
    binary = triton.compile(*compile_args, **compile_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\Lib\site-packages\triton\compiler\compiler.py", line 288, in compile
    metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\Lib\site-packages\triton\runtime\cache.py", line 122, in put
    with open(temp_path, mode) as f:
         ^^^^^^^^^^^^^^^^^^^^^
torch._inductor.exc.InductorError: FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\<your username>\\AppData\\Local\\Temp\\torchinductor_<your username>\\triton\\0\\...LONG...FILE...NAME...'
```
Or errors like:
```
[WinError 206] The filename or extension is too long
```
The solution is to [enable Windows' long path support](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later). A reboot is required after the modification.

### fp8 is not supported on RTX 30xx and older GPUs

If you see errors like
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
then it's because in the official Triton, fp8 only works on Nvidia GPUs with sm >= 89, such as RTX 40xx and newer.

Since `triton-windows 3.5.0.post21`, fp8 on RTX 30xx is supported.

### Error with `os.rename`

If you see errors like
```
FileExistsError: [WinError 183] Cannot create a file when that file already exists: ...
```
then you need: https://github.com/pytorch/pytorch/issues/138211

This has been fixed since PyTorch 2.6 .

### Error with model offloading

If you're using ComfyUI, the model is compiled, and you see error messages like
```
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```
then you may use `--gpu-only` when launching ComfyUI to disable model offloading, see https://github.com/woct0rdho/triton-windows/issues/61

### No module named 'triton.ops'

`triton.ops` was removed in Triton 3.1, and this is because some of your Python package is outdated (most likely `bitsandbytes`), see https://github.com/woct0rdho/triton-windows/issues/65

### `Exception Code: 0x80000003`

If you see `Exception Code: 0x80000003` in `libtriton.pyd` in a function like `registerImplicitTypeID`, that may not actually be an issue in Triton, but a CUDA error happened earlier, see https://github.com/thu-ml/SageAttention/issues/270 . If you want to help debug, you may set the environment variable `CUDA_LAUNCH_BLOCKING=1` and run again.

## Build from source

See [BUILD.md](https://github.com/woct0rdho/triton-windows/blob/readme/BUILD.md). This is for developers.
