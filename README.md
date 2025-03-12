# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `v3.2.x-windows` branch for the code, forked from `release/3.2.x` branch of the official repo.

Based on [andreigh](https://github.com/andreigh/triton/tree/windows), [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), [anmyachev](https://github.com/triton-lang/triton/issues?q=author%3Aanmyachev), and more development in the community. Thank you all!

## Why?

* Free software should run on non-free platforms, as per Richard Stallman
* This is required by `torch.compile`, and used by torchao, SageAttention, ParaAttention, and more packages
* Memory management on WSL is hard
* Catgirl matters

## Progress

**Announcement**: Since the release `triton-windows==3.2.0.post11`, the wheels are published to PyPI, and no longer to GitHub. You still need to manually install MSVC, and we're discussing with people from the ComfyUI community to try to figure out a simpler way to set up everything.

* `triton.jit` and `torch.compile` just work
* All unit tests passed
* When I run Flux or HunyuanVideo in ComfyUI on Windows, it's almost as fast as on WSL on the same machine
* Windows 10 and 11 are supported
* Only MSVC is supported to build the package, because it's much more stable than GCC and Clang when working with CUDA on Windows
* MSVC, GCC, and Clang are supported for JIT compilation on the user's computer. Help wanted to bundle a minimal C compiler in the wheels
* Only Nvidia GPU is supported, help wanted to support other backends
    * For AMD GPU, you may try https://github.com/Repeerc/triton-amdgpu-windows
    * For Intel XPU, you may try https://github.com/intel/intel-xpu-backend-for-triton
* TODO: Set up CI (help wanted)

## Install from wheel

Triton accelerates your AI model by compiling things on your computer. It's not a simple package that just works with `pip install`, and you need to set up the compiler and the libraries used by it. This may be  unfamiliar for Windows users, and you can follow the instructions below.

### 1. GPU

Check your GPU model. Technically they're categorized by 'compute capability' (also known as 'CUDA arch' or 'sm'), and here I use RTX models for example:

<details>
<summary>RTX 50xx (Blackwell)</summary>

This only works with Triton >= 3.3 (pre-release), PyTorch >= 2.7 (nightly), and CUDA 12.8 .
</details>

<details>
<summary>RTX 40xx (Ada)</summary>

This is officially supported by Triton.
</details>

<details>
<summary>RTX 30xx (Ampere)</summary>

This is officially supported by Triton, but fp8 (also known as float8) will not work, see the [known issue](https://github.com/woct0rdho/triton-windows#fp8-is-not-supported-on-rtx-30xx-and-older-gpus). I recommend to use GGUF instead of fp8 models in this case.
</details>

<details>
<summary>RTX 20xx (Turing) or older</summary>

This is not officially supported by Triton. It can run some simple AI models, but not always. fp8 (also known as float8) and bf16 (also known as bfloat16) will not work. I recommend to use GGUF instead of fp8 or bf16 models in this case.
</details>

### 2. Python environment

Check how your Python is installed. Either of the following environments is supported:
* **Embeded**: You use an all-in-one package of ComfyUI (or some other AI software), and there is a folder `python_embeded` in it
    * In this case, don't directly run `python`, but use the full path `C:\path\to\python_embeded\python.exe`
    * Also, don't directly run `pip`, but instead run `C:\path\to\python_embeded\python.exe -m pip`
    * By default there is no `pip.exe` in the folder `python_embeded`. If you directly run `pip`, you're actually running a `pip.exe` installed somewhere else on your computer
* **System-wide**: You install Python at a location like `C:\Python312\` and directly use it
* **User-wide**: You install Python at a location like `C:\Users\<your username>\AppData\Local\Programs\Python\Python312\` and directly use it
* **conda**: You create a virtual environment using `conda`
* **Python venv**: You create a virtual environment using `venv` or `virtualenv`

For other environment managers like poetry or uv, if you find problems, please open an issue.

Make sure what environment you're using. You can run `Get-Command -All python` in PowerShell (or `where python` in cmd) to see the installation path of Python, and `python --version` to see its version. If you see multiple Python installations, make sure that you install and run everything from the first one.
* For example, if you think you're using Python 3.12, but pip downloads a wheel with `cp311` in its name, then it means you're not using the Python environment you think

Don't mix two environments, unless you know them very well.
* If you're using ComfyUI with embeded Python, then don't use conda or venv
* If you're already using conda, then always create a new env using conda, and don't use Python venv

### 3. PyTorch

Although technically Triton can be used alone, in the following let's assume you use it with PyTorch. Check your PyTorch version:

Triton 3.2 works with PyTorch >= 2.6 . If you're using PyTorch < 2.6, I recommend to upgrade to 2.6 because there are several improvements to `torch.compile`.

Triton 3.3 (pre-release) works with PyTorch >= 2.7 (nightly).

PyTorch tagged with CUDA 12 is required. CUDA 11 is not supported.

### 4. CUDA

Since the release `triton-windows==3.2.0.post11`, a minimal CUDA toolchain is bundled in the Triton wheels, so you don't need to manually install it.

Triton 3.2 bundles CUDA 12.4, and Triton 3.3 bundles CUDA 12.8 . They should be compatible with other CUDA 12.x because of the [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) of CUDA. CUDA 11 and older versions are not supported.

<details>
<summary>Instructions for older or custom wheels without bundled CUDA</summary>

Choose either of the following ways to install CUDA:

**a) System-wide**: Recommended for most people
<details>
<summary>Expand</summary>

1. Install PyTorch with CUDA using pip
2. Install CUDA toolkit from [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. When installing, you need to choose both 'CUDA Development' and 'CUDA Runtime'. Make sure these folders exist on your computer: (Change the version number according to your installation)
    ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64
    ```
4. Then you need to add the path of CUDA to the Windows `PATH`:
    * The path is like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
    * Make sure this folder exists
5. If you open a new PowerShell, type `ptxas --version`, and it shows your CUDA version like `Cuda compilation tools, release 12.6, V12.6.85`, then you're doing right
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
    * Download it from https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/cuda_12.6_lib.zip
    * Choose 12.6 or 12.8 according to your CUDA version
    * Put the folder `lib` into `cuda_runtime`
</details>

For details about version compatibility of various pip packages and CUDA, see https://github.com/woct0rdho/triton-windows/issues/43
</details>

### 5. MSVC and Windows SDK

C compiler is required. If you don't have one, I recommend to install MSVC and Windows SDK.
* You can install them in Visual Studio
    * If you don't want to install the whole Visual Studio, you can just install [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)
* Visual Studio >= 2017 is supported
* Choose the latest version of MSVC and Windows SDK from the list

Then you need to add the path containing `cl.exe` to the Windows `PATH`:
* The path is like `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64`
* Change the version numbers according to your installation, and make sure this folder accually exists on your computer
* If you open a new PowerShell, type `cl`, and it shows `Microsoft (R) C/C++ Optimizing Compiler ...`, then you're doing right

<details>
<summary>Note on automatically adding the path</summary>
(Do this if you don't want to permanently modify the Windows `PATH`)

Run 'Developer PowerShell for VS 2022' (or 'Developer Command Prompt for VS 2022') from the Start menu (or in VS), and it will automatically add the paths containing `cl.exe` and other relevant VS components.
</details>

### 6. vcredist

vcredist is required (also known as 'Visual C++ Redistributable for Visual Studio 2015-2022', `msvcp140.dll`, `vcruntime140.dll`). Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe

### 7. Triton

Now you can install the wheel, or upgrade the already installed version:
```pwsh
pip install -U triton-windows
```
Note again that if you're using the embeded Python, then instead of directly run `pip`, you need:
```pwsh
C:\path\to\python_embeded\python.exe -m pip install -U triton-windows
```
For Triton 3.3 (pre-release), you need:
```pwsh
pip install -U --pre triton-windows
```

### 8. Special notes for ComfyUI with embeded Python

* There should be a folder `python_embeded` in your ComfyUI installation path
* You need to put two folders `include` and `libs` into `python_embeded` to make Triton work
    * Be careful: It is 'libs', not 'lib'. There may already be a folder `Lib` in `python_embeded`, containing things like `site-packages` or `__future__.py`. You should not modify the `Lib` folder
    * If you're using ComfyUI_windows_portable >= 0.2.4 with Python 3.12.7, you can download the two folders here: https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip
    * If you're using another Python version, you can find the two folders at https://github.com/woct0rdho/triton-windows/releases/v3.0.0-windows.post1/
* (For developers: This is equivalent to `python-dev` on Linux, and you can get the two folders from nuget when packaging the embeded Python with your app, see https://bugs.python.org/issue38224 )

## Test if it works

Before using Triton in larger projects like ComfyUI, please run the following script to test if Triton itself works. You need to save the code in a file, such as `test_triton.py`, then run `python test_triton.py`.
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

### ModuleNotFoundError: No module named 'triton.language'; 'triton' is not a package

Don't name the test script `triton.py`. Also, check if there is a folder named `triton` in your current directory. If so, Python will think it's the 'triton' package and fail to import.

### AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'

This is because your `setuptools` is outdated. Run the following and try again:
```pwsh
python -m ensurepip -U
python -m pip install -U pip
python -m pip install -U setuptools
```

### ImportError: DLL load failed while importing libtriton

If you see this and there are `vcruntime140.dll` and `vcruntime140_1.dll` in the folder containing `python.exe`, then you may try:
1. Install the latest version of vcredist from https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Copy-paste `msvcp140.dll`, `vcruntime140.dll`, and `vcruntime140_1.dll` from `C:\Windows\System32\` to the folder containing `python.exe`, and replace the existing DLLs

If you're using conda, then you may try:
```pwsh
conda install -c conda-forge vc14_runtime
```

### ImportError: DLL load failed while importing cuda_utils

1. If these cache folders exist on your computer, delete them:
    ```
    C:\Users\<your username>\.triton\cache\
    C:\Users\<your username>\AppData\Local\Temp\torchinductor_<your username>\
    ```
    You may also need to delete these cache folders when you change the Python version, install another version of Triton, or change the version of MSVC, Windows SDK, or CUDA
2. Double check your Python version: You can run `Get-Command -All python` in PowerShell (or `where python` in cmd) to see the installation path of Python, and `python --version` to see its version. If you see multiple Python installations, make sure that you install and run everything from the first one
3. If you're using ComfyUI with embeded Python, make sure that you copy-pasted the folders `include` and `libs` from the correct version of Python

### dlltracer

If the above still doesn't work, you may try:
* Install [dlltracer](https://github.com/microsoft/dlltracer-python) in the same Python environment
* In an administrator PowerShell, run the following script:
```python
import torch

import sys
import dlltracer
with dlltracer.Trace(out=sys.stdout):
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
* Open an issue and paste the results

If it shows `Failed \Device\...\cuda_utils.pyd`, please also:
* Find `cuda_utils.pyd` at this location
* Use [DependenciesGui](https://github.com/lucasg/Dependencies) (or similar tools) to check what DLLs this `cuda_utils.pyd` depends on, and send a screenshot (or other related information) in the issue

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

This has been fixed since PyTorch 2.6 .

### Error with model offloading

If you're using ComfyUI, the model is compiled, and you see error messages like
```
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```
then you may use `--gpu-only` when launching ComfyUI to disable model offloading. See https://github.com/woct0rdho/triton-windows/issues/61

### No module named 'triton.ops'

`triton.ops` was removed in Triton 3.1, and this is because some of your Python package is outdated (most likely `bitsandbytes`). See https://github.com/woct0rdho/triton-windows/issues/65

## Build from source

See [BUILD.md](https://github.com/woct0rdho/triton-windows/blob/readme/BUILD.md). This is for developers.
