import functools
import os
import re
import subprocess
import sys
import sysconfig
import winreg
from functools import partial
from glob import glob
from pathlib import Path


def find_in_program_files(rel_path):
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    path = Path(program_files) / rel_path
    if path.exists():
        return path

    program_files = os.environ.get("ProgramW6432", r"C:\Program Files")
    path = Path(program_files) / rel_path
    if path.exists():
        return path

    return None


def parse_version(s, prefix=""):
    s = s.removeprefix(prefix)
    try:
        return tuple(int(x) for x in s.split("."))
    except ValueError:
        return None


def unparse_version(t, prefix=""):
    return prefix + ".".join([str(x) for x in t])


def max_version(versions, prefix="", check=lambda x: True):
    versions = [parse_version(x, prefix) for x in versions]
    versions = [
        x for x in versions if x is not None and check(unparse_version(x, prefix))
    ]
    if not versions:
        return None
    version = unparse_version(max(versions), prefix)
    return version


def check_msvc(msvc_base_path, version):
    return all(
        x.exists()
        for x in [
            msvc_base_path / version / "include" / "vcruntime.h",
            msvc_base_path / version / "lib" / "x64" / "vcruntime.lib",
        ]
    )


def find_msvc_vswhere():
    vswhere_path = find_in_program_files(
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if vswhere_path is None:
        return None, None

    command = [
        str(vswhere_path),
        "-prerelease",
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-requires",
        "Microsoft.VisualStudio.Component.Windows10SDK",
        "-latest",
        "-property",
        "installationPath",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
    except subprocess.CalledProcessError:
        return None, None

    msvc_base_path = Path(output) / "VC" / "Tools" / "MSVC"
    if not msvc_base_path.exists():
        return None, None

    version = max_version(
        os.listdir(msvc_base_path), check=partial(check_msvc, msvc_base_path)
    )
    if version is None:
        return None, None

    return msvc_base_path, version


def find_msvc_envpath():
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        path = path.replace("/", "\\")
        match = re.compile(r".*\\VC\\Tools\\MSVC\\").match(path)
        if not match:
            continue

        msvc_base_path = Path(match.group(0))
        if not msvc_base_path.exists():
            continue

        version = max_version(
            os.listdir(msvc_base_path), check=partial(check_msvc, msvc_base_path)
        )
        if version is None:
            continue

        return msvc_base_path, version

    return None, None


def find_msvc_hardcoded():
    vs_path = find_in_program_files("Microsoft Visual Studio")
    if not vs_path.exists():
        return None, None

    paths = glob(str(vs_path / "*" / "*" / "VC" / "Tools" / "MSVC"))
    # First try the highest version
    paths = sorted(paths)[::-1]
    for msvc_base_path in paths:
        msvc_base_path = Path(msvc_base_path)
        version = max_version(
            os.listdir(msvc_base_path), check=partial(check_msvc, msvc_base_path)
        )
        if version is None:
            continue
        return msvc_base_path, version

    return None, None


def find_msvc():
    msvc_base_path, version = find_msvc_vswhere()
    if msvc_base_path is None:
        msvc_base_path, version = find_msvc_envpath()
    if msvc_base_path is None:
        msvc_base_path, version = find_msvc_hardcoded()
    if msvc_base_path is None:
        print("WARNING: Failed to find MSVC.")
        return [], []

    return (
        [str(msvc_base_path / version / "include")],
        [str(msvc_base_path / version / "lib" / "x64")],
    )


def check_winsdk(winsdk_base_path, version):
    return all(
        x.exists()
        for x in [
            winsdk_base_path / "Include" / version / "ucrt" / "stdlib.h",
            winsdk_base_path / "Lib" / version / "ucrt" / "x64" / "ucrt.lib",
        ]
    )


def find_winsdk_registry():
    try:
        reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        key = winreg.OpenKeyEx(
            reg, r"SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0"
        )
        folder = winreg.QueryValueEx(key, "InstallationFolder")[0]
        winreg.CloseKey(key)
    except OSError:
        return None

    winsdk_base_path = Path(folder)
    if not winsdk_base_path.exists():
        return None, None

    version = max_version(
        os.listdir(winsdk_base_path / "Include"),
        check=partial(check_winsdk, winsdk_base_path),
    )
    if version is None:
        return None, None

    return winsdk_base_path, version


def find_winsdk_hardcoded():
    winsdk_base_path = find_in_program_files(r"Windows Kits\10")
    if not winsdk_base_path.exists():
        return None, None

    version = max_version(
        os.listdir(winsdk_base_path / "Include"),
        check=partial(check_winsdk, winsdk_base_path),
    )
    if version is None:
        return None, None

    return winsdk_base_path, version


def find_winsdk():
    winsdk_base_path, version = find_winsdk_registry()
    if winsdk_base_path is None:
        winsdk_base_path, version = find_winsdk_hardcoded()
    if winsdk_base_path is None:
        print("WARNING: Failed to find Windows SDK.")
        return [], []

    return (
        [
            str(winsdk_base_path / "Include" / version / "shared"),
            str(winsdk_base_path / "Include" / version / "ucrt"),
            str(winsdk_base_path / "Include" / version / "um"),
        ],
        [
            str(winsdk_base_path / "Lib" / version / "ucrt" / "x64"),
            str(winsdk_base_path / "Lib" / version / "um" / "x64"),
        ],
    )


@functools.cache
def find_msvc_winsdk():
    msvc_inc_dirs, msvc_lib_dirs = find_msvc()
    winsdk_inc_dirs, winsdk_lib_dirs = find_winsdk()
    return msvc_inc_dirs + winsdk_inc_dirs, msvc_lib_dirs + winsdk_lib_dirs


@functools.cache
def find_python():
    version = sysconfig.get_python_version().replace(".", "")
    for python_base_path in [
        sys.exec_prefix,
        sys.base_exec_prefix,
        os.path.dirname(sys.executable),
        rf"C:\Python{version}",
    ]:
        python_lib_dir = Path(python_base_path) / "libs"
        if (python_lib_dir / f"python{version}.lib").exists():
            return [str(python_lib_dir)]

    print("WARNING: Failed to find Python libs.")
    return []


def check_cuda_pip(nvidia_base_path):
    return all(
        x.exists()
        for x in [
            nvidia_base_path / "cuda_nvcc" / "bin" / "ptxas.exe",
            nvidia_base_path / "cuda_runtime" / "include" / "cuda.h",
            nvidia_base_path / "cuda_runtime" / "lib" / "x64" / "cuda.lib",
        ]
    )


def find_cuda_pip():
    nvidia_base_path = Path(sysconfig.get_paths()["platlib"]) / "nvidia"
    if check_cuda_pip(nvidia_base_path):
        return (
            str(nvidia_base_path / "cuda_nvcc" / "bin"),
            [str(nvidia_base_path / "cuda_runtime" / "include")],
            [str(nvidia_base_path / "cuda_runtime" / "lib" / "x64")],
        )

    return None, [], []


def check_cuda(cuda_base_path):
    return all(
        x.exists()
        for x in [
            cuda_base_path / "bin" / "ptxas.exe",
            cuda_base_path / "include" / "cuda.h",
            cuda_base_path / "lib" / "x64" / "cuda.lib",
        ]
    )


def find_cuda_env():
    for cuda_base_path in ["CUDA_PATH", "CUDA_HOME"]:
        cuda_base_path = os.environ.get(cuda_base_path)
        if cuda_base_path is None:
            continue

        cuda_base_path = Path(cuda_base_path)
        if check_cuda(cuda_base_path):
            return cuda_base_path

    return None


def find_cuda_hardcoded():
    parent = find_in_program_files(r"NVIDIA GPU Computing Toolkit\CUDA")
    if parent is None:
        return None

    paths = glob(str(parent / "v12*"))
    # First try the highest version
    paths = sorted(paths)[::-1]
    for path in paths:
        cuda_base_path = Path(path)
        if check_cuda(cuda_base_path):
            return cuda_base_path

    return None


@functools.cache
def find_cuda():
    cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs = find_cuda_pip()
    if cuda_bin_path:
        return cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs

    cuda_base_path = find_cuda_env()
    if cuda_base_path is None:
        cuda_base_path = find_cuda_hardcoded()
    if cuda_base_path is None:
        print("WARNING: Failed to find CUDA.")
        return None, [], []

    return (
        str(cuda_base_path / "bin"),
        [str(cuda_base_path / "include")],
        [str(cuda_base_path / "lib" / "x64")],
    )
