import functools
import os
import re
import subprocess
import sys
import sysconfig
import warnings
import winreg
from collections.abc import Iterable
from functools import partial
from glob import glob
from pathlib import Path
from typing import Callable, Optional


def find_in_program_files(rel_path: str) -> Optional[Path]:
    program_files = os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    path = Path(program_files) / rel_path
    if path.exists():
        return path

    program_files = os.getenv("ProgramW6432", r"C:\Program Files")
    path = Path(program_files) / rel_path
    if path.exists():
        return path

    return None


def parse_version(s: str, prefix: str = "") -> Optional[tuple[int, ...]]:
    s = s.removeprefix(prefix)
    try:
        return tuple(int(x) for x in s.split("."))
    except ValueError:
        return None


def unparse_version(t: Iterable[int], prefix: str = "") -> str:
    return prefix + ".".join([str(x) for x in t])


def max_version(
    versions: Iterable[str],
    prefix: str = "",
    check: Callable[[str], bool] = lambda x: True,
) -> Optional[str]:
    versions = [x for x in versions if check(x)]
    versions = [parse_version(x, prefix) for x in versions]
    versions = [x for x in versions if x is not None]
    if not versions:
        return None
    version = unparse_version(max(versions), prefix)
    return version


def check_msvc(msvc_base_path: Path, version: str) -> bool:
    return all(
        x.exists()
        for x in [
            msvc_base_path / version / "bin" / "Hostx64" / "x64" / "cl.exe",
            msvc_base_path / version / "include" / "vcruntime.h",
            msvc_base_path / version / "lib" / "x64" / "vcruntime.lib",
        ]
    )


def find_msvc_env() -> tuple[Optional[Path], Optional[str]]:
    msvc_base_path = os.getenv("VCINSTALLDIR")
    if msvc_base_path is None:
        return None, None
    msvc_base_path = Path(msvc_base_path) / "Tools" / "MSVC"

    version = os.getenv("VCToolsVersion")
    if not check_msvc(msvc_base_path, version):
        warnings.warn(
            f"Environment variables VCINSTALLDIR = {os.getenv('VCINSTALLDIR')}, "
            f"VCToolsVersion = {os.getenv('VCToolsVersion')} are set, "
            "but this MSVC installation is incomplete."
        )
        return None, None

    return msvc_base_path, version


def find_msvc_vswhere() -> tuple[Optional[Path], Optional[str]]:
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


def find_msvc_envpath() -> tuple[Optional[Path], Optional[str]]:
    paths = os.getenv("PATH", "").split(os.pathsep)
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


def find_msvc_hardcoded() -> tuple[Optional[Path], Optional[str]]:
    vs_path = find_in_program_files("Microsoft Visual Studio")
    if vs_path is None:
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


def find_msvc(env_only: bool) -> tuple[Optional[str], list[str], list[str]]:
    if env_only:
        fs = [find_msvc_env]
    else:
        fs = [
            find_msvc_env,
            find_msvc_vswhere,
            find_msvc_envpath,
            find_msvc_hardcoded,
        ]
    for f in fs:
        msvc_base_path, version = f()
        if msvc_base_path:
            return (
                str(msvc_base_path / version / "bin" / "Hostx64" / "x64" / "cl.exe"),
                [str(msvc_base_path / version / "include")],
                [str(msvc_base_path / version / "lib" / "x64")],
            )

    if not env_only:
        warnings.warn("Failed to find MSVC.")
    return None, [], []


def check_winsdk(winsdk_base_path: Path, version: str) -> bool:
    return all(
        x.exists()
        for x in [
            winsdk_base_path / "Include" / version / "ucrt" / "stdlib.h",
            winsdk_base_path / "Lib" / version / "ucrt" / "x64" / "ucrt.lib",
        ]
    )


def find_winsdk_env() -> tuple[Optional[Path], Optional[str]]:
    winsdk_base_path = os.getenv("WindowsSdkDir")
    if winsdk_base_path is None:
        return None, None
    winsdk_base_path = Path(winsdk_base_path)

    version = os.getenv("WindowsSDKVersion")
    if version:
        version = version.rstrip("\\")
    if not check_winsdk(winsdk_base_path, version):
        warnings.warn(
            f"Environment variables WindowsSdkDir = {os.getenv('WindowsSdkDir')}, "
            f"WindowsSDKVersion = {os.getenv('WindowsSDKVersion')} are set, "
            "but this Windows SDK installation is incomplete."
        )
        return None, None

    return winsdk_base_path, version


def find_winsdk_registry() -> tuple[Optional[Path], Optional[str]]:
    try:
        reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        key = winreg.OpenKeyEx(
            reg, r"SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0"
        )
        folder = winreg.QueryValueEx(key, "InstallationFolder")[0]
        winreg.CloseKey(key)
    except OSError:
        return None, None

    winsdk_base_path = Path(folder)
    if not (winsdk_base_path / "Include").exists():
        return None, None

    version = max_version(
        os.listdir(winsdk_base_path / "Include"),
        check=partial(check_winsdk, winsdk_base_path),
    )
    if version is None:
        return None, None

    return winsdk_base_path, version


def find_winsdk_hardcoded() -> tuple[Optional[Path], Optional[str]]:
    winsdk_base_path = find_in_program_files(r"Windows Kits\10")
    if winsdk_base_path is None:
        return None, None
    if not (winsdk_base_path / "Include").exists():
        return None, None

    version = max_version(
        os.listdir(winsdk_base_path / "Include"),
        check=partial(check_winsdk, winsdk_base_path),
    )
    if version is None:
        return None, None

    return winsdk_base_path, version


def find_winsdk(env_only: bool) -> tuple[list[str], list[str]]:
    if env_only:
        fs = [find_winsdk_env]
    else:
        fs = [
            find_winsdk_env,
            find_winsdk_registry,
            find_winsdk_hardcoded,
        ]
    for f in fs:
        winsdk_base_path, version = f()
        if winsdk_base_path:
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

    if not env_only:
        warnings.warn("Failed to find Windows SDK.")
    return [], []


@functools.cache
def find_msvc_winsdk(
    env_only: bool = False,
) -> tuple[Optional[str], list[str], list[str]]:
    msvc_bin_path, msvc_inc_dirs, msvc_lib_dirs = find_msvc(env_only)
    winsdk_inc_dirs, winsdk_lib_dirs = find_winsdk(env_only)
    return (
        msvc_bin_path,
        msvc_inc_dirs + winsdk_inc_dirs,
        msvc_lib_dirs + winsdk_lib_dirs,
    )


@functools.cache
def find_python() -> list[str]:
    for python_base_path in [
        sys.exec_prefix,
        sys.base_exec_prefix,
        os.path.dirname(sys.executable),
    ]:
        python_lib_dir = Path(python_base_path) / "libs"
        if (python_lib_dir / "python3.lib").exists():
            return [str(python_lib_dir)]

    warnings.warn("Failed to find Python libs.")
    return []


def check_and_find_cuda(base_path: Path) -> tuple[Optional[str], list[str], list[str]]:
    # pip
    if all(
        x.exists()
        for x in [
            base_path / "cuda_nvcc" / "bin" / "ptxas.exe",
            base_path / "cuda_runtime" / "include" / "cuda.h",
            base_path / "cuda_runtime" / "lib" / "x64" / "cuda.lib",
        ]
    ):
        return (
            str(base_path / "cuda_nvcc" / "bin"),
            [str(base_path / "cuda_runtime" / "include")],
            [str(base_path / "cuda_runtime" / "lib" / "x64")],
        )

    # conda
    if all(
        x.exists()
        for x in [
            base_path / "bin" / "ptxas.exe",
            base_path / "include" / "cuda.h",
            base_path / "lib" / "cuda.lib",
        ]
    ):
        return (
            str(base_path / "bin"),
            [str(base_path / "include")],
            [str(base_path / "lib")],
        )

    # bundled or system-wide
    if all(
        x.exists()
        for x in [
            base_path / "bin" / "ptxas.exe",
            base_path / "include" / "cuda.h",
            base_path / "lib" / "x64" / "cuda.lib",
        ]
    ):
        return (
            str(base_path / "bin"),
            [str(base_path / "include")],
            [str(base_path / "lib" / "x64")],
        )

    return None, [], []


def find_cuda_env() -> tuple[Optional[str], list[str], list[str]]:
    for cuda_base_path in ["CUDA_PATH", "CUDA_HOME"]:
        cuda_base_path = os.getenv(cuda_base_path)
        if cuda_base_path is None:
            continue

        cuda_base_path = Path(cuda_base_path)
        cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs = check_and_find_cuda(
            cuda_base_path
        )
        if cuda_bin_path:
            return cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs

    return None, [], []


def find_cuda_bundled() -> tuple[Optional[str], list[str], list[str]]:
    cuda_base_path = (
        Path(sysconfig.get_paths()["platlib"]) / "triton" / "backends" / "nvidia"
    )
    return check_and_find_cuda(cuda_base_path)


def find_cuda_pip() -> tuple[Optional[str], list[str], list[str]]:
    nvidia_base_path = Path(sysconfig.get_paths()["platlib"]) / "nvidia"
    return check_and_find_cuda(nvidia_base_path)


def find_cuda_conda() -> tuple[Optional[str], list[str], list[str]]:
    cuda_base_path = Path(sys.exec_prefix) / "Library"
    return check_and_find_cuda(cuda_base_path)


def find_cuda_hardcoded() -> tuple[Optional[str], list[str], list[str]]:
    parent = find_in_program_files(r"NVIDIA GPU Computing Toolkit\CUDA")
    if parent is None:
        return None, [], []

    paths = glob(str(parent / "v12*"))
    # First try the highest version
    paths = sorted(paths)[::-1]
    for cuda_base_path in paths:
        cuda_base_path = Path(cuda_base_path)
        cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs = check_and_find_cuda(
            cuda_base_path
        )
        if cuda_bin_path:
            return cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs

    return None, [], []


@functools.cache
def find_cuda() -> tuple[Optional[str], list[str], list[str]]:
    for f in [
        find_cuda_env,
        find_cuda_bundled,
        find_cuda_pip,
        find_cuda_conda,
        find_cuda_hardcoded,
    ]:
        cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs = f()
        if cuda_bin_path:
            return cuda_bin_path, cuda_inc_dirs, cuda_lib_dirs

    warnings.warn("Failed to find CUDA.")
    return None, [], []
