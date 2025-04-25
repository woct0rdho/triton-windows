from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sysconfig
import tempfile

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs

if os.name == "nt":
    from triton.windows_utils import find_msvc_winsdk, find_python


@functools.lru_cache
def get_cc():
    cc = os.environ.get("CC")
    if cc is None:
        # Find and check MSVC and Windows SDK from environment variables set by Launch-VsDevShell.ps1 or VsDevCmd.bat
        cc, _, _ = find_msvc_winsdk(env_only=True)
    if cc is None:
        # Bundled TinyCC
        cc = os.path.join(sysconfig.get_paths()["platlib"], "triton", "runtime", "tcc", "tcc.exe")
        if not os.path.exists(cc):
            cc = None
    if cc is None:
        cc = shutil.which("cl")
    if cc is None:
        cc = shutil.which("gcc")
    if cc is None:
        cc = shutil.which("clang")
    if cc is None:
        raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    return cc


def is_tcc(cc):
    cc = os.path.basename(cc).lower()
    return cc == "tcc" or cc == "tcc.exe"


def is_msvc(cc):
    cc = os.path.basename(cc).lower()
    return cc == "cl" or cc == "cl.exe"


def is_clang(cc):
    cc = os.path.basename(cc).lower()
    return cc == "clang" or cc == "clang.exe"


def _cc_cmd(cc: str, src: str, out: str, include_dirs: list[str], library_dirs: list[str], libraries: list[str],
            ccflags: list[str]) -> list[str]:
    if is_msvc(cc):
        out_base = os.path.splitext(out)[0]
        cc_cmd = [cc, src, "/nologo", "/O2", "/LD", "/wd4819"]
        cc_cmd += [f"/I{dir}" for dir in include_dirs if dir is not None]
        cc_cmd += [f"/Fo{out_base + '.obj'}"]
        cc_cmd += ["/link"]
        cc_cmd += [f"/LIBPATH:{dir}" for dir in library_dirs]
        cc_cmd += [f'{lib}.lib' for lib in libraries]
        cc_cmd += [f"/OUT:{out}"]
        cc_cmd += [f"/IMPLIB:{out_base + '.lib'}"]
        cc_cmd += [f"/PDB:{out_base + '.pdb'}"]
    else:
        # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
        cc_cmd = [cc, src, "-O3", "-shared", "-Wno-psabi", "-o", out]
        if not (os.name == "nt" and is_clang(cc)):
            # Clang does not support -fPIC on Windows
            cc_cmd += ["-fPIC"]
        if is_tcc(cc):
            cc_cmd += ["-D_Py_USE_GCC_BUILTIN_ATOMICS"]
        cc_cmd += [f'-l{lib}' for lib in libraries]
        cc_cmd += [f"-L{dir}" for dir in library_dirs]
        cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd += ccflags
    return cc_cmd


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str]) -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    cc = get_cc()
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    if os.name == "nt":
        library_dirs += find_python()
        version = sysconfig.get_python_version().replace(".", "")
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            version += "t"
        libraries += [f"python{version}"]
    if is_msvc(cc):
        _, msvc_winsdk_inc_dirs, msvc_winsdk_lib_dirs = find_msvc_winsdk()
        include_dirs += msvc_winsdk_inc_dirs
        library_dirs += msvc_winsdk_lib_dirs
    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries, ccflags)

    try:
        ret = subprocess.check_call(cc_cmd)
    except Exception as e:
        print("Failed to compile. cc_cmd:", cc_cmd)
        raise e

    return so


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _load_module_from_path(name: str, path: str) -> ModuleType:
    # Loading module with relative path may cause error
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None) -> ModuleType:
    key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")

    if cache_path is not None:
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    return _load_module_from_path(name, cache_path)
