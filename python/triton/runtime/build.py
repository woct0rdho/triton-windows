import contextlib
import functools
import sys
import io
import sysconfig
import os
import shutil
import subprocess
import setuptools

if os.name == "nt":
    from triton.windows_utils import find_msvc_winsdk, find_python


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


@functools.cache
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


def is_msvc(cc):
    cc = os.path.basename(cc).lower()
    return cc == "cl" or cc == "cl.exe"


def _cc_cmd(cc, src, out, include_dirs, library_dirs, libraries):
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
        cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", out]
        cc_cmd += [f'-l{lib}' for lib in libraries]
        cc_cmd += [f"-L{dir}" for dir in library_dirs]
        cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    return cc_cmd


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = get_cc()
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    if os.name == "nt":
        library_dirs += find_python()
    # Link against Python stable ABI
    # libraries is modified in place
    if "python3" not in libraries:
        libraries += ["python3"]
    if is_msvc(cc):
        _, msvc_winsdk_inc_dirs, msvc_winsdk_lib_dirs = find_msvc_winsdk()
        include_dirs += msvc_winsdk_inc_dirs
        library_dirs += msvc_winsdk_lib_dirs
    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)

    try:
        ret = subprocess.check_call(cc_cmd)
        if ret == 0:
            return so
    except Exception as e:
        print("Failed to compile. cc_cmd:", cc_cmd)
        raise e

    # fallback on setuptools
    extra_compile_args = []
    if is_msvc(cc):
        extra_compile_args += ["/O2"]
    else:
        extra_compile_args += ["-O3"]
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so
