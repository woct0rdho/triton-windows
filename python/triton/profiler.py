"""
Triton Profiler Module

This module provides a conditional import wrapper for the Proton profiler.
When TRITON_BUILD_PROTON is enabled, it imports the full Proton profiler.
When disabled, it provides stub implementations that gracefully handle missing functionality.
"""

import os
import warnings
from typing import Any, Optional, Union, Dict, List


# Check if Proton is actually available
_PROTON_AVAILABLE = False
_proton_impl = None
_viewer_impl = None

try:
    # This will work when Proton is built and symlinked
    # Check if the profiler directory exists and contains the expected modules
    import importlib.util
    profiler_dir = os.path.join(os.path.dirname(__file__), 'profiler')
    if os.path.isdir(profiler_dir):
        proton_spec = importlib.util.find_spec('triton.profiler.proton')
        viewer_spec = importlib.util.find_spec('triton.profiler.viewer')
        if proton_spec is not None and viewer_spec is not None:
            import triton.profiler.proton as _proton_impl
            import triton.profiler.viewer as _viewer_impl
            _PROTON_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    pass


class _ProtonStub:
    """Stub class for Proton profiler when not available."""
    
    def __init__(self):
        self._warned = False
    
    def _warn_once(self, method_name: str):
        """Warn once about missing Proton functionality."""
        if not self._warned:
            warnings.warn(
                f"Proton profiler is not available. The '{method_name}' call will be ignored. "
                "To enable Proton profiler, build Triton with -DTRITON_BUILD_PROTON=ON.",
                UserWarning,
                stacklevel=3
            )
            self._warned = True
    
    def start(self, name: str = "profile", hook: str = "triton", *args, **kwargs):
        """Stub for proton.start()"""
        self._warn_once("start")
        return None
    
    def finalize(self, *args, **kwargs):
        """Stub for proton.finalize()"""
        self._warn_once("finalize")
        return None
    
    def activate(self, *args, **kwargs):
        """Stub for proton.activate()"""
        self._warn_once("activate")
        return None
    
    def deactivate(self, *args, **kwargs):
        """Stub for proton.deactivate()"""
        self._warn_once("deactivate")
        return None
    
    def scope(self, name: str, *args, **kwargs):
        """Stub for proton.scope() context manager"""
        class _ScopeStub:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        
        self._warn_once("scope")
        return _ScopeStub()


class _ViewerStub:
    """Stub class for Proton viewer when not available."""
    
    def __init__(self):
        self._warned = False
    
    def _warn_once(self, method_name: str):
        """Warn once about missing Proton functionality."""
        if not self._warned:
            warnings.warn(
                f"Proton viewer is not available. The '{method_name}' call will be ignored. "
                "To enable Proton profiler, build Triton with -DTRITON_BUILD_PROTON=ON.",
                UserWarning,
                stacklevel=3
            )
            self._warned = True
    
    def parse(self, metric_names: List[str], file_name: str, *args, **kwargs):
        """Stub for viewer.parse()"""
        self._warn_once("parse")
        return None, None
    
    def print_tree(self, tree: Any, metrics: Any, *args, **kwargs):
        """Stub for viewer.print_tree()"""
        self._warn_once("print_tree")
        return None
    
    def read(self, file_path: str, *args, **kwargs):
        """Stub for viewer.read()"""
        self._warn_once("read")
        return None, None, None, None
    
    def main(self, *args, **kwargs):
        """Stub for viewer.main()"""
        self._warn_once("main")
        return None


# Export the appropriate implementation
if _PROTON_AVAILABLE:
    # Re-export all Proton functionality
    from triton.profiler.proton import *
    
    # Import viewer as a submodule
    import triton.profiler.viewer as viewer
    
else:
    # Use stub implementations
    _proton_stub = _ProtonStub()
    _viewer_stub = _ViewerStub()
    
    # Export stub functions at module level
    start = _proton_stub.start
    finalize = _proton_stub.finalize
    activate = _proton_stub.activate
    deactivate = _proton_stub.deactivate
    scope = _proton_stub.scope
    
    # Create viewer submodule
    class ViewerModule:
        parse = _viewer_stub.parse
        print_tree = _viewer_stub.print_tree
        read = _viewer_stub.read
        main = _viewer_stub.main
    
    viewer = ViewerModule()


# Export availability flag
PROTON_AVAILABLE = _PROTON_AVAILABLE


def is_available() -> bool:
    """Check if Proton profiler is available."""
    return _PROTON_AVAILABLE