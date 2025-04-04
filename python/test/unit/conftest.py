import os
import pytest
import tempfile


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
    except OSError:
        # On Windows, the compiled binary may not be deleted when tmpdir cleans up,
        # because it's still loaded by the Python process
        pass
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
