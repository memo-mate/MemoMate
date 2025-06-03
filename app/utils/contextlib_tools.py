import contextlib
import os


@contextlib.contextmanager
def temporary_no_proxy():
    """Temporarily set NO_PROXY environment variable and restore after use"""
    original_no_proxy = os.environ.get("NO_PROXY", "")
    try:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        yield
    finally:
        if original_no_proxy:
            os.environ["NO_PROXY"] = original_no_proxy
        else:
            os.environ.pop("NO_PROXY", None)
