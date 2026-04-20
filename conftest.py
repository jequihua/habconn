"""Root conftest: ensures tests import habconn from THIS checkout.

Problem: if another habconn checkout on the same machine has run
``pip install -e .`` more recently, the editable-install .pth file
in site-packages will point to the OTHER checkout. Tests would then
silently import the wrong code.

Fix: before any test collection, insert this checkout's ``src/``
directory at the front of ``sys.path`` so it takes priority over
any .pth-based path. Then verify the import resolved correctly.

This is NOT a PYTHONPATH hack exposed to the user — it is a repo-
local test-infrastructure guard that makes the test suite self-
contained regardless of ambient editable-install state.
"""

import sys
from pathlib import Path

_THIS_SRC = str(Path(__file__).resolve().parent / "src")
_EXPECTED_PKG = Path(__file__).resolve().parent / "src" / "habconn"


def pytest_configure(config):
    """Ensure this checkout's src/ is importable, then verify."""
    # 1. Put this checkout's src/ first on sys.path
    if _THIS_SRC not in sys.path:
        sys.path.insert(0, _THIS_SRC)

    # 2. If habconn was already imported from elsewhere (e.g. by a
    #    pytest plugin or early collection), force a re-resolve.
    if "habconn" in sys.modules:
        del sys.modules["habconn"]
        # Also remove sub-modules so they reimport cleanly
        to_remove = [k for k in sys.modules if k.startswith("habconn.")]
        for k in to_remove:
            del sys.modules[k]

    # 3. Import and verify
    import habconn

    actual = Path(habconn.__file__).resolve().parent
    if actual != _EXPECTED_PKG:
        raise RuntimeError(
            f"habconn imported from wrong location even after sys.path fix.\n"
            f"  Expected: {_EXPECTED_PKG}\n"
            f"  Actual:   {actual}\n"
            f"  sys.path[0]: {sys.path[0]}\n"
            f"This should not happen. Check for namespace-package conflicts."
        )
