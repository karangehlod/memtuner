"""Auto-mark everything under tests/unit as `unit`.

CI selects tests with `-m "unit or contract"`. Files here that forgot the
explicit marker were silently deselected — at one point 182 of 944 unit tests
never ran in CI. Location in this directory IS the marker; no per-file
annotation needed (explicit markers remain harmless duplicates).
"""

from pathlib import Path

import pytest

_UNIT_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(items):
    for item in items:
        if _UNIT_DIR in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.unit)
