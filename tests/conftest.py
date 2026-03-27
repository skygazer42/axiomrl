from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.support.markers import classify_test_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        test_path = item.nodeid.split("::", maxsplit=1)[0]
        for marker in sorted(classify_test_path(test_path)):
            item.add_marker(marker)
