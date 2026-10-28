import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def spark():
    from tsrecon.spark_session import get_spark

    s = get_spark("local[2]", "tsrecon-tests")
    yield s
    s.stop()
