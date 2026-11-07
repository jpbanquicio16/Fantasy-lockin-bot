import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
