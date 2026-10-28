import pytest

from fdagym.tasks.config import GenConfig
from fdagym.tasks.generators import generate
from tests.tasks.fake_snapshot import FIXTURE_DRUGS, load


@pytest.fixture(scope="session")
def snapshot():
    return load()


@pytest.fixture(scope="session")
def tasks(snapshot):
    return generate(snapshot, GenConfig(), drugs=FIXTURE_DRUGS)
