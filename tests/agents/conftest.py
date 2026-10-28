import pytest


@pytest.fixture(autouse=True)
def block_openrouter(monkeypatch):
    """Pytest never calls OpenRouter. Tests inject a scripted completion."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("tests must not call LiteLLM/OpenRouter")

    monkeypatch.setattr("fdagym.agents.litellm_agent._completion", boom)
    monkeypatch.setattr("litellm.completion", boom, raising=False)
