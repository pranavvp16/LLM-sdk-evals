"""Settings helpers and runtime OSS model registration.

Each Settings(...) call below explicitly overrides any keys that pydantic-
settings would otherwise load from a local ``.env``, so the tests behave the
same whether the developer's ``.env`` has OPENCODE_API_KEY set or not.
"""

from __future__ import annotations

import pytest

from sdk.registry import get_model

from services.api.config import Settings
from services.api.oss_registry import register_oss_models

register_oss_models()


# Common shape: zero out anything the .env might fill in.
_BLANK = {
    "anthropic_api_key": "",
    "huggingface_api_key": "",
    "opencode_api_key": "",
    "openai_api_key": "",
    "google_api_key": "",
    "vllm_base_url": "",
    "ollama_base_url": "",
    "oss_provider": "",
    "oss_model": "",
}


def _settings(**overrides) -> Settings:
    return Settings(**{**_BLANK, **overrides})


def test_resolve_oss_defaults_to_huggingface():
    s = _settings(huggingface_api_key="hf_test")
    assert s.resolve_oss() == ("huggingface", "qwen2.5-0.5b-instruct")


def test_resolve_oss_defaults_to_opencode_go_when_key_present():
    s = _settings(opencode_api_key="sk-test", huggingface_api_key="hf_test")
    assert s.resolve_oss() == ("opencode-go", "deepseek-v4-flash")


def test_resolve_oss_vllm_without_url_raises():
    s = _settings(oss_provider="vllm", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="VLLM_BASE_URL"):
        s.resolve_oss()


def test_resolve_oss_vllm_with_url_ok():
    s = _settings(
        oss_provider="vllm",
        vllm_base_url="http://localhost:8001/v1",
        huggingface_api_key="hf",
    )
    assert s.resolve_oss() == ("vllm", "qwen2.5-1.5b-instruct")


def test_resolve_oss_opencode_without_key_raises():
    s = _settings(oss_provider="opencode", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        s.resolve_oss()


def test_resolve_oss_unknown_provider_raises():
    s = _settings(oss_provider="bad", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="Invalid OSS_PROVIDER"):
        s.resolve_oss()


def test_resolve_oss_no_keys_raises():
    with pytest.raises(ValueError, match="No OSS backend configured"):
        _settings().resolve_oss()


def test_base_urls_opencode_only_with_key():
    assert _settings().base_urls() == {}
    s = _settings(opencode_api_key="sk-test")
    urls = s.base_urls()
    assert urls["opencode"] == "https://opencode.ai/zen/v1"
    assert urls["opencode-go"] == "https://opencode.ai/zen/go/v1"


def test_oss_registry_vllm():
    m = get_model("vllm", "qwen2.5-0.5b-instruct")
    assert m.provider == "vllm"
    assert m.api.value == "openai-completions"
    assert m.supports_tools is False


def test_oss_registry_opencode_zen():
    m = get_model("opencode", "kimi-k2.5")
    assert m.provider == "opencode"
    assert m.api.value == "openai-completions"


def test_oss_registry_opencode_go_glm5():
    m = get_model("opencode-go", "glm-5")
    assert m.provider == "opencode-go"
    assert m.supports_tools is True


def test_oss_registry_opencode_go_deepseek():
    m = get_model("opencode-go", "deepseek-v4-flash")
    assert m.provider == "opencode-go"
    assert m.supports_tools is True
    assert m.supports_reasoning is True


def test_resolve_oss_ollama_without_url_raises():
    s = _settings(oss_provider="ollama", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        s.resolve_oss()


def test_resolve_oss_ollama_with_url_ok():
    s = _settings(
        oss_provider="ollama",
        ollama_base_url="http://ollama:11434/v1",
    )
    assert s.resolve_oss() == ("ollama", "qwen2.5:1.5b")


def test_base_urls_includes_ollama_when_set():
    s = _settings(ollama_base_url="http://ollama:11434/v1")
    urls = s.base_urls()
    assert urls["ollama"] == "http://ollama:11434/v1"


def test_oss_registry_ollama_qwen():
    m = get_model("ollama", "qwen2.5:1.5b")
    assert m.provider == "ollama"
    assert m.supports_tools is True
