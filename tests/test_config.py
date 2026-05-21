"""Settings helpers for OSS provider selection (eval CLI)."""

from __future__ import annotations

import pytest

from services.api.config import Settings


def test_resolve_oss_defaults_to_huggingface():
    s = Settings(huggingface_api_key="hf_test")
    assert s.resolve_oss() == ("huggingface", "qwen2.5-0.5b-instruct")


def test_resolve_oss_vllm_without_url_raises():
    s = Settings(oss_provider="vllm", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="VLLM_BASE_URL"):
        s.resolve_oss()


def test_resolve_oss_vllm_with_url_ok():
    s = Settings(
        oss_provider="vllm",
        vllm_base_url="http://localhost:8001/v1",
        huggingface_api_key="hf",
    )
    assert s.resolve_oss() == ("vllm", "qwen2.5-0.5b-instruct")


def test_resolve_oss_opencode_without_key_raises():
    s = Settings(oss_provider="opencode", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        s.resolve_oss()


def test_resolve_oss_unknown_provider_raises():
    s = Settings(oss_provider="bad", huggingface_api_key="hf")
    with pytest.raises(ValueError, match="Invalid OSS_PROVIDER"):
        s.resolve_oss()


def test_resolve_oss_default_without_hf_key_raises():
    with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY"):
        Settings().resolve_oss()


def test_base_urls_opencode_only_with_key():
    assert Settings().base_urls() == {}
    s = Settings(opencode_api_key="sk-test")
    urls = s.base_urls()
    assert urls["opencode"] == "https://opencode.ai/zen/v1"
    assert urls["opencode-go"] == "https://opencode.ai/zen/go/v1"
