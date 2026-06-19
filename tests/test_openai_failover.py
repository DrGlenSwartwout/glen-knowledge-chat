"""Tests for OpenAI embeddings multi-key failover.

OpenAI is the only embeddings provider whose vectors match the existing Pinecone
index, so resilience to an OPENAI_API_KEY quota/auth failure = a SECOND OpenAI key
for the SAME model (OPENAI_API_KEY_FALLBACK). With no fallback configured the wrapper
behaves exactly like a single client.
"""
import pytest
from dashboard.openai_failover import (
    _EmbeddingFailover, OpenAIWithEmbedFailover, build_openai_client)


class _Emb:
    def __init__(self, result=None, boom=False):
        self.result, self.boom, self.calls = result, boom, 0

    def create(self, **kw):
        self.calls += 1
        if self.boom:
            raise RuntimeError("429 insufficient_quota")
        return self.result


class _Client:
    def __init__(self, result=None, boom=False):
        self.embeddings = _Emb(result, boom)


def test_uses_primary_when_healthy():
    a, b = _Client(result="A"), _Client(result="B")
    fo = _EmbeddingFailover([a, b])
    assert fo.create(input=["x"], model="m") == "A"
    assert a.embeddings.calls == 1 and b.embeddings.calls == 0


def test_fails_over_to_secondary_on_error():
    a, b = _Client(boom=True), _Client(result="B")
    fo = _EmbeddingFailover([a, b])
    assert fo.create(input=["x"], model="m") == "B"
    assert a.embeddings.calls == 1 and b.embeddings.calls == 1


def test_raises_last_error_when_all_fail():
    fo = _EmbeddingFailover([_Client(boom=True), _Client(boom=True)])
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        fo.create(input=["x"], model="m")


def test_raises_when_no_clients():
    with pytest.raises(RuntimeError):
        _EmbeddingFailover([]).create(input=["x"], model="m")


def test_wrapper_delegates_unknown_attributes():
    class _C:
        def __init__(self):
            self.embeddings = _Emb(result="A")
            self.foo = "bar"
    w = OpenAIWithEmbedFailover([_C()])
    assert w.foo == "bar"                                   # delegated to primary
    assert w.embeddings.create(input=["x"], model="m") == "A"


def test_build_single_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.delenv("OPENAI_API_KEY_FALLBACK", raising=False)
    assert len(build_openai_client()._clients) == 1


def test_build_with_fallback_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("OPENAI_API_KEY_FALLBACK", "k2")
    assert len(build_openai_client()._clients) == 2


def test_build_no_keys_surfaces_loudly_like_bare_sdk(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FALLBACK", raising=False)
    from openai import OpenAIError
    # No masking: a missing key raises at startup exactly as the bare SDK would.
    with pytest.raises(OpenAIError):
        build_openai_client()
