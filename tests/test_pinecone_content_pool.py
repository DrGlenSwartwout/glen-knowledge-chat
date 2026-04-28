"""Tests for the Pinecone content pool — audience-filtered candidate
topic + source-text fetch for the incentive engine."""

import pytest
from unittest.mock import MagicMock


def _build_fake_pinecone(entries):
    """Build a mock Pinecone index. entries is a list of dicts with
    id, audience, topics, text."""
    svc = MagicMock()
    svc.list.return_value = MagicMock(
        ids=[e["id"] for e in entries],
        pagination={},
    )
    fake_vectors = {
        e["id"]: MagicMock(metadata={
            "topics": e.get("topics", []),
            "text": e.get("text", ""),
            "audience": e.get("audience", "both"),
            "question": e.get("question", "What about it"),
        })
        for e in entries
    }
    svc.fetch.return_value = MagicMock(vectors=fake_vectors)
    return svc


def test_fetch_pool_filters_practitioner_for_client(monkeypatch):
    """Client-audience query should exclude practitioner-only entries."""
    from pinecone_content_pool import fetch_pool_for_audience
    monkeypatch.setattr("pinecone_content_pool._list_clinical_qa_ids",
                        lambda: ("qa-1", "qa-2", "qa-3"))
    fake = _build_fake_pinecone([
        {"id": "qa-1", "audience": "client",       "topics": ["leaky-gut"], "text": "client text"},
        {"id": "qa-2", "audience": "practitioner", "topics": ["mechanism"], "text": "depth text"},
        {"id": "qa-3", "audience": "both",         "topics": ["EMF"],       "text": "both text"},
    ])
    monkeypatch.setattr("pinecone_content_pool._get_pinecone_index",
                        lambda: fake)
    pool = fetch_pool_for_audience("client")
    audiences = {e["audience"] for e in pool}
    assert "practitioner" not in audiences
    assert "client" in audiences
    assert "both" in audiences


def test_fetch_pool_filters_client_for_practitioner(monkeypatch):
    from pinecone_content_pool import fetch_pool_for_audience
    monkeypatch.setattr("pinecone_content_pool._list_clinical_qa_ids",
                        lambda: ("qa-1", "qa-2"))
    fake = _build_fake_pinecone([
        {"id": "qa-1", "audience": "client", "topics": ["t1"], "text": "x"},
        {"id": "qa-2", "audience": "practitioner", "topics": ["t2"], "text": "y"},
    ])
    monkeypatch.setattr("pinecone_content_pool._get_pinecone_index",
                        lambda: fake)
    pool = fetch_pool_for_audience("practitioner")
    audiences = {e["audience"] for e in pool}
    assert "client" not in audiences


def test_candidate_topics_returns_unique_primary_topics(monkeypatch):
    from pinecone_content_pool import candidate_topics_for_audience
    monkeypatch.setattr("pinecone_content_pool._list_clinical_qa_ids",
                        lambda: ("qa-1", "qa-2", "qa-3"))
    fake = _build_fake_pinecone([
        {"id": "qa-1", "audience": "both", "topics": ["leaky-gut"], "text": "a"},
        {"id": "qa-2", "audience": "both", "topics": ["leaky-gut", "fiber"], "text": "b"},
        {"id": "qa-3", "audience": "both", "topics": ["EMF"], "text": "c"},
    ])
    monkeypatch.setattr("pinecone_content_pool._get_pinecone_index",
                        lambda: fake)
    topics = candidate_topics_for_audience("both")
    assert "leaky-gut" in topics
    assert "EMF" in topics
    # Should be deduped
    assert len(topics) == 2


def test_fetch_source_text_returns_longest_match(monkeypatch):
    from pinecone_content_pool import fetch_source_text_for_topic
    monkeypatch.setattr("pinecone_content_pool._list_clinical_qa_ids",
                        lambda: ("qa-1", "qa-2"))
    fake = _build_fake_pinecone([
        {"id": "qa-1", "audience": "both", "topics": ["leaky-gut"], "text": "short"},
        {"id": "qa-2", "audience": "both", "topics": ["leaky-gut"], "text": "this is much longer"},
    ])
    monkeypatch.setattr("pinecone_content_pool._get_pinecone_index",
                        lambda: fake)
    text = fetch_source_text_for_topic("leaky-gut", "both")
    assert text == "this is much longer"


def test_fetch_source_text_returns_empty_for_no_match(monkeypatch):
    from pinecone_content_pool import fetch_source_text_for_topic
    monkeypatch.setattr("pinecone_content_pool._list_clinical_qa_ids",
                        lambda: ("qa-1",))
    fake = _build_fake_pinecone([
        {"id": "qa-1", "audience": "both", "topics": ["EMF"], "text": "x"},
    ])
    monkeypatch.setattr("pinecone_content_pool._get_pinecone_index",
                        lambda: fake)
    assert fetch_source_text_for_topic("nonexistent", "both") == ""
