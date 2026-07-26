import sqlite3

import pytest

from dashboard import eye_vision_suggestion_review as review


def test_approval_requires_wording_and_safety_review():
    cx = sqlite3.connect(":memory:")
    with pytest.raises(ValueError):
        review.save(cx, "client@example.com", "abc", "approved", "p1")


def test_review_is_persisted_and_overlaid():
    cx = sqlite3.connect(":memory:")
    stored = review.save(
        cx, "client@example.com", "abc", "approved", "p1",
        "Client-safe wording", "Interactions reviewed")
    assert stored["decision"] == "approved"
    contract = {"candidates": [{
        "suggestion_id": "abc", "status": "pending_practitioner_approval"}]}
    review.overlay(contract, review.decisions(cx, "client@example.com"))
    assert contract["candidates"][0]["status"] == "approved"
    assert contract["candidates"][0]["review"]["reviewer_id"] == "p1"
