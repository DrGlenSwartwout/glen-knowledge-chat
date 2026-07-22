"""Console-action tests for the free product review feature (no app import)."""
import sqlite3

import pytest

from dashboard import supplement_reviews as sr
from dashboard import supplement_reviews_actions as sra
from dashboard.actions import get_action


def _cx():
    cx = sqlite3.connect(":memory:")
    sr.init_table(cx)
    return cx


def test_register_idempotent():
    sra.register()
    sra.register()  # must not raise (guarded by get_action)
    assert get_action("product_review.confirm") is not None
    assert get_action("product_review.reject") is not None


def test_confirm_executor_promotes():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "P", "B")
    sr.set_draft(cx, r["id"], "review body")
    out = sra._exec_confirm({"id": r["id"]}, {"cx": cx, "actor": {"name": "Glen"}})
    assert out == {"id": r["id"], "status": "confirmed"}
    assert sr.get(cx, r["id"])["status"] == "confirmed"


def test_reject_executor():
    cx = _cx()
    r = sr.create_request(cx, "a@x.com", "P", "B")
    out = sra._exec_reject({"id": r["id"]}, {"cx": cx})
    assert out["status"] == "rejected"


def test_executor_requires_id():
    cx = _cx()
    with pytest.raises(ValueError):
        sra._exec_confirm({}, {"cx": cx})
