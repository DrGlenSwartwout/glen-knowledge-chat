import sqlite3
import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_record_and_list_request():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "low-energy", "A@Example.com")
    tp.record_request(cx, "low-energy", "")  # ignored
    rows = tp.requesters_to_email(cx, "low-energy")
    assert [r["email"] for r in rows] == ["a@example.com"]


def test_notify_links_to_learn_and_marks_once():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "low-energy", "a@example.com")
    sent = []
    tp.notify_on_approve(cx, "low-energy", "Low Energy", "https://x.test",
                         send=lambda to, subj, body: sent.append((to, subj, body)))
    assert len(sent) == 1
    assert "https://x.test/learn/low-energy" in sent[0][2]
    # second call sends nothing (already emailed)
    tp.notify_on_approve(cx, "low-energy", "Low Energy", "https://x.test",
                         send=lambda to, subj, body: sent.append((to, subj, body)))
    assert len(sent) == 1


def test_notify_one_bad_send_does_not_stop_others():
    tp = _mod()
    cx = _cx(tp)
    tp.record_request(cx, "detox", "bad@example.com")
    tp.record_request(cx, "detox", "good@example.com")
    ok = []

    def _send(to, subj, body):
        if to == "bad@example.com":
            raise RuntimeError("smtp down")
        ok.append(to)

    tp.notify_on_approve(cx, "detox", "Detox", "https://x.test", send=_send)
    assert ok == ["good@example.com"]
    # bad one is still un-emailed and retryable
    assert [r["email"] for r in tp.requesters_to_email(cx, "detox")] == ["bad@example.com"]
