import sqlite3
from dashboard import sales_page_viewers as spv


def _cx():
    return sqlite3.connect(":memory:")


def test_record_idempotent_and_to_email():
    cx = _cx()
    spv.record_viewer(cx, "longevity", "A@x.com", "Ann")
    spv.record_viewer(cx, "longevity", "a@x.com", "Ann2")   # same (lowercased) -> ignored
    rows = spv.viewers_to_email(cx, "longevity")
    assert len(rows) == 1 and rows[0]["email"] == "a@x.com" and rows[0]["name"] == "Ann"
    spv.record_viewer(cx, "longevity", "b@x.com")            # no name
    assert {r["email"] for r in spv.viewers_to_email(cx, "longevity")} == {"a@x.com", "b@x.com"}


def test_mark_emailed_excludes():
    cx = _cx()
    spv.record_viewer(cx, "x", "a@x.com", "Ann")
    spv.record_viewer(cx, "x", "b@x.com", "Bob")
    spv.mark_emailed(cx, "x", ["a@x.com"])
    assert [r["email"] for r in spv.viewers_to_email(cx, "x")] == ["b@x.com"]


def test_notify_on_approve_sends_once_and_marks():
    cx = _cx()
    spv.record_viewer(cx, "x", "a@x.com", "Ann")
    spv.record_viewer(cx, "x", "b@x.com", "")
    sent = []
    def fake_send(to, subject, body, from_name=None):
        sent.append({"to": to, "subject": subject, "body": body, "from_name": from_name})
    n = spv.notify_on_approve(cx, "x", "Longevity", "https://illtowell.com",
                              send=fake_send, strip=lambda s: s.replace("—", ","))
    assert n == 2 and len(sent) == 2
    by = {s["to"]: s for s in sent}
    assert by["a@x.com"]["body"].startswith("Aloha Ann,")
    assert by["b@x.com"]["body"].startswith("Aloha,")           # no name
    assert "Your Longevity page is ready, reviewed by Dr. Glen" == by["a@x.com"]["subject"]
    assert "https://illtowell.com/begin/product/x" in by["a@x.com"]["body"]
    assert by["a@x.com"]["from_name"] == "Dr. Glen Swartwout"
    assert "—" not in by["a@x.com"]["body"]                # em dash stripped
    # idempotent: a re-run sends nobody
    assert spv.notify_on_approve(cx, "x", "Longevity", "https://illtowell.com", send=fake_send) == 0
    assert len(sent) == 2


def test_notify_per_recipient_failure_isolated():
    cx = _cx()
    spv.record_viewer(cx, "x", "good@x.com", "G")
    spv.record_viewer(cx, "x", "bad@x.com", "B")
    def flaky_send(to, subject, body, from_name=None):
        if to == "bad@x.com":
            raise RuntimeError("smtp down")
    n = spv.notify_on_approve(cx, "x", "P", "https://b", send=flaky_send)
    assert n == 1                                              # only good@x.com emailed
    assert [r["email"] for r in spv.viewers_to_email(cx, "x")] == ["bad@x.com"]   # bad left for retry
