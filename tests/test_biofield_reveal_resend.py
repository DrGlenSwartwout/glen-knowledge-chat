"""biofield_reveal.resend: re-send a client their reveal-ready link WITHOUT
approving/un-blurring, so they can open the (blurred) reveal and click
'request review'. Distinct from biofield_reveal.send, which is approval-gated."""
import sqlite3

from dashboard import biofield_reveal_actions as bra
from dashboard import biofield_reveals as br


def _cx_with_reveal(approved=False):
    cx = sqlite3.connect(":memory:")
    br.init_table(cx)
    rid, _ = br.upsert(cx, "c@x.com", "2026-07-18", {}, [], "t")
    if approved:
        br.approve_first(cx, rid, "test")
    return cx, rid


def test_resend_sends_link_even_when_not_approved():
    cx, rid = _cx_with_reveal(approved=False)
    sent = []
    bra.configure(send_reveal_link=lambda r: (sent.append(r) or True))
    res = bra._exec_resend({"id": rid}, {"cx": cx, "actor": None})
    assert res == {"sent": True}
    assert sent == [rid]          # sent despite first_approved == 0


def test_resend_reports_missing_reveal():
    cx, _ = _cx_with_reveal()
    bra.configure(send_reveal_link=lambda r: True)
    assert bra._exec_resend({"id": 99999}, {"cx": cx, "actor": None}) == {
        "sent": False, "reason": "not_found"}
