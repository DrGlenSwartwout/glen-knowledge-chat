"""Gmail client-email reader for the testimonial-invite scan."""
from dashboard import gmail_feedback as gf


class _Exec:
    def __init__(self, val): self._v = val
    def execute(self): return self._v


class _FakeGmail:
    """Mimics service.users().messages().list(...).execute() / .get(...).execute()."""
    def __init__(self, msgs): self._msgs = msgs
    def users(self): return self
    def messages(self): return self
    def list(self, **k): return _Exec({"messages": [{"id": m["id"]} for m in self._msgs]})
    def get(self, *, id, **k):
        return _Exec(next(m for m in self._msgs if m["id"] == id))


def _msg(mid, frm, subject, snippet):
    return {"id": mid, "snippet": snippet,
            "payload": {"headers": [{"name": "From", "value": frm},
                                    {"name": "Subject", "value": subject}]}}


def test_filters_to_known_clients_only():
    svc = _FakeGmail([
        _msg("1", "Happy Client <happy@x.com>", "Update", "thank you, I feel so much better"),
        _msg("2", "spam@vendor.com", "SALE", "buy now"),
        _msg("3", "stranger@nope.com", "hi", "random"),
    ])
    out = gf.recent_client_messages({"happy@x.com", "other@x.com"}, service=svc)
    assert set(out.keys()) == {"happy@x.com"}
    assert "better" in out["happy@x.com"] and "Update" in out["happy@x.com"]


def test_empty_known_or_no_messages():
    assert gf.recent_client_messages(set(), service=_FakeGmail([_msg("1", "a@x.com", "s", "x")])) == {}
    assert gf.recent_client_messages({"a@x.com"}, service=_FakeGmail([])) == {}


def test_dedups_multiple_emails_from_same_client():
    svc = _FakeGmail([
        _msg("1", "a@x.com", "First", "doing great"),
        _msg("2", "A@X.com", "Second", "even better now"),
    ])
    out = gf.recent_client_messages({"a@x.com"}, service=svc)
    assert "doing great" in out["a@x.com"] and "even better now" in out["a@x.com"]
