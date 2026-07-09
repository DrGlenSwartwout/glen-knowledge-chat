import io, json
import dashboard.biofield_invoice as bi


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _fake_urlopen(payload):
    def _open(req, timeout=0):
        return _Resp(json.dumps(payload).encode())
    return _open


def _env(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://illtowell.com")


def test_fetch_catalog_parses_products(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"products": [{"slug": "vitality", "name": "Vitality"}]}))
    assert bi.default_fetch_catalog() == [{"slug": "vitality", "name": "Vitality"}]


def test_create_order_ok(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": True, "order_id": 42, "external_ref": "INH-AB",
                                       "totals": {"total_cents": 12345}}))
    out = bi.default_create_order({"name": "D", "email": "d@x.com"}, [{"slug": "biofield-analysis", "qty": 1}])
    assert out["ok"] and out["order_id"] == 42 and out["external_ref"] == "INH-AB"
    assert out["total_cents"] == 12345


def _capturing_urlopen(payload, sink):
    def _open(req, timeout=0):
        sink.append(json.loads(req.data.decode()))
        return _Resp(json.dumps(payload).encode())
    return _open


def test_create_order_does_not_force_pickup(monkeypatch):
    """A Biofield hand-off is NOT a pickup. It used to post pickup=True, which
    zeroed shipping on the physical remedy bottles riding on the same invoice.
    Shipping is now computed normally; the analysis fee contributes no bottle
    because it is a service (see dashboard.shipping.is_shippable), so an
    analysis-only invoice still ships for $0 without the flag."""
    _env(monkeypatch)
    sent = []
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _capturing_urlopen({"ok": True, "order_id": 1, "external_ref": "INH-A",
                                            "totals": {}}, sent))
    bi.default_create_order({"name": "D", "email": "d@x.com"},
                            [{"slug": "biofield-analysis", "qty": 1}, {"slug": "vitality", "qty": 2}])
    assert not sent[0].get("pickup"), f"hand-off must not force pickup: {sent[0]!r}"


def test_create_order_still_sends_lines_and_replace_open(monkeypatch):
    """Guard the rest of the hand-off contract while removing the pickup key."""
    _env(monkeypatch)
    sent = []
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _capturing_urlopen({"ok": True, "order_id": 1, "external_ref": "INH-A",
                                            "totals": {}}, sent))
    bi.default_create_order({"name": "D", "email": "d@x.com"},
                            [{"slug": "biofield-analysis", "qty": 1}], replace_open=True)
    body = sent[0]
    assert body["replace_open"] is True
    assert body["lines"] == [{"slug": "biofield-analysis", "qty": 1}]
    assert "Payable by check" in body["invoice_note"]


def test_create_order_server_not_ok_is_explicit(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": False, "error": "no valid products"}))
    out = bi.default_create_order({"email": "d@x.com"}, [])
    assert out["ok"] is False and out["error"] == "no valid products"


def test_create_order_no_console_is_explicit(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    out = bi.default_create_order({"email": "d@x.com"}, [{"slug": "biofield-analysis", "qty": 1}])
    assert out["ok"] is False and out["error"]


def test_invoice_link_ok(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(bi.urllib.request, "urlopen",
                        _fake_urlopen({"ok": True, "link": "https://illtowell.com/invoice/tok?print=1"}))
    out = bi.default_invoice_link(42)
    assert out["ok"] and out["print_url"].endswith("print=1")
