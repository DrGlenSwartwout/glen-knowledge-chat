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
