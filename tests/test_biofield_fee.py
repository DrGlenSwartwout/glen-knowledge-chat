import json as _json
import pytest
from dashboard import biofield_fee as bf


def test_constants():
    assert bf.BIOFIELD_SLUG == "biofield-analysis"
    assert bf.STANDARD_CENTS == 30000 and bf.VALUE_CENTS == 99700


@pytest.mark.parametrize("v,cents", [("300", 30000), (100, 10000), (0, 0), ("697.50", 69750), (97.0, 9700)])
def test_dollars_to_cents_ok(v, cents):
    assert bf.dollars_to_cents(v) == cents


@pytest.mark.parametrize("bad", [-1, "-5", "abc", "", None])
def test_dollars_to_cents_rejects(bad):
    with pytest.raises(ValueError):
        bf.dollars_to_cents(bad)


@pytest.mark.parametrize("cents,s", [(30000, "300"), (0, "0"), (69750, "697.50"), (9700, "97")])
def test_cents_to_dollars(cents, s):
    assert bf.cents_to_dollars(cents) == s


def test_parse_courtesy_found():
    resp = {"ok": True, "prices": [{"slug": "biofield-analysis", "price_cents": 10000, "note": "special"},
                                   {"slug": "other", "price_cents": 5}]}
    assert bf.parse_courtesy(resp) == {"courtesy_cents": 10000, "note": "special"}


def test_parse_courtesy_absent():
    assert bf.parse_courtesy({"ok": True, "prices": []}) == {"courtesy_cents": None, "note": ""}
    assert bf.parse_courtesy({}) == {"courtesy_cents": None, "note": ""}


def test_build_fee_state_no_email():
    st = bf.build_fee_state("", fee_get=lambda e: {"available": True, "courtesy_cents": None, "note": ""})
    assert st["has_email"] is False and st["available"] is False
    assert st["standard_cents"] == 30000 and st["value_cents"] == 99700


def test_build_fee_state_with_courtesy():
    st = bf.build_fee_state("j@x.com", fee_get=lambda e: {"available": True, "courtesy_cents": 10000, "note": "special"})
    assert st["has_email"] and st["available"] and st["courtesy_cents"] == 10000 and st["note"] == "special"


def test_build_fee_state_unavailable():
    st = bf.build_fee_state("j@x.com", fee_get=lambda e: {"available": False, "courtesy_cents": None, "note": ""})
    assert st["has_email"] and st["available"] is False


def test_default_fee_get_no_secret_is_unavailable(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    got = bf.default_fee_get("j@x.com")
    assert got == {"available": False, "courtesy_cents": None, "note": ""}


def test_default_fee_get_parses_prod_response(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    class _Resp:
        def read(self): return b'{"ok":true,"prices":[{"slug":"biofield-analysis","price_cents":10000,"note":"special"}]}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(bf.urllib.request, "urlopen", lambda *a, **k: _Resp())
    got = bf.default_fee_get("j@x.com")
    assert got == {"available": True, "courtesy_cents": 10000, "note": "special"}


def test_default_fee_get_network_failure_is_unavailable(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    def boom(*a, **k): raise OSError("prod down")
    monkeypatch.setattr(bf.urllib.request, "urlopen", boom)
    assert bf.default_fee_get("j@x.com")["available"] is False


def test_default_fee_get_malformed_json_array_is_unavailable(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    class _Resp:
        def read(self): return b'["unexpected","array"]'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(bf.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert bf.default_fee_get("j@x.com") == {"available": False, "courtesy_cents": None, "note": ""}


def test_default_fee_get_malformed_prices_row_is_unavailable(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    class _Resp:
        def read(self): return b'{"ok":true,"prices":[{"slug":"biofield-analysis"}]}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(bf.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert bf.default_fee_get("j@x.com") == {"available": False, "courtesy_cents": None, "note": ""}


def test_default_fee_set_no_secret_is_not_ok(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    assert bf.default_fee_set("j@x.com", 10000, "n") == {"ok": False}


def test_default_fee_set_posts_ok(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    seen = {}
    class _Resp:
        def read(self): return b'{"ok":true}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def _open(req, timeout=None):
        seen["method"] = req.get_method(); seen["url"] = req.full_url
        seen["body"] = _json.loads(req.data.decode())
        return _Resp()
    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    out = bf.default_fee_set("j@x.com", 10000, "special")
    assert out == {"ok": True}
    assert seen["method"] == "POST"
    assert seen["body"] == {"email": "j@x.com", "slug": "biofield-analysis",
                            "price_cents": 10000, "note": "special"}


def test_default_fee_clear_deletes(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    seen = {}
    class _Resp:
        def read(self): return b'{"ok":true}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def _open(req, timeout=None):
        seen["method"] = req.get_method(); seen["body"] = _json.loads(req.data.decode())
        return _Resp()
    monkeypatch.setattr(bf.urllib.request, "urlopen", _open)
    assert bf.default_fee_clear("j@x.com") == {"ok": True}
    assert seen["method"] == "DELETE"
    assert seen["body"] == {"email": "j@x.com", "slug": "biofield-analysis"}
