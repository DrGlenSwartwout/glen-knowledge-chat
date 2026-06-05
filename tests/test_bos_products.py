import json, os, sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _products(tmp_path, products):
    d = tmp_path / "data"; d.mkdir()
    (d / "products.json").write_text(json.dumps({"products": products}))
    os.environ["DATA_DIR"] = str(d)
    from dashboard import products as P
    P._FIXED_CACHE = None
    return P


def test_stale_pages_excludes_fixed(tmp_path, monkeypatch):
    P = _products(tmp_path, {
        "a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True, "gk_stale_reason": "missing X"},
        "b": {"name": "B", "ingredients": [{"name": "Y"}]},
    })
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    sp = P.stale_pages()
    assert len(sp) == 1 and sp[0]["slug"] == "a"


def test_products_signal_amber_on_stale(tmp_path, monkeypatch):
    from dashboard import signals as S
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    sig = P.products_signal(None, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1


def test_products_signal_green_when_clear(tmp_path, monkeypatch):
    from dashboard import signals as S
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}]}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    assert P.products_signal(None, None)["level"] == S.GREEN


def test_mark_page_fixed_action(tmp_path, monkeypatch):
    import sqlite3
    from dashboard import dispatch as D, events as E, rbac as R, actions as A
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    assert A.get_action("products.mark_page_fixed") is not None
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row; E.init_event_tables(cx)
    res = D.dispatch_action(cx, "products.mark_page_fixed", {"slug": "a"}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert P.stale_pages() == []  # 'a' now fixed -> queue empty


def test_products_signal_registered():
    from dashboard import products as P, signals as S  # noqa
    assert S.SIGNAL_REGISTRY.get("products") is not None
