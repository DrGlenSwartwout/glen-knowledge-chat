"""Regression: /api/console/client-invoice must NOT reflect the console key
(CONSOLE_SECRET) into edit_url in its JSON response. The console UI appends its
own key() client-side, so the server embedding it only leaked the secret."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        O.upsert_order(cx, source="qbo", external_ref="INV-1",
                       email="client@example.com", total_cents=12345)
        cx.commit()
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"app not importable: {e}")
    # console-gated on a non-None actor (no role requirement on this read endpoint)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: {"role": "owner"})
    return appmod, appmod.app.test_client()


def test_edit_url_does_not_leak_console_key(tmp_path, monkeypatch):
    _appmod, c = _client(tmp_path, monkeypatch)
    # pass a key on the request exactly as the real console does
    r = c.get("/api/console/client-invoice"
              "?email=client@example.com&key=super-secret-console-key")
    assert r.status_code == 200
    order = r.get_json()["order"]
    assert order is not None
    edit_url = order["edit_url"]
    # the order is still editable...
    assert "edit_order=" in edit_url
    # ...but the secret must NOT be reflected back anywhere in the response
    assert "key=" not in edit_url
    assert "super-secret-console-key" not in r.get_data(as_text=True)
