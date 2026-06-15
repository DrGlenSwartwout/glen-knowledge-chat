"""Group-bundle qualifying gate: only a PAID Biofield Analysis client qualifies
(the free E4L scan does NOT). Forward-looking via the biofield_readiness record."""
import importlib


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    return _app


def test_has_paid_biofield_true_only_with_paid_record(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    import sqlite3
    from dashboard import biofield_store as bs
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        bs.init_table(cx)
        bs.seed_paid(cx, "paid@x.com", via="stripe", order_ref="INV1")
    assert app._has_paid_biofield("paid@x.com") is True
    # case-insensitive
    assert app._has_paid_biofield("PAID@X.COM") is True
    # unknown email -> not a paid Biofield client
    assert app._has_paid_biofield("nobody@x.com") is False


def test_has_paid_biofield_false_for_photo_only_no_payment(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    import sqlite3
    from dashboard import biofield_store as bs
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        bs.init_table(cx)
        # a row exists (e.g. a photo uploaded) but no payment -> does NOT qualify
        bs.set_photo_on_file(cx, "noPay@x.com", "x.jpg")
    assert app._has_paid_biofield("nopay@x.com") is False


def test_has_paid_biofield_safe_when_table_absent(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    # no biofield_readiness rows / fresh DB -> False, never raises
    assert app._has_paid_biofield("anyone@x.com") is False
