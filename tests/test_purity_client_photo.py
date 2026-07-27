import io, sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr, fullscript as fs, purity_acquire as pa_mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "cp.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); fs.init_tables(cx)
    # a real catalog product to resolve the slug against
    fs.sync_from_seed(cx, {"products": [{"name": "Test Mag", "brand": "BrandX",
                                         "product_slug": "test-mag", "external_id": "EID",
                                         "focus_tags": [], "best_ff": None, "relation": None}],
                           "focus_area_products": [], "focus_area_items": []})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: True)
    monkeypatch.setattr(app_mod, "_portal_record_for", lambda cx, tok: {"email": "a@b.com"} if tok == "TOK" else None)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _img(form):
    return {**form, "photo": (io.BytesIO(b"\xff\xd8fake"), "label.jpg")}


def test_client_photo_screens_and_lands_screened_not_confirmed(client, monkeypatch):
    monkeypatch.setattr(pa_mod, "acquire_from_image", lambda product, blob, ct, **k: {
        "raw": "magnesium stearate", "parsed": ["magnesium stearate"], "source": "photo", "ok": True})
    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and "review" in body["message"].lower()
    assert "color" not in body                              # never returns a color to the client
    # the row landed screened (red), NOT confirmed
    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT status, color FROM product_ratings WHERE product_key=?",
                     ("fullscript::test-mag",)).fetchone()
    cx.close()
    assert row["status"] == "screened" and row["color"] == "red"


def test_client_photo_flag_off_404(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: False)
    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 404


def test_client_photo_bad_token_404(client):
    r = client.post("/api/portal/NOPE/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 404


def test_client_photo_unknown_product_404(client):
    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "does-not-exist"}), content_type="multipart/form-data")
    assert r.status_code == 404


def test_client_photo_requires_file(client):
    r = client.post("/api/portal/TOK/purity/photo",
                    data={"product_slug": "test-mag"}, content_type="multipart/form-data")
    assert r.status_code == 400
