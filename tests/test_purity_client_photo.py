import io, sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr, fullscript as fs, purity_acquire as pa_mod
from dashboard import purity_ratings_access as acc


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "cp.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); fs.init_tables(cx); acc.init_table(cx)
    # a real catalog product to resolve the slug against
    fs.sync_from_seed(cx, {"products": [{"name": "Test Mag", "brand": "BrandX",
                                         "product_slug": "test-mag", "external_id": "EID",
                                         "focus_tags": [], "best_ff": None, "relation": None}],
                           "focus_area_products": [], "focus_area_items": []})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: True)
    monkeypatch.setattr(app_mod, "_portal_record_for", lambda cx, tok: {"email": "a@b.com"} if tok == "TOK" else None)
    # entitled by default ('full' membership -> can_request short-circuits True)
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "full", raising=False)
    # Console-ok by default, so console-gated tests don't depend on CONSOLE_SECRET being
    # unset in the shell (Glen's profile exports it, which confounds the vacuous default).
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: True)
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


def test_client_photo_not_entitled_403(client, monkeypatch):
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "none", raising=False)
    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 403
    assert r.get_json()["error"] == "not_entitled"


def test_client_photo_no_launder_existing_screened(client, monkeypatch):
    key = "fullscript::test-mag"
    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    pr.record_screen(cx, key, brand="BrandX", product_name="Test Mag",
                     other_ingredients_raw="titanium dioxide",
                     other_ingredients_parsed=["titanium dioxide"],
                     screen={"color": "red", "red_hits": ["titanium dioxide"],
                            "yellow_hits": [], "avoidlist_version": "v1"})
    cx.commit(); cx.close()

    def _boom(*a, **k):
        raise AssertionError("acquire_from_image must not be called")
    monkeypatch.setattr(pa_mod, "acquire_from_image", _boom)

    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "already being reviewed" in body["message"].lower()

    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    row = cx.execute("SELECT status, color, other_ingredients_raw FROM product_ratings "
                     "WHERE product_key=?", (key,)).fetchone()
    cx.close()
    assert row["status"] == "screened" and row["color"] == "red"
    assert row["other_ingredients_raw"] == "titanium dioxide"   # unchanged


def test_client_photo_rejects_bad_type(client):
    r = client.post("/api/portal/TOK/purity/photo",
                    data={"product_slug": "test-mag",
                         "photo": (io.BytesIO(b"not an image"), "note.txt", "text/plain")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["error"] == "image_only"


from dashboard import purity_photos as _pp


def test_client_photo_persists_the_image(client, monkeypatch):
    from dashboard import purity_acquire as _pa
    monkeypatch.setattr(_pa, "acquire_from_image", lambda product, blob, ct, **k: {
        "raw": "silica", "parsed": ["silica"], "source": "photo", "ok": True})
    r = client.post("/api/portal/TOK/purity/photo",
                    data=_img({"product_slug": "test-mag"}), content_type="multipart/form-data")
    assert r.status_code == 200
    import sqlite3, app as app_mod
    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    row = _pp.get(cx, "fullscript::test-mag"); cx.close()
    assert row is not None and bytes(row["image_blob"]) and row["email"] == "a@b.com"


def test_console_serves_the_photo(client):
    import sqlite3, app as app_mod
    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    _pp.save(cx, "fullscript::test-mag", "a@b.com", b"\xff\xd8JPEGBYTES", "image/jpeg"); cx.close()
    r = client.get("/api/console/purity/photo/fullscript::test-mag")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    assert r.data == b"\xff\xd8JPEGBYTES"


def test_console_serve_photo_missing_404(client):
    assert client.get("/api/console/purity/photo/fullscript::none").status_code == 404


def test_console_serve_photo_unauthorized(client, monkeypatch):
    import app as app_mod
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: False)
    assert client.get("/api/console/purity/photo/fullscript::test-mag").status_code == 401


def test_ratings_list_flags_has_photo(client):
    import sqlite3, app as app_mod
    from dashboard import product_ratings as pr, purity_photos as pp
    cx = sqlite3.connect(app_mod.LOG_DB); cx.row_factory = sqlite3.Row
    pr.record_screen(cx, "fullscript::test-mag", brand="B", product_name="P",
                     other_ingredients_raw="silica", other_ingredients_parsed=["silica"],
                     screen={"color": "yellow", "red_hits": [], "yellow_hits": ["silica"], "avoidlist_version": "v1"})
    pp.save(cx, "fullscript::test-mag", "a@b.com", b"img", "image/png"); cx.close()
    r = client.get("/api/console/purity-ratings")
    rows = {row["product_key"]: row for row in r.get_json()["ratings"]}
    assert rows["fullscript::test-mag"]["has_photo"] is True
