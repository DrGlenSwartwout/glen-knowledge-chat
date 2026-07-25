# Body Map Multi-Photo + Alignment — Slice 1 (Foundation) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store many photos per client keyed by `(email, system, side)`, each carrying a saved `{mx,my,tx,ty}` similarity transform; serve them and save/load their transforms over HTTP; and wire the Body Map so a saved alignment is applied directly (no re-detection) and the alignment the code already computes gets persisted.

**Architecture:** A new persistence module `dashboard/body_map_photos.py` (a `BYTEA` blob + JSON transform per slot). New Flask routes for per-slot photo upload/serve and transform save/load, token-scoped for the client and console-gated twins for later curation. The portal bodymap-data payload reports the current slot's photo + transform. `static/body-map.js` reconstructs a saved transform and skips MediaPipe when one exists, and PUTs the transform whenever a new alignment is established. `client_photos` is never modified; the `face` slot falls back to it.

**Tech Stack:** Python 3, Flask (`app.py`), the `dashboard/` module convention, SQLite (tests) / Postgres (prod) via `dashboard/db.py` + `dashboard/pgcompat.py`, vanilla JS in `static/body-map.js`, MediaPipe (unchanged).

## Global Constraints

- **Binary columns are `BYTEA`, never `BLOB`.** Runtime `pgcompat.translate_sql` does not translate `BLOB`; it fails on Postgres. `BYTEA` round-trips bytes losslessly on SQLite. (Established by `client_documents`.)
- **`client_photos` is never modified.** It is the identity portrait for biofield/onboarding/sync. This feature only *reads* it, via the `face`-slot fallback.
- **Photo bytes are served through the allowlist.** Reuse `_doc_response_content_type` / `_doc_safe_filename` / `_DOC_INLINE_TYPES` (already in `app.py` from #1172) plus `X-Content-Type-Options: nosniff` and `Cache-Control: private, no-store`. A client-influenced image must never render inline as `text/html`/`svg`.
- **The slot key is `(email, system, side)`.** `side ∈ {'left','right','foot',''}`; `None`→`''`. A new photo for a slot **replaces** the old one and **clears** its transform.
- **The transform is 4 finite numbers `{mx,my,tx,ty}`** in the fixed 600×600 viewBox space (resolution-independent). A malformed transform is never stored (400 / rejected).
- **`system` is validated against `bodymap_store.system_catalog()` ids** so junk slots can't be created.
- **Token-scoping:** a portal token resolves to its owner's email (`_portal_record_for`) and may only ever read/write that email's slots.
- **NEVER `cur.lastrowid`** (raises on Postgres). **NEVER leak `cx.row_factory`** in a store module.
- **Running tests:** run the targeted files during development; do NOT run a bare full suite from a shell with real creds (it sends real email) — use `bash ci/run-tests.sh` for the gate.

---

## File Structure

**Create:**
- `dashboard/body_map_photos.py` — slot photo + transform persistence.
- Tests: `tests/test_body_map_photos_store.py`, `tests/test_bodymap_photo_routes.py`, `tests/test_bodymap_transform_routes.py`, `tests/test_bodymap_payload_slot.py`, `tests/test_bodymap_transform_math.js`

**Modify:**
- `app.py` — photo routes, transform routes, console twins, and the `_portal_bodymap_data` payload.
- `static/body-map.js` — transform reconstruct/save wiring.

---

### Task 1: `body_map_photos` store

**Files:**
- Create: `dashboard/body_map_photos.py`
- Test: `tests/test_body_map_photos_store.py`

**Interfaces:**
- Produces:
  - `init_table(cx)`
  - `put(cx, email, system, side, blob, content_type, source) -> bool` (upsert; CLEARS transform)
  - `get(cx, email, system, side) -> {blob, content_type, transform, source} | None`
  - `set_transform(cx, email, system, side, transform) -> bool` (transform = `{mx,my,tx,ty}` or None)
  - `get_transform(cx, email, system, side) -> dict | None`
  - `list_for_email(cx, email) -> [{system, side, has_transform, updated_at}]` (no blobs)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_body_map_photos_store.py
import sqlite3
from dashboard import body_map_photos as bmp


def _cx():
    cx = sqlite3.connect(":memory:")
    bmp.init_table(cx)
    return cx


def test_bytea_blob_round_trips_all_byte_values():
    cx = _cx()
    blob = bytes(range(256)) * 30
    assert bmp.put(cx, "c@x.com", "face", "", blob, "image/jpeg", "portal-self") is True
    got = bmp.get(cx, "c@x.com", "face", "")
    assert got["blob"] == blob and got["content_type"] == "image/jpeg"
    assert got["transform"] is None and got["source"] == "portal-self"


def test_slot_key_is_email_system_side():
    cx = _cx()
    bmp.put(cx, "C@x.com", "iris", "left", b"L", "image/png", "portal-self")
    bmp.put(cx, "c@x.com", "iris", "right", b"R", "image/png", "portal-self")
    assert bmp.get(cx, "c@x.com", "iris", "left")["blob"] == b"L"
    assert bmp.get(cx, "c@x.com", "iris", "right")["blob"] == b"R"
    rows = bmp.list_for_email(cx, "c@x.com")
    assert {(r["system"], r["side"]) for r in rows} == {("iris", "left"), ("iris", "right")}


def test_none_side_normalizes_to_empty():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", None, b"F", "image/jpeg", "console")
    assert bmp.get(cx, "c@x.com", "face", "")["blob"] == b"F"
    assert bmp.get(cx, "c@x.com", "face", None)["blob"] == b"F"


def test_reput_replaces_photo_and_clears_transform():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"one", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "face", "", {"mx": 1, "my": 0, "tx": 2, "ty": 3})
    bmp.put(cx, "c@x.com", "face", "", b"two", "image/jpeg", "portal-self")  # new photo
    got = bmp.get(cx, "c@x.com", "face", "")
    assert got["blob"] == b"two" and got["transform"] is None       # transform cleared
    assert len(bmp.list_for_email(cx, "c@x.com")) == 1              # still one row


def test_set_transform_round_trips_and_clears():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    assert bmp.set_transform(cx, "c@x.com", "face", "",
                             {"mx": 1.5, "my": -0.5, "tx": 300, "ty": 12.25}) is True
    assert bmp.get_transform(cx, "c@x.com", "face", "") == {"mx": 1.5, "my": -0.5,
                                                            "tx": 300.0, "ty": 12.25}
    assert bmp.set_transform(cx, "c@x.com", "face", "", None) is True   # clear
    assert bmp.get_transform(cx, "c@x.com", "face", "") is None


def test_set_transform_rejects_malformed():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    for bad in ({"mx": 1, "my": 0, "tx": 2}, {"mx": "x", "my": 0, "tx": 2, "ty": 3},
                {"mx": float("nan"), "my": 0, "tx": 2, "ty": 3}, "notadict", []):
        assert bmp.set_transform(cx, "c@x.com", "face", "", bad) is False
    assert bmp.get_transform(cx, "c@x.com", "face", "") is None   # nothing persisted


def test_get_missing_returns_none():
    assert bmp.get(_cx(), "nobody@x.com", "face", "") is None


def test_list_excludes_blobs_and_reports_has_transform():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    bmp.put(cx, "c@x.com", "hand", "", b"h", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "hand", "", {"mx": 1, "my": 0, "tx": 0, "ty": 0})
    rows = {r["system"]: r for r in bmp.list_for_email(cx, "c@x.com")}
    assert "blob" not in rows["face"] and "image_blob" not in rows["face"]
    assert rows["face"]["has_transform"] is False
    assert rows["hand"]["has_transform"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_body_map_photos_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.body_map_photos'`

- [ ] **Step 3: Write the implementation**

```python
# dashboard/body_map_photos.py
"""Per-slot Body Map photo + saved alignment store.

A slot is (email, system, side); each holds one photo and an optional
{mx,my,tx,ty} similarity transform in the map's fixed 600x600 viewBox space
(resolution-independent). Persistence only -- no HTTP, no rendering.

`image_blob` is BYTEA, not BLOB: runtime pgcompat does not translate BLOB, so a
BLOB column fails outright on Postgres. BYTEA round-trips bytes on SQLite. See
docs/superpowers/specs/2026-07-25-bodymap-multiphoto-alignment-foundation-design.md

This is a SEPARATE table from client_photos (the identity portrait). client_photos
is never touched here; the face-slot fallback to it lives in the HTTP layer.
"""
import json
import math
from datetime import datetime, timezone

_TKEYS = ("mx", "my", "tx", "ty")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def _side(side):
    return (side or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS body_map_photos (
        email TEXT, system TEXT, side TEXT,
        image_blob BYTEA, content_type TEXT, transform_json TEXT,
        source TEXT, updated_at TEXT,
        PRIMARY KEY (email, system, side))""")
    cx.commit()


def _valid_transform(t):
    if not isinstance(t, dict):
        return None
    out = {}
    for k in _TKEYS:
        v = t.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return None
        out[k] = float(v)
    return out


def put(cx, email, system, side, blob, content_type, source):
    """Upsert the slot's photo. A new photo CLEARS any saved transform (it needs
    re-aligning). Returns True on write, False for empty email/system/blob."""
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e or not sys_ or not blob:
        return False
    init_table(cx)
    cx.execute(
        "INSERT INTO body_map_photos"
        "(email, system, side, image_blob, content_type, transform_json, source, updated_at) "
        "VALUES(?,?,?,?,?,NULL,?,?) "
        "ON CONFLICT(email, system, side) DO UPDATE SET "
        "image_blob=excluded.image_blob, content_type=excluded.content_type, "
        "transform_json=NULL, source=excluded.source, updated_at=excluded.updated_at",
        (e, sys_, sd, blob, content_type or "image/jpeg", source or "", _now()))
    cx.commit()
    return True


def get(cx, email, system, side):
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e:
        return None
    init_table(cx)
    r = cx.execute(
        "SELECT image_blob, content_type, transform_json, source FROM body_map_photos "
        "WHERE email=? AND system=? AND side=?", (e, sys_, sd)).fetchone()
    if not r or r[0] is None:
        return None
    try:
        transform = json.loads(r[2]) if r[2] else None
    except (TypeError, ValueError):
        transform = None
    return {"blob": r[0], "content_type": r[1] or "image/jpeg",
            "transform": transform, "source": r[3] or ""}


def set_transform(cx, email, system, side, transform):
    """Save (or clear, when transform is None) the slot's {mx,my,tx,ty}. Rejects
    anything that is not four finite numbers. Returns True on a write/clear,
    False on a malformed transform."""
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e:
        return False
    init_table(cx)
    if transform is None:
        val = None
    else:
        clean = _valid_transform(transform)
        if clean is None:
            return False
        val = json.dumps(clean)
    cx.execute("UPDATE body_map_photos SET transform_json=?, updated_at=? "
               "WHERE email=? AND system=? AND side=?", (val, _now(), e, sys_, sd))
    cx.commit()
    return True


def get_transform(cx, email, system, side):
    rec = get(cx, email, system, side)
    return rec["transform"] if rec else None


def list_for_email(cx, email):
    e = _norm(email)
    if not e:
        return []
    init_table(cx)
    rows = cx.execute(
        "SELECT system, side, transform_json, updated_at FROM body_map_photos "
        "WHERE email=? ORDER BY system, side", (e,)).fetchall()
    return [{"system": r[0], "side": r[1], "has_transform": bool(r[2]),
             "updated_at": r[3]} for r in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_body_map_photos_store.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/body_map_photos.py tests/test_body_map_photos_store.py
git commit -m "feat(bodymap): body_map_photos store — per-slot photo + saved transform"
```

---

### Task 2: Photo routes (upload + serve, with face fallback)

**Files:**
- Modify: `app.py` (add after `api_portal_photo_serve`, ~line 20690)
- Test: `tests/test_bodymap_photo_routes.py`

**Interfaces:**
- Consumes: `body_map_photos` (Task 1), `_portal_record_for`, `_present_console_key`/`_owner_token_ok`, `_PHOTO_TYPES`, `_PHOTO_MAX`, `_doc_response_content_type`, `_doc_safe_filename`, `bodymap_store.system_catalog`.
- Produces:
  - `_bodymap_valid_system(system) -> bool`
  - `POST /api/portal/<token>/bodymap-photo?system=&side=`
  - `GET /api/portal/<token>/bodymap-photo?system=&side=` (face falls back to `client_photos`)
  - `POST /api/console/bodymap-photo` (email, system, side; console-gated)
  - `GET /api/console/bodymap-photo?email=&system=&side=`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bodymap_photo_routes.py
import importlib, io, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _token(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit(); cx.close()
    return tok


def _up(client, url, data=b"\xff\xd8\xffimg", name="face.jpg", ctype="image/jpeg"):
    return client.post(url, data={"photo": (io.BytesIO(data), name, ctype)},
                       content_type="multipart/form-data")


def test_upload_and_serve_a_slot(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand").status_code == 200
    r = c.get(f"/api/portal/{tok}/bodymap-photo?system=hand")
    assert r.status_code == 200 and r.data == b"\xff\xd8\xffimg"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "private, no-store"


def test_side_makes_distinct_slots(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    _up(c, f"/api/portal/{tok}/bodymap-photo?system=iris&side=left", data=b"LEYE")
    _up(c, f"/api/portal/{tok}/bodymap-photo?system=iris&side=right", data=b"REYE")
    assert c.get(f"/api/portal/{tok}/bodymap-photo?system=iris&side=left").data == b"LEYE"
    assert c.get(f"/api/portal/{tok}/bodymap-photo?system=iris&side=right").data == b"REYE"


def test_face_falls_back_to_client_photos(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit(); cx.close()
    # no body_map_photos face row -> face serves the client_photos portrait
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=face")
    assert r.status_code == 200 and r.data == b"PORTRAIT"


def test_face_slot_wins_over_client_photos(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit(); cx.close()
    _up(appmod.app.test_client(), f"/api/portal/{tok}/bodymap-photo?system=face", data=b"FACESLOT")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=face")
    assert r.data == b"FACESLOT"


def test_nonface_missing_slot_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/bodymap-photo?system=hand").status_code == 404


def test_token_scoping(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com")
    tok_a = _token(appmod, "a@x.com")
    _up(appmod.app.test_client(), f"/api/portal/{tok_a}/bodymap-photo?system=hand", data=b"AHAND")
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get(cx, "a@x.com", "hand", "") is not None
    assert bmp.get(cx, "other@x.com", "hand", "") is None


def test_unknown_system_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert _up(appmod.app.test_client(),
               f"/api/portal/{tok}/bodymap-photo?system=notasystem").status_code == 400


def test_size_and_type_rejected(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand", data=b"", ).status_code == 400
    big = b"x" * (5 * 1024 * 1024 + 1)
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand", data=big).status_code == 400
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand",
               data=b"pdf", ctype="application/pdf").status_code == 400


def test_html_slot_served_as_attachment(tmp_path, monkeypatch):
    # A slot somehow stored as text/html must never render inline.
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"<script>", "text/html", "console"); cx.commit(); cx.close()
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=hand")
    assert r.headers["Content-Type"].startswith("application/octet-stream")
    assert "attachment" in r.headers.get("Content-Disposition", "")


def test_console_photo_requires_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().post(
        "/api/console/bodymap-photo",
        data={"email": "c@x.com", "system": "hand",
              "photo": (io.BytesIO(b"H"), "h.jpg", "image/jpeg")},
        content_type="multipart/form-data")
    assert r.status_code == 401


def test_console_photo_with_key_stores_console_source(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().post(
        "/api/console/bodymap-photo?key=test-secret",
        data={"email": "C@x.com", "system": "hand",
              "photo": (io.BytesIO(b"H"), "h.jpg", "image/jpeg")},
        content_type="multipart/form-data")
    assert r.status_code == 200
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get(cx, "c@x.com", "hand", "")["source"] == "console"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_bodymap_photo_routes.py -v`
Expected: FAIL — the routes 404.

- [ ] **Step 3: Write the implementation**

Insert into `app.py` after `api_portal_photo_serve` (~line 20690):

```python
def _bodymap_valid_system(system):
    """True if `system` is a real Body Map system id (guards junk slots)."""
    import bodymap_store as _bm
    try:
        return (system or "").strip() in {s["id"] for s in _bm.system_catalog()}
    except Exception:
        return False


def _accept_bodymap_photo(cx, email, system, side, f, source):
    """Validate + store one slot photo. Shared by the portal and console routes."""
    from dashboard import body_map_photos as _bmp
    system = (system or "").strip()
    if not _bodymap_valid_system(system):
        return {"ok": False, "error": "unknown system"}, 400
    blob = f.read() if f else b""
    if not blob:
        return {"ok": False, "error": "no image uploaded"}, 400
    ctype = (getattr(f, "mimetype", "") or "").lower()
    if ctype not in _PHOTO_TYPES:
        return {"ok": False, "error": "use a JPG, PNG, or WEBP image"}, 400
    if len(blob) > _PHOTO_MAX:
        return {"ok": False, "error": "image too large (max 5 MB)"}, 400
    _bmp.put(cx, email, system, side, blob, ctype, source)
    return {"ok": True}, 200


def _serve_bodymap_photo(rec, filename="photo"):
    """A stored slot photo -> hardened Response (allowlist + nosniff)."""
    ctype, disp = _doc_response_content_type(rec["content_type"])
    resp = Response(rec["blob"], mimetype=ctype)
    resp.headers["Cache-Control"] = "private, no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Disposition"] = f'{disp}; filename="{_doc_safe_filename(filename)}"'
    return resp


@app.route("/api/portal/<token>/bodymap-photo", methods=["POST"])
def api_portal_bodymap_photo_upload(token):
    """Client self-uploads a Body Map slot photo (token-scoped)."""
    from dashboard import client_portal as _cp
    system = request.args.get("system", "")
    side = request.args.get("side", "")
    with _db_lock, db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        if not email:
            return jsonify({"ok": False, "error": "not found"}), 404
        body, status = _accept_bodymap_photo(cx, email, system, side,
                                             request.files.get("photo"), "portal-self")
    return jsonify(body), status


@app.route("/api/portal/<token>/bodymap-photo", methods=["GET"])
def api_portal_bodymap_photo_serve(token):
    """Serve the token owner's slot photo. system=face with no slot row falls
    back to the client_photos identity portrait (today's behavior)."""
    from dashboard import client_portal as _cp
    from dashboard import body_map_photos as _bmp
    from dashboard import client_photos as _cph
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    with db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        rec = _bmp.get(cx, email, system, side) if email else None
        if not rec and system == "face" and email:
            rec = _cph.get(cx, email)   # {blob, content_type} identity-portrait fallback
    if not rec:
        return Response("", status=404)
    return _serve_bodymap_photo(rec)


@app.route("/api/console/bodymap-photo", methods=["POST"])
def api_console_bodymap_photo_upload():
    """Console-side slot photo upload (for later curation)."""
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
    email = (request.form.get("email") or "").strip().lower()
    system = request.form.get("system", "") or request.args.get("system", "")
    side = request.form.get("side", "") or request.args.get("side", "")
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        body, status = _accept_bodymap_photo(cx, email, system, side,
                                             request.files.get("photo"), "console")
    return jsonify(body), status


@app.route("/api/console/bodymap-photo", methods=["GET"])
def api_console_bodymap_photo_serve():
    """Console-gated slot photo viewer."""
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import body_map_photos as _bmp
    email = (request.args.get("email") or "").strip().lower()
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    with db.connect(LOG_DB) as cx:
        rec = _bmp.get(cx, email, system, side) if email else None
    if not rec:
        return Response("", status=404)
    return _serve_bodymap_photo(rec)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_bodymap_photo_routes.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_bodymap_photo_routes.py
git commit -m "feat(bodymap): per-slot photo upload/serve routes with face fallback"
```

---

### Task 3: Transform routes (save + load)

**Files:**
- Modify: `app.py` (add after the photo routes from Task 2)
- Test: `tests/test_bodymap_transform_routes.py`

**Interfaces:**
- Consumes: `body_map_photos.set_transform` / `get_transform` (Task 1), `_bodymap_valid_system`, the guards from Task 2.
- Produces:
  - `PUT /api/portal/<token>/bodymap-transform?system=&side=` (JSON body `{mx,my,tx,ty}`)
  - `GET /api/portal/<token>/bodymap-transform?system=&side=`
  - `PUT /api/console/bodymap-transform` / `GET` (console-gated, explicit email)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bodymap_transform_routes.py
import importlib, io, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _token(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit(); cx.close()
    return tok


def _seed_photo(appmod, email, system, side=""):
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, email, system, side, b"img", "image/jpeg", "portal-self"); cx.commit(); cx.close()


T = {"mx": 1.5, "my": -0.5, "tx": 300.0, "ty": 12.25}


def test_put_then_get_transform(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face", json=T).status_code == 200
    assert c.get(f"/api/portal/{tok}/bodymap-transform?system=face").get_json() == T


def test_get_missing_transform_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/bodymap-transform?system=face").status_code == 404


def test_put_malformed_transform_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face",
                 json={"mx": 1, "my": 0, "tx": 2}).status_code == 400
    assert c.get(f"/api/portal/{tok}/bodymap-transform?system=face").status_code == 404


def test_transform_token_scoping(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com"); _seed_photo(appmod, "other@x.com", "face")
    tok_a = _token(appmod, "a@x.com"); _seed_photo(appmod, "a@x.com", "face")
    appmod.app.test_client().put(f"/api/portal/{tok_a}/bodymap-transform?system=face", json=T)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get_transform(cx, "a@x.com", "face", "") == T
    assert bmp.get_transform(cx, "other@x.com", "face", "") is None


def test_unknown_system_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert appmod.app.test_client().put(
        f"/api/portal/{tok}/bodymap-transform?system=notasystem", json=T).status_code == 400


def test_console_transform_requires_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().put(
        "/api/console/bodymap-transform?email=c@x.com&system=face", json=T).status_code == 401


def test_console_transform_roundtrip(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 json=T).status_code == 200
    assert c.get("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face"
                 ).get_json() == T
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_bodymap_transform_routes.py -v`
Expected: FAIL — the routes 404.

- [ ] **Step 3: Write the implementation**

Append to `app.py` after the Task-2 routes:

```python
@app.route("/api/portal/<token>/bodymap-transform", methods=["PUT"])
def api_portal_bodymap_transform_set(token):
    from dashboard import client_portal as _cp
    from dashboard import body_map_photos as _bmp
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    if not _bodymap_valid_system(system):
        return jsonify({"ok": False, "error": "unknown system"}), 400
    t = request.get_json(silent=True)
    with _db_lock, db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        if not email:
            return jsonify({"ok": False, "error": "not found"}), 404
        ok = _bmp.set_transform(cx, email, system, side, t)
    return (jsonify({"ok": True}), 200) if ok else \
           (jsonify({"ok": False, "error": "invalid transform"}), 400)


@app.route("/api/portal/<token>/bodymap-transform", methods=["GET"])
def api_portal_bodymap_transform_get(token):
    from dashboard import client_portal as _cp
    from dashboard import body_map_photos as _bmp
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    with db.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        t = _bmp.get_transform(cx, email, system, side) if email else None
    if not t:
        return Response("", status=404)
    return jsonify(t)


@app.route("/api/console/bodymap-transform", methods=["PUT"])
def api_console_bodymap_transform_set():
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import body_map_photos as _bmp
    email = (request.args.get("email") or "").strip().lower()
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    if not email or not _bodymap_valid_system(system):
        return jsonify({"ok": False, "error": "email and valid system required"}), 400
    t = request.get_json(silent=True)
    with _db_lock, db.connect(LOG_DB) as cx:
        ok = _bmp.set_transform(cx, email, system, side, t)
    return (jsonify({"ok": True}), 200) if ok else \
           (jsonify({"ok": False, "error": "invalid transform"}), 400)


@app.route("/api/console/bodymap-transform", methods=["GET"])
def api_console_bodymap_transform_get():
    if CONSOLE_SECRET:
        key = _present_console_key()
        if key != CONSOLE_SECRET and not _owner_token_ok(key):
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import body_map_photos as _bmp
    email = (request.args.get("email") or "").strip().lower()
    system = (request.args.get("system", "") or "").strip()
    side = request.args.get("side", "")
    with db.connect(LOG_DB) as cx:
        t = _bmp.get_transform(cx, email, system, side) if email else None
    if not t:
        return Response("", status=404)
    return jsonify(t)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_bodymap_transform_routes.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_bodymap_transform_routes.py
git commit -m "feat(bodymap): per-slot alignment transform save/load routes"
```

---

### Task 4: Extend the portal bodymap-data payload with the current slot

**Files:**
- Modify: `app.py` — `_portal_bodymap_data` (~line 21279)
- Test: `tests/test_bodymap_payload_slot.py`

**Interfaces:**
- The payload dict gains: `slot_side` (the canonical side for this system/view) and `slot_transform` (`{mx,my,tx,ty}` or null). `has_photo` continues to report whether the CURRENT slot has a servable photo (slot row, or the client_photos fallback for face).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bodymap_payload_slot.py
import importlib, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_payload_reports_slot_transform_for_current_system(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"H", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "hand", "", {"mx": 1, "my": 0, "tx": 5, "ty": 6})
    cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is True
    assert out["slot_transform"] == {"mx": 1.0, "my": 0.0, "tx": 5.0, "ty": 6.0}


def test_payload_no_transform_when_unaligned(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"H", "image/jpeg", "portal-self"); cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is True and out["slot_transform"] is None


def test_payload_face_photo_via_client_photos_fallback(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="face")
    assert out["has_photo"] is True          # fallback still counts as a photo
    assert out["slot_transform"] is None      # no saved transform for the portrait


def test_payload_no_photo_no_slot(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is False and out["slot_transform"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_bodymap_payload_slot.py -v`
Expected: FAIL — `KeyError: 'slot_transform'` / `has_photo` wrong for a non-face slot.

- [ ] **Step 3: Write the implementation**

In `app.py`, `_portal_bodymap_data`, the current `has_photo` block reads:

```python
    out = {"system": system, "view": VIEW, "has_photo": False,
           "findings": [], "lit_zones": [], "count": 0}
    email = (email or "").strip().lower()
    if not email:
        return out
    try:
        out["has_photo"] = bool(_cph.has(cx, email))
    except Exception:
        pass
```

Replace it with (add `slot_side`/`slot_transform`, and make `has_photo` slot-aware with the face fallback):

```python
    slot_side = RESOLVE_SIDE if RESOLVE_SIDE in ("left", "right", "foot") else ""
    out = {"system": system, "view": VIEW, "has_photo": False,
           "slot_side": slot_side, "slot_transform": None,
           "findings": [], "lit_zones": [], "count": 0}
    email = (email or "").strip().lower()
    if not email:
        return out
    try:
        from dashboard import body_map_photos as _bmp
        rec = _bmp.get(cx, email, system, slot_side)
        if rec:
            out["has_photo"] = True
            out["slot_transform"] = rec["transform"]
        elif system == "face" and _cph.has(cx, email):
            out["has_photo"] = True            # client_photos portrait fallback (no transform)
    except Exception:
        pass
```

(`RESOLVE_SIDE` is already bound from `cfg["resolve_side"]` earlier in the function; confirm and reuse it. If `RESOLVE_SIDE` can be a value outside left/right/foot for symmetric systems, the guard maps it to `''` so the slot key matches what the routes store.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_bodymap_payload_slot.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_bodymap_payload_slot.py
git commit -m "feat(bodymap): payload reports current-slot photo + saved transform"
```

---

### Task 5: Body Map JS — apply a saved transform, persist a new one

**Files:**
- Modify: `static/body-map.js`
- Test: `tests/test_bodymap_transform_math.js` (node) — pins the persist/reconstruct round-trip.

**Interfaces:**
- Consumes: the Task-4 payload (`slot_side`, `slot_transform`), the Task-2 photo GET, the Task-3 transform PUT.
- Produces: `bmTransformFromParams(p) -> fn`; `bmTransformParams(steps|anchors) -> {mx,my,tx,ty}` (extracted from the existing fit math). Both exported for the node test via a guarded `module.exports` at the file's end (mirroring the pattern used by `portal-documents.js`).

**Note:** `.js` tests are not CI-gated (`ci/run-tests.sh` runs pytest only). This node test pins the transform math; the render behavior (skip-detect when a transform exists, save after align) MUST be verified in a real browser at the end — a green node test is not evidence the page warps correctly.

- [ ] **Step 1: Write the failing test**

```javascript
// tests/test_bodymap_transform_math.js
// Run: node tests/test_bodymap_transform_math.js
const assert = require('assert');
const { bmTransformFromParams, bmTransformParams } = require('../static/body-map.js');

// A saved {mx,my,tx,ty} reconstructs the SAME mapping fitSimilarity produced.
const steps = [
  { template: { x: 0, y: 0 }, key: 'a' },
  { template: { x: 1, y: 0 }, key: 'b' },
];
const anchors = { a: { x: 100, y: 100 }, b: { x: 300, y: 100 } };  // scale 200, no rotation
const p = bmTransformParams(steps, anchors);
const fn = bmTransformFromParams(p);

// template (0,0)->(100,100); (1,0)->(300,100); (0,1)-> rotated by the same similarity
const A = fn({ x: 0, y: 0 }), B = fn({ x: 1, y: 0 });
assert.ok(Math.abs(A.x - 100) < 1e-6 && Math.abs(A.y - 100) < 1e-6);
assert.ok(Math.abs(B.x - 300) < 1e-6 && Math.abs(B.y - 100) < 1e-6);

// round-trip through JSON (what the endpoint stores) is identical
const p2 = JSON.parse(JSON.stringify(p));
const fn2 = bmTransformFromParams(p2);
const C = fn({ x: 0.37, y: 0.81 }), D = fn2({ x: 0.37, y: 0.81 });
assert.ok(Math.abs(C.x - D.x) < 1e-9 && Math.abs(C.y - D.y) < 1e-9);

console.log('ok - bodymap transform math round-trips');
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node tests/test_bodymap_transform_math.js`
Expected: FAIL — `bmTransformFromParams`/`bmTransformParams` are not exported.

- [ ] **Step 3: Write the implementation**

In `static/body-map.js`:

1. Add the two helpers near `fitSimilarity` (~line 528). `bmTransformParams` extracts the same math `fitSimilarity` already computes, returning the raw params; `fitSimilarity` becomes a thin wrapper so behavior is unchanged:

```javascript
  // The persistable similarity params {mx,my,tx,ty} from the first two anchor
  // correspondences (translation + rotation + uniform scale). Same math as the
  // old fitSimilarity; split out so the params can be SAVED, not just applied.
  function bmTransformParams(steps, anchorMap) {
    const a0 = steps[0].template, a1 = steps[1].template;
    const b0 = anchorMap[steps[0].key], b1 = anchorMap[steps[1].key];
    const dax = a1.x - a0.x, day = a1.y - a0.y;
    const dbx = b1.x - b0.x, dby = b1.y - b0.y;
    const denom = dax * dax + day * day || 1e-9;
    const mx = (dbx * dax + dby * day) / denom;
    const my = (dby * dax - dbx * day) / denom;
    const tx = b0.x - (mx * a0.x - my * a0.y);
    const ty = b0.y - (my * a0.x + mx * a0.y);
    return { mx, my, tx, ty };
  }

  function bmTransformFromParams(p) {
    return (n) => ({ x: p.mx * n.x - p.my * n.y + p.tx,
                     y: p.my * n.x + p.mx * n.y + p.ty });
  }
```

Refactor `fitSimilarity(steps)` to `return bmTransformFromParams(bmTransformParams(steps, anchors));` (uses the module-level `anchors`, unchanged behavior).

2. In `placeOverlay(steps)` (~line 542), after setting `state.transform`, capture and persist the params for the current slot. For the two-anchor path the params are `bmTransformParams(steps, anchors)`; for the iris `computeSimilarity` fallback, derive `{mx,my,tx,ty}` from its `scale`/rotation the same way (expose them from `computeSimilarity` as a `{fn, params}` return, or recompute: `mx=scale*cos, my=scale*sin, tx=P.x, ty=P.y`). Store `state.savedParams = params` and call `saveBodymapTransform(params)`.

3. Add `saveBodymapTransform(params)` — best-effort PUT to `/api/portal/<token>/bodymap-transform?system=<current>&side=<slot_side>` (from the payload), swallowing errors:

```javascript
  function saveBodymapTransform(params) {
    if (!state.portalToken || !params) return;
    fetch("/api/portal/" + encodeURIComponent(state.portalToken)
          + "/bodymap-transform?system=" + encodeURIComponent(state.payload.system)
          + "&side=" + encodeURIComponent(state.slotSide || ""),
          { method: "PUT", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params) }).catch(function () {});
  }
```

4. In `bootstrapPortal` (~line 382), capture `state.slotSide = pz.slot_side || ""` and `state.slotTransform = pz.slot_transform || null`. Point the photo load at the slot endpoint and apply a saved transform when present. Replace `loadPortalPhoto(token)` (~line 404) so:
   - it sets `img.src` to `/api/portal/<token>/bodymap-photo?system=<current>&side=<slotSide>&t=...` (face still works via the server-side fallback);
   - on load, **if `state.slotTransform` exists**, `state.transform = bmTransformFromParams(state.slotTransform); setMode(true); renderChart();` and **return without calling `beginAnchoring()`/`autoDetect()`** (the skip-redetect win);
   - else run today's `beginAnchoring()` + `autoDetect()`.

5. Ensure `onUpload` (local file upload) still works for non-portal use (unchanged), and that a fresh portal upload (which clears the transform server-side) results in `state.slotTransform` being null on the next bootstrap so it re-detects.

**Node-requireability (critical — the file is a browser IIFE).** `static/body-map.js` is a single `(function () { … })()` whose LAST two lines run at load time and touch `window`/`document`:

```javascript
  window.__bm = { clockToNormalized, arcSectorPoints, computeSimilarity, state };
  document.addEventListener("DOMContentLoaded", wire);
```

A `require()` in node executes the IIFE and would throw `ReferenceError: window is not defined` at those lines, before any export runs. The exported functions also live *inside* the closure, so an export block placed after `})();` cannot see them. Fix both by editing the closing lines of the IIFE (still INSIDE it) to guard the browser tail and export in node:

```javascript
  // node (tests): export the pure transform helpers. browser: wire the page.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { bmTransformFromParams: bmTransformFromParams,
                       bmTransformParams: bmTransformParams };
  }
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    window.__bm = { clockToNormalized, arcSectorPoints, computeSimilarity, state };
    document.addEventListener("DOMContentLoaded", wire);
  }
})();
```

Everything else in the IIFE only *defines* functions (which touch `document` when called, not at load), so it is safe under node. Do NOT move the helpers out of the closure — guard the tail instead.

- [ ] **Step 4: Run the node test to verify it passes**

Run: `node tests/test_bodymap_transform_math.js`
Expected: `ok - bodymap transform math round-trips`

- [ ] **Step 5: Verify in a real browser (cannot be faked)**

Boot the app against a seeded DB (a portal token, a `body_map_photos` face slot with a saved transform, and a second system slot). Open `/portal/<token>/bodymap`. Confirm by eye: the face map loads warped **without a detect pause** when a transform is saved; switching to a system with a slot photo loads that photo; completing an alignment persists (reload → still aligned, no re-detect). A green node test is NOT this evidence. If the app cannot be booted in your environment, say so plainly and leave this for a human.

- [ ] **Step 6: Commit**

```bash
git add static/body-map.js tests/test_bodymap_transform_math.js
git commit -m "feat(bodymap): apply saved slot transform (skip re-detect), persist new alignments"
```

---

### Task 6: Full-suite gate

- [ ] **Step 1: Run the whole gated suite**

Run: `bash ci/run-tests.sh`
Expected: PASS (ratchets against `tests/known_failures.txt`, fails only on a NEW failure; sets fake keys and unsets `DOPPLER_TOKEN`).

- [ ] **Step 2: If a NEW failure appears, fix the cause**

Likely sources: a test module importing `app` that needs a dummy key added to its own helper; or an existing test asserting a fixed route count. Confirm the new behavior is correct before adjusting an assertion; never add to `known_failures.txt`.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(bodymap): keep the CI ratchet green"
```

---

## Self-Review notes (for the implementer)

- `client_photos` must never be written by any of this — only read via the face fallback in Task 2 and Task 4. If a diff writes `client_photos`, it's wrong.
- The transform is stored/served in the 600×600 viewBox space. Do not introduce any pixel/screen-size data into the stored transform.
- The `slot_side` the payload reports and the `side` the routes store must agree for a given system — Task 4's `RESOLVE_SIDE→slot_side` mapping is the single source of that value; the JS passes it straight through.
- Task 5's browser step is real. A green node test proves the math, not the warp.

## Out of scope (Slices 2 & 3)

- The nudge/rotate/scale adjust editor (Slice 2).
- The console curation screen (Slice 3) — its endpoints ship here; its UI does not.
- Any migration of existing single photos into slots — the face fallback covers it.
