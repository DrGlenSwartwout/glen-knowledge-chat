# Purity client-photo fixes (#1 entitlement-gate affordance, #2 store + surface photo) — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Before merging #1214, close the two review Minors: (1) show the client photo-upload affordance ONLY to entitled clients; (2) STORE the client-uploaded label photo and let Glen view it in console review, so a photo/product mismatch can be caught at confirm.

**Architecture:** A new BYTEA-backed `purity_photos` store (mirrors `body_map_photos`). The client photo route persists the uploaded image; the console gains a serve route + a `has_photo` flag on the ratings list. `_fullscript_for` exposes a per-client `purity_photo_ok` entitlement bool the frontend gates the affordance on.

**Tech Stack:** Python 3.9 / Flask (sqlite dev, Postgres adapter prod — BYTEA for blobs), vanilla JS.

## Global Constraints

- **BYTEA, never BLOB** for the image column (runtime pgcompat does not translate BLOB — see `body_map_photos.py` / reference_pgcompat_runtime_blob_bytea). Writes use SELECT-then-INSERT/UPDATE + Python timestamps (consistent with the other purity modules — no `ON CONFLICT`/`datetime('now')`).
- **Store the photo only on a path that actually screened** — never on the "already under review" early return, and never before the token/entitlement/product gates pass.
- **Console serve route is console-gated** (`_portal_console_ok()` → 401); it returns the raw image bytes with the stored `content_type`.
- **Entitlement bool** `purity_photo_ok = _purity_badges_enabled() AND can_request(cx, email, membership_category(email))` — same entitlement the route enforces, so the affordance is only shown to clients who can actually use it.
- No new dependency; tests inject `acquire_from_image` and never do real vision/network.

---

### Task 1: `purity_photos` store

**Files:**
- Create: `dashboard/purity_photos.py`
- Test: `tests/test_purity_photos.py`

**Interfaces:**
- Produces:
  - `init_table(cx)` — `purity_photos(product_key TEXT PRIMARY KEY, email TEXT, image_blob BYTEA, content_type TEXT, updated_at TEXT)`.
  - `save(cx, product_key, email, blob, content_type) -> bool` — upsert (latest photo per product); False on empty key/blob.
  - `get(cx, product_key) -> dict | None` — `{image_blob, content_type, email, updated_at}`.
  - `keys_with_photos(cx) -> set[str]` — all product_keys that have a photo (for the `has_photo` list flag).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_photos.py
import sqlite3
from dashboard import purity_photos as pp


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pp.init_table(cx); return cx


def test_save_and_get_roundtrips_bytes():
    cx = _cx()
    assert pp.save(cx, "fullscript::x", "a@b.com", b"\xff\xd8\xff-image", "image/jpeg") is True
    row = pp.get(cx, "fullscript::x")
    assert bytes(row["image_blob"]) == b"\xff\xd8\xff-image"
    assert row["content_type"] == "image/jpeg" and row["email"] == "a@b.com"


def test_save_upserts_latest():
    cx = _cx()
    pp.save(cx, "fullscript::x", "a@b.com", b"first", "image/png")
    pp.save(cx, "fullscript::x", "c@d.com", b"second", "image/jpeg")
    row = pp.get(cx, "fullscript::x")
    assert bytes(row["image_blob"]) == b"second" and row["email"] == "c@d.com"


def test_save_rejects_empty():
    cx = _cx()
    assert pp.save(cx, "", "a@b.com", b"x", "image/png") is False
    assert pp.save(cx, "fullscript::x", "a@b.com", b"", "image/png") is False


def test_get_missing_is_none():
    assert pp.get(_cx(), "fullscript::nope") is None


def test_keys_with_photos():
    cx = _cx()
    pp.save(cx, "fullscript::a", "e", b"1", "image/png")
    pp.save(cx, "fullscript::b", "e", b"2", "image/png")
    assert pp.keys_with_photos(cx) == {"fullscript::a", "fullscript::b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_photos.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# dashboard/purity_photos.py
"""Client-uploaded supplement LABEL photos, kept so Glen can verify a
client-submitted purity screen against the actual photo at confirm time (catch a
photo/product mismatch). One photo per product_key (latest wins).

image_blob is BYTEA, not BLOB: runtime pgcompat does not translate BLOB, so a
BLOB column fails on Postgres; BYTEA round-trips bytes on SQLite too (see
dashboard/body_map_photos.py). Writes use SELECT-then-INSERT/UPDATE + a Python
timestamp -- no ON CONFLICT / datetime('now') -- matching the other purity
modules for cross-backend safety.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purity_photos (
        product_key TEXT PRIMARY KEY,
        email TEXT, image_blob BYTEA, content_type TEXT, updated_at TEXT)""")
    cx.commit()


def save(cx, product_key, email, blob, content_type):
    """Upsert the latest label photo for a product. False on empty key/blob."""
    key = (product_key or "").strip()
    if not key or not blob:
        return False
    init_table(cx)
    now = _now()
    exists = cx.execute("SELECT 1 FROM purity_photos WHERE product_key=?", (key,)).fetchone()
    if exists:
        cx.execute("UPDATE purity_photos SET email=?, image_blob=?, content_type=?, "
                   "updated_at=? WHERE product_key=?",
                   ((email or "").strip().lower(), blob, content_type or "image/jpeg", now, key))
    else:
        cx.execute("INSERT INTO purity_photos "
                   "(product_key, email, image_blob, content_type, updated_at) VALUES (?,?,?,?,?)",
                   (key, (email or "").strip().lower(), blob, content_type or "image/jpeg", now))
    cx.commit()
    return True


def get(cx, product_key):
    key = (product_key or "").strip()
    if not key:
        return None
    r = cx.execute("SELECT product_key, email, image_blob, content_type, updated_at "
                   "FROM purity_photos WHERE product_key=?", (key,)).fetchone()
    return dict(r) if r else None


def keys_with_photos(cx):
    return {row[0] for row in cx.execute("SELECT product_key FROM purity_photos").fetchall()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_photos.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/purity_photos.py tests/test_purity_photos.py
git commit -m "feat(purity): purity_photos BYTEA store for client-uploaded label photos"
```

---

### Task 2: wire storage + entitlement + console serve + has_photo

**Files:**
- Modify: `app.py` — `_fullscript_for` (add `purity_photo_ok`); `api_portal_purity_photo` (store the photo after screen); `api_console_purity_ratings_list` (add `has_photo`); add `api_console_purity_photo_serve`
- Test: `tests/test_purity_client_photo.py` (append)

**Interfaces:**
- Consumes: `purity_photos` (Task 1), `purity_ratings_access.can_request`, `membership_category`.
- Produces:
  - `_fullscript_for(...)` return gains `"purity_photo_ok": bool`.
  - `POST /api/portal/<token>/purity/photo` persists the blob via `purity_photos.save` after a successful screen.
  - `GET /api/console/purity/photo/<path:product_key>` → raw image bytes (console-gated) or 404.
  - `/api/console/purity-ratings` rows gain `has_photo: bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_client_photo.py — append. (Reuses the file's existing `client`
# fixture, which grants entitlement + sets a valid TOK.)
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_client_photo.py -k "persists or serves or has_photo or serve_photo" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

(a) In `_fullscript_for`, compute and add `purity_photo_ok`. After the `groups` are built (where entitlement can be read), change the return to include it — compute inside a short read connection using the client `email` the function already has:

```python
        _enrich_fullscript_purity(groups)
        photo_ok = False
        if _purity_badges_enabled():
            try:
                from dashboard import purity_ratings_access as _acc
                with db.connect(LOG_DB) as cx:
                    _acc.init_table(cx)
                    photo_ok = _acc.can_request(cx, (email or "").strip().lower(),
                                                membership_category(email))
            except Exception:
                photo_ok = False
        return {"dispensary_url": _fullscript_dispensary_url(), "groups": groups,
                "purity_enabled": _purity_badges_enabled(), "purity_photo_ok": photo_ok}
```

(Use the actual local variable name `_fullscript_for` uses for the client email — grep the function; it is the `email`/`email_norm` in scope. If it's `email_norm`, use that.)

(b) In `api_portal_purity_photo`, after the successful `record_screen` (inside or right after the `with _db_lock` write block), persist the photo:

```python
        from dashboard import purity_photos as _pp
        _pp.save(cx, key, (portal.get("email") or ""), blob, ctype)
```

Place this INSIDE the same `with _db_lock, db.connect(...) as cx:` block as `record_screen` (so it shares the lock + connection), AFTER `record_screen`. `portal` is already resolved earlier in the read block — capture the email into a local (e.g. `email = (portal.get("email") or "")`) before the lock so it's in scope.

(c) Add the console serve route (place near `api_console_purity_ratings_list`):

```python
@app.route("/api/console/purity/photo/<path:product_key>", methods=["GET"])
def api_console_purity_photo_serve(product_key):
    """Console-gated: the raw client-uploaded label photo for a product, so Glen
    can verify a client-submitted screen against the actual label before
    confirming."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import purity_photos as _pp
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pp.init_table(cx)
        row = _pp.get(cx, product_key)
    if not row or not row.get("image_blob"):
        return jsonify({"error": "no_photo"}), 404
    return Response(bytes(row["image_blob"]),
                    mimetype=row.get("content_type") or "application/octet-stream")
```

(d) In `api_console_purity_ratings_list`, add `has_photo` per row:

```python
    from dashboard import product_ratings as _pr, purity_photos as _pp
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_tables(cx); _pp.init_table(cx)
        rows = [dict(r) for r in cx.execute(
            "SELECT product_key, brand, product_name, color, status, avoidlist_version, "
            "updated_at FROM product_ratings ORDER BY updated_at DESC").fetchall()]
        with_photos = _pp.keys_with_photos(cx)
    for r in rows:
        r["has_photo"] = r["product_key"] in with_photos
    return jsonify({"ok": True, "ratings": rows})
```

(Adapt to the exact existing body of `api_console_purity_ratings_list` — only ADD the `_pp` import, the `init_table`, the `with_photos` set, and the per-row `has_photo`.)

- [ ] **Step 4: Run to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_client_photo.py -v`
Expected: PASS (all — existing + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add app.py tests/test_purity_client_photo.py
git commit -m "feat(purity): store client label photo + console serve/has_photo + purity_photo_ok entitlement in payload"
```

---

### Task 3: gate the affordance on entitlement (frontend)

**Files:**
- Modify: `static/client-portal.html` — `fullscriptBodyHtml` affordance condition
- Verify: headless render (controller)

**Interfaces:**
- Consumes: `fsData.purity_photo_ok` (Task 2).
- Produces: the affordance renders only when `fsData.purity_photo_ok && !p.purity && p.product_slug`.

- [ ] **Step 1: Change the condition**

In `static/client-portal.html`, in `fullscriptBodyHtml`, change the affordance gate from `fsData.purity_enabled` to `fsData.purity_photo_ok`:

```javascript
      const canPhoto = fsData.purity_photo_ok && !p.purity && p.product_slug;
```

(Leave everything else — the `photo` markup, the `${photo}` in the `<li>`, the delegated handler — unchanged.)

- [ ] **Step 2: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add static/client-portal.html
git commit -m "feat(purity): show client photo affordance only to entitled clients (purity_photo_ok)"
```

- [ ] **Step 3: Verify (controller, headless)**

Eval `fullscriptBodyHtml` with an unrated product and: `purity_photo_ok:true` → affordance present; `purity_photo_ok:false` (even with `purity_enabled:true`) → affordance ABSENT. Confirm a rated product still shows no affordance.

---

## Self-Review

**Coverage of the two Minors:**
- #1 affordance shown to non-entitled clients → Task 2 (`purity_photo_ok` = flag AND `can_request`) + Task 3 (affordance gates on it). ✅
- #2 photo not stored / mismatch uncatchable → Task 1 (`purity_photos` BYTEA store) + Task 2 (persist on upload, console serve route, `has_photo` flag so Glen sees which ratings have a photo to verify). ✅

**Placeholder scan:** none — complete code every step; two "adapt to the exact existing body" notes are real integration instructions (grep the email var; add to the existing list route), not placeholders.

**Type consistency:** `purity_photos.save/get/keys_with_photos` (Task 1) consumed by Task 2's route wiring; `purity_photo_ok` produced in `_fullscript_for` (Task 2) read as `fsData.purity_photo_ok` in Task 3; `has_photo` bool added to list rows. Consistent.
