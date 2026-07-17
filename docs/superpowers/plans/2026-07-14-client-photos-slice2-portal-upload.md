# Client Photos — Slice 2 (Portal Self-Upload) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a client upload their own photo from their portal; it flows into the shared `client_photos` store and appears on the intake page and console reveals like any other source.

**Architecture:** Two token-scoped routes on `app.py` (`POST`/`GET /api/portal/<token>/photo`) that resolve the client's email from their portal token via the existing `_portal_record_for(cx, token)` and read/write the Slice 1 store `dashboard/client_photos.py`. A photo control in `static/client-portal.html` shows the current photo and uploads a new one. No store changes: `client_photos.put` upserts, and `portal-self` is the highest-precedence source, so a client upload correctly overwrites an FMP/GHL photo.

**Tech Stack:** Python 3 / Flask (`app.py`), SQLite (`client_photos` table, Slice 1), vanilla JS (`static/client-portal.html`), pytest with the repo's `_app` reload harness.

## Global Constraints

- Source precedence `portal-self` > `fmp` > `ghl`. Slice 2 writes only `source="portal-self"`, which the existing upsert `put` already makes win — **no precedence code in this slice** (added in Slice 3).
- Accepted image types: `image/jpeg`, `image/png`, `image/webp` only.
- Max upload size: 5 MB (`5 * 1024 * 1024` bytes).
- Serve is **token-scoped**: a route resolves the email from the token and serves/writes only that email's photo — never an email passed by the client.
- Portal identity is resolved with `_portal_record_for(cx, token)` (returns the portal dict with `"email"`, or `None`) — the same helper `/api/portal/<token>/share-consent` uses. Do not invent a new resolver.
- The store API (Slice 1, `dashboard/client_photos.py`): `put(cx, email, blob, content_type, source="upload") -> email|None`; `get(cx, email) -> {"blob": bytes, "content_type": str}|None`.
- Tests that import `app` must use the repo's reload harness (`monkeypatch.setenv("DATA_DIR", tmp_path)`, `importlib.reload(app)`) and skip if app is not importable. Run with `doppler run -p remedy-match -c dev -- pytest` (app-importing tests silently skip under bare pytest).

---

### Task 1: Backend — token-scoped portal photo upload + serve

**Files:**
- Modify: `app.py` — add two routes next to the other `/api/portal/<token>/*` routes (near `api_portal_share_consent`, ~line 17811).
- Test: `tests/test_portal_photo.py` (create)

**Interfaces:**
- Consumes: `_portal_record_for(cx, token) -> dict|None` (existing, in `app.py`); `dashboard.client_photos.put/get` (Slice 1).
- Produces: `POST /api/portal/<token>/photo` (multipart field `photo`) → `{"ok": true}` / `{"ok": false, "error": ...}`; `GET /api/portal/<token>/photo` → image bytes (200) or empty 404.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portal_photo.py`:

```python
import base64, importlib, io, sqlite3, sys
from pathlib import Path
import pytest

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed_portal(appmod, email):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "Test Client", {})
        cx.commit()
    return token


def _upload(client, token, blob=PNG, ctype="image/png", name="m.png"):
    return client.post(
        f"/api/portal/{token}/photo",
        data={"photo": (io.BytesIO(blob), name, ctype)},
        content_type="multipart/form-data")


def test_upload_then_serve_own_photo(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "client@x.com")
    c = appmod.app.test_client()
    r = _upload(c, token)
    assert r.status_code == 200 and r.get_json()["ok"] is True
    g = c.get(f"/api/portal/{token}/photo")
    assert g.status_code == 200
    assert g.data == PNG
    assert g.mimetype == "image/png"


def test_serve_is_token_scoped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    t1 = _seed_portal(appmod, "a@x.com")
    t2 = _seed_portal(appmod, "b@x.com")
    c = appmod.app.test_client()
    _upload(c, t1)
    # t2's owner has no photo; the route serves only the token's own email -> 404
    assert c.get(f"/api/portal/{t2}/photo").status_code == 404


def test_rejects_non_image_and_oversize(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "client@x.com")
    c = appmod.app.test_client()
    assert _upload(c, token, blob=b"not-an-image", ctype="text/plain", name="x.txt").status_code == 400
    big = b"\x89PNG" + b"\x00" * (5 * 1024 * 1024 + 1)
    assert _upload(c, token, blob=big, ctype="image/png").status_code == 400


def test_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.get("/api/portal/nope/photo").status_code == 404
    assert _upload(c, "nope").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c dev -- pytest tests/test_portal_photo.py -v`
Expected: FAIL — routes return 404 (Flask "not found" for an undefined route) so the 200/round-trip assertions fail.

- [ ] **Step 3: Add the two routes**

In `app.py`, immediately before `@app.route("/api/portal/<token>/share-consent", methods=["POST"])`:

```python
_PHOTO_TYPES = ("image/jpeg", "image/png", "image/webp")
_PHOTO_MAX = 5 * 1024 * 1024


@app.route("/api/portal/<token>/photo", methods=["POST"])
def api_portal_photo_upload(token):
    """Client self-uploads their portal photo. Token-scoped: writes only the token
    owner's email. source='portal-self' (highest precedence — overwrites FMP/GHL)."""
    from dashboard import client_photos as _cph
    f = request.files.get("photo")
    blob = f.read() if f else b""
    if not blob:
        return jsonify({"ok": False, "error": "no image uploaded"}), 400
    ctype = (getattr(f, "mimetype", "") or "").lower()
    if ctype not in _PHOTO_TYPES:
        return jsonify({"ok": False, "error": "use a JPG, PNG, or WEBP image"}), 400
    if len(blob) > _PHOTO_MAX:
        return jsonify({"ok": False, "error": "image too large (max 5 MB)"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        if not email:
            return jsonify({"ok": False, "error": "not found"}), 404
        _cph.put(cx, email, blob, ctype, source="portal-self")
    return jsonify({"ok": True})


@app.route("/api/portal/<token>/photo", methods=["GET"])
def api_portal_photo_serve(token):
    """Serve the token owner's OWN photo (token-scoped). 404 when none so the
    portal <img> hides cleanly."""
    from dashboard import client_photos as _cph
    with sqlite3.connect(LOG_DB) as cx:
        portal = _portal_record_for(cx, token)
        email = (portal.get("email") or "").strip().lower() if portal else ""
        rec = _cph.get(cx, email) if email else None
    if not rec:
        return Response("", status=404)
    resp = Response(rec["blob"], mimetype=rec["content_type"])
    resp.headers["Cache-Control"] = "private, no-store"
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c dev -- pytest tests/test_portal_photo.py -v`
Expected: PASS (4 tests). If it reports "app not importable" and skips, run under doppler dev as shown (bare pytest skips app-importing tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_photo.py
git commit -m "Client photos Slice 2: token-scoped portal photo upload + serve"
```

---

### Task 2: Frontend — portal photo control

**Files:**
- Modify: `static/client-portal.html` — add a photo control in the greeting/header area (near the `pd-greeting` header the page already renders).

**Interfaces:**
- Consumes: `POST`/`GET /api/portal/<token>/photo` from Task 1; the page's existing `token`/`seg` and `esc` helpers.
- Produces: none (leaf UI).

- [ ] **Step 1: Add the photo control markup + logic**

In `static/client-portal.html`, in the greeting/header render (where `pd-greeting`/the client name is shown), insert a photo block. Use the page's existing `token` const and `encodeURIComponent`:

```html
<div class="client-photo-box" style="display:flex;gap:14px;align-items:center;margin:0 0 14px">
  <img id="client-photo" alt="" style="width:88px;height:88px;object-fit:cover;border-radius:12px;display:none">
  <label class="btn" style="cursor:pointer;display:inline-block">
    <span id="client-photo-label">Add your photo</span>
    <input id="client-photo-file" type="file" accept="image/jpeg,image/png,image/webp" style="display:none">
  </label>
  <span id="client-photo-stat" class="dose"></span>
</div>
```

And a script (place with the page's other init code, after `token` is defined):

```javascript
(function initClientPhoto(){
  const img = document.getElementById("client-photo");
  const file = document.getElementById("client-photo-file");
  const label = document.getElementById("client-photo-label");
  const stat = document.getElementById("client-photo-stat");
  if (!img || !file) return;
  function show(){
    img.onload = function(){ img.style.display = "block"; if (label) label.textContent = "Change photo"; };
    img.onerror = function(){ img.style.display = "none"; };
    img.src = "/api/portal/" + encodeURIComponent(token) + "/photo?t=" + Date.now();
  }
  show();
  file.addEventListener("change", async function(){
    if (!file.files || !file.files[0]) return;
    stat.textContent = "Uploading…";
    const fd = new FormData(); fd.append("photo", file.files[0]);
    try {
      const r = await fetch("/api/portal/" + encodeURIComponent(token) + "/photo",
                            {method: "POST", body: fd, credentials: "same-origin"});
      const j = await r.json();
      if (j.ok) { stat.textContent = "Saved."; show(); }
      else { stat.textContent = j.error || "Upload failed"; }
    } catch (e) { stat.textContent = "Upload failed"; }
  });
})();
```

- [ ] **Step 2: Render-verify against the running app**

There is no JS test harness in this repo; verify by driving the real page (matches how Slice 1's portal-facing UI was verified):
1. Start the app locally (`doppler run -p remedy-match -c dev -- python3 app.py`) or use a prod portal token.
2. Open a client portal `/portal/<token>`; confirm the "Add your photo" control renders in the header and no broken image shows when there's no photo.
3. Pick a JPG/PNG → confirm it uploads (stat "Saved.") and the photo appears at 88px; reload → it persists (served from `GET /api/portal/<token>/photo`).
4. Confirm it now also shows on that client's intake page + reveal thumbnail (same store).

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "Client photos Slice 2: portal photo control (upload + display)"
```

---

## Self-Review

- **Spec coverage:** POST upload (Task 1) ✓; token-scoped GET serve (Task 1) ✓; portal header control (Task 2) ✓; `source="portal-self"` + precedence-overwrites-FMP (Task 1, via upsert) ✓; type/size caps (Task 1) ✓; token-scoped (never trust a client-passed email) (Task 1 tests) ✓. Moderation decision = trust + validate only (no code) — matches spec recommendation.
- **Placeholders:** none — every step has full code/commands.
- **Type consistency:** `put(cx, email, blob, content_type, source)` and `get(cx, email) -> {"blob","content_type"}` match Slice 1 `dashboard/client_photos.py`; `_portal_record_for(cx, token) -> dict|None` matches `app.py`.
- **Deferred:** precedence-aware `put` (Slice 3, first slice that must NOT overwrite); cropping/multiple photos (out of scope).
