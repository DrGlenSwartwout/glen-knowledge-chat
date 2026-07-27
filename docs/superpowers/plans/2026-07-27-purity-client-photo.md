# Purity 2c part 2 — client label-photo upload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a client's Fullscript card shows an unrated product, let them upload a label photo to have its purity checked — it vision-extracts, screens (lands `screened`, pending Glen's confirm), and the client sees a "thanks, under review" message (never an unconfirmed result).

**Architecture:** Reuses the operator photo core built in #1212 (`purity_acquire.acquire_from_image` + the guarded vision extractor). Adds a token-gated portal route `POST /api/portal/<token>/purity/photo` and a client-portal affordance shown only on unrated products when the badge flag is on. Everything is gated by the existing `PURITY_BADGES_ENABLED` (already on in prod — so this ships live on merge, but is inert until a card actually carries an unrated product, since all 24 are currently confirmed).

**Tech Stack:** Python 3.9 / Flask (sqlite dev, Postgres adapter prod), vanilla JS in `static/client-portal.html`, multipart upload.

## Global Constraints

- **Gated by `PURITY_BADGES_ENABLED`** (reuse, Glen's choice): the portal route returns 404 when the flag is off; the frontend affordance shows only when `fsData.purity_enabled` is true. One switch for the whole client purity experience.
- **Lands `screened`, never confirmed by the client.** A client upload runs the screen and `record_screen` (status `screened`) — it does NOT confirm. Glen's existing confirm gate is the only thing that makes a rating count or show a badge. The route returns NO color to the client ("thanks, under review" — Glen's choice).
- **`record_screen` never-downgrades a confirmed row** (existing behavior), so a client can never overwrite a confirmed rating.
- **Real products only.** The product must resolve via `fullscript.product_by_slug` (a real catalog product); otherwise 404. `product_key = "fullscript::" + slug` (the established convention).
- **Token identity only.** Client identity/authorization comes from `_portal_record_for(cx, token)` (404 if not found) — never from the request body.
- **Vision call OUTSIDE `_db_lock`** (tens of seconds); only `record_screen`/`get` inside; the write connection sets `cx.row_factory = sqlite3.Row`.
- Upload validated (mirror the operator route + `api_cert_upload`): file present (400), content-type in image allowlist (400), size ≤ 10 MB (400).
- **Unrated-never-green preserved:** a miss screens `unrated`, never green. No new dependency.

---

### Task 1: `purity_enabled` in payload + portal photo route

**Files:**
- Modify: `app.py` — add `"purity_enabled"` to `_fullscript_for`'s return; add `api_portal_purity_photo` near the other `/api/portal/<token>/purity/*` route (~line 28053 area)
- Test: `tests/test_purity_client_photo.py` (create)

**Interfaces:**
- Consumes: `_purity_badges_enabled()`, `purity_acquire.acquire_from_image`, `purity_screen.screen_label`, `purity_avoidlist.load_avoidlist`, `product_ratings.record_screen`/`get`, `fullscript.product_by_slug`, `_portal_record_for`, `_db_lock`, `db`, `sqlite3`, `request`, `jsonify`.
- Produces:
  - `_fullscript_for(...)` return dict gains `"purity_enabled": <bool>`.
  - `POST /api/portal/<token>/purity/photo` (multipart: `photo` file + `product_slug` form field) → `{"ok": True, "message": "Thanks — we'll review this and update your card."}` on success; 404 when flag off / token invalid / unknown product; 400 on missing/invalid file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_client_photo.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_client_photo.py -v`
Expected: FAIL — route 404 for the wrong reason / missing.

- [ ] **Step 3: Implement**

(a) In `_fullscript_for` (app.py ~line 23273), change the return to include `purity_enabled`:

```python
        _enrich_fullscript_purity(groups)
        return {"dispensary_url": _fullscript_dispensary_url(), "groups": groups,
                "purity_enabled": _purity_badges_enabled()}
```

(b) Add the portal route (place it right after `api_portal_purity_request`, ~line 28053+):

```python
@app.route("/api/portal/<token>/purity/photo", methods=["POST"])
def api_portal_purity_photo(token):
    """A client uploads a label photo for an UNRATED product on their card. The
    photo is vision-extracted, screened, and recorded as 'screened' (pending
    Glen's confirm -- the client never sees an unconfirmed color). Gated by
    PURITY_BADGES_ENABLED. Identity is the portal token only. The vision call
    runs OUTSIDE _db_lock; only record_screen is inside."""
    if not _purity_badges_enabled():
        return jsonify({"error": "not_available"}), 404
    from dashboard import (product_ratings as _pr, purity_screen as _ps,
                           purity_avoidlist as _pa, purity_acquire as _acq,
                           fullscript as _fs)
    slug = (request.form.get("product_slug") or "").strip()
    f = request.files.get("photo") or request.files.get("file")
    if not f or not (f.filename or "").strip():
        return jsonify({"error": "photo_required"}), 400
    ctype = (f.mimetype or "").lower()
    if ctype not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
        return jsonify({"error": "image_only"}), 400
    blob = f.read()
    if len(blob) > 10 * 1024 * 1024:
        return jsonify({"error": "file_too_large"}), 400
    # Authorize the token and resolve the product to a REAL catalog row (never
    # trust an arbitrary slug); read-only, own connection.
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _fs.init_tables(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not_found"}), 404
        prow = _fs.product_by_slug(cx, slug)
    if not prow:
        return jsonify({"error": "unknown_product"}), 404
    key = "fullscript::" + slug
    name, brand = prow.get("name") or "", prow.get("brand") or ""
    # Slow vision call OUTSIDE the lock.
    res = _acq.acquire_from_image({"name": name, "brand": brand}, blob, ctype)
    avoidlist = _pa.load_avoidlist()
    screen = _ps.screen_label(None, res["parsed"], avoidlist)   # parsed None -> unrated
    with _db_lock, db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_tables(cx)
        _pr.record_screen(cx, key, brand=brand, product_name=name,
                          other_ingredients_raw=res["raw"],
                          other_ingredients_parsed=(res["parsed"] or []), screen=screen)
    return jsonify({"ok": True,
                    "message": "Thanks — we'll review this and update your card."})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_client_photo.py -v`
Expected: PASS (6 passed). (The `test_fullscript_payload_has_purity_enabled` sanity test just asserts the function exists; the real payload-key coverage is the Task-2 render-verify + the route tests.)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add app.py tests/test_purity_client_photo.py
git commit -m "feat(purity): client portal label-photo upload route + purity_enabled payload flag"
```

---

### Task 2: client-portal upload affordance + handler

**Files:**
- Modify: `static/client-portal.html` — add the affordance in `fullscriptBodyHtml` (unrated products when `fsData.purity_enabled`) + a delegated change handler wired once
- Verify: headless render (controller; see Step 4)

**Interfaces:**
- Consumes: `fsData.purity_enabled` (Task 1), the per-product `p.purity` (absent = unrated), the module-level `token`, `esc`, `fetch`.
- Produces: for each unrated product (when enabled), a "📷 Have this? Add a label photo" file-input affordance + a status span; a single delegated `change` handler posts to `/api/portal/<token>/purity/photo` and shows the under-review message.

- [ ] **Step 1: Add the affordance to `fullscriptBodyHtml`**

In `static/client-portal.html`, inside `fullscriptBodyHtml`'s product `.map(p => …)`, after the existing badge/note computation and BEFORE the `return \`<li…\`` line, add:

```javascript
      // Phase 2c client photo: an unrated product (no confirmed color) invites a
      // label-photo upload when the purity feature is on. Absent p.purity = unrated.
      const canPhoto = fsData.purity_enabled && !p.purity && p.product_slug;
      const photo = canPhoto
        ? `<div class="small" style="margin:.2rem 0 .1rem"><label style="cursor:pointer;color:#2f6f5e">📷 Have this? Add a label photo to check its purity<input type="file" class="purity-photo" data-slug="${esc(p.product_slug)}" accept="image/jpeg,image/png,image/webp" style="display:none"></label> <span class="purity-photo-stat muted"></span></div>`
        : "";
```

Then append `${photo}` to the returned `<li>` (after `${why}${note}`):

```javascript
      return `<li${liStyle}>${badge}<a href="/fs/${fsToken}/${esc(p.product_slug||"")}" target="_blank" rel="noopener">${esc(p.name||"")}</a>${brand}${ff}${why}${note}${photo}</li>`;
```

- [ ] **Step 2: Add the delegated upload handler (wired once)**

Find where the page wires other one-time listeners (search for `addEventListener("change"` near the `client-photo-file` handler, ~line 3439). Add, at page-init scope (NOT inside `fullscriptBodyHtml`), a single delegated handler:

```javascript
  // Delegated once: any purity label-photo input (rendered per unrated Fullscript
  // product) uploads to the portal purity/photo route; the result is review-gated,
  // so the client only ever sees a thank-you, never an unconfirmed color.
  document.addEventListener("change", async function(ev){
    const inp = ev.target;
    if (!inp || !inp.classList || !inp.classList.contains("purity-photo")) return;
    if (!inp.files || !inp.files[0]) return;
    const stat = inp.parentElement && inp.parentElement.parentElement
      ? inp.parentElement.parentElement.querySelector(".purity-photo-stat") : null;
    if (stat) stat.textContent = " Uploading…";
    const fd = new FormData();
    fd.append("photo", inp.files[0]);
    fd.append("product_slug", inp.getAttribute("data-slug") || "");
    try {
      const r = await fetch("/api/portal/" + encodeURIComponent(token) + "/purity/photo",
                            {method: "POST", body: fd, credentials: "same-origin"});
      const j = await r.json();
      if (stat) stat.textContent = (j && j.ok) ? " Thanks — we'll review this." : " Upload failed.";
    } catch (e) { if (stat) stat.textContent = " Upload failed."; }
  });
```

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add static/client-portal.html
git commit -m "feat(purity): client-portal label-photo upload affordance for unrated products"
```

- [ ] **Step 4: Verify (controller, headless)**

Serve `static/` locally, load `client-portal.html`, set `window.token`, and eval `fullscriptBodyHtml(fsData)` with a synthetic payload where `purity_enabled:true` and a group has (a) a product WITH `purity:{color:"green"}` and (b) a product WITHOUT `purity` (unrated) and a `product_slug`. Assert the returned HTML string:
- contains a `class="purity-photo"` file input with `data-slug="<the unrated slug>"` AND the "Add a label photo" text — ONLY for the unrated product,
- does NOT render the affordance for the green (rated) product,
- renders NO affordance at all when `purity_enabled:false` (re-eval with it false).
Then insert the HTML into the DOM and confirm the affordance is visible for the unrated product.

---

## Self-Review

**Spec coverage (Source C client path):**
- Client uploads on an unrated product → Task 2 affordance (only when `purity_enabled && !p.purity`). ✅
- Vision extract → screen → **screened, not confirmed**; client sees "under review", no color → Task 1 route (returns message, no color; `record_screen` lands screened). ✅
- Reuses `PURITY_BADGES_ENABLED`; route 404 when off; affordance hidden when off → Task 1 (`purity_enabled` in payload + route gate). ✅
- Token identity only; real products only; vision outside lock; Postgres-safe → Task 1 route. ✅
- Never-green on miss; never overwrite a confirmed row → `screen_label(None,…)` + `record_screen` never-downgrade. ✅

**Placeholder scan:** none — complete code every step.

**Type consistency:** `purity_enabled` produced by Task 1 (`_fullscript_for` return) is read as `fsData.purity_enabled` in Task 2; `product_slug` form field posted by Task 2's handler is read by Task 1's route via `request.form.get("product_slug")`; the route resolves it with `product_by_slug` and keys `record_screen` as `fullscript::<slug>` (the established convention, matching the badge reader). Consistent.
