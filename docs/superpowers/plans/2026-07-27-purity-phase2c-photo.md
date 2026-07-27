# Purity Phase 2c — operator label-photo fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator rate a product by uploading a photo of its label in the console — vision-extract the Other Ingredients (fabrication-guarded), screen, and record — as the fallback when online sources can't resolve a product.

**Architecture:** A single-product label photo has no cross-product-borrow risk, so this reuses the purity grounding WITHOUT the `_quote_near_anchor` step: the guard is `verify_quotes` against the model's OWN verbatim transcription of the label. A shared `_ground_oi(payload, source_text, anchors)` helper (anchors=None skips the anchor) backs both the existing online text extractor and the new image extractor. `acquire_from_image` returns the same `{raw, parsed, source, ok}` shape as `acquire`; a console upload route runs it OUTSIDE `_db_lock`, then screens + records exactly like `/api/console/purity/acquire`.

**Tech Stack:** Python 3.9 / Flask (sqlite dev, Postgres adapter prod), `anthropic` vision (already used by `document_extract`), multipart upload.

## Global Constraints

- **Same three outcomes as the online path:** a verified line (has excipients), `""` (verified explicit-none → green), or `None` (miss → unrated). Unrated-never-green preserved end to end.
- **Fabrication guard = `verify_quotes` against the model's own `label_text` transcription** (the #1172 discipline). NO `_quote_near_anchor` for the photo path (single product → no neighbor to borrow). A payload with no/blank `label_text` fails closed to `None` (the guard would have no haystack).
- **`_ground_oi` refactor must NOT change the online text-extractor behavior.** The text path passes its anchors list and must produce identical results — the existing `tests/test_document_extract_other_ingredients.py` must still pass unchanged.
- **Vision call runs OUTSIDE `_db_lock`** (tens of seconds); only `record_screen` is inside. Both DB connections that hit `product_ratings.get()` set `cx.row_factory = sqlite3.Row`.
- **Console-gated** (`_portal_console_ok()` → 401). Upload validated: `product_key` required (400), a file required (400), content-type in image/pdf allowlist (400), size ≤ 10 MB (400) — mirror `api_cert_upload` (app.py:29085).
- **Best-effort / never raises:** `acquire_from_image` is DB-free and returns the miss shape on any exception.
- **No new dependencies.** Tests inject `call_model` / monkeypatch `acquire_from_image` — no real vision calls, no real network.

---

### Task 1: shared `_ground_oi` + image extractor

**Files:**
- Modify: `dashboard/document_extract.py` (extract shared `_ground_oi`; refactor `extract_other_ingredients` to use it; add `extract_other_ingredients_from_image`, `_OTHER_ING_IMAGE_PROMPT`, `_default_call_model_image`)
- Test: `tests/test_document_extract_image.py` (create)

**Interfaces:**
- Consumes: existing `verify_quotes`, `_quote_near_anchor`, `_norm_text`, `_MODEL`, `json`, `os`.
- Produces:
  - `_ground_oi(payload, source_text, anchors) -> str | None` — resolves `{other_ingredients_line?, none_source_quote?}` to a verified line / `""` / `None`; `anchors=None` skips `_quote_near_anchor`.
  - `extract_other_ingredients_from_image(blob, content_type, *, name="", brand="", sku="", call_model=None) -> str | None` — vision extract for a label photo, grounded against the model's `label_text`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_document_extract_image.py
from dashboard import document_extract as dx

LABEL = ("Magnesium Taurate. Supplement Facts. Magnesium 100mg. "
         "Other Ingredients: microcrystalline cellulose, magnesium stearate, silicon dioxide. "
         "Keep out of reach of children.")


def test_image_returns_verified_line():
    def fake(blob, ct):
        return {"label_text": LABEL,
                "other_ingredients_line": "microcrystalline cellulose, magnesium stearate, silicon dioxide"}
    got = dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake)
    assert got == "microcrystalline cellulose, magnesium stearate, silicon dioxide"


def test_image_fabricated_line_fails_closed():
    def fake(blob, ct):
        return {"label_text": LABEL, "other_ingredients_line": "titanium dioxide and pharmaceutical glaze"}
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake) is None


def test_image_explicit_none_returns_empty():
    def fake(blob, ct):
        return {"label_text": "Creatine. Other Ingredients: None. Store cool.",
                "none_source_quote": "Other Ingredients: None"}
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=fake) == ""


def test_image_no_transcription_fails_closed():
    for bad in [{"other_ingredients_line": "magnesium stearate"}, {"label_text": ""}, {"label_text": 123}, None, "x"]:
        assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg",
                                                       call_model=lambda b, c, _b=bad: _b) is None


def test_image_model_error_fails_closed():
    def boom(blob, ct):
        raise RuntimeError("vision down")
    assert dx.extract_other_ingredients_from_image(b"img", "image/jpeg", call_model=boom) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_document_extract_image.py -v`
Expected: FAIL — `extract_other_ingredients_from_image` missing.

- [ ] **Step 3: Refactor + implement**

In `dashboard/document_extract.py`, FIRST add the shared helper immediately ABOVE `extract_other_ingredients`:

```python
def _ground_oi(payload, source_text, anchors):
    """Resolve a model payload {other_ingredients_line?, none_source_quote?} to a
    verified line / "" (verified explicit-none -> caller screens green) / None
    (miss -> unrated). Every non-None outcome must pass verify_quotes against
    source_text; when `anchors` is a list the quote must ALSO fall within a
    target anchor (_quote_near_anchor) -- for multi-product online pages.
    `anchors=None` skips the anchor: a single-product LABEL photo has no neighbor
    to borrow from, so verify_quotes against the label's own transcription is the
    whole guard."""
    line = payload.get("other_ingredients_line")
    if isinstance(line, str) and line.strip():
        kept, _d = verify_quotes([{"source_quote": line}], source_text)
        if kept and (anchors is None or _quote_near_anchor(source_text, line, anchors)):
            return line.strip()
        return None
    none_q = payload.get("none_source_quote")
    if isinstance(none_q, str) and none_q.strip():
        kept, _d = verify_quotes([{"source_quote": none_q}], source_text)
        nq = _norm_text(none_q)
        if (kept and "none" in nq and "ingredient" in nq
                and (anchors is None or _quote_near_anchor(source_text, none_q, anchors))):
            return ""
    return None
```

Then REPLACE the body of `extract_other_ingredients` AFTER its `if not isinstance(payload, dict): return None` line (i.e. replace the `anchors = [...]` block and everything below it through the final `return None`) with a single call:

```python
    return _ground_oi(payload, source_text, [name or "", brand or "", sku or "", slug or ""])
```

(The `call`, `try/except`, and `isinstance(payload, dict)` guard at the top of `extract_other_ingredients` stay exactly as they are.)

Then append the image extractor at the END of the file:

```python
_OTHER_ING_IMAGE_PROMPT = (
    "You are reading a photo of a single dietary supplement product label. "
    "Return STRICT JSON with:\n"
    '  "label_text": a VERBATIM transcription of ALL text visible on the label, '
    "exactly as printed. Every other value is checked against this, so any text "
    "not present here is discarded.\n"
    "AND ONE of:\n"
    '  "other_ingredients_line": the VERBATIM "Other Ingredients" (or '
    '"Non-Medicinal Ingredients") text copied exactly, WITHOUT the label word; '
    "OR\n"
    '  "none_source_quote": if the label explicitly states it has none (e.g. '
    '"Other Ingredients: None"), that verbatim declaration INCLUDING the word '
    '"None"; OR\n'
    '  "other_ingredients_line": "" if the label is unreadable or shows no '
    "other-ingredients section.\n"
    "Never invent an ingredient not printed on the label. No markdown fences, no "
    "prose outside the JSON."
)


def _default_call_model_image(blob, content_type):
    """Real Anthropic vision call for a supplement-label photo. Lazy-imported so
    tests that inject call_model never pull the SDK. Mirrors _default_call_model's
    image/PDF source shaping."""
    import base64
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    b64 = base64.standard_b64encode(blob).decode("ascii")
    if (content_type or "").lower() == "application/pdf":
        doc = {"type": "document", "source": {"type": "base64",
               "media_type": "application/pdf", "data": b64}}
    else:
        doc = {"type": "image", "source": {"type": "base64",
               "media_type": content_type, "data": b64}}
    resp = cli.messages.create(
        model=_MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": [doc, {"type": "text",
                                                     "text": _OTHER_ING_IMAGE_PROMPT}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def extract_other_ingredients_from_image(blob, content_type, *, name="", brand="",
                                         sku="", call_model=None):
    """Vision variant of extract_other_ingredients for a single-product LABEL
    photo. Returns a verified line / "" (verified explicit-none) / None (miss),
    grounded by verify_quotes against the model's OWN verbatim transcription
    (`label_text`) -- NO anchor, since a single label has no neighbor product to
    borrow from. Fails closed to None on a model error, non-dict reply, or a
    missing/blank/non-str transcription. name/brand/sku are accepted for parity
    with the text extractor but unused here (no anchor)."""
    call = call_model or _default_call_model_image
    try:
        payload = call(blob, content_type)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    src = payload.get("label_text")
    if not isinstance(src, str) or not src.strip():
        return None
    return _ground_oi(payload, src, None)
```

- [ ] **Step 4: Run the new tests AND the existing text-extractor tests (no regression)**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_document_extract_image.py tests/test_document_extract_other_ingredients.py -v`
Expected: ALL pass — the 5 new image tests AND every existing text-extractor test (the `_ground_oi` refactor is behavior-preserving for the text path).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/document_extract.py tests/test_document_extract_image.py
git commit -m "feat(purity): image label extractor + shared _ground_oi (no-anchor photo path)"
```

---

### Task 2: `purity_acquire.acquire_from_image`

**Files:**
- Modify: `dashboard/purity_acquire.py` (append `acquire_from_image`)
- Test: `tests/test_purity_acquire.py` (append)

**Interfaces:**
- Consumes: `document_extract.extract_other_ingredients_from_image` (Task 1), existing `split_other_ingredients`.
- Produces: `acquire_from_image(product, blob, content_type, *, call_model=None) -> {"raw","parsed","source","ok"}` with `source="photo"`. DB-free; never raises; miss → `{"raw":"","parsed":None,"source":"photo","ok":False}`; explicit-none → `{"raw":"None (no other ingredients listed)","parsed":[],...ok:True}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_acquire.py — append
_PHOTO_MISS = {"raw": "", "parsed": None, "source": "photo", "ok": False}


def test_acquire_from_image_line_screens():
    res = pa.acquire_from_image(
        {"name": "Mag Taurate", "brand": "X"}, b"img", "image/jpeg",
        call_model=lambda blob, ct: {"label_text": "Other Ingredients: magnesium stearate, silica",
                                     "other_ingredients_line": "magnesium stearate, silica"})
    assert res["ok"] is True and res["source"] == "photo"
    assert "magnesium stearate" in res["parsed"] and "silica" in res["parsed"]


def test_acquire_from_image_explicit_none_is_green():
    res = pa.acquire_from_image(
        {"name": "Creatine", "brand": "Thorne"}, b"img", "image/jpeg",
        call_model=lambda blob, ct: {"label_text": "Other Ingredients: None.",
                                     "none_source_quote": "Other Ingredients: None"})
    assert res["ok"] is True and res["parsed"] == [] and res["source"] == "photo"


def test_acquire_from_image_miss_is_unrated():
    res = pa.acquire_from_image(
        {"name": "X", "brand": "Y"}, b"img", "image/jpeg",
        call_model=lambda blob, ct: {"label_text": "unrelated text", "other_ingredients_line": ""})
    assert res == _PHOTO_MISS


def test_acquire_from_image_never_raises():
    res = pa.acquire_from_image({"name": "X"}, b"img", "image/jpeg",
                                call_model=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert res == _PHOTO_MISS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_acquire.py -k image -v`
Expected: FAIL — `acquire_from_image` missing.

- [ ] **Step 3: Implement**

Append to `dashboard/purity_acquire.py`:

```python
def acquire_from_image(product, blob, content_type, *, call_model=None):
    """Acquire Other Ingredients from a single-product LABEL photo. Same
    {raw,parsed,source,ok} contract as acquire(); source='photo'. DB-free; holds
    no lock; never raises. Miss -> parsed None -> caller screens unrated."""
    miss = {"raw": "", "parsed": None, "source": "photo", "ok": False}
    try:
        p = product or {}
        line = _dx.extract_other_ingredients_from_image(
            blob, content_type, name=p.get("name") or "", brand=p.get("brand") or "",
            sku=p.get("sku") or "", call_model=call_model)
        if line is None:
            return dict(miss)
        if line == "":
            return {"raw": "None (no other ingredients listed)", "parsed": [],
                    "source": "photo", "ok": True}
        parsed = split_other_ingredients(line)
        if not parsed:
            return dict(miss)
        return {"raw": line, "parsed": parsed, "source": "photo", "ok": True}
    except Exception:
        return dict(miss)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_acquire.py -v`
Expected: PASS (all — the 4 new image tests plus the existing acquire tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/purity_acquire.py tests/test_purity_acquire.py
git commit -m "feat(purity): acquire_from_image orchestrator (source=photo)"
```

---

### Task 3: console photo-upload route

**Files:**
- Modify: `app.py` (add `api_console_purity_acquire_photo` after `api_console_purity_acquire`, ~line 27997+)
- Test: `tests/test_purity_routes.py` (append)

**Interfaces:**
- Consumes: `purity_acquire.acquire_from_image` (Task 2), `purity_screen.screen_label`, `purity_avoidlist.load_avoidlist`, `product_ratings.record_screen`/`get`, `_portal_console_ok`, `_db_lock`, `db`, `sqlite3`, `request`, `jsonify`.
- Produces: `POST /api/console/purity/acquire-photo` (multipart) → `{"ok",True,"status","color","source","raw"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_routes.py — append. Uses this file's existing `client` fixture
# (console ok=True) and app_mod. Import io + purity_acquire at the top with the others.
import io
from dashboard import purity_acquire as _pa_mod


def _img(client_form):
    return {**client_form, "photo": (io.BytesIO(b"\xff\xd8fakejpeg"), "label.jpg")}


def test_acquire_photo_screens_red(client, monkeypatch):
    monkeypatch.setattr(_pa_mod, "acquire_from_image", lambda product, blob, ct, **k: {
        "raw": "magnesium stearate, silica", "parsed": ["magnesium stearate", "silica"],
        "source": "photo", "ok": True})
    r = client.post("/api/console/purity/acquire-photo",
                    data=_img({"product_key": "brand::prod", "product_name": "P", "brand": "B"}),
                    content_type="multipart/form-data")
    assert r.status_code == 200
    b = r.get_json()
    assert b["color"] == "red" and b["status"] == "screened" and b["source"] == "photo"


def test_acquire_photo_miss_unrated(client, monkeypatch):
    monkeypatch.setattr(_pa_mod, "acquire_from_image", lambda product, blob, ct, **k: {
        "raw": "", "parsed": None, "source": "photo", "ok": False})
    r = client.post("/api/console/purity/acquire-photo",
                    data=_img({"product_key": "brand::unk"}), content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["color"] is None and r.get_json()["status"] == "unrated"


def test_acquire_photo_requires_key(client):
    r = client.post("/api/console/purity/acquire-photo",
                    data=_img({}), content_type="multipart/form-data")
    assert r.status_code == 400


def test_acquire_photo_requires_file(client):
    r = client.post("/api/console/purity/acquire-photo",
                    data={"product_key": "k"}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_acquire_photo_rejects_bad_type(client):
    r = client.post("/api/console/purity/acquire-photo",
                    data={"product_key": "k", "photo": (io.BytesIO(b"x"), "note.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_acquire_photo_unauthorized(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: False)
    r = client.post("/api/console/purity/acquire-photo",
                    data=_img({"product_key": "k"}), content_type="multipart/form-data")
    assert r.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_routes.py -k photo -v`
Expected: FAIL — route 404.

- [ ] **Step 3: Implement**

Insert into `app.py` immediately after `api_console_purity_acquire` (before `@app.route("/api/console/purity/tier2"...)`):

```python
@app.route("/api/console/purity/acquire-photo", methods=["POST"])
def api_console_purity_acquire_photo():
    """Operator uploads a product LABEL photo; vision-extract the Other
    Ingredients, screen, and record. Phase-2c fallback for products online
    sources can't resolve. The vision call runs OUTSIDE _db_lock (same contract
    as /purity/acquire); only record_screen is inside. A miss screens 'unrated'
    (never green)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import (product_ratings as _pr, purity_screen as _ps,
                           purity_avoidlist as _pa, purity_acquire as _acq)
    key = (request.form.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    f = request.files.get("photo") or request.files.get("file")
    if not f or not (f.filename or "").strip():
        return jsonify({"error": "photo_required"}), 400
    ctype = (f.mimetype or "").lower()
    if ctype not in ("image/png", "image/jpeg", "image/jpg", "image/webp",
                     "image/gif", "application/pdf"):
        return jsonify({"error": "image_or_pdf_only"}), 400
    blob = f.read()
    if len(blob) > 10 * 1024 * 1024:
        return jsonify({"error": "file_too_large"}), 400
    brand = request.form.get("brand") or ""
    name = request.form.get("product_name") or ""
    # Slow vision call: OUTSIDE the lock.
    res = _acq.acquire_from_image({"name": name, "brand": brand}, blob, ctype)
    avoidlist = _pa.load_avoidlist()
    screen = _ps.screen_label(None, res["parsed"], avoidlist)   # parsed None -> unrated
    with _db_lock, db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_tables(cx)
        _pr.record_screen(cx, key, brand=brand, product_name=name,
                          other_ingredients_raw=res["raw"],
                          other_ingredients_parsed=(res["parsed"] or []), screen=screen)
        row = _pr.get(cx, key)
    return jsonify({"ok": True, "status": row["status"], "color": row["color"],
                    "source": res["source"], "raw": res["raw"]})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_routes.py -v`
Expected: PASS (all — the 6 new photo tests plus the existing route tests). If the file's console fixture is named differently, adapt to it (read the top of `tests/test_purity_routes.py`); do not invent a new auth mechanism.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add app.py tests/test_purity_routes.py
git commit -m "feat(purity): console /purity/acquire-photo operator label-photo route (Phase 2c)"
```

---

## Self-Review

**Spec coverage (Section 2 Source C, Phase 2c):**
- Operator console photo upload, full slice → Tasks 1-3. ✅
- Vision extract grounded by `verify_quotes` against the label transcription, NO anchor → Task 1 (`_ground_oi(..., None)` + `label_text` haystack). ✅
- Same 3 outcomes / unrated-never-green → Tasks 1-2 (`""`/`None`/line) + Task 3 (`screen_label(None, parsed)`). ✅
- Vision outside `_db_lock`, Postgres-safe, console-gated, upload-validated → Task 3. ✅
- Client-portal prompt = deferred (not in this plan). ✅
- `_ground_oi` refactor preserves the online text path → Task 1 Step 4 re-runs the text tests. ✅

**Placeholder scan:** none — complete code every step.

**Type consistency:** `_ground_oi(payload, source_text, anchors) -> str|None` consumed by both extractors; `extract_other_ingredients_from_image(blob, content_type, *, name, brand, sku, call_model) -> str|None` consumed by `acquire_from_image`, whose `{raw,parsed,source,ok}` (source="photo") is consumed by the route exactly as `acquire`'s shape is. `record_screen` kwargs match the online route. Consistent.
