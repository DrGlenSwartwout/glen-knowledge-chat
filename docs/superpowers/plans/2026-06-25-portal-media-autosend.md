# Portal-Hosted Audio + PDF + Auto-Email — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "Publish to portal" host the client's audio mp3 + report PDF on prod, render them on the portal page (inline `<audio>` + PDF download), and auto-email the client the portal link.

**Architecture:** Two prod additions (a `/portal-asset` upload+serve route pair mirroring the existing `/clips` routes; `/api/portal` + `client-portal.html` rendering two new content fields `audio`/`report_pdf`) plus local-connector additions (`upload_asset`, opaque filenames, `build_portal_content` audio/pdf kwargs, a `send` param on `publish_to_portal`, and route wiring that uploads then publishes with `send=True`).

**Tech Stack:** Python 3.11, Flask, sqlite3, `requests`, pytest. Reuses `dashboard/biofield_report_pdf.report_pdf_bytes`, `dashboard/biofield_report_present.render_present`, `biofield_local_app.AUDIO_DIR`, and the existing `dashboard/biofield_portal_publish.py` module from PR #320.

## Global Constraints

- Prod file route mirrors `/clips` (app.py:16043-16068): console-gated (`X-Console-Key`==`CONSOLE_SECRET`), filename `^[\w\-]+\.(mp3|pdf)$`, served from `_PORTAL_ASSETS_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "portal-assets"`, base url `os.environ.get("RENDER_EXTERNAL_URL", "https://glen-knowledge-chat.onrender.com")`.
- `/api/portal` exposes `audio`/`report_pdf` ONLY when `bf_confirmed` (they name remedies).
- Asset filenames are opaque (`biofield-<16 hex>.<ext>`) — no PHI in the name.
- Email auto-sends via the existing upsert `send` path (fires only when a NEW token is minted = first publish; re-publish updates silently). Connector passes `send=True`.
- Missing audio file must NOT block publish (PDF + content still go).
- `app.py` cannot be imported offline (validates Pinecone at import) — Tasks 1 & 2's app.py routes are verified LIVE post-deploy via curl + headless render; their plan steps document the exact live checks instead of an offline pytest. The local-connector tasks (3 & 4) are fully offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: Prod `/portal-asset` upload + serve routes

**Files:**
- Modify: `app.py` (add near the `/clips` routes, ~line 16068, after `clips_delete`)

**Interfaces:**
- Produces: `PUT /portal-asset/upload?filename=<name>` → `{"ok":True,"url":"<base>/portal-asset/<name>"}`; `GET /portal-asset/<filename>` serving the bytes.

**Why no offline test:** `app.py` imports validate Pinecone over the network at import time, so it cannot be imported under pytest offline (the existing `/clips` routes are likewise untested in-repo). This task is verified live post-deploy (Step 3). Keep the code a faithful mirror of `/clips` so review is diff-only.

- [ ] **Step 1: Add the routes**

Insert into `app.py` after the `clips_delete` function (~line 16080):

```python
_PORTAL_ASSETS_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "portal-assets"
_PORTAL_ASSETS_DIR.mkdir(exist_ok=True)
_PORTAL_ASSET_RE = r'^[\w\-]+\.(mp3|pdf)$'
_PORTAL_ASSET_MIME = {"mp3": "audio/mpeg", "pdf": "application/pdf"}


@app.route("/portal-asset/upload", methods=["PUT"])
def portal_asset_upload():
    secret = request.headers.get("X-Console-Key", "")
    cs = os.environ.get("CONSOLE_SECRET", "")
    if cs and secret != cs:
        return jsonify({"error": "unauthorized"}), 401
    filename = request.args.get("filename", "")
    if not filename or not re.match(_PORTAL_ASSET_RE, filename):
        return jsonify({"error": "invalid filename (alphanumeric, hyphens, .mp3/.pdf only)"}), 400
    (_PORTAL_ASSETS_DIR / filename).write_bytes(request.data)
    base_url = os.environ.get("RENDER_EXTERNAL_URL", "https://glen-knowledge-chat.onrender.com")
    return jsonify({"ok": True, "url": f"{base_url}/portal-asset/{filename}"})


@app.route("/portal-asset/<filename>")
def portal_asset_serve(filename):
    m = re.match(_PORTAL_ASSET_RE, filename)
    if not m:
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(str(_PORTAL_ASSETS_DIR), filename,
                               mimetype=_PORTAL_ASSET_MIME[m.group(1)])
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat(portal-media): prod /portal-asset upload+serve routes (mp3/pdf)"
```

- [ ] **Step 3: Live verification (post-merge/deploy — record commands in the report, run at go-live)**

```bash
# upload (console key), expect 200 + url:
doppler run -p remedy-match -c prd -- sh -c 'printf "hello" | curl -s -X PUT "https://glen-knowledge-chat.onrender.com/portal-asset/upload?filename=test-pa.pdf" -H "X-Console-Key: $CONSOLE_SECRET" --data-binary @- '
# serve back, expect 200 application/pdf:
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" "https://glen-knowledge-chat.onrender.com/portal-asset/test-pa.pdf"
# no key -> 401:
curl -s -o /dev/null -w "%{http_code}\n" -X PUT "https://glen-knowledge-chat.onrender.com/portal-asset/upload?filename=x.pdf" --data-binary "x"
# bad ext -> 400:
doppler run -p remedy-match -c prd -- sh -c 'curl -s -o /dev/null -w "%{http_code}\n" -X PUT "https://glen-knowledge-chat.onrender.com/portal-asset/upload?filename=x.exe" -H "X-Console-Key: $CONSOLE_SECRET" --data-binary "x"'
```

---

### Task 2: Prod portal render — `/api/portal` passthrough + `client-portal.html` cards

**Files:**
- Modify: `app.py` (the `/api/portal/<token>` return dict, ~line 10891)
- Modify: `static/client-portal.html` (after the video card render, ~line 263)

**Interfaces:**
- Consumes: content fields `audio {url,label}` and `report_pdf {url}` produced by Task 3's `build_portal_content`.
- Produces: `/api/portal` payload keys `audio`, `report_pdf`; portal page renders an `<audio>` player + a PDF download button.

**Why no offline test:** same `app.py` import constraint as Task 1; the portal render is verified live by the headless render-verify at go-live (Step 4).

- [ ] **Step 1: Add the API passthrough**

In `app.py`, in the `/api/portal/<token>` return `jsonify({...})` (the dict starting ~line 10886), add these two keys right after the `"video": bf_content.get("video") or {},` line:

```python
        "audio": (bf_content.get("audio") or {}) if bf_confirmed else {},
        "report_pdf": (bf_content.get("report_pdf") or {}) if bf_confirmed else {},
```

- [ ] **Step 2: Add the portal-page cards**

In `static/client-portal.html`, immediately after the video-card block closes (the `}` at ~line 263, before the `// Cold portal:` comment), insert:

```javascript
  if(d.audio && d.audio.url){
    html += `
      <div class="card audio-card">
        <h2>Your audio walkthrough</h2>
        <audio controls preload="none" src="${esc(d.audio.url)}" style="width:100%"></audio>
        ${d.audio.label?`<p class="muted">${esc(d.audio.label)}</p>`:""}
      </div>`;
  }
  if(d.report_pdf && d.report_pdf.url){
    html += `
      <div class="card report-card">
        <h2>Your written report</h2>
        <a class="btn" href="${esc(d.report_pdf.url)}" target="_blank" rel="noopener" download>Download your report (PDF)</a>
      </div>`;
  }
```

(If `esc`, `class="card"`, `class="btn"`, or `muted` differ in this file, match the file's actual helpers/classes — read the surrounding render code. The video card above is the reference pattern.)

- [ ] **Step 3: Commit**

```bash
git add app.py static/client-portal.html
git commit -m "feat(portal-media): render audio player + PDF download on client portal"
```

- [ ] **Step 4: Live render-verify (post-deploy, at go-live — record in report)**

After a real publish (Task 4 wired), load `/portal/<token>` in a headless browser and assert: an `<audio>` element exists with a resolving `src` (network 200 on the mp3), the PDF download `<a>` exists and its href returns 200, and the page console has ZERO errors. (This is the project's mandatory render-verify, not an injection check.)

---

### Task 3: Connector module — asset upload, opaque names, content kwargs, send param

**Files:**
- Modify: `dashboard/biofield_portal_publish.py`
- Test: `tests/test_biofield_portal_publish_assets.py`

**Interfaces:**
- Consumes: existing `build_portal_content`, `publish_to_portal` (PR #320).
- Produces: `upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None) -> str`; `_asset_name(ext) -> str`; `build_portal_content(..., audio_url=None, report_pdf_url=None)`; `publish_to_portal(payload, *, base_url, console_key, send=False, http_post=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_assets.py
import re
import sqlite3
import pytest
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row

CATALOG = {"vitality": {"name": "Vitality"}}

class _Resp:
    def __init__(self, status, body):
        self.status_code = status; self._b = body
        import json as _j; self.text = _j.dumps(body)
    def json(self): return self._b

def test_asset_name_opaque_and_unique():
    a = bpp._asset_name("mp3"); b = bpp._asset_name("mp3")
    assert re.match(r'^biofield-[0-9a-f]{16}\.mp3$', a)
    assert bpp._asset_name("pdf").endswith(".pdf")
    assert a != b

def test_upload_asset_puts_bytes_with_key_and_returns_url():
    cap = {}
    def fake_put(url, data=None, headers=None, timeout=None):
        cap["url"] = url; cap["data"] = data; cap["headers"] = headers
        return _Resp(200, {"ok": True, "url": "https://h/portal-asset/biofield-x.pdf"})
    out = bpp.upload_asset(b"PDFBYTES", "biofield-x.pdf",
                           base_url="https://h", console_key="secret", http_put=fake_put)
    assert out == "https://h/portal-asset/biofield-x.pdf"
    assert cap["url"] == "https://h/portal-asset/upload?filename=biofield-x.pdf"
    assert cap["data"] == b"PDFBYTES"
    assert cap["headers"]["X-Console-Key"] == "secret"

def test_upload_asset_raises_on_non_2xx():
    def fake_put(url, data=None, headers=None, timeout=None):
        return _Resp(401, {"error": "unauthorized"})
    with pytest.raises(RuntimeError):
        bpp.upload_asset(b"x", "biofield-x.pdf", base_url="https://h",
                         console_key="bad", http_put=fake_put)

def test_build_content_includes_audio_and_pdf_when_urls_given():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "K", "k@example.com", "2026-06-25")
    add_chain_row(cx, f"a{tid}", layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    out = bpp.build_portal_content(cx, f"a{tid}", special_price_cents=5000, catalog=CATALOG,
                                   audio_url="https://h/portal-asset/a.mp3",
                                   report_pdf_url="https://h/portal-asset/r.pdf")
    c = out["content"]
    assert c["audio"] == {"url": "https://h/portal-asset/a.mp3", "label": "Listen to your walkthrough"}
    assert c["report_pdf"] == {"url": "https://h/portal-asset/r.pdf"}

def test_build_content_omits_audio_pdf_when_not_given():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "K", "k@example.com", "2026-06-25")
    add_chain_row(cx, f"a{tid}", layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    c = bpp.build_portal_content(cx, f"a{tid}", special_price_cents=5000, catalog=CATALOG)["content"]
    assert "audio" not in c and "report_pdf" not in c

def test_publish_send_param_controls_body():
    cap = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        cap["send"] = json.get("send"); return _Resp(200, {"ok": True, "url": "u"})
    bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://h",
                          console_key="s", send=True, http_post=fake_post)
    assert cap["send"] is True
    bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://h",
                          console_key="s", http_post=fake_post)
    assert cap["send"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_assets.py -v`
Expected: FAIL — `_asset_name`/`upload_asset` missing; `build_portal_content` rejects new kwargs; `publish_to_portal` rejects `send`.

- [ ] **Step 3: Implement**

In `dashboard/biofield_portal_publish.py`: add `import secrets` at the top. Add:

```python
def _asset_name(ext):
    return f"biofield-{secrets.token_hex(8)}.{ext}"


def upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None):
    """PUT raw bytes to the prod /portal-asset/upload; return the served url.
    Raises RuntimeError on non-2xx. http_put injectable (defaults requests.put)."""
    put = http_put or requests.put
    url = f"{base_url.rstrip('/')}/portal-asset/upload?filename={filename}"
    r = put(url, data=data_bytes, headers={"X-Console-Key": console_key}, timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"asset upload failed {r.status_code}: {r.text[:300]}")
    return r.json()["url"]
```

Change `publish_to_portal`'s signature to add `send=False` and use it (replace the hard-coded `"send": False`):

```python
def publish_to_portal(payload, *, base_url, console_key, send=False, http_post=None):
    post = http_post or requests.post
    url = f"{base_url.rstrip('/')}/admin/portal/upsert"
    body = {**payload, "send": bool(send)}
    r = post(url, json=body, headers={"X-Console-Key": console_key}, timeout=30)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"portal upsert failed {r.status_code}: {r.text[:300]}")
    return r.json()
```

In `build_portal_content`, add `audio_url=None, report_pdf_url=None` to the signature, and just before the `return`, after the `content = {...}` dict is built, insert:

```python
    if audio_url:
        content["audio"] = {"url": audio_url, "label": "Listen to your walkthrough"}
    if report_pdf_url:
        content["report_pdf"] = {"url": report_pdf_url}
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_assets.py -v`
Expected: PASS (6 tests). Also run the PR #320 suite to confirm no regression:
`~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_post.py tests/test_biofield_portal_publish_build.py tests/test_biofield_portal_publish_route.py -q` (the `send` default stays False, so route tests pass).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_assets.py
git commit -m "feat(portal-media): upload_asset + opaque names + content audio/pdf + send param"
```

---

### Task 4: Route wiring — upload assets, publish with send=True

**Files:**
- Modify: `biofield_local_app.py` (the `publish_portal` route added in PR #320)
- Test: `tests/test_biofield_portal_publish_route.py` (add cases)

**Interfaces:**
- Consumes: Task 3's `upload_asset`, `build_portal_content(audio_url=,report_pdf_url=)`, `publish_to_portal(send=)`; existing `render_present`, `report_pdf_bytes` (biofield_local_app imports at lines 44-45), `AUDIO_DIR` (line 47).

**Read first:** the current `publish_portal` route in `biofield_local_app.py` and the existing `tests/test_biofield_portal_publish_route.py` to match the seed/monkeypatch pattern.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_biofield_portal_publish_route.py`:

```python
def test_publish_route_uploads_assets_and_autosends(tmp_path, monkeypatch):
    import os as _os
    import biofield_local_app
    from dashboard import biofield_portal_publish as bpp
    from dashboard.biofield_authoring import create_test, add_chain_row
    db = str(tmp_path / "t.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    # a dummy audio file so the route finds one
    _os.makedirs(biofield_local_app.AUDIO_DIR, exist_ok=True)
    with open(_os.path.join(biofield_local_app.AUDIO_DIR, f"test_{aid}.mp3"), "wb") as f:
        f.write(b"ID3AUDIO")

    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    uploads = []
    monkeypatch.setattr(bpp, "upload_asset",
        lambda data, name, **kw: (uploads.append(name) or f"https://h/portal-asset/{name}"))
    captured = {}
    def fake_publish(payload, **kw):
        captured["content"] = payload["content"]; captured["send"] = kw.get("send")
        return {"ok": True, "url": "https://illtowell.com/portal/xyz", "emailed": True}
    monkeypatch.setattr(bpp, "publish_to_portal", fake_publish)
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")

    app = biofield_local_app.create_app(db_path=db)
    r = app.test_client().post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    assert captured["send"] is True
    assert captured["content"]["audio"]["url"].endswith(".mp3")
    assert captured["content"]["report_pdf"]["url"].endswith(".pdf")
    assert len(uploads) == 2     # pdf + mp3 both uploaded

def test_publish_route_missing_audio_still_publishes_pdf(tmp_path, monkeypatch):
    import biofield_local_app
    from dashboard import biofield_portal_publish as bpp
    from dashboard.biofield_authoring import create_test, add_chain_row
    db = str(tmp_path / "t2.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k2@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    # ensure NO audio file for this aid
    import os as _os
    p = _os.path.join(biofield_local_app.AUDIO_DIR, f"test_{aid}.mp3")
    if _os.path.exists(p): _os.remove(p)

    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "upload_asset", lambda data, name, **kw: f"https://h/portal-asset/{name}")
    captured = {}
    monkeypatch.setattr(bpp, "publish_to_portal",
        lambda payload, **kw: (captured.update(content=payload["content"]) or {"ok": True, "url": "u"}))
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    app = biofield_local_app.create_app(db_path=db)
    r = app.test_client().post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    assert "report_pdf" in captured["content"]
    assert "audio" not in captured["content"]    # no mp3 -> no audio field
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_route.py -v`
Expected: FAIL — the route doesn't upload assets / pass `send=True` yet.

- [ ] **Step 3: Implement the route changes**

In `biofield_local_app.py`, rewrite the body of the `publish_portal` route (keep the existing 400/409/500/502 structure) so that, after the 409 unresolved check and after reading `base`/`key`, it generates the PDF, uploads both assets, and publishes with `send=True`:

```python
    @app.route("/test/<test_id>/publish-portal", methods=["POST"])
    def publish_portal(test_id):
        from dashboard import biofield_portal_publish as _bpp
        body = request.get_json(silent=True) or {}
        try:
            special = int(body.get("special_price_cents") or 0)
        except (TypeError, ValueError):
            return {"ok": False, "error": "special_price_cents must be an integer"}, 400
        with sqlite3.connect(db_path) as cx:
            pre = _bpp.build_portal_content(cx, test_id, special_price_cents=special)
            if pre["unresolved"]:
                return {"ok": False, "unresolved": pre["unresolved"]}, 409
            # report HTML -> pdf bytes (reuse the report renderer)
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        base = os.environ.get("PORTAL_PUBLISH_BASE_URL", "")
        key = os.environ.get("CONSOLE_SECRET", "")
        if not base:
            return {"ok": False, "error": "PORTAL_PUBLISH_BASE_URL not set"}, 500
        try:
            pdf_bytes = report_pdf_bytes(render_present(rep, narrative))
            pdf_url = _bpp.upload_asset(pdf_bytes, _bpp._asset_name("pdf"),
                                        base_url=base, console_key=key)
            audio_url = None
            audio_path = os.path.join(AUDIO_DIR, f"test_{test_id}.mp3")
            if os.path.exists(audio_path):
                with open(audio_path, "rb") as af:
                    audio_url = _bpp.upload_asset(af.read(), _bpp._asset_name("mp3"),
                                                  base_url=base, console_key=key)
            with sqlite3.connect(db_path) as cx:
                payload = _bpp.build_portal_content(cx, test_id, special_price_cents=special,
                                                    audio_url=audio_url, report_pdf_url=pdf_url)
            res = _bpp.publish_to_portal(payload, base_url=base, console_key=key, send=True)
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}, 502
        return {"ok": True, "url": res.get("url", ""),
                "updated": bool(res.get("updated")), "note": res.get("note", ""),
                "emailed": bool(res.get("emailed")), "unresolved": []}
```

(Confirm `_report_for`, `get_narrative`, `render_present`, `report_pdf_bytes`, `AUDIO_DIR` are in scope in `biofield_local_app.py` — they are, from the imports at lines 44-47 and the `_report_for` helper used by the report routes. If `report_pdf_bytes` raises without a PDF backend, the route returns 502, which the tests don't hit because they monkeypatch nothing there — the tests DO call the real `report_pdf_bytes`; if that needs a browser, wrap pdf generation so tests can run: see Step 3b.)

- [ ] **Step 3b: Keep the route test offline-safe**

`report_pdf_bytes` uses Playwright (not in the test venv). To keep Task 4's tests offline, make the pdf-generation call indirect through a module-level name the test can monkeypatch, OR have the test monkeypatch `report_pdf_bytes` in `biofield_local_app`. Use the latter — add to BOTH route tests (Step 1), right after the other monkeypatches:

```python
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")
```

(This patches the symbol the route calls. The real PDF path is exercised at go-live, not in unit tests.)

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_route.py -v`
Expected: PASS (the original 4 + 2 new = 6).

- [ ] **Step 5: Run the whole connector suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_resolve.py tests/test_biofield_portal_publish_text.py tests/test_biofield_portal_publish_build.py tests/test_biofield_portal_publish_post.py tests/test_biofield_portal_publish_assets.py tests/test_biofield_portal_publish_route.py -v
git add biofield_local_app.py tests/test_biofield_portal_publish_route.py
git commit -m "feat(portal-media): publish route uploads audio+pdf and auto-sends link"
```

---

## Self-Review

**1. Spec coverage:**
- Prod hosting route (mp3/pdf, console-gated, opaque-served) → Task 1. ✅
- `/api/portal` audio/report_pdf passthrough gated on confirmed → Task 2 Step 1. ✅
- Portal page audio player + PDF download → Task 2 Step 2. ✅
- `upload_asset`, `_asset_name`, `build_portal_content` kwargs, `publish_to_portal` send param → Task 3. ✅
- Route uploads pdf (always) + audio (if present), publishes `send=True`, missing-audio non-blocking → Task 4. ✅
- Auto-email via existing upsert send path (first-publish only) → Task 4 passes `send=True`; behavior is the upsert's (unchanged). ✅
- Live curl + headless render-verify → Task 1 Step 3, Task 2 Step 4. ✅

**2. Placeholder scan:** No TBD/TODO. Tasks 1 & 2 have explicit "no offline test" rationale + concrete live commands (not vague). Step 3b explicitly resolves the Playwright-offline problem with a concrete monkeypatch. ✅

**3. Type consistency:** `upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None) -> str`; `_asset_name(ext)`; `build_portal_content(..., audio_url=None, report_pdf_url=None)`; `publish_to_portal(payload, *, base_url, console_key, send=False, http_post=None)` — identical across Tasks 3 & 4 and the route. Content keys `audio {url,label}` / `report_pdf {url}` match between Task 3 (producer), Task 2 (api passthrough + html consumer). ✅
