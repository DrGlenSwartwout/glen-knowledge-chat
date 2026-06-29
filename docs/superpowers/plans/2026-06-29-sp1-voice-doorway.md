# SP1 — Native Voice Doorway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the external ScoreApp quiz with a native `/begin` voice doorway where a visitor speaks, the existing journal pipeline reads them (TCM/elements/treasures/polyvagal/congruence + remedy match), the page reflects one hinge, and an opt-in captures the lead with full GHL parity.

**Architecture:** A new static page `static/begin-doorway.html` records mic audio and reuses the EXISTING endpoints `POST /journal/analyze` (full analysis) and `POST /journal/match` (remedies) — no new analysis backend. A new `POST /begin/doorway/opt-in` mirrors `/begin/quiz/opt-in` and ports the `/webhook/scoreapp` GHL/lead-logging behaviors. The funnel "quiz" land and the affiliate `healing.scoreapp.com` links are repointed to the internal `/begin/doorway`. The `/webhook/scoreapp` route is left in place but dormant.

**Tech Stack:** Python 3 / Flask (raw `@app.route`), sqlite3 (`chat_log.db`), vanilla-JS MediaRecorder frontend, pytest. Pipeline deps already present: OpenAI Whisper + ada-002, Anthropic Haiku, Pinecone.

## Global Constraints

- **Voice/copy rules (user-facing text in `begin-doorway.html`):** NO em dashes; no ALL CAPS; calm and consultative; lead with validation of the reader's lived experience. (Per `00 System/brand-dna/01-VOICE-AND-TONE.md`.)
- **App-import tests need real secrets + a writable DATA_DIR:** run as `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test python3 -m pytest <file> -v` (run `mkdir -p /tmp/sp1test` first). `import app` builds a Pinecone client at import (network 401 without a real key). Pure-module tests (`import begin_funnel`, `from dashboard.voice_doorway import ...`) run with plain `python3 -m pytest`.
- **Frontend changes MUST be render-verified** (headless browser asserting DOM + zero console errors), not just "served". (Per `feedback_render_verify_not_just_inject`.)
- **No direct-to-main:** a merge guard blocks pushes to `main`; integrate via PR + squash merge.
- **Reuse, don't reinvent:** the analysis is `POST /journal/analyze`; the remedy match is `POST /journal/match`. Do not re-implement Whisper/Haiku/Pinecone calls.
- **DB:** `LOG_DB = Path(os.environ.get("DATA_DIR", ...)) / "chat_log.db"`; writes go `with _db_lock, sqlite3.connect(LOG_DB) as cx:`.

---

## File Structure

- **Create** `dashboard/voice_doorway.py` — pure, dependency-free helpers (tag building) so they unit-test without importing `app.py`.
- **Create** `static/begin-doorway.html` — the doorway page (recorder → analyze → match → reflect → opt-in).
- **Create** `tests/test_voice_doorway.py` — pure-helper unit tests (plain pytest).
- **Create** `tests/test_begin_doorway_routes.py` — route tests (app import; doppler+DATA_DIR).
- **Modify** `app.py` — import `voice_doorway`; add `GET /begin/doorway` + `POST /begin/doorway/opt-in`; repoint `QUIZ_URL` (line 9028) and the affiliate URL builders (lines 8137, 8996, 9569) and the receipt literal (line 9274).
- **Modify** `begin_funnel.py` — repoint the "quiz" land: `WANT_TARGETS["quiz"]` (line 134) and `CARD_CATALOG["quiz"]` (lines 386–388) to internal `/begin/doorway`.

---

### Task 1: `voice_signal_tags` helper

**Files:**
- Create: `dashboard/voice_doorway.py`
- Test: `tests/test_voice_doorway.py`

**Interfaces:**
- Produces: `voice_signal_tags(signals: dict) -> list[str]` — builds GHL tags from a doorway analysis summary. `signals` keys: `dominant_element` (str), `dominant_treasure` (str), `polyvagal_state` (dict of state→0-100, or str), `top_themes` (list[str]). Returns e.g. `["element:water", "treasure:jing", "state:ventral-vagal", "theme:grounding"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice_doorway.py
from dashboard.voice_doorway import voice_signal_tags


def test_voice_signal_tags_builds_prefixed_tags():
    tags = voice_signal_tags({
        "dominant_element": "Water",
        "dominant_treasure": "Jing",
        "polyvagal_state": {"ventral_vagal": 60, "sympathetic": 28, "dorsal_vagal": 12},
        "top_themes": ["grounding", "kidney-essence depletion"],
    })
    assert "element:water" in tags
    assert "treasure:jing" in tags
    assert "state:ventral-vagal" in tags
    assert "theme:grounding" in tags
    assert "theme:kidney-essence-depletion" in tags


def test_voice_signal_tags_empty_is_empty():
    assert voice_signal_tags({}) == []


def test_voice_signal_tags_polyvagal_string_ok():
    assert "state:sympathetic" in voice_signal_tags({"polyvagal_state": "sympathetic"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_voice_doorway.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.voice_doorway'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/voice_doorway.py
"""Pure helpers for the /begin voice doorway (the native first-scan).
Dependency-free so tests run without importing app.py."""
import re


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def voice_signal_tags(signals: dict) -> list:
    """GHL tags from a doorway voice-scan analysis summary.
    signals: {dominant_element, dominant_treasure, polyvagal_state(dict|str), top_themes:[str]}."""
    signals = signals or {}
    tags = []
    el = _slug(signals.get("dominant_element"))
    if el:
        tags.append(f"element:{el}")
    tr = _slug(signals.get("dominant_treasure"))
    if tr:
        tags.append(f"treasure:{tr}")
    pv = signals.get("polyvagal_state")
    if isinstance(pv, dict) and pv:
        top = max(pv, key=lambda k: pv.get(k) or 0)
        if (pv.get(top) or 0) > 0:
            tags.append(f"state:{_slug(top)}")
    elif isinstance(pv, str) and pv.strip():
        tags.append(f"state:{_slug(pv)}")
    for t in (signals.get("top_themes") or [])[:5]:
        s = _slug(t)
        if s:
            tags.append(f"theme:{s}")
    return tags
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_voice_doorway.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/voice_doorway.py tests/test_voice_doorway.py
git commit -m "feat(doorway): voice_signal_tags helper (GHL tags from voice-scan signals)"
```

---

### Task 2: `POST /begin/doorway/opt-in` (lead-capture parity)

**Files:**
- Modify: `app.py` (add the route; import the helper near the other `from dashboard import ...` lines)
- Test: `tests/test_begin_doorway_routes.py`

**Interfaces:**
- Consumes: `voice_signal_tags` (Task 1); existing `ghl_onboard_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None) -> dict`; `begin_funnel.record_unlock(cx, *, session_id, trigger, email="", detail="", first_name="", last_name="", tos=False, ref_slug="", tos_version="")`; `_log_inbound_lead(source, email, first_name, last_name, phone, raw, ghl_result)`; `_capture_concierge_referral(email, first_name, last_name, ref_slug)`; `_mint_lead_magnet_guide_link(email)`; constants `LOG_DB`, `_db_lock`, `BEGIN_TOS_VERSION`.
- Produces: `POST /begin/doorway/opt-in` returning `{"ok": True, "current_rung": str, "guide_token": str}` (400 on bad email / missing tos).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_doorway_routes.py
import json, time
import pytest


@pytest.fixture
def appmod(monkeypatch):
    import app as appmod
    calls = {"unlocks": [], "ghl": []}

    def fake_record_unlock(cx, **kw):
        calls["unlocks"].append(kw)
        return {"current_rung": "assess", "email": kw.get("email", "")}

    def fake_ghl(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
        calls["ghl"].append({"email": email, "source_tag": source_tag, "tags": extra_tags or []})
        return {"contact_id": "c1"}

    monkeypatch.setattr(appmod.begin_funnel, "record_unlock", fake_record_unlock)
    monkeypatch.setattr(appmod, "ghl_onboard_contact", fake_ghl)
    monkeypatch.setattr(appmod, "_log_inbound_lead", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_capture_concierge_referral", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_mint_lead_magnet_guide_link", lambda email: "tok123")
    appmod._test_calls = calls
    return appmod


def test_doorway_optin_requires_email(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={"tos": True})
    assert r.status_code == 400


def test_doorway_optin_requires_tos(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={"email": "a@b.com", "tos": False})
    assert r.status_code == 400


def test_doorway_optin_captures_and_records_gates(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={
        "name": "Jane Doe", "email": "Jane@B.com", "tos": True,
        "signals": {"dominant_element": "Water", "top_themes": ["grounding"]},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["guide_token"] == "tok123"
    triggers = [u["trigger"] for u in appmod._test_calls["unlocks"]]
    assert "tos" in triggers and "quiz" in triggers
    # the GHL onboard runs in a daemon thread; allow it to flush
    for _ in range(50):
        if appmod._test_calls["ghl"]:
            break
        time.sleep(0.05)
    assert appmod._test_calls["ghl"], "ghl_onboard_contact should be called"
    g = appmod._test_calls["ghl"][0]
    assert g["source_tag"] == "source:voice"
    assert "voice-doorway" in g["tags"] and "element:water" in g["tags"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mkdir -p /tmp/sp1test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test python3 -m pytest tests/test_begin_doorway_routes.py::test_doorway_optin_captures_and_records_gates -v`
Expected: FAIL with 404 (route not defined)

- [ ] **Step 3: Write minimal implementation**

Add near the existing `from dashboard import ...` imports in `app.py`:

```python
from dashboard.voice_doorway import voice_signal_tags
```

Add the route (place it right after the `begin_quiz_optin` handler, ~line 1959 in `app.py`):

```python
@app.route("/begin/doorway/opt-in", methods=["POST", "OPTIONS"])
def begin_doorway_optin():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    parts = name.split(None, 1)
    first_name = parts[0] if parts else ""
    last_name = parts[1] if len(parts) > 1 else ""
    email = (data.get("email") or "").strip().lower()
    tos = bool(data.get("tos"))
    if not email or "@" not in email:
        return jsonify({"error": "valid email required"}), 400
    if not tos:
        return jsonify({"error": "tos required"}), 400
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip() or uuid.uuid4().hex)
    ref_slug = (request.cookies.get("rm_ref") or (data.get("ref") or "")).strip()
    signals = data.get("signals") or {}
    sig_tags = voice_signal_tags(signals)

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        state = begin_funnel.record_unlock(
            cx, session_id=session_id, trigger="tos", email=email,
            first_name=first_name, tos=True, ref_slug=ref_slug,
            tos_version=BEGIN_TOS_VERSION)
        begin_funnel.record_unlock(
            cx, session_id=session_id, trigger="quiz", email=email,
            detail="doorway:voice", ref_slug=ref_slug)

    import threading as _threading

    def _onboard():
        try:
            tags = ["begin", "voice-doorway"] + sig_tags
            if ref_slug:
                tags.append(f"ref:{ref_slug}")
                _capture_concierge_referral(email, first_name, last_name, ref_slug)
            ghl_result = ghl_onboard_contact(
                email, first_name, last_name,
                source_tag="source:voice", extra_tags=tags)
            _log_inbound_lead("voice", email, first_name, last_name, "",
                              json.dumps({"signals": signals}), ghl_result)
        except Exception as e:
            print(f"[doorway-optin] {e!r}", flush=True)

    _threading.Thread(target=_onboard, daemon=True).start()

    guide_token = _mint_lead_magnet_guide_link(email)
    resp = jsonify({"ok": True, "current_rung": state.get("current_rung"),
                    "guide_token": guide_token})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test python3 -m pytest tests/test_begin_doorway_routes.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_doorway_routes.py
git commit -m "feat(doorway): POST /begin/doorway/opt-in with GHL + lead-log parity"
```

---

### Task 3: `GET /begin/doorway` + `static/begin-doorway.html`

**Files:**
- Modify: `app.py` (add the `GET /begin/doorway` route after the opt-in route)
- Create: `static/begin-doorway.html`
- Test: render-verify (headless) — see Step 4.

**Interfaces:**
- Consumes: existing `POST /journal/analyze` (multipart `audio` + `duration_seconds`; returns `{id, transcript, dominant_element, treasure_scores, dominant_treasure, polyvagal_state, congruence, top_themes, top_emotions, ...}`); existing `POST /journal/match` (JSON `{transcript, dominant_element, dominant_treasure}` → `{matches:[{id,namespace,score,metadata}]}`); `POST /begin/doorway/opt-in` (Task 2).
- Produces: `GET /begin/doorway` serving the page.

- [ ] **Step 1: Add the route (no test-first; it serves a file)**

In `app.py`, right above `begin_doorway_optin`:

```python
@app.route("/begin/doorway")
def begin_doorway():
    resp = send_from_directory(STATIC, "begin-doorway.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 2: Build `static/begin-doorway.html`**

Build a single page with this structure (match the dark-green/gold funnel theme used by `static/begin-quiz.html`). REQUIRED element ids the JS below depends on: `#recBtn`, `#stopBtn`, `#status`, `#timer`, `#reflection` (hidden until analysis), `#optin` (hidden until reflection), `#nameInput`, `#emailInput`, `#tosChk`, `#optinBtn`, `#onward` (hidden until opt-in success). Copy must follow the Global Constraints voice rules (validation first, no em dashes). Opening copy frames the doorway: invite them to speak about what they are living with and what they want to heal.

Use this exact JavaScript (the recorder is adapted from `journal.html:950-1018`; the flow chains the existing endpoints):

```html
<script>
const ANALYZE_URL = '/journal/analyze';
const MATCH_URL   = '/journal/match';
const OPTIN_URL   = '/begin/doorway/opt-in';
const REF = new URLSearchParams(location.search).get('ref') || '';
let stream, mediaRecorder, chunks = [], startedAt = 0, timer = null, analysis = null;
const $ = id => document.getElementById(id);

async function record() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (e) { $('status').textContent = 'Mic blocked: ' + e.message; return; }
  chunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
    : (MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '');
  mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = onStopped;
  mediaRecorder.start();
  startedAt = Date.now();
  $('recBtn').disabled = true; $('stopBtn').disabled = false;
  $('status').textContent = 'Listening...';
  timer = setInterval(() => {
    const s = (Date.now() - startedAt) / 1000;
    $('timer').textContent = Math.floor(s/60) + ':' + String(Math.floor(s%60)).padStart(2,'0');
    if (s >= 300) stop();
  }, 200);
}

function stop() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  clearInterval(timer);
  $('recBtn').disabled = false; $('stopBtn').disabled = true;
}

async function onStopped() {
  const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
  const dur = (Date.now() - startedAt) / 1000;
  const form = new FormData();
  form.append('audio', blob, 'doorway.webm');
  form.append('duration_seconds', dur.toFixed(1));
  $('status').textContent = 'Hearing you...';
  try {
    const a = await fetch(ANALYZE_URL, { method: 'POST', body: form });
    if (!a.ok) { $('status').textContent = 'Could not read that. Please try again.'; return; }
    analysis = await a.json();
    let remedy = null;
    try {
      const m = await fetch(MATCH_URL, { method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ transcript: analysis.transcript,
          dominant_element: analysis.dominant_element, dominant_treasure: analysis.dominant_treasure }) });
      if (m.ok) { const md = await m.json(); remedy = (md.matches || [])[0] || null; }
    } catch (e) {}
    renderReflection(analysis, remedy);
  } catch (e) { $('status').textContent = 'Network error. Please try again.'; }
}

function renderReflection(a, remedy) {
  const themes = (a.top_themes || []).slice(0, 3);
  const remedyName = remedy && remedy.metadata
    ? (remedy.metadata.name || remedy.metadata.title || remedy.metadata.formulation || '') : '';
  $('reflection').innerHTML =
    '<p>Here is what I heard in your voice.</p>' +
    (themes.length ? '<ul>' + themes.map(t => '<li>' + t + '</li>').join('') + '</ul>' : '') +
    (a.dominant_element ? '<p>Your body is speaking most through the ' + a.dominant_element + ' element right now.</p>' : '') +
    (remedyName ? '<p>One place to begin: <strong>' + remedyName + '</strong>.</p>' : '');
  $('reflection').style.display = 'block';
  $('optin').style.display = 'block';
  $('status').textContent = '';
}

async function submitOptin() {
  const email = ($('emailInput').value || '').trim();
  if (!email || email.indexOf('@') < 0) { $('status').textContent = 'Please enter a valid email.'; return; }
  if (!$('tosChk').checked) { $('status').textContent = 'Please agree to continue.'; return; }
  const signals = {
    dominant_element: analysis && analysis.dominant_element,
    dominant_treasure: analysis && analysis.dominant_treasure,
    polyvagal_state: analysis && analysis.polyvagal_state,
    top_themes: analysis && analysis.top_themes,
  };
  const r = await fetch(OPTIN_URL, { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ name: $('nameInput').value, email, tos: true, ref: REF, signals }) });
  if (!r.ok) { $('status').textContent = 'Something went wrong. Please try again.'; return; }
  $('optin').style.display = 'none';
  $('onward').style.display = 'block';
}

window.addEventListener('DOMContentLoaded', () => {
  $('recBtn').addEventListener('click', record);
  $('stopBtn').addEventListener('click', stop);
  $('optinBtn').addEventListener('click', submitOptin);
});
</script>
```

- [ ] **Step 3: Commit the page + route**

```bash
git add app.py static/begin-doorway.html
git commit -m "feat(doorway): GET /begin/doorway page (record -> analyze -> match -> reflect -> opt-in)"
```

- [ ] **Step 4: Render-verify (headless)**

Start the app locally and headless-load the page, asserting the controls render with zero console errors. Run:

```bash
mkdir -p /tmp/sp1test
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test \
  gunicorn --worker-class gevent -b 127.0.0.1:8099 app:app &
sleep 4
python3 - <<'PY'
from playwright.sync_api import sync_playwright
errs = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8099/begin/doorway", wait_until="networkidle")
    assert pg.query_selector("#recBtn"), "record button missing"
    assert pg.query_selector("#emailInput"), "opt-in email missing"
    assert pg.query_selector("#tosChk"), "tos checkbox missing"
    b.close()
print("console errors:", errs)
assert not errs, f"console errors: {errs}"
print("RENDER OK")
PY
kill %1 2>/dev/null
```

Expected: `RENDER OK`, no console errors. (Gunicorn must use `--worker-class gevent`; a sync worker yields a chrome error per `feedback_chat_sse_test_seam`/render-verify notes.)

- [ ] **Step 5: Commit any render fixes**

```bash
git add -A && git commit -m "fix(doorway): render-verify pass for /begin/doorway"
```

---

### Task 4: Funnel cutover — repoint the "quiz" land to the doorway

**Files:**
- Modify: `begin_funnel.py:134` (`WANT_TARGETS["quiz"]`) and `begin_funnel.py:386-388` (`CARD_CATALOG["quiz"]`)
- Test: `tests/test_voice_doorway.py` (append; pure-module)

**Interfaces:**
- Produces: `begin_funnel.CARD_CATALOG["quiz"]["base_url"] == "/begin/doorway"`, `["internal"] is True`; `begin_funnel.WANT_TARGETS["quiz"] == "/begin/doorway"`.

- [ ] **Step 1: Write the failing test (append to `tests/test_voice_doorway.py`)**

```python
def test_quiz_land_repointed_to_internal_doorway():
    import importlib, begin_funnel
    importlib.reload(begin_funnel)
    card = begin_funnel.CARD_CATALOG["quiz"]
    assert card["base_url"] == "/begin/doorway"
    assert card["internal"] is True
    assert begin_funnel.WANT_TARGETS["quiz"] == "/begin/doorway"
    # no scoreapp URL left in the funnel land config
    import json as _j
    assert "scoreapp.com" not in _j.dumps(begin_funnel.CARD_CATALOG["quiz"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_voice_doorway.py::test_quiz_land_repointed_to_internal_doorway -v`
Expected: FAIL (base_url is still `https://healing.scoreapp.com`, internal is False)

- [ ] **Step 3: Edit `begin_funnel.py`**

Line 134, change:
```python
    "quiz":    "https://healing.scoreapp.com",
```
to:
```python
    "quiz":    "/begin/doorway",
```

Lines 386-388, change the `CARD_CATALOG["quiz"]` entry to:
```python
"quiz": {
    "title": "Speak with your guide",
    "sub": "Say what you are living with. In a minute, hear what your body is asking for.",
    "base_url": "/begin/doorway",
    "internal": True
},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_voice_doorway.py::test_quiz_land_repointed_to_internal_doorway -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add begin_funnel.py tests/test_voice_doorway.py
git commit -m "feat(doorway): repoint funnel quiz land to internal /begin/doorway"
```

---

### Task 5: Repoint affiliate / tracking URLs off ScoreApp

**Files:**
- Modify: `app.py:9028` (`QUIZ_URL`), `app.py:8137`, `app.py:8996`, `app.py:9569` (affiliate URL builders), `app.py:9274` (receipt literal)
- Test: `tests/test_begin_doorway_routes.py` (append) + a grep check

**Interfaces:**
- Produces: affiliate tracking URLs of the form `{PUBLIC_BASE_URL}/begin/doorway?ref=<slug>`; no `healing.scoreapp.com` remaining outside the dormant `/webhook/scoreapp` route.

- [ ] **Step 1: Edit `QUIZ_URL` (app.py:9028)**

```python
QUIZ_URL = f"{PUBLIC_BASE_URL}/begin/doorway"
```
(If `PUBLIC_BASE_URL` is not in scope at line 9028, use the same base constant the membership cancel-link uses; confirm with `grep -n "PUBLIC_BASE_URL" app.py | head -1`.)

- [ ] **Step 2: Edit the three builder sites to use `?ref=` instead of the scoreapp utm string**

- `app.py:8137` (affiliate_offers seed): replace the literal
  `'https://healing.scoreapp.com?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz'`
  with `QUIZ_URL + '?ref={slug}'`.
- `app.py:8996`: replace `f"https://healing.scoreapp.com?utm_source={utm_src}&utm_medium={utm_med}&utm_campaign={utm_camp}"`
  with `f"{QUIZ_URL}?ref={slug}"` (use the affiliate `slug` in scope; drop the utm params).
- `app.py:9569`: replace `f"{QUIZ_URL}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"`
  with `f"{QUIZ_URL}?ref={slug}"`.
- `app.py:9274` (receipt email body): replace the literal `"https://healing.scoreapp.com"` with `QUIZ_URL`.

- [ ] **Step 3: Write the verification test (append to `tests/test_begin_doorway_routes.py`)**

```python
def test_no_scoreapp_url_outside_dormant_webhook():
    import re, pathlib
    src = pathlib.Path("app.py").read_text()
    # allow the route name/handler to mention scoreapp, but no built URLs
    bad = [ln for ln in src.splitlines()
           if "healing.scoreapp.com" in ln]
    assert bad == [], f"scoreapp URLs still present: {bad}"
```

- [ ] **Step 4: Run + grep check**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test python3 -m pytest tests/test_begin_doorway_routes.py::test_no_scoreapp_url_outside_dormant_webhook -v`
Then: `grep -n "healing.scoreapp.com" app.py begin_funnel.py` → expected: no output.
Expected: PASS, grep empty.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_doorway_routes.py
git commit -m "feat(doorway): repoint affiliate/receipt URLs from ScoreApp to internal doorway"
```

---

### Task 6: Whole-branch verification + go-live notes

**Files:** none (verification only)

- [ ] **Step 1: Run the full new test set**

```bash
mkdir -p /tmp/sp1test
python3 -m pytest tests/test_voice_doorway.py -v
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/sp1test python3 -m pytest tests/test_begin_doorway_routes.py -v
```
Expected: all pass.

- [ ] **Step 2: Whole-branch review**

Read the full diff (`git diff origin/main..HEAD`). Confirm: opt-in mirrors the quiz opt-in parity (tos + quiz gates, threaded GHL, cookie); `voice_signal_tags` is the only new pure helper; the funnel land + affiliate URLs are internal; `/webhook/scoreapp` is untouched (dormant). Check `feedback_enum_value_blast_radius` is not triggered (no enum/status set changes here).

- [ ] **Step 3: Render-verify the live funnel path**

Re-run the Task 3 render-verify, and additionally confirm `GET /begin/state` (or the funnel) returns the quiz land as `internal: true` pointing at `/begin/doorway`:
```bash
curl -s http://127.0.0.1:8099/begin/quiz-data >/dev/null   # quiz engine untouched, still 200
```
(The native eye-brain quiz routes remain; they are simply no longer the funnel's "quiz" land.)

- [ ] **Step 4: Open the PR (do not direct-push to main)**

```bash
git push -u origin <branch>
gh pr create --base main --title "SP1: native voice doorway (replaces ScoreApp quiz)" \
  --body "Native /begin/doorway voice scan (reuses /journal/analyze + /journal/match), opt-in with GHL+lead parity, funnel + affiliate cutover off ScoreApp. /webhook/scoreapp left dormant."
```

- [ ] **Step 5: Go-live checklist (post-merge, in the PR body)**

Document for Glen: (a) render-verify on prod `illtowell.com/begin/doorway` after deploy; (b) confirm a real opt-in writes a GHL contact with `source:voice` + `element:*` tags; (c) the funnel "quiz" card now opens the doorway in-app; (d) SP1.5 = voice-out in Glen's ElevenLabs clone; SP2 = multi-turn ally + persistent ASH map; the proven Track B (real E4L scan) is the next scan upgrade.

---

## Self-Review

**Spec coverage:** voice-in + native scan (Task 3 reuses `/journal/analyze`+`/journal/match`); reflect one hinge (Task 3 `renderReflection`); capture with lead parity (Task 2); funnel cutover (Tasks 4–5); leave webhook dormant (Task 6 review). Voice-out and multi-turn ally are explicitly deferred to SP1.5/SP2 (spec §9–10). Covered.

**Placeholder scan:** every code step has full code; no TBD/TODO. Render-verify and test commands are concrete.

**Type consistency:** `voice_signal_tags(signals: dict) -> list` used identically in Task 1 (def) and Task 2 (call). The analysis dict keys consumed in Task 3 (`dominant_element`, `top_themes`, `polyvagal_state`, `dominant_treasure`, `transcript`) match the `/journal/analyze` response schema from the interface extraction.

**Open verifications for the implementer (not blockers):** confirm `PUBLIC_BASE_URL` is in scope at `app.py:9028` (Task 5 Step 1 gives the grep); confirm the affiliate `slug` variable name in scope at `app.py:8996` matches the surrounding function. Both are 1-line confirmations against the live file.
