# Chat Creates-a-Page Offer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a `/begin` chat is about a health topic with no page yet, offer to build a guide; the person's accept records a *suggestion* Glen reviews and builds through the existing topic-page pipeline.

**Architecture:** Reuse the `topic_pages` table with a new `suggested` state (no new table). A cheap haiku call extracts a normalized `{name, kind, slug}` candidate in the `/chat` done handler; if no page exists, an offer card (merged via the existing `merge_cards`) links to a `/learn/suggest/<slug>` form that records the suggestion + asker email. A console queue ranks suggestions by demand; **Build** reuses `topic_page.regenerate`, **Dismiss** is the one new action. Dark behind `CHAT_TOPIC_OFFER_ENABLED`.

**Tech Stack:** Python 3, Flask, sqlite3 (`LOG_DB`), the shared `_cl` haiku client, pytest. No new deps. Reuses the `{key,title,sub,href}` card shape (no client change) and `dashboard/page_links.merge_cards`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-21-chat-topic-offer-design.md`. Every task inherits its requirements.
- **Never `import app` in tests** — Pinecone builds at import and fails in the sandbox. Test `dashboard/*` directly; verify `app.py` via `python3 -m py_compile app.py`. Use `python3`, not `python`.
- A suggestion is a `topic_pages` row in state `suggested` (no new table); askers go in the existing `topic_page_requests`. `record_suggestion` MUST NOT downgrade an `approved`/`draft`/`gated` row.
- AI helpers follow the existing contract: one synchronous haiku call, **never raise**, safe default on any error. `extract_topic_candidate` returns `None` on empty/bad/error.
- `kind` is one of `("symptom","condition","function")`; `slug = dashboard.ingredients.slugify(name)`.
- Public path serves **approved-only** (unchanged); `suggested`/`dismissed` rows never leak (the public `/learn/<slug>` and the page-links index already gate on `approved`).
- Nothing publishes without Glen building **and** approving; the compliance gate is unchanged.
- Ships **DARK behind `CHAT_TOPIC_OFFER_ENABLED`** (default off). Flag off → no extraction, no offer, no routes.
- All new chat-path work wrapped in try/except so it can never break the chat stream; hrefs built from the normalized slug only.
- Card shape: `{"key": f"suggest:{slug}", "title": ..., "sub": ..., "href": ...}`.

---

### Task 1: `topic_copy.extract_topic_candidate`

**Files:**
- Modify: `dashboard/topic_copy.py` (append a function)
- Test: `tests/test_topic_copy.py` (append tests)

**Interfaces:**
- Produces: `extract_topic_candidate(query, answer, client) -> {"name","kind","slug"} | None` — one haiku call; returns `None` when there is no single clear health topic, on a bad/unknown kind, on bad JSON, or on any error. `slug = dashboard.ingredients.slugify(name)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_topic_copy.py`:

```python
# --- extract_topic_candidate (chat creates-a-page offer) ---

class _FakeJSONClient:
    def __init__(self, text):
        self._text = text
        self.messages = self
    def create(self, **kw):
        class _C:  # minimal content block
            def __init__(s, t): s.text = t; s.type = "text"
        class _M:
            def __init__(s, t): s.content = [_C(t)]
        return _M(self._text)


def test_extract_returns_normalized_candidate():
    tc = _mod()
    client = _FakeJSONClient('{"name": "Magnesium Deficiency", "kind": "condition"}')
    out = tc.extract_topic_candidate("I think I'm low on magnesium", "You may be deficient.", client)
    assert out == {"name": "Magnesium Deficiency", "kind": "condition", "slug": "magnesium-deficiency"}


def test_extract_none_when_no_topic():
    tc = _mod()
    client = _FakeJSONClient("{}")
    assert tc.extract_topic_candidate("hello", "hi there", client) is None


def test_extract_none_on_bad_kind():
    tc = _mod()
    client = _FakeJSONClient('{"name": "Stuff", "kind": "banana"}')
    assert tc.extract_topic_candidate("q", "a", client) is None


def test_extract_none_on_bad_json():
    tc = _mod()
    client = _FakeJSONClient("not json at all")
    assert tc.extract_topic_candidate("q", "a", client) is None


def test_extract_none_on_client_error():
    tc = _mod()
    class _Boom:
        messages = None
        def create(self, **kw): raise RuntimeError("api down")
    boom = _Boom(); boom.messages = boom
    assert tc.extract_topic_candidate("q", "a", boom) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q -k extract`
Expected: FAIL — `AttributeError: module 'dashboard.topic_copy' has no attribute 'extract_topic_candidate'`.

- [ ] **Step 3: Implement**

Append to `dashboard/topic_copy.py`:

```python
def extract_topic_candidate(query, answer, client):
    """Name the single health topic a conversation is about, for the create-a-page offer.

    Returns {"name","kind","slug"} or None. One haiku call; never raises.
    """
    try:
        from dashboard import ingredients as _ingredients
        system = (
            "You decide whether a chat is centrally about ONE health topic a person would search "
            "for (a symptom, a named condition, or a physiological function). Return ONLY JSON. "
            "If yes: {\"name\": \"Title Case Topic\", \"kind\": \"symptom|condition|function\"}. "
            "If it is small talk, multiple unrelated topics, or not health, return {}. "
            "No commentary, no markdown."
        )
        user = f"User: {query}\n\nAssistant answer: {answer[:600]}\n\nReturn the JSON now."
        msg = client.messages.create(model=_MODEL, max_tokens=120, system=system,
                                     messages=[{"role": "user", "content": user}])
        raw = _text_of(msg)
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
        data = json.loads(raw)
        name = (data.get("name") or "").strip()
        kind = (data.get("kind") or "").strip().lower()
        if not name or kind not in ("symptom", "condition", "function"):
            return None
        return {"name": name, "kind": kind, "slug": _ingredients.slugify(name)}
    except Exception as exc:  # noqa: BLE001 - never raises
        print(f"[topic-copy] extract_topic_candidate failed: {exc}", flush=True)
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py -q`
Expected: PASS (existing + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_copy.py tests/test_topic_copy.py
git commit -m "feat(topic-offer): extract_topic_candidate (AI-normalize the chat topic)"
```

---

### Task 2: `topic_pages.record_suggestion` + `list_suggestions`

**Files:**
- Modify: `dashboard/topic_pages.py` (append two functions)
- Test: `tests/test_topic_suggestions_store.py`

**Interfaces:**
- Consumes: existing `get_page`, `set_name`, `set_kind`, `set_state`, `record_request`.
- Produces:
  - `record_suggestion(cx, slug, name, kind, email) -> str` — returns resulting state. Never downgrades an `approved`/`draft`/`gated` row (records the request only); otherwise upserts name/kind + `state="suggested"` + records the request.
  - `list_suggestions(cx) -> list[dict]` — rows with `state=="suggested"`, each `{slug, name, kind, demand}` (demand = count of `topic_page_requests` for the slug), ordered by demand desc.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_suggestions_store.py
import sqlite3, sys
from pathlib import Path
import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_record_new_suggestion_sets_suggested_and_records_request():
    tp = _mod(); cx = _cx(tp)
    st = tp.record_suggestion(cx, "magnesium-deficiency", "Magnesium Deficiency", "condition", "a@x.com")
    assert st == "suggested"
    page = tp.get_page(cx, "magnesium-deficiency")
    assert page["state"] == "suggested" and page["kind"] == "condition" and page["name"] == "Magnesium Deficiency"
    assert tp.requesters_to_email(cx, "magnesium-deficiency")  # one asker recorded


def test_second_asker_increments_demand_keeps_suggested():
    tp = _mod(); cx = _cx(tp)
    tp.record_suggestion(cx, "dry-skin", "Dry Skin", "symptom", "a@x.com")
    st = tp.record_suggestion(cx, "dry-skin", "Dry Skin", "symptom", "b@x.com")
    assert st == "suggested"
    rows = [r for r in tp.list_suggestions(cx) if r["slug"] == "dry-skin"]
    assert rows and rows[0]["demand"] == 2


def test_record_does_not_downgrade_existing_pipeline_row():
    tp = _mod(); cx = _cx(tp)
    tp.upsert_section(cx, "low-energy", "overview", "x")
    tp.set_state(cx, "low-energy", "approved", by="glen")
    st = tp.record_suggestion(cx, "low-energy", "Low Energy", "symptom", "c@x.com")
    assert st == "approved"
    assert tp.get_page(cx, "low-energy")["state"] == "approved"  # not downgraded
    assert tp.requesters_to_email(cx, "low-energy")             # request still recorded


def test_list_suggestions_only_suggested_ordered_by_demand():
    tp = _mod(); cx = _cx(tp)
    tp.record_suggestion(cx, "a-topic", "A Topic", "function", "1@x.com")
    tp.record_suggestion(cx, "b-topic", "B Topic", "function", "1@x.com")
    tp.record_suggestion(cx, "b-topic", "B Topic", "function", "2@x.com")
    tp.upsert_section(cx, "c-topic", "overview", "x")  # a non-suggested row must not appear
    out = tp.list_suggestions(cx)
    slugs = [r["slug"] for r in out]
    assert slugs[:2] == ["b-topic", "a-topic"]   # b has demand 2, a has 1
    assert "c-topic" not in slugs
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_suggestions_store.py -q`
Expected: FAIL — `record_suggestion`/`list_suggestions` not defined.

- [ ] **Step 3: Implement**

Append to `dashboard/topic_pages.py`:

```python
def record_suggestion(cx, slug, name, kind, email):
    """Record a create-a-page suggestion. Never downgrades a row already in the build pipeline."""
    init_table(cx)
    page = get_page(cx, slug)
    state = (page or {}).get("state")
    if state in ("approved", "draft", "gated"):
        record_request(cx, slug, email)
        return state
    set_name(cx, slug, name)
    set_kind(cx, slug, kind)
    set_state(cx, slug, "suggested")
    record_request(cx, slug, email)
    return "suggested"


def list_suggestions(cx):
    """Suggested rows with demand counts, highest demand first."""
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT p.slug, p.name, p.kind, "
        "  (SELECT COUNT(*) FROM topic_page_requests r WHERE r.slug=p.slug) AS demand "
        "FROM topic_pages p WHERE p.state='suggested' "
        "ORDER BY demand DESC, p.updated_at DESC"
    ).fetchall()
    return [{"slug": r["slug"], "name": r["name"] or r["slug"],
             "kind": r["kind"] or "", "demand": r["demand"]} for r in rows]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_suggestions_store.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_pages.py tests/test_topic_suggestions_store.py
git commit -m "feat(topic-offer): record_suggestion + list_suggestions (suggested state, no new table)"
```

---

### Task 3: `topic_render.render_suggest_html`

**Files:**
- Modify: `dashboard/topic_render.py` (append a function)
- Test: `tests/test_topic_render.py` (append tests)

**Interfaces:**
- Consumes: existing `_document`, `_esc`.
- Produces: `render_suggest_html(slug, name, *, submitted=False) -> str` — styled, `noindex`. Form variant posts to `/learn/suggest/<slug>`; submitted variant shows a confirmation and no form.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic_render.py`:

```python
def test_suggest_form_posts_and_is_noindex():
    tr = _mod()
    html = tr.render_suggest_html("magnesium-deficiency", "Magnesium Deficiency")
    assert 'action="/learn/suggest/magnesium-deficiency"' in html
    assert 'name="robots" content="noindex"' in html
    assert "Magnesium Deficiency" in html
    assert "brand-name" in html  # site chrome


def test_suggest_submitted_shows_confirmation_no_form():
    tr = _mod()
    html = tr.render_suggest_html("dry-skin", "Dry Skin", submitted=True)
    assert "<form" not in html
    assert "email you" in html.lower()


def test_suggest_escapes_name():
    tr = _mod()
    html = tr.render_suggest_html("x", "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_render.py -q -k suggest`
Expected: FAIL — `render_suggest_html` not defined.

- [ ] **Step 3: Implement**

Append to `dashboard/topic_render.py`:

```python
def render_suggest_html(slug, name, *, submitted=False):
    """Styled, noindex page: offer-to-build form, or a post-submit confirmation."""
    if submitted:
        body_inner = (
            f"<main><h1>{_esc(name)}</h1>"
            "<p>Thank you. We will create this guide and email you when it is ready.</p>"
            '<p><a class="cta-btn" href="/begin">Continue exploring</a></p>'
            "</main>"
        )
    else:
        body_inner = (
            f"<main><h1>{_esc(name)}</h1>"
            f"<p>We do not have a guide on {_esc(name)} yet. Leave your email and we will create "
            "one and send it to you.</p>"
            f'<form method="post" action="/learn/suggest/{_esc(slug)}">'
            '<input type="email" name="email" required placeholder="you@example.com">'
            '<button class="cta-btn" type="submit">Yes, create this guide</button></form>'
            "</main>"
        )
    return _document(name, None, "", body_inner, noindex=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_render.py -q`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_render.py tests/test_topic_render.py
git commit -m "feat(topic-offer): render_suggest_html (styled offer form + confirmation)"
```

---

### Task 4: `topic_page.dismiss` console action

**Files:**
- Modify: `dashboard/topic_page_actions.py` (add executor + register)
- Test: `tests/test_topic_page_actions.py` (append a test)

**Interfaces:**
- Consumes: existing `topic_pages.set_state`, the `Action`/`register_action`/`get_action` spine.
- Produces: action key `topic_page.dismiss` — sets state `"dismissed"`; `OWNER/OPS`, `LOW_WRITE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic_page_actions.py`:

```python
def test_dismiss_sets_state_dismissed_and_drops_from_suggestions():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.record_suggestion(cx, "junk-topic", "Junk Topic", "symptom", "a@x.com")
    assert any(r["slug"] == "junk-topic" for r in tp.list_suggestions(cx))
    res = _get_action("topic_page.dismiss").executor({"slug": "junk-topic"}, {"cx": cx, "actor": _actor()})
    assert res["state"] == "dismissed"
    assert tp.get_page(cx, "junk-topic")["state"] == "dismissed"
    assert not any(r["slug"] == "junk-topic" for r in tp.list_suggestions(cx))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_page_actions.py -q -k dismiss`
Expected: FAIL — no `topic_page.dismiss` action registered.

- [ ] **Step 3: Implement**

In `dashboard/topic_page_actions.py`, add the executor near the others:

```python
def _exec_dismiss(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _tp.set_state(ctx["cx"], slug, "dismissed", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "dismissed"}
```

and in `register()`, after the existing `register_action(...)` calls, add:

```python
    register_action(Action(
        key="topic_page.dismiss", module="topic_pages", title="Dismiss topic suggestion",
        description="Drop a create-a-page suggestion (sets it dismissed; never public).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_dismiss))
```

(The `register()` guard `if get_action("topic_page.approve"): return` already prevents double-registration of the whole set.)

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_page_actions.py -q`
Expected: PASS (existing + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add dashboard/topic_page_actions.py tests/test_topic_page_actions.py
git commit -m "feat(topic-offer): topic_page.dismiss action"
```

---

### Task 5: `app.py` wiring — flag, /chat offer, /learn/suggest, console queue

**Files:**
- Modify: `app.py` (flag near `TOPIC_PAGES_ENABLED`; `/chat` offer block after the page-links merge; `/learn/suggest` routes near the other `/learn` routes; console route + api near the topic-pages console; register-already-covered by Task 4)
- Create: `static/console-topic-suggestions.html` (clone of `static/console-topic-pages.html`)
- Modify: `static/console-search-index.json`
- Test: `tests/test_topic_offer_wiring.py`

**Interfaces:**
- Consumes: `topic_copy.extract_topic_candidate`, `topic_pages.record_suggestion`/`list_suggestions`, `topic_render.render_suggest_html`, `page_links.merge_cards`; existing `query`/`answer`/`surfaced_cards`/`_cl`/`_db_lock`/`LOG_DB`/`_sales_console_ok`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_offer_wiring.py
import json, subprocess, sys
from pathlib import Path


def _repo():
    return Path(__file__).resolve().parent.parent


def test_app_compiles():
    r = subprocess.run([sys.executable, "-m", "py_compile", str(_repo() / "app.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_flag_present_default_off():
    src = (_repo() / "app.py").read_text()
    assert 'os.environ.get("CHAT_TOPIC_OFFER_ENABLED"' in src


def test_wiring_references():
    src = (_repo() / "app.py").read_text()
    assert "extract_topic_candidate" in src
    assert "/learn/suggest/" in src
    assert "record_suggestion" in src
    assert "list_suggestions" in src


def test_console_search_index_lists_suggestions():
    blob = json.dumps(json.loads((_repo() / "static" / "console-search-index.json").read_text()))
    assert "/console/topic-suggestions" in blob


def test_console_suggestions_html_exists():
    assert (_repo() / "static" / "console-topic-suggestions.html").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_offer_wiring.py -q`
Expected: FAIL — flag/refs/html/index missing.

- [ ] **Step 3a: Add the flag**

In `app.py`, right after the `CHAT_PAGE_LINKS_ENABLED` line, add:

```python
CHAT_TOPIC_OFFER_ENABLED = os.environ.get("CHAT_TOPIC_OFFER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 3b: Add the `/chat` offer block**

In `app.py`, in the `/chat` `generate()` done handler, immediately AFTER the `if CHAT_PAGE_LINKS_ENABLED:` block (grep `[chat-page-links]`) and BEFORE `_done_payload = {`, insert:

```python
        if CHAT_TOPIC_OFFER_ENABLED:
            try:
                # only offer when no existing-page link card already fired this turn
                _has_link = any(str(c.get("key", "")).split(":")[0] in ("topic", "ingredient", "product")
                                for c in (surfaced_cards or []))
                if not _has_link:
                    from dashboard import topic_copy as _tc2, topic_pages as _tp2, page_links as _pl2
                    import urllib.parse as _up
                    _cand = _tc2.extract_topic_candidate(query or "", answer or "", _cl)
                    if _cand:
                        _cslug = _cand["slug"]
                        with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                            _crow = _tp2.get_page(_cx, _cslug)
                        if not (_crow and _crow.get("state") in ("approved", "draft", "gated")):
                            _offer = {
                                "key": f"suggest:{_cslug}",
                                "title": f"Want a guide on {_cand['name']}?",
                                "sub": "We'll create it and email you",
                                "href": f"/learn/suggest/{_cslug}?kind={_cand['kind']}"
                                        f"&name={_up.quote(_cand['name'])}",
                            }
                            surfaced_cards = _pl2.merge_cards([_offer], surfaced_cards or [])
            except Exception as _toe:  # noqa: BLE001 - never break chat
                print(f"[chat-topic-offer] {_toe!r}", flush=True)
```

- [ ] **Step 3c: Add the `/learn/suggest` routes**

In `app.py`, near the other `/learn` routes, add:

```python
@app.route("/learn/suggest/<slug>", methods=["GET"])
def learn_suggest_form(slug):
    from dashboard import topic_render as _tr
    if not CHAT_TOPIC_OFFER_ENABLED:
        return ("Not found", 404)
    name = (request.values.get("name") or slug.replace("-", " ").title()).strip()
    return Response(_tr.render_suggest_html(slug, name), mimetype="text/html")


@app.route("/learn/suggest/<slug>", methods=["POST"])
def learn_suggest_submit(slug):
    from dashboard import topic_pages as _tp, topic_render as _tr
    if not CHAT_TOPIC_OFFER_ENABLED:
        return ("Not found", 404)
    email = (request.form.get("email") or request.values.get("email") or "").strip()
    name = (request.values.get("name") or slug.replace("-", " ").title()).strip()
    kind = (request.values.get("kind") or "symptom").strip()
    if not email:
        # re-render the form (browser 'required' normally prevents this)
        return Response(_tr.render_suggest_html(slug, name), mimetype="text/html")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _tp.record_suggestion(cx, slug, name, kind, email)
    return Response(_tr.render_suggest_html(slug, name, submitted=True), mimetype="text/html")
```

- [ ] **Step 3d: Add the console route + api**

In `app.py`, after the topic-pages console block, add:

```python
@app.route("/console/topic-suggestions")
def console_topic_suggestions_page():
    bad = _sales_console_ok()
    if bad:
        return bad
    resp = send_from_directory(STATIC, "console-topic-suggestions.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/console/topic-suggestions", methods=["GET"])
def api_console_topic_suggestions_list():
    bad = _sales_console_ok()
    if bad:
        return bad
    from dashboard import topic_pages as _tp
    with sqlite3.connect(LOG_DB) as cx:
        rows = _tp.list_suggestions(cx)
    return jsonify({"ok": True, "suggestions": rows})
```

(The `topic_page.dismiss` action is already registered via Task 4's `register()`; no extra registration here.)

- [ ] **Step 3e: Create the console HTML**

```bash
cd /tmp/wt-deploy-chat-252dcf59
cp static/console-topic-pages.html static/console-topic-suggestions.html
```

Then edit `static/console-topic-suggestions.html`: change the page title/heading to "Topic Suggestions"; point the list call at `/api/console/topic-suggestions` and read `data.suggestions` (each `{slug, name, kind, demand}`, show the demand count); the detail/edit panel reuses `/api/console/topic-page/<slug>`; the **Build** button dispatches `topic_page.regenerate` (drafts + scans → draft/gated, then the page flows to the normal `/console/topic-pages` approve queue) and the **Dismiss** button dispatches `topic_page.dismiss`; after either action, reload the list. Read the existing file first to match its JS structure.

- [ ] **Step 3f: Add to the search index**

Edit `static/console-search-index.json`: add a `/console/topic-suggestions` entry matching the existing entry shape (title "Topic Suggestions", url `/console/topic-suggestions`, short description "Review create-a-page suggestions from chat, ranked by demand; build or dismiss."). Keep the JSON valid.

- [ ] **Step 4: Verify**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m py_compile app.py && python3 -m json.tool static/console-search-index.json > /dev/null && python3 -m pytest tests/test_topic_offer_wiring.py -q`
Expected: compile OK, JSON valid, 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add app.py static/console-topic-suggestions.html static/console-search-index.json tests/test_topic_offer_wiring.py
git commit -m "feat(topic-offer): flag + /chat offer + /learn/suggest + console suggestions queue"
```

---

### Task 6: Regression + dark-default check

**Files:** none (verification only)

- [ ] **Step 1: Run the feature suites**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/test_topic_copy.py tests/test_topic_suggestions_store.py tests/test_topic_render.py tests/test_topic_page_actions.py tests/test_topic_offer_wiring.py -q`
Expected: all green.

- [ ] **Step 2: Broader sanity (ignore Pinecone app-import errors)**

Run: `cd /tmp/wt-deploy-chat-252dcf59 && python3 -m pytest tests/ -q -k "topic or page_links" --continue-on-collection-errors -p no:cacheprovider`
Expected: the topic/page_links dashboard tests pass; any `PineconeConfigurationError` collection errors are environmental (tests that `import app`), not regressions.

- [ ] **Step 3: Confirm dark-by-default**

Read the diff: `CHAT_TOPIC_OFFER_ENABLED` default `"false"`; the `/chat` offer block and both `/learn/suggest` routes are fully gated by it (offer block inside `if CHAT_TOPIC_OFFER_ENABLED:`, routes 404 when off). No code change — a read-check.

- [ ] **Step 4: Commit (only if fixups were needed)**

```bash
cd /tmp/wt-deploy-chat-252dcf59
git add -A && git commit -m "test(topic-offer): regression pass" || echo "nothing to commit"
```

---

## Rollout (post-merge, not a code task)

- PR `chat-topic-offer` → `main`; merge (direct-push guarded).
- Go-live: set `CHAT_TOPIC_OFFER_ENABLED=true` in Doppler `remedy-match/prd` after a live check: a chat about an uncovered topic surfaces the offer card → the suggest form records a suggestion → it appears in `/console/topic-suggestions` ranked by demand → Build (regenerate) → approve → ready-email. Reversible by flipping the flag back.

## Self-Review notes (author)

- **Spec coverage:** §3 flow → Tasks 1 (extract) + 5 (offer/routes); §4 `extract_topic_candidate` → T1, `record_suggestion`/`list_suggestions` → T2, `render_suggest_html` → T3, `topic_page.dismiss` → T4, app wiring/console/flag → T5; §5 safety (no downgrade, approved-only, flag-off inert, wrapped) → T2 + T5; §6 testing → T1-T6; §7 rollout → T5 flag + rollout. "Build = regenerate" reused (no new build action) per spec §4. No gaps.
- **Type consistency:** card `{key,title,sub,href}` from the offer block consumed by `merge_cards` (unchanged); `record_suggestion(cx,slug,name,kind,email)->str` and `list_suggestions(cx)->[{slug,name,kind,demand}]` defined T2 and consumed by T5's routes/api; `extract_topic_candidate(query,answer,client)->{name,kind,slug}|None` T1 consumed by T5; `render_suggest_html(slug,name,*,submitted=False)` T3 consumed by T5.
- **Watch-item:** T5 Step 3e (console HTML clone) is the least mechanical step — the implementer must read `console-topic-pages.html` first and wire Build→`topic_page.regenerate` / Dismiss→`topic_page.dismiss` against its real JS, surfacing the `demand` field.
