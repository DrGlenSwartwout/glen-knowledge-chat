# Community-Aware Chat (Layer C, slice C3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a member asks the "Ask Dr. Glen" portal chat something, surface the top 1-2 relevant Community content items as tier-aware "From the community" cards under the answer, reusing C1's embeddings.

**Architecture:** A `_community_related` helper (app.py) cosine-matches the member's already-embedded query against C1's content vectors and tier-gates the results; the existing `POST /api/portal/<token>/chat` route computes it once, injects a short note into the chat context, and emits a final `related` SSE event; the portal chat UI renders `related` items as cards (mirroring the existing `suggestion` card path). Reuses C1; no new store, no new model.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`), OpenAI embeddings via `embed()` (app.py), SSE via `sse()` (app.py:1932), vanilla JS in `static/client-portal.html`.

## Global Constraints

- **Privacy / no leak:** the query is the member's own; matching is one-directional. A related card carries only `{id, title, kind}` — NEVER a Rumble `video_ref`. Free members get `kind:"teaser"` cards (no full link); paid get `kind:"full"`. The card links to the gated `/community` page, not a raw video.
- **Fail-open:** related-content retrieval is best-effort. Any failure logs and is skipped; the chat answer must stream normally regardless.
- **Reuse the query embedding:** the chat route already embeds the query for knowledge-base RAG — capture that vector ONCE and reuse it for `_community_related`. Do not embed the query twice.
- **No lazy embed on the chat path:** items lacking a current-model vector (not yet embedded by the C1 feed) are simply skipped in retrieval, to keep chat latency low.
- **Copy:** card labels and the assistant acknowledgement have no em dashes and no ALL CAPS.
- **Scope:** the members' portal chat only. Do NOT touch the public widget/funnel chat.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- C1 helpers (already in app.py): `_community_candidates(cx, is_paid) -> (candidates, full_by_id)` — paid candidates are `list_full` items (with `id`, `title`, `video_ref`, ...); free candidates are teaser dicts (`id`, `title`, `interest_tags`, ..., NO `video_ref`). `COMMUNITY_FEED_MODEL = "text-embedding-ada-002"`.
- `dashboard/community.py:get_embeddings(cx, content_ids, model) -> {id: [float]}`.
- `dashboard/community_feed.py:cosine(a, b) -> float`.
- `embed(text) -> [float]` (app.py:1763). `sse(payload_dict) -> str` (app.py:1932) — formats one `data: ...\n\n` SSE line. `_is_paid_member(email)`, `LOG_DB`.
- The chat route `POST /api/portal/<token>/chat` (app.py:15159): it looks up `portal`/`email`, builds `_sys` + a `context_str` via `matches = _match_query_namespaces(embed(query))` then `build_context(matches)`, assembles `messages` with a `user_block = ("CONTEXT:\n{context_str}\n\n" if context_str else "") + query`, and streams inside a nested `def generate()` that yields `sse({"token": ...})` and finally `sse({"done": True})` (grep the route for the exact `done` emit).
- The portal chat UI (`static/client-portal.html` ~line 1529): the SSE reader loop parses `data: ` lines into `evt` and branches on `evt.token` / `evt.suggestion` (→ `renderSuggestion(evt.suggestion)` appended to `#chatMsgs`) / `evt.error` / `evt.done`. The member's portal token is the JS `token`/`seg` var (~line 284).

**Testing note (READ FIRST):** route/helper tests `import app`; the prd Doppler config points `DATA_DIR` at a nonexistent prod path, so override it:
```
export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
```

---

### Task 1: `_community_related` retrieval helper

**Files:**
- Modify: `app.py` (add the helper near `_community_candidates`)
- Test: `tests/test_community_related.py`

**Interfaces:**
- Consumes: `_community_candidates`, `COMMUNITY_FEED_MODEL`, `dashboard/community.py:get_embeddings`, `dashboard/community_feed.py:cosine`.
- Produces: `_community_related(cx, query_vec, is_paid, *, k=2, min_sim=0.72) -> [dict]` — each item `{"id", "title", "kind"}` where `kind` is `"full"` (paid) or `"teaser"` (free). Never raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_related.py
import sqlite3
import app as appmod
from dashboard import community as _c


def _seed(*, tags_per_item):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        ids = []
        for i, tags in enumerate(tags_per_item):
            cid = _c.upsert_full(cx, type="coaching_replay", title=f"T{i}", description="",
                                 video_ref=f"https://rumble.com/v-{i}", interest_tags=tags,
                                 transcript=""); _c.publish(cx, cid); ids.append(cid)
        # embed item 0 near [1,0,0], item 1 near [0,1,0]
        _c.set_embedding(cx, ids[0], [1.0, 0.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        _c.set_embedding(cx, ids[1], [0.0, 1.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        cx.commit()
    return ids


def test_related_returns_nearest_first_paid():
    ids = _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        out = appmod._community_related(cx, [1.0, 0.05, 0.0], is_paid=True, k=2)
    assert out[0]["id"] == ids[0]                      # nearest to [1,0,0]
    assert out[0]["kind"] == "full"
    assert "video_ref" not in out[0]                   # never leaks the Rumble link


def test_related_free_is_teaser_kind():
    _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        out = appmod._community_related(cx, [1.0, 0.0, 0.0], is_paid=False, k=2)
    assert out and all(it["kind"] == "teaser" for it in out)
    assert all("video_ref" not in it for it in out)


def test_related_excludes_below_min_sim():
    _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        # query orthogonal-ish to both → nothing clears a high bar
        out = appmod._community_related(cx, [0.0, 0.0, 1.0], is_paid=True, k=2, min_sim=0.72)
    assert out == []


def test_related_empty_when_no_embeddings():
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="NE", description="",
                             video_ref="https://rumble.com/v-ne", interest_tags=[],
                             transcript=""); _c.publish(cx, cid); cx.commit()
        out = appmod._community_related(cx, [1.0, 0.0, 0.0], is_paid=True, k=2)
    assert out == []                                   # unembedded item skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_related.py -q`
Expected: FAIL — `_community_related` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` near `_community_candidates`:

```python
def _community_related(cx, query_vec, is_paid, *, k=2, min_sim=0.72):
    """Top-k tier-visible Community items most similar to the query vector, above
    min_sim. Card-shaped {id, title, kind}; never carries video_ref; never raises.
    Items without a current-model embedding are skipped (no lazy embed here)."""
    try:
        from dashboard import community as _cm, community_feed as _cf
        cands, _ = _community_candidates(cx, is_paid)
        vecs = _cm.get_embeddings(cx, [c["id"] for c in cands], COMMUNITY_FEED_MODEL)
        scored = []
        for c in cands:
            v = vecs.get(c["id"])
            if not v:
                continue
            s = _cf.cosine(query_vec, v)
            if s >= min_sim:
                scored.append((s, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        kind = "full" if is_paid else "teaser"
        return [{"id": c["id"], "title": c.get("title", ""), "kind": kind}
                for _, c in scored[:k]]
    except Exception:
        app.logger.exception("community_related failed")
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_related.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_community_related.py
git commit -m "feat(community): _community_related retrieval helper (reuses C1)"
```

---

### Task 2: Wire retrieval into the chat route

**Files:**
- Modify: `app.py` — the `api_portal_chat` route (app.py:15159)
- Test: `tests/test_community_chat_related.py`

**Interfaces:**
- Consumes: `_community_related` (Task 1), `embed`, `sse`, `_is_paid_member`, the existing route internals.
- Produces: the chat route now (a) reuses one query embedding for both KB RAG and community retrieval, (b) injects a one-line note into the context, (c) emits a final `sse({"related": [...]})` event when related items exist.

**Contract:** the answer stream is unchanged; when related items exist a `related` SSE event carrying `[{id,title,kind}]` is emitted before `done`. Retrieval failure never breaks the stream. Related items respect the member's tier.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_chat_related.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member_and_content(email):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="Sleep Reset", description="",
                             video_ref="https://rumble.com/v-s", interest_tags=["sleep"],
                             transcript=""); _c.publish(cx, cid)
        _c.set_embedding(cx, cid, [1.0, 0.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        token = _ev.ensure_portal_token(cx, email, "You")
        cx.commit()
    return token, cid


def _sse_events(resp):
    text = resp.get_data(as_text=True)
    import json
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try: out.append(json.loads(line[6:]))
            except Exception: pass
    return out


def test_chat_emits_related_event(monkeypatch):
    c = _client(); tok, cid = _seed_member_and_content("m@x.com")
    # deterministic query embedding near the seeded item; stub the LLM stream + KB RAG
    monkeypatch.setattr(appmod, "embed", lambda t: [1.0, 0.0, 0.0])
    monkeypatch.setattr(appmod, "_match_query_namespaces", lambda v: [])
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_stream_claude_tokens", create=True):
        # If the route streams via anthropic directly, this test asserts on the related
        # event regardless of tokens; see Step 3 note on stubbing the LLM.
        resp = c.post(f"/api/portal/{tok}/chat", json={"query": "help me sleep", "history": []})
    evts = _sse_events(resp)
    related = [e for e in evts if "related" in e]
    assert related and related[0]["related"][0]["title"] == "Sleep Reset"
    assert related[0]["related"][0]["kind"] == "full"
    assert "video_ref" not in related[0]["related"][0]


def test_chat_related_failure_does_not_break(monkeypatch):
    c = _client(); tok, cid = _seed_member_and_content("m2@x.com")
    monkeypatch.setattr(appmod, "embed", lambda t: [1.0, 0.0, 0.0])
    monkeypatch.setattr(appmod, "_match_query_namespaces", lambda v: [])
    monkeypatch.setattr(appmod, "_community_related",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        resp = c.post(f"/api/portal/{tok}/chat", json={"query": "hi", "history": []})
    assert resp.status_code == 200   # stream still returned, no crash
```

Note: the LLM stream in this route calls Anthropic. If the test environment has no key, the route's own `try/except` around the stream yields an error token but STILL reaches the `related`/`done` emit. The implementer MUST ensure the `related` event is emitted in a `finally`-style position that runs whether or not the LLM stream succeeded (see Step 3). If stubbing the Anthropic client is cleaner, the implementer may monkeypatch the module-level client (`_cl`) to a fake that yields no tokens — either way the `related` event and answer-independence are what these tests assert.

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_chat_related.py -q`
Expected: FAIL — no `related` event emitted.

- [ ] **Step 3: Write minimal implementation**

Modify `api_portal_chat` (app.py:15159). Three edits:

1. **Embed once + compute related** (in the request scope, before `def generate()`). Find the existing RAG block:
```python
    # RAG (best-effort, fail-open)
    context_str = ""
    try:
        matches = _match_query_namespaces(embed(query))
        context_str, _ = build_context(matches) if matches else ("", [])
    except Exception as e:
        print(f"[portal-concierge] retrieval: {e}", flush=True)
```
Replace with (capture the vector once, reuse for community retrieval):
```python
    # RAG (best-effort, fail-open) — embed the query ONCE and reuse for both the
    # knowledge base and Community content retrieval.
    context_str = ""
    qvec = None
    try:
        qvec = embed(query)
        matches = _match_query_namespaces(qvec)
        context_str, _ = build_context(matches) if matches else ("", [])
    except Exception as e:
        print(f"[portal-concierge] retrieval: {e}", flush=True)
    community_related = []
    if qvec is not None:
        try:
            with sqlite3.connect(LOG_DB) as ccx:
                ccx.row_factory = sqlite3.Row
                community_related = _community_related(ccx, qvec, _is_paid_member(email), k=2)
        except Exception as e:
            print(f"[community-chat] related: {e}", flush=True)
    if community_related:
        titles = "; ".join(r["title"] for r in community_related)
        context_str = (context_str + "\n" if context_str else "") + \
            f"Relevant community sessions the member can open: {titles}."
```

2. The `user_block` line already uses `context_str`, so the community note rides along automatically — no change there.

3. **Emit the `related` event at the START of `def generate()`**, BEFORE the token-stream `try`/`with _cl.messages.stream(...)` block. The existing `generate()` does an early `return` in its `except` if the LLM stream fails, so a `related` emit placed after the stream would be skipped on failure (and would be untestable without an Anthropic key). Emitting it first guarantees it fires regardless of the answer, and it still renders BELOW the answer in the UI because the assistant bubble DOM element is created before the fetch. Add as the first lines inside `generate()`:
```python
    def generate():
        if community_related:
            yield sse({"related": community_related})
        full = []
        try:
            with _cl.messages.stream(...):
                ...
```
(Insert only the two `if community_related: yield ...` lines at the top; leave the rest of `generate()` unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_chat_related.py -q`
Expected: PASS (2 passed). If the LLM-stream stubbing is awkward, adjust per the Step 1 note (monkeypatch the module-level Anthropic client `_cl`) so the test exercises the `related` emit deterministically.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_community_chat_related.py
git commit -m "feat(community): chat surfaces related community content (SSE related event)"
```

---

### Task 3: Related-cards in the portal chat UI

**Files:**
- Modify: `static/client-portal.html` (the SSE reader loop ~line 1529 and a new `renderRelated` helper near `renderSuggestion`)
- Test: manual JS parse check.

**Interfaces:**
- Consumes: the `related` SSE event `{related: [{id, title, kind}]}` from Task 2.

**Design note:** read the chat SSE reader loop (~1529) and the existing `renderSuggestion` helper first. Mirror the `evt.suggestion` pattern.

- [ ] **Step 1: Add the `related` branch to the SSE reader**

In the `for` loop over parsed `evt` objects (the one with `if(evt.token){...} else if(evt.suggestion){...}`), add a branch:
```javascript
            } else if(evt.related){
              var relEl = renderRelated(evt.related);
              if(relEl && msgs){ msgs.appendChild(relEl); msgs.scrollTop = msgs.scrollHeight; }
```

- [ ] **Step 2: Add the `renderRelated` helper**

Near `renderSuggestion`, add a function that builds a small "From the community" block. It must:
- Take an array of `{id, title, kind}`.
- Return a DOM element (a `div`) containing a small heading "From the community" and, per item, a link to `"/community?token=" + encodeURIComponent(token)` whose visible text is the item `title` (set via `textContent`, never innerHTML). For `kind === "teaser"` items, append a quiet nudge line: "Become a member to watch the full session."
- Use the `token` var already in scope (the member's portal token).
- Copy: no em dashes, no ALL CAPS. Heading exactly "From the community".

- [ ] **Step 3: Verify the page JS parses**

Run: `cd /tmp/wt-deploy-chat-cca589e9 && node --check <(python3 -c "import re; h=open('static/client-portal.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse).

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(community): render related-community cards in the portal chat"
```

---

## Definition of Done

- Asking the members' "Ask Dr. Glen" chat surfaces the top 1-2 relevant Community items (above a similarity bar) as tier-aware "From the community" cards linking to `/community`; the assistant can acknowledge them via an injected context note.
- Free members get teaser cards with a membership nudge and NO full Rumble link; paid get full cards.
- Retrieval is fail-open — the answer streams normally on any retrieval error, and the query is embedded once.
- All new tests pass; the public widget chat and Layer A/B/C1 stores are untouched.

## Deferred (not in this plan)

- C2 (opt-in introductions).
- Public widget/funnel chat community-awareness.
- Per-item deep links / scroll-to-item on `/community` (cards link to the page).
- Lazy-embedding on the chat path (relies on the C1 feed having embedded items).
