# Community Signal Layer (Layer B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let members react to Community content (anonymous aggregate counts) and privately like/block topics and people, capturing the signals Layer C will curate on.

**Architecture:** A new pure-sqlite `dashboard/community_signals.py` store (reactions + like/block), four portal-token-gated routes in `app.py`, and reaction + interest affordances added to the existing `static/community.html`. No feed is built here; Layer B only captures signals.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), vanilla JS/HTML in `static/`.

## Global Constraints

- **Privacy (load-bearing):** reaction *counts* are aggregate; **who** reacted is NEVER exposed by any endpoint. Like/block signals are visible only to the member who set them. No endpoint returns another member's email or identity.
- **Gate:** any logged-in member (portal token via `_evox_ident`), free or paid. Do NOT gate on `_is_paid_member` — free members' signals feed their own later curation.
- **Fixed vocabularies:** `REACTIONS = ["helpful", "inspiring", "this_is_me"]`, `TARGET_TYPES = ["topic", "person"]`, `SIGNALS = ["like", "block"]`. Unknown values → HTTP 400.
- **Copy:** client-facing copy has no em dashes and no ALL CAPS. Reaction labels render as "Helpful", "Inspiring", "This is me". Blocking is framed as tuning your own experience, never shaming.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; emails lowercased before storage/lookup.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- `dashboard/community.py:get_content(cx, content_id) -> dict|None` — returns a content row (has `published` column, 1 when published). Used to validate a reaction target exists and is published.
- `app.py` helpers/constants: `_evox_ident(cx, token)` → identity with `.email` or None; `_evox` portal token minting for tests is `dashboard/evox.py:ensure_portal_token(cx, email, name)`; `LOG_DB`; `_db_lock`; `sqlite3` imported at module top; routes return via `jsonify`.
- `static/community.html` (Layer A): reads the portal token into a `token` var; has an `$` template helper `(h)=>{const t=document.createElement("template");t.innerHTML=h.trim();return t.content.firstChild;}`; builds one card per library item; each item (paid full, free teaser, and out-take) carries `id` and `interest_tags`; `tagsRow(item.interest_tags)` renders the tag chips; `render(data)` is the top-level render after `fetch("/api/community/library?token="+…)`.

**Testing note (READ FIRST):**
- The store test (Task 1) does NOT import `app` — run with plain `python3 -m pytest tests/test_community_signals_store.py -q`.
- Route tests (Task 2) `import app`, which opens `LOG_DB = DATA_DIR/chat_log.db`; the prd Doppler config points `DATA_DIR` at a prod path that does not exist locally, so override it:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```

---

### Task 1: Signal store (`dashboard/community_signals.py`)

**Files:**
- Create: `dashboard/community_signals.py`
- Test: `tests/test_community_signals_store.py`

**Interfaces:**
- Consumes: nothing (pure sqlite).
- Produces:
  - `REACTIONS = ["helpful", "inspiring", "this_is_me"]`, `TARGET_TYPES = ["topic", "person"]`, `SIGNALS = ["like", "block"]`
  - `init_signal_tables(cx)`
  - `toggle_reaction(cx, email, content_id, reaction) -> bool` (True if now on, False if removed)
  - `reaction_counts(cx, content_id) -> {reaction: count}` (aggregate only)
  - `my_reactions(cx, email, content_id) -> [reaction]`
  - `set_signal(cx, email, target_type, target_key, signal)` (upsert)
  - `clear_signal(cx, email, target_type, target_key)`
  - `my_signals(cx, email) -> {"likes": [{"target_type","target_key"}], "blocks": [...]}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_signals_store.py
import sqlite3
from dashboard import community_signals as _s


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _s.init_signal_tables(cx)
    return cx


def test_toggle_reaction_on_then_off():
    cx = _cx()
    assert _s.toggle_reaction(cx, "A@B.com", 5, "helpful") is True   # added
    assert _s.my_reactions(cx, "a@b.com", 5) == ["helpful"]
    assert _s.toggle_reaction(cx, "a@b.com", 5, "helpful") is False  # removed
    assert _s.my_reactions(cx, "a@b.com", 5) == []


def test_reaction_counts_aggregate_no_emails():
    cx = _cx()
    _s.toggle_reaction(cx, "a@b.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "inspiring")
    counts = _s.reaction_counts(cx, 5)
    assert counts["helpful"] == 2 and counts["inspiring"] == 1
    # aggregate structure only: values are ints, keys are reaction names
    assert all(isinstance(v, int) for v in counts.values())
    assert "a@b.com" not in counts and "c@d.com" not in counts


def test_my_reactions_only_caller():
    cx = _cx()
    _s.toggle_reaction(cx, "a@b.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "inspiring")
    assert _s.my_reactions(cx, "a@b.com", 5) == ["helpful"]  # not c@d's


def test_set_signal_upsert_replaces():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "block")  # same target
    sig = _s.my_signals(cx, "a@b.com")
    assert sig["blocks"] == [{"target_type": "topic", "target_key": "sleep"}]
    assert sig["likes"] == []          # like replaced, not doubled
    row_count = cx.execute("SELECT COUNT(*) FROM community_signals").fetchone()[0]
    assert row_count == 1              # one row per (email, target)


def test_clear_signal_deletes():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.clear_signal(cx, "a@b.com", "topic", "sleep")
    assert _s.my_signals(cx, "a@b.com") == {"likes": [], "blocks": []}


def test_my_signals_splits_and_scopes():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.set_signal(cx, "a@b.com", "person", "x@y.com", "block")
    _s.set_signal(cx, "c@d.com", "topic", "adrenals", "like")  # other member
    sig = _s.my_signals(cx, "a@b.com")
    assert {"target_type": "topic", "target_key": "sleep"} in sig["likes"]
    assert {"target_type": "person", "target_key": "x@y.com"} in sig["blocks"]
    assert all(x["target_key"] != "adrenals" for x in sig["likes"])  # not c@d's
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_signals_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.community_signals'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/community_signals.py
"""Community signal layer (Layer B): reactions on content + private like/block
on topics and people. Pure sqlite; no app-layer imports. These signals feed
Layer C's curation. Reaction COUNTS are aggregate; who reacted is never exposed
by this module. Like/block signals are per-member private."""

REACTIONS = ["helpful", "inspiring", "this_is_me"]
TARGET_TYPES = ["topic", "person"]
SIGNALS = ["like", "block"]

_DDL = """
CREATE TABLE IF NOT EXISTS community_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    content_id INTEGER NOT NULL,
    reaction TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(email, content_id, reaction)
);
CREATE INDEX IF NOT EXISTS ix_reactions_content ON community_reactions(content_id);
CREATE TABLE IF NOT EXISTS community_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    signal TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(email, target_type, target_key)
);
CREATE INDEX IF NOT EXISTS ix_signals_email ON community_signals(email);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_signal_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def toggle_reaction(cx, email, content_id, reaction):
    """Add the reaction if absent, remove it if present. Returns True if now on."""
    email = _lc(email)
    row = cx.execute("SELECT id FROM community_reactions WHERE email=? AND content_id=? "
                     "AND reaction=?", (email, content_id, reaction)).fetchone()
    if row:
        cx.execute("DELETE FROM community_reactions WHERE id=?", (row[0],))
        cx.commit()
        return False
    cx.execute("INSERT INTO community_reactions (email,content_id,reaction,created_at) "
               "VALUES (?,?,?,?)", (email, content_id, reaction, _now()))
    cx.commit()
    return True


def reaction_counts(cx, content_id):
    """Aggregate counts per reaction for one content item. No emails, ever."""
    rows = cx.execute("SELECT reaction, COUNT(*) c FROM community_reactions "
                      "WHERE content_id=? GROUP BY reaction", (content_id,)).fetchall()
    return {r["reaction"]: r["c"] for r in rows}


def my_reactions(cx, email, content_id):
    rows = cx.execute("SELECT reaction FROM community_reactions WHERE email=? AND content_id=? "
                      "ORDER BY reaction", (_lc(email), content_id)).fetchall()
    return [r["reaction"] for r in rows]


def set_signal(cx, email, target_type, target_key, signal):
    """Upsert a like/block on a target. One row per (email, target_type, target_key)."""
    cx.execute(
        "INSERT INTO community_signals (email,target_type,target_key,signal,created_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(email,target_type,target_key) DO UPDATE SET signal=excluded.signal",
        (_lc(email), target_type, target_key, signal, _now()))
    cx.commit()


def clear_signal(cx, email, target_type, target_key):
    cx.execute("DELETE FROM community_signals WHERE email=? AND target_type=? AND target_key=?",
               (_lc(email), target_type, target_key))
    cx.commit()


def my_signals(cx, email):
    rows = cx.execute("SELECT target_type, target_key, signal FROM community_signals "
                      "WHERE email=? ORDER BY created_at", (_lc(email),)).fetchall()
    out = {"likes": [], "blocks": []}
    for r in rows:
        entry = {"target_type": r["target_type"], "target_key": r["target_key"]}
        out["likes" if r["signal"] == "like" else "blocks"].append(entry)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_signals_store.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community_signals.py tests/test_community_signals_store.py
git commit -m "feat(community): signal store (reactions + like/block)"
```

---

### Task 2: Signal routes

**Files:**
- Modify: `app.py` (add four routes near the community routes; grep `@app.route("/api/community/library")`)
- Test: `tests/test_community_signals_api.py`

**Interfaces:**
- Consumes: `dashboard/community_signals.py` (all functions + `REACTIONS`/`TARGET_TYPES`/`SIGNALS`), `dashboard/community.py:get_content`, `_evox_ident`, `LOG_DB`, `_db_lock`.
- Produces: `POST /api/community/react`, `GET /api/community/reactions`, `POST /api/community/signal`, `GET /api/community/signals`.

**Contract:**
- All: bad/absent token → 404 `{"error":"not_found"}`.
- `POST /api/community/react {content_id, reaction}` → 400 `{"error":"bad_reaction"}` if `reaction not in REACTIONS`; 404 `{"error":"not_found"}` if `get_content` is None or its `published != 1`; else toggle → `{"ok":true, "on":bool, "counts":{...}}`.
- `GET /api/community/reactions?content_id=…` → `{"counts":{...}, "mine":[...]}` (aggregate + caller's own; no identities).
- `POST /api/community/signal {target_type, target_key, signal}` → 400 `{"error":"bad_signal"}` if `target_type not in TARGET_TYPES` or `signal not in (SIGNALS + ["none"])`; if `signal=="none"` → `clear_signal`; else `set_signal`; return `{"ok":true}`.
- `GET /api/community/signals` → `{"likes":[...], "blocks":[...]}` (caller's own only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_signals_api.py
import sqlite3
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, community_signals as _s
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _s.init_signal_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="W1", description="",
                             video_ref="https://rumble.com/v-b", interest_tags=["sleep"],
                             transcript=""); _c.publish(cx, cid)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token, cid


def test_react_toggle_and_counts():
    c = _client(); tok, cid = _seed()
    r1 = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "helpful"})
    d1 = r1.get_json()
    assert d1["ok"] and d1["on"] is True and d1["counts"]["helpful"] == 1
    r2 = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "helpful"})
    assert r2.get_json()["on"] is False  # toggled off


def test_react_bad_reaction_400():
    c = _client(); tok, cid = _seed()
    r = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "nope"})
    assert r.status_code == 400


def test_react_unknown_content_404():
    c = _client(); tok, _ = _seed()
    r = c.post(f"/api/community/react?token={tok}", json={"content_id": 999999, "reaction": "helpful"})
    assert r.status_code == 404


def test_reactions_get_aggregate_no_identity():
    c = _client(); tok, cid = _seed()
    c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "inspiring"})
    d = c.get(f"/api/community/reactions?token={tok}&content_id={cid}").get_json()
    assert d["counts"]["inspiring"] == 1
    assert d["mine"] == ["inspiring"]
    # no identity fields anywhere in the payload
    assert "email" not in repr(d)


def test_signal_set_clear_and_scope():
    c = _client(); tok, _ = _seed()
    c.post(f"/api/community/signal?token={tok}",
           json={"target_type": "topic", "target_key": "sleep", "signal": "like"})
    d = c.get(f"/api/community/signals?token={tok}").get_json()
    assert d["likes"] == [{"target_type": "topic", "target_key": "sleep"}]
    c.post(f"/api/community/signal?token={tok}",
           json={"target_type": "topic", "target_key": "sleep", "signal": "none"})
    assert c.get(f"/api/community/signals?token={tok}").get_json()["likes"] == []


def test_signal_bad_type_400():
    c = _client(); tok, _ = _seed()
    r = c.post(f"/api/community/signal?token={tok}",
               json={"target_type": "planet", "target_key": "x", "signal": "like"})
    assert r.status_code == 400


def test_bad_token_404():
    c = _client()
    assert c.get("/api/community/signals?token=nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_signals_api.py -q`
Expected: FAIL — routes 404 (not registered).

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near `community_library`):

```python
@app.route("/api/community/react", methods=["POST"])
def community_react():
    from dashboard import community as _cm, community_signals as _cs
    body = request.get_json(force=True) or {}
    reaction = (body.get("reaction") or "").strip()
    content_id = body.get("content_id")
    if reaction not in _cs.REACTIONS:
        return jsonify({"error": "bad_reaction"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cm.init_community_tables(cx); _cs.init_signal_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        item = _cm.get_content(cx, content_id)
        if item is None or item["published"] != 1:
            return jsonify({"error": "not_found"}), 404
        on = _cs.toggle_reaction(cx, ident.email, content_id, reaction)
        counts = _cs.reaction_counts(cx, content_id)
        return jsonify({"ok": True, "on": on, "counts": counts})


@app.route("/api/community/reactions")
def community_reactions():
    from dashboard import community_signals as _cs
    content_id = request.args.get("content_id", type=int)
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_signal_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"counts": _cs.reaction_counts(cx, content_id),
                        "mine": _cs.my_reactions(cx, ident.email, content_id)})


@app.route("/api/community/signal", methods=["POST"])
def community_signal():
    from dashboard import community_signals as _cs
    body = request.get_json(force=True) or {}
    ttype = (body.get("target_type") or "").strip()
    tkey = (body.get("target_key") or "").strip()
    signal = (body.get("signal") or "").strip()
    if ttype not in _cs.TARGET_TYPES or signal not in (_cs.SIGNALS + ["none"]):
        return jsonify({"error": "bad_signal"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_signal_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if signal == "none":
            _cs.clear_signal(cx, ident.email, ttype, tkey)
        else:
            _cs.set_signal(cx, ident.email, ttype, tkey, signal)
        return jsonify({"ok": True})


@app.route("/api/community/signals")
def community_signals():
    from dashboard import community_signals as _cs
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_signal_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify(_cs.my_signals(cx, ident.email))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_signals_api.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_community_signals_api.py
git commit -m "feat(community): signal routes (react + like/block)"
```

---

### Task 3: Member surface (extend `static/community.html`)

**Files:**
- Modify: `static/community.html`
- Test: manual JS parse check (no framework harness in this repo)

**Interfaces:**
- Consumes: `POST /api/community/react`, `GET /api/community/reactions`, `POST /api/community/signal`, `GET /api/community/signals` (Task 2).

**Design note:** this task extends the existing Layer A page. Read `static/community.html` first to match its existing helpers (`$` template helper, `token` var, `tagsRow`, `render`). Add three things, wrapped in `<!-- BEGIN signal-layer script -->` / `<!-- END signal-layer script -->` for the new JS:

1. **Reaction row on each content card.** After a card is built (both paid full items and free teaser items carry `item.id`), append a reaction row: one small button per reaction in `["helpful","inspiring","this_is_me"]`, labelled "Helpful" / "Inspiring" / "This is me", each showing its count. On load, fetch `GET /api/community/reactions?token=<TOKEN>&content_id=<item.id>` to fill counts and mark the member's own reactions active. On click, `POST /api/community/react?token=<TOKEN>` with `{content_id, reaction}`, and update the button's count + active state from the response (`d.on`, `d.counts`).
2. **Like / block on each topic chip.** In `tagsRow` (or alongside it), give each topic tag a small like and block control that `POST /api/community/signal?token=<TOKEN>` with `{target_type:"topic", target_key:<tag>, signal:"like"|"block"|"none"}` (clicking an active control again sends `"none"` to clear). Reflect current state from `GET /api/community/signals`.
3. **"Your interests" section** near the top or bottom of the page: fetch `GET /api/community/signals?token=<TOKEN>` and list the member's liked and blocked topics, each with a "clear" control (`signal:"none"`). Heading "Your interests". If none yet, a quiet line: "Like a topic to see more of it here."

Rules:
- All server-supplied strings (tag text, etc.) inserted via `textContent`, never `innerHTML`.
- Copy: no em dashes, no ALL CAPS. Reaction labels exactly "Helpful", "Inspiring", "This is me".
- Reactions show counts only; the UI must never display who reacted (the API does not expose it).
- Person-target signals are out of scope for the UI here (no people are surfaced until Layer C); only topic like/block ships.

- [ ] **Step 1: Read the existing page and add the reaction row**

Read `static/community.html`. In the card-building code, after tags are appended, append a reaction row per the design note. Wire the on-load `GET /api/community/reactions` fill and the click `POST /api/community/react`.

- [ ] **Step 2: Add topic like/block controls**

Extend the tag rendering so each topic chip has like/block controls posting to `/api/community/signal`, reflecting state from `/api/community/signals`.

- [ ] **Step 3: Add the "Your interests" section**

Add the section that lists liked/blocked topics from `/api/community/signals` with clear controls.

- [ ] **Step 4: Verify the page JS parses**

Run: `node --check <(python3 -c "import re; h=open('static/community.html').read(); print('\n;\n'.join(re.findall(r'<script>(.*?)</script>', h, re.S)))")`
Expected: no output (clean parse). If it errors, fix the JS.

- [ ] **Step 5: Commit**

```bash
git add static/community.html
git commit -m "feat(community): reaction + interest affordances on the library page"
```

---

## Definition of Done

- Members can react to content (Helpful / Inspiring / This is me) with aggregate counts shown and identities never exposed, and can privately like/block topics; both are captured for Layer C.
- Four portal-token-gated routes; any member (free or paid) can use them; unknown reaction/target/signal values return 400; bad token 404.
- The `/community` page shows reaction rows, topic like/block controls, and a "Your interests" section.
- All new tests pass; Layer A (library, store, publish) and EVOX/consult/triage/masterclass/onboarding are untouched.

## Deferred (not in this plan)

- Layer C: the curated "for you" feed built from these signals, opt-in like-minded-member introductions, and the community-aware AI chat.
- Person-target like/block UI (the store supports person targets now; the UI lands with Layer C when people are surfaced).
- Folding reaction counts into the `/api/community/library` payload to save the per-item fetch (a perf optimization; per-item fetch is fine at current volume).
