# Community Content Library + AI Cataloging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tier-gated Community video library (paid full replays/course sessions on Rumble + free self-hosted out-take teasers), plus a local Mac tool that transcribes a recording, suggests title/tags/out-takes with AI, and publishes into it.

**Architecture:** Prod side (deploy-chat) = a `community_content` sqlite store, a member-facing library page gated on `_is_paid_member`, and a `CONSOLE_SECRET`-gated JSON publish endpoint. Local side = an in-repo Flask app Glen runs on his Mac (pattern: `biofield_local_app.py`) that runs Whisper + an LLM suggestion pass over a source recording, cuts approved out-takes with ffmpeg, uploads the clips via the existing `/portal-asset/upload`, and POSTs the catalog entry to the publish endpoint. Both halves live on one branch/PR.

**Tech Stack:** Python 3 / Flask (single `app.py` + `dashboard/*.py`), sqlite (`chat_log.db`, `?` placeholders, `_db_lock`, `cx.row_factory = sqlite3.Row`), vanilla JS/HTML in `static/`, OpenAI client (Whisper `whisper-1` + `gpt-4o` json_object, per `dashboard/fmp_biofield.py`), ffmpeg via `dashboard/video_trim.py`, Rumble unlisted for full videos.

## Global Constraints

- **Video hosting:** full recordings (paid) live on **Rumble unlisted** — never YouTube/Vimeo (they censor Glen's health content). `video_ref` for a full item is its Rumble embed URL. Out-take clips (free) are cut by ffmpeg and hosted via the existing `/portal-asset/upload` asset mechanism (Task 3 extends its filename allowlist to accept `.mp4`); the out-take `video_ref` is the returned served URL. This reuse supersedes the spec's literal `GET /community/clip/<id>` route — the existing asset route serves the clips.
- **Tier gate:** full recordings are visible only to active members (`_is_paid_member(email)`). The free-member library payload MUST NOT include any full item's `video_ref` (field allowlist — the Rumble link never reaches a non-member). Out-takes are ungated.
- **Privacy (applies to later layers, do not violate now):** never expose a member's private journal/chat to other members. This slice does not touch journals; just do not add any cross-member exposure.
- **Copy:** all client-facing copy (library page, teases, captions) has no em dashes and no ALL CAPS. Warm, inviting; the free tease names the full session's value and points at membership without disparaging.
- sqlite writes under `with _db_lock, sqlite3.connect(LOG_DB)`; network/LLM/ffmpeg calls run OUTSIDE any DB lock.
- `interest_tags` round-trips as a JSON array string in the DB.
- DRY, YAGNI, TDD, frequent commits.

**Repo facts the implementer needs:**
- `dashboard/video_trim.py:trim_video(src_path, dst_path, start, end, *, runner=None)` cuts `[start,end]` seconds of `src_path` into `dst_path` (ffmpeg). `runner` is injectable for tests (defaults to real subprocess).
- Asset upload (already in prod): `PUT /portal-asset/upload?filename=<name>` with header `X-Console-Key: <CONSOLE_SECRET>`, body = raw bytes; returns `{"url": "<served url>"}`. Client helper pattern: `dashboard/biofield_portal_publish.py:upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None)`.
- OpenAI client pattern (see `dashboard/fmp_biofield.py:draft_prose`): `import openai; client = openai.OpenAI()`. Transcription: `client.audio.transcriptions.create(model="whisper-1", file=<fileobj>, response_format="verbose_json")` → object with `.text` and `.segments` (each `{"start","end","text"}`). Chat: `client.chat.completions.create(model="gpt-4o", response_format={"type":"json_object"}, messages=[...])`.
- app.py helpers/constants: `_evox_ident(cx, token)` → identity with `.email` or None; `_is_paid_member(email)`; `CONSOLE_SECRET`; `PUBLIC_BASE_URL`; `STATIC`; `send_from_directory`; `_db_lock`; `LOG_DB`; `_portal_console_ok()` or the `X-Console-Key == CONSOLE_SECRET` check used by other console routes (grep an existing `@app.route("/api/console/...")` to copy the exact auth line).
- Local-app pattern: `biofield_local_app.py` (a standalone `create_app().run(host="127.0.0.1", port=…)` Flask app in the repo root, run manually on the Mac; not part of the deployed service).

**Testing note (READ FIRST):**
- Prod tasks (1-3) `import app`, which opens `LOG_DB = DATA_DIR/chat_log.db` at import; the prd Doppler config points `DATA_DIR` at a prod path that does not exist locally, so override it:
  ```
  export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest <paths> -q
  ```
- Pure/local tasks (`dashboard/community.py`, `dashboard/community_catalog.py`, `community_local_app.py`) do NOT import `app`; run them with plain `python3 -m pytest <paths> -q` (no Doppler), mocking the OpenAI client and `requests`.

---

### Task 1: Content store (`dashboard/community.py`)

**Files:**
- Create: `dashboard/community.py`
- Test: `tests/test_community_store.py`

**Interfaces:**
- Consumes: nothing (pure sqlite).
- Produces:
  - `init_community_tables(cx)`
  - `create_content(cx, *, type, title, description, video_ref, tier, interest_tags, parent_id=None, transcript=None) -> int`
  - `get_content(cx, content_id) -> dict | None`
  - `publish(cx, content_id)`
  - `upsert_full(cx, *, type, title, description, video_ref, interest_tags, transcript) -> int` — create-or-update a full item keyed on `video_ref` (idempotent publish); returns its id and clears its old out-takes.
  - `add_outtake(cx, *, parent_id, title, video_ref, interest_tags) -> int`
  - `list_full(cx) -> [dict]` — published `paid` full items, newest `published_at` first, each with `outtakes` list.
  - `list_outtakes(cx, parent_id=None) -> [dict]` — published `free` out-takes, optionally filtered to one parent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_store.py
import sqlite3, json
from dashboard import community as _c


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _c.init_community_tables(cx)
    return cx


def test_create_and_get():
    cx = _cx()
    cid = _c.create_content(cx, type="coaching_replay", title="Week 1",
                            description="d", video_ref="https://rumble.com/v-abc",
                            tier="paid", interest_tags=["sleep", "adrenals"])
    row = _c.get_content(cx, cid)
    assert row["title"] == "Week 1"
    assert row["tier"] == "paid"
    assert json.loads(row["interest_tags"]) == ["sleep", "adrenals"]
    assert row["published"] == 0


def test_list_full_only_published_newest_first():
    cx = _cx()
    a = _c.create_content(cx, type="coaching_replay", title="A", description="",
                          video_ref="r/a", tier="paid", interest_tags=[])
    b = _c.create_content(cx, type="course_session", title="B", description="",
                          video_ref="r/b", tier="paid", interest_tags=[])
    _c.publish(cx, a); _c.publish(cx, b)
    cx.execute("UPDATE community_content SET published_at='2026-01-01' WHERE id=?", (a,))
    cx.execute("UPDATE community_content SET published_at='2026-02-01' WHERE id=?", (b,)); cx.commit()
    titles = [r["title"] for r in _c.list_full(cx)]
    assert titles == ["B", "A"]  # newest first


def test_list_full_excludes_unpublished_and_outtakes():
    cx = _cx()
    f = _c.create_content(cx, type="coaching_replay", title="F", description="",
                          video_ref="r/f", tier="paid", interest_tags=[]); _c.publish(cx, f)
    _c.create_content(cx, type="coaching_replay", title="Draft", description="",
                      video_ref="r/d", tier="paid", interest_tags=[])  # unpublished
    o = _c.add_outtake(cx, parent_id=f, title="clip", video_ref="/asset/x.mp4",
                       interest_tags=[]); _c.publish(cx, o)
    assert [r["title"] for r in _c.list_full(cx)] == ["F"]
    full = _c.list_full(cx)[0]
    assert [ot["title"] for ot in full["outtakes"]] == ["clip"]


def test_upsert_full_is_idempotent_on_video_ref():
    cx = _cx()
    id1 = _c.upsert_full(cx, type="coaching_replay", title="v1", description="",
                         video_ref="r/same", interest_tags=["x"], transcript="t1")
    o1 = _c.add_outtake(cx, parent_id=id1, title="old-clip", video_ref="/a.mp4",
                        interest_tags=[])
    id2 = _c.upsert_full(cx, type="coaching_replay", title="v2", description="",
                         video_ref="r/same", interest_tags=["y"], transcript="t2")
    assert id1 == id2  # same row
    row = _c.get_content(cx, id2)
    assert row["title"] == "v2" and row["transcript"] == "t2"
    # old out-takes cleared by the re-upsert
    assert cx.execute("SELECT COUNT(*) FROM community_content WHERE parent_id=?",
                      (id2,)).fetchone()[0] == 0


def test_add_outtake_is_free_and_linked():
    cx = _cx()
    f = _c.upsert_full(cx, type="course_session", title="F", description="",
                       video_ref="r/f", interest_tags=[], transcript="")
    o = _c.add_outtake(cx, parent_id=f, title="teaser", video_ref="/asset/t.mp4",
                       interest_tags=["thyroid"]); _c.publish(cx, o)
    row = _c.get_content(cx, o)
    assert row["tier"] == "free" and row["type"] == "outtake" and row["parent_id"] == f
    assert [r["title"] for r in _c.list_outtakes(cx, parent_id=f)] == ["teaser"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.community'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/community.py
"""Community content library store (slice 1, Layer A).

Pure sqlite. Full items (coaching_replay / course_session) are tier='paid' and
carry a Rumble embed video_ref; out-takes are tier='free', type='outtake', and
point at a full parent via parent_id. The membership gate (_is_paid_member) lives
in the route layer, so this module has no app-layer imports."""

import json

_DDL = """
CREATE TABLE IF NOT EXISTS community_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    title TEXT,
    description TEXT,
    video_ref TEXT,
    tier TEXT NOT NULL,
    interest_tags TEXT,
    parent_id INTEGER,
    transcript TEXT,
    published INTEGER DEFAULT 0,
    published_at TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_community_parent ON community_content(parent_id);
CREATE INDEX IF NOT EXISTS ix_community_videoref ON community_content(video_ref);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def init_community_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def create_content(cx, *, type, title, description, video_ref, tier,
                   interest_tags, parent_id=None, transcript=None):
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,parent_id,transcript,published,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,0,?)",
        (type, title, description, video_ref, tier,
         json.dumps(list(interest_tags or [])), parent_id, transcript, _now()))
    cx.commit()
    return cur.lastrowid


def get_content(cx, content_id):
    row = cx.execute("SELECT * FROM community_content WHERE id=?", (content_id,)).fetchone()
    return dict(row) if row else None


def publish(cx, content_id):
    cx.execute("UPDATE community_content SET published=1, published_at=? WHERE id=?",
               (_now(), content_id))
    cx.commit()


def upsert_full(cx, *, type, title, description, video_ref, interest_tags, transcript):
    """Create-or-update a full (paid) item keyed on video_ref. On update, clear the
    item's existing out-takes so a re-publish replaces rather than duplicates."""
    tags = json.dumps(list(interest_tags or []))
    existing = cx.execute("SELECT id FROM community_content WHERE video_ref=? AND type!='outtake'",
                          (video_ref,)).fetchone()
    if existing:
        cid = existing[0]
        cx.execute("UPDATE community_content SET type=?, title=?, description=?, "
                   "tier='paid', interest_tags=?, transcript=? WHERE id=?",
                   (type, title, description, tags, transcript, cid))
        cx.execute("DELETE FROM community_content WHERE parent_id=?", (cid,))
        cx.commit()
        return cid
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,transcript,published,created_at) "
        "VALUES (?,?,?,?, 'paid', ?,?,0,?)",
        (type, title, description, video_ref, tags, transcript, _now()))
    cx.commit()
    return cur.lastrowid


def add_outtake(cx, *, parent_id, title, video_ref, interest_tags):
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,parent_id,published,created_at) "
        "VALUES ('outtake', ?, '', ?, 'free', ?, ?, 0, ?)",
        (title, video_ref, json.dumps(list(interest_tags or [])), parent_id, _now()))
    cx.commit()
    return cur.lastrowid


def _row_tags(row):
    try:
        return json.loads(row["interest_tags"] or "[]")
    except Exception:
        return []


def list_outtakes(cx, parent_id=None):
    if parent_id is None:
        rows = cx.execute("SELECT * FROM community_content WHERE type='outtake' "
                          "AND published=1 ORDER BY published_at DESC").fetchall()
    else:
        rows = cx.execute("SELECT * FROM community_content WHERE type='outtake' "
                          "AND published=1 AND parent_id=? ORDER BY published_at DESC",
                          (parent_id,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "video_ref": r["video_ref"],
             "parent_id": r["parent_id"], "interest_tags": _row_tags(r)} for r in rows]


def list_full(cx):
    rows = cx.execute("SELECT * FROM community_content WHERE type!='outtake' "
                      "AND tier='paid' AND published=1 "
                      "ORDER BY published_at DESC").fetchall()
    out = []
    for r in rows:
        out.append({"id": r["id"], "type": r["type"], "title": r["title"],
                    "description": r["description"], "video_ref": r["video_ref"],
                    "interest_tags": _row_tags(r), "published_at": r["published_at"],
                    "outtakes": list_outtakes(cx, parent_id=r["id"])})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_store.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community.py tests/test_community_store.py
git commit -m "feat(community): content library store (slice 1)"
```

---

### Task 2: Member library page + tier-gated API

**Files:**
- Modify: `app.py` (add two routes near the other portal/member routes; grep `@app.route("/api/onboarding/state")` for a placement anchor)
- Create: `static/community.html`
- Test: `tests/test_community_library_api.py`

**Interfaces:**
- Consumes: `dashboard/community.py` (`init_community_tables`, `list_full`, `list_outtakes`), `_evox_ident`, `_is_paid_member`, `LOG_DB`, `STATIC`, `send_from_directory`.
- Produces: `GET /community` (serves the page), `GET /api/community/library?token=…` (tier-aware JSON).

**Contract:**
- Bad/absent token → 404 `{"error":"not_found"}`.
- Paid member (`_is_paid_member(email)` True) → `{"tier":"paid", "full":[{id,type,title,description,video_ref,interest_tags,published_at,outtakes:[…]}], "outtakes":[…]}` (from `list_full` + `list_outtakes`).
- Free member (not an active member) → `{"tier":"free", "full":[{id,type,title,description,interest_tags,published_at,teaser_outtakes:[…]}]}` — **each full item's `video_ref` is REMOVED**; only metadata + that item's out-takes (as `teaser_outtakes`) are exposed. The Rumble link is never in a free payload.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_library_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx)
        token = _ev.ensure_portal_token(cx, email, "")
        f = _c.upsert_full(cx, type="coaching_replay", title="Week 1", description="d",
                           video_ref="https://rumble.com/v-secret", interest_tags=["sleep"],
                           transcript="t"); _c.publish(cx, f)
        o = _c.add_outtake(cx, parent_id=f, title="teaser", video_ref="/portal-asset/x.mp4",
                           interest_tags=["sleep"]); _c.publish(cx, o)
        cx.commit()
    return token


def test_paid_member_sees_full_video_ref():
    c = _client(); tok = _seed_member("p@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        d = c.get(f"/api/community/library?token={tok}").get_json()
    assert d["tier"] == "paid"
    assert d["full"][0]["video_ref"] == "https://rumble.com/v-secret"
    assert d["full"][0]["outtakes"][0]["title"] == "teaser"


def test_free_member_never_sees_full_video_ref():
    c = _client(); tok = _seed_member("f@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        d = c.get(f"/api/community/library?token={tok}").get_json()
    assert d["tier"] == "free"
    item = d["full"][0]
    assert "video_ref" not in item                 # Rumble link withheld
    assert item["title"] == "Week 1"               # metadata still shown
    assert item["teaser_outtakes"][0]["title"] == "teaser"  # free clip shown


def test_bad_token_404():
    c = _client()
    assert c.get("/api/community/library?token=nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_library_api.py -q`
Expected: FAIL — route 404 (not registered).

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the onboarding routes):

```python
@app.route("/community")
def community_page():
    return send_from_directory(STATIC, "community.html")


@app.route("/api/community/library")
def community_library():
    from dashboard import community as _cm
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cm.init_community_tables(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        full = _cm.list_full(cx)
        if _is_paid_member(ident.email):
            return jsonify({"tier": "paid", "full": full,
                            "outtakes": _cm.list_outtakes(cx)})
        # Free member: strip every full item's Rumble video_ref; expose metadata
        # + the item's free out-takes only. The full link never reaches a non-member.
        teasers = []
        for it in full:
            teasers.append({"id": it["id"], "type": it["type"], "title": it["title"],
                            "description": it["description"],
                            "interest_tags": it["interest_tags"],
                            "published_at": it["published_at"],
                            "teaser_outtakes": it["outtakes"]})
        return jsonify({"tier": "free", "full": teasers})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_library_api.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Build the page (`static/community.html`)**

Create a self-contained page (inline CSS/JS, no external assets) that:
- Reads the portal token from `location.search` (`?token=…`) or `location.pathname` — match how `static/client-portal.html` obtains its token (grep it).
- `fetch('/api/community/library?token='+TOKEN)`. On `tier==="paid"`: render each `full` item as a card with its title, description, tags, and the Rumble embed (`<iframe>` from `video_ref`), plus its out-takes. On `tier==="free"`: render each item's title/description/tags and its `teaser_outtakes` (each out-take is a `<video>` playing `video_ref`), with a warm line: "Become a member to watch the full session." and a link to membership.
- Use `textContent` for all title/description/tag values (no innerHTML injection of server data).
- Title: `Community · Healing Oasis`. Copy: no em dashes, no ALL CAPS.
- Wrap the page JS in `<!-- BEGIN community script -->` / `<!-- END community script -->` and verify it parses: `node --check <(extract the script block)`.

- [ ] **Step 6: Commit**

```bash
git add app.py static/community.html tests/test_community_library_api.py
git commit -m "feat(community): member library page + tier-gated API"
```

---

### Task 3: Console publish endpoint + mp4 asset allowlist

**Files:**
- Modify: `app.py` (add one console-gated route near the community routes; extend the portal-asset filename allowlist at ~app.py:21317)
- Test: `tests/test_community_publish_api.py`

**Interfaces:**
- Consumes: `dashboard/community.py` (`upsert_full`, `add_outtake`, `publish`), `CONSOLE_SECRET`, `_db_lock`, `LOG_DB`.
- Produces: `POST /api/console/community/publish`; `.mp4` accepted by `/portal-asset/upload`.

**Sub-part A — allow mp4 out-take clips through the existing asset route.** The out-take clips are uploaded by the local tool via the existing `PUT /portal-asset/upload`, but its filename allowlist currently accepts only `mp3|pdf`, so `.mp4` is rejected. Extend it additively (mp3/pdf unchanged). At ~app.py:21317:

```python
_PORTAL_ASSET_RE = r'^[\w\-]+\.(mp3|pdf)$'
_PORTAL_ASSET_MIME = {"mp3": "audio/mpeg", "pdf": "application/pdf"}
```
becomes:
```python
_PORTAL_ASSET_RE = r'^[\w\-]+\.(mp3|pdf|mp4)$'
_PORTAL_ASSET_MIME = {"mp3": "audio/mpeg", "pdf": "application/pdf", "mp4": "video/mp4"}
```

**Contract:** header `X-Console-Key` must equal `CONSOLE_SECRET` (else 401). JSON body:
```json
{"type":"coaching_replay","title":"...","description":"...","video_ref":"https://rumble.com/v-...",
 "interest_tags":["..."],"transcript":"...",
 "outtakes":[{"title":"...","video_ref":"/portal-asset/....mp4","interest_tags":["..."]}]}
```
Behavior: `upsert_full` the full item (idempotent on `video_ref`, clears old out-takes), `add_outtake` each out-take with `parent_id` = the full id, `publish` the full item and every out-take. Returns `{"ok":true,"content_id":<id>,"outtakes":<n>}`. The out-take clip files are already uploaded (by the local tool via `/portal-asset/upload`); this endpoint receives their served URLs, not bytes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_publish_api.py
import sqlite3
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _payload(video_ref="https://rumble.com/v-1"):
    return {"type": "coaching_replay", "title": "Week 1", "description": "d",
            "video_ref": video_ref, "interest_tags": ["sleep"], "transcript": "t",
            "outtakes": [{"title": "clip A", "video_ref": "/portal-asset/a.mp4",
                          "interest_tags": ["sleep"]},
                         {"title": "clip B", "video_ref": "/portal-asset/b.mp4",
                          "interest_tags": []}]}


def test_publish_requires_console_key():
    c = _client()
    r = c.post("/api/console/community/publish", json=_payload())
    assert r.status_code == 401


def test_publish_creates_full_and_outtakes():
    c = _client()
    r = c.post("/api/console/community/publish", json=_payload(),
               headers={"X-Console-Key": appmod.CONSOLE_SECRET})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["outtakes"] == 2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx)
        full = _c.list_full(cx)
        mine = [f for f in full if f["video_ref"] == "https://rumble.com/v-1"][0]
        assert mine["title"] == "Week 1"
        assert sorted(o["title"] for o in mine["outtakes"]) == ["clip A", "clip B"]


def test_publish_is_idempotent():
    c = _client()
    h = {"X-Console-Key": appmod.CONSOLE_SECRET}
    c.post("/api/console/community/publish", json=_payload("https://rumble.com/v-dup"), headers=h)
    c.post("/api/console/community/publish", json=_payload("https://rumble.com/v-dup"), headers=h)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx)
        dup = [f for f in _c.list_full(cx) if f["video_ref"] == "https://rumble.com/v-dup"]
        assert len(dup) == 1                 # one full row, not two
        assert len(dup[0]["outtakes"]) == 2  # out-takes replaced, not doubled


def test_portal_asset_accepts_mp4_and_still_mp3():
    c = _client()
    h = {"X-Console-Key": appmod.CONSOLE_SECRET}
    r4 = c.put("/portal-asset/upload?filename=outtake-0.mp4", data=b"\x00\x01", headers=h)
    assert r4.status_code == 200 and r4.get_json()["url"].endswith("outtake-0.mp4")
    g4 = c.get("/portal-asset/outtake-0.mp4")
    assert g4.status_code == 200 and g4.mimetype == "video/mp4"
    # regression: mp3 still accepted
    r3 = c.put("/portal-asset/upload?filename=snippet.mp3", data=b"\x00", headers=h)
    assert r3.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_publish_api.py -q`
Expected: FAIL — route 404.

- [ ] **Step 3: Write minimal implementation**

First apply Sub-part A above (the `_PORTAL_ASSET_RE` / `_PORTAL_ASSET_MIME` mp4 extension at ~app.py:21317). Then add the publish route to `app.py`:

```python
@app.route("/api/console/community/publish", methods=["POST"])
def community_publish():
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import community as _cm
    body = request.get_json(force=True) or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cm.init_community_tables(cx)
        cid = _cm.upsert_full(cx, type=body.get("type", "coaching_replay"),
                              title=body.get("title", ""), description=body.get("description", ""),
                              video_ref=body.get("video_ref", ""),
                              interest_tags=body.get("interest_tags", []),
                              transcript=body.get("transcript", ""))
        n = 0
        for ot in (body.get("outtakes") or []):
            oid = _cm.add_outtake(cx, parent_id=cid, title=ot.get("title", ""),
                                  video_ref=ot.get("video_ref", ""),
                                  interest_tags=ot.get("interest_tags", []))
            _cm.publish(cx, oid); n += 1
        _cm.publish(cx, cid)
    return jsonify({"ok": True, "content_id": cid, "outtakes": n})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export DATA_DIR="$(mktemp -d)" && doppler run -p remedy-match -c prd -- env DATA_DIR="$DATA_DIR" python3 -m pytest tests/test_community_publish_api.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_community_publish_api.py
git commit -m "feat(community): console publish endpoint"
```

---

### Task 4: Cataloging helpers — transcribe + AI suggestions (`dashboard/community_catalog.py`)

**Files:**
- Create: `dashboard/community_catalog.py`
- Test: `tests/test_community_catalog.py`

**Interfaces:**
- Consumes: OpenAI client (injectable for tests).
- Produces:
  - `transcribe(audio_path, *, client=None) -> {"text": str, "segments": [{"start":float,"end":float,"text":str}]}`
  - `suggest_catalog(transcript_text, *, client=None) -> {"title": str, "interest_tags": [str], "outtakes": [{"start":float,"end":float,"title":str,"reason":str}]}`

**Design note:** both call the OpenAI client (`openai.OpenAI()` when `client is None`), matching `dashboard/fmp_biofield.py`. `suggest_catalog` uses `gpt-4o` with `response_format={"type":"json_object"}` and returns the parsed object; it degrades to empty structures on any failure (never raises).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_catalog.py
import json
from dashboard import community_catalog as _cat


class _FakeTranscription:
    text = "hello world this is the talk"
    segments = [{"start": 0.0, "end": 2.0, "text": "hello world"},
                {"start": 2.0, "end": 5.0, "text": "this is the talk"}]


class _FakeClient:
    """Mimics the openai client surface used by the module."""
    def __init__(self, chat_json):
        self._chat_json = chat_json
        self.audio = self
        self.transcriptions = self
        self.chat = self
        self.completions = self

    def create(self, **kw):
        if kw.get("model") == "whisper-1":
            return _FakeTranscription()
        # chat completion
        class _M:
            def __init__(s, c): s.message = type("x", (), {"content": c})
        return type("R", (), {"choices": [_M(self._chat_json)]})


def test_transcribe_returns_text_and_segments():
    client = _FakeClient("{}")
    out = _cat.transcribe("/tmp/whatever.mp4", client=client)
    assert out["text"].startswith("hello world")
    assert out["segments"][0]["end"] == 2.0


def test_suggest_catalog_parses_json():
    payload = json.dumps({"title": "Sleep and Adrenals",
                          "interest_tags": ["sleep", "adrenals"],
                          "outtakes": [{"start": 2.0, "end": 5.0, "title": "The adrenal tip",
                                        "reason": "punchy standalone tip"}]})
    client = _FakeClient(payload)
    out = _cat.suggest_catalog("hello world this is the talk", client=client)
    assert out["title"] == "Sleep and Adrenals"
    assert out["interest_tags"] == ["sleep", "adrenals"]
    assert out["outtakes"][0]["title"] == "The adrenal tip"


def test_suggest_catalog_degrades_on_bad_json():
    client = _FakeClient("not json")
    out = _cat.suggest_catalog("x", client=client)
    assert out == {"title": "", "interest_tags": [], "outtakes": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_catalog.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/community_catalog.py
"""Local cataloging helpers (run on Glen's Mac, not in the deployed app).

transcribe() runs Whisper over a recording; suggest_catalog() asks an LLM for a
title, interest tags, and out-take moments from the transcript. Both accept an
injectable client for tests and use openai.OpenAI() otherwise (pattern:
dashboard/fmp_biofield.py)."""

import json
import os

_SUGGEST_SYSTEM = (
    "You catalog a recorded health coaching or course session for a members "
    "community. Given the transcript, return STRICT JSON with keys: "
    "title (short, warm, no em dashes, no ALL CAPS), "
    "interest_tags (3-7 lowercase topic tags), "
    "outtakes (2-4 objects, each {start, end, title, reason}) picking short "
    "self-contained highlight moments that tease the full session. "
    "start/end are seconds. Only JSON."
)


def _client(client):
    if client is not None:
        return client
    import openai
    return openai.OpenAI()


def transcribe(audio_path, *, client=None):
    """Whisper transcription → {"text", "segments":[{"start","end","text"}]}."""
    c = _client(client)
    with open(audio_path, "rb") as fh:
        r = c.audio.transcriptions.create(model="whisper-1", file=fh,
                                          response_format="verbose_json")
    segs = []
    for s in (getattr(r, "segments", None) or []):
        # segments may be dicts or objects depending on client version
        get = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
        segs.append({"start": float(get("start")), "end": float(get("end")),
                     "text": get("text")})
    return {"text": getattr(r, "text", "") or "", "segments": segs}


def suggest_catalog(transcript_text, *, client=None):
    """LLM → {"title","interest_tags","outtakes"}; degrades to empty on any error."""
    empty = {"title": "", "interest_tags": [], "outtakes": []}
    try:
        c = _client(client)
        r = c.chat.completions.create(
            model=os.environ.get("COMMUNITY_CATALOG_MODEL", "gpt-4o"),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _SUGGEST_SYSTEM},
                      {"role": "user", "content": transcript_text[:120000]}])
        data = json.loads(r.choices[0].message.content)
        return {"title": data.get("title", "") or "",
                "interest_tags": list(data.get("interest_tags", []) or []),
                "outtakes": list(data.get("outtakes", []) or [])}
    except Exception as e:
        print(f"[community-catalog] suggest failed: {e!r}", flush=True)
        return empty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_catalog.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community_catalog.py tests/test_community_catalog.py
git commit -m "feat(community): local cataloging helpers (whisper + AI suggest)"
```

---

### Task 5: Cut out-takes + publish client (`dashboard/community_catalog.py`)

**Files:**
- Modify: `dashboard/community_catalog.py`
- Test: `tests/test_community_publish_client.py`

**Interfaces:**
- Consumes: `dashboard/video_trim.py:trim_video`, `dashboard/biofield_portal_publish.py:upload_asset`, `requests` (injectable).
- Produces:
  - `cut_outtakes(src_path, outtakes, *, workdir, trimmer=None) -> [{"title","interest_tags","path"}]` — cut each approved out-take `{start,end,title,interest_tags}` into `workdir/outtake-<i>.mp4` via `trim_video`.
  - `publish_session(*, base_url, console_key, full, outtake_files, uploader=None, poster=None) -> dict` — upload each out-take file (via `upload_asset`), then POST the assembled catalog entry to `/api/console/community/publish`; return the response JSON. `uploader`/`poster` injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_publish_client.py
from dashboard import community_catalog as _cat


def test_cut_outtakes_calls_trimmer_per_range(tmp_path):
    calls = []
    def fake_trim(src, dst, start, end, *, runner=None):
        calls.append((start, end)); open(dst, "wb").close()
    outs = _cat.cut_outtakes("/src.mp4",
        [{"start": 2.0, "end": 5.0, "title": "A", "interest_tags": ["x"]},
         {"start": 10.0, "end": 14.0, "title": "B", "interest_tags": []}],
        workdir=str(tmp_path), trimmer=fake_trim)
    assert calls == [(2.0, 5.0), (10.0, 14.0)]
    assert outs[0]["title"] == "A" and outs[0]["path"].endswith(".mp4")


def test_publish_session_uploads_then_posts(tmp_path):
    f1 = tmp_path / "o0.mp4"; f1.write_bytes(b"x")
    uploaded = []
    def fake_upload(data, filename, *, base_url, console_key, http_put=None):
        uploaded.append(filename); return f"/portal-asset/{filename}"
    posted = {}
    def fake_post(url, *, json=None, headers=None, timeout=None):
        posted["url"] = url; posted["body"] = json; posted["key"] = headers.get("X-Console-Key")
        class _R:
            status_code = 200
            def json(self): return {"ok": True, "content_id": 7, "outtakes": 1}
        return _R()
    out = _cat.publish_session(
        base_url="https://prod.example", console_key="SEKRET",
        full={"type": "coaching_replay", "title": "T", "description": "d",
              "video_ref": "https://rumble.com/v-1", "interest_tags": ["s"], "transcript": "t"},
        outtake_files=[{"title": "A", "interest_tags": ["s"], "path": str(f1)}],
        uploader=fake_upload, poster=fake_post)
    assert out["content_id"] == 7
    assert posted["url"].endswith("/api/console/community/publish")
    assert posted["key"] == "SEKRET"
    assert posted["body"]["outtakes"][0]["video_ref"] == "/portal-asset/o0.mp4"
    assert posted["body"]["video_ref"] == "https://rumble.com/v-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_publish_client.py -q`
Expected: FAIL — `cut_outtakes`/`publish_session` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/community_catalog.py`:

```python
def cut_outtakes(src_path, outtakes, *, workdir, trimmer=None):
    """Cut each approved out-take range into workdir/outtake-<i>.mp4. Returns a list
    of {title, interest_tags, path}. trimmer injectable (defaults to video_trim)."""
    import os as _os
    if trimmer is None:
        from dashboard.video_trim import trim_video as trimmer
    results = []
    for i, ot in enumerate(outtakes):
        dst = _os.path.join(workdir, f"outtake-{i}.mp4")
        trimmer(src_path, dst, float(ot["start"]), float(ot["end"]))
        results.append({"title": ot.get("title", ""),
                        "interest_tags": list(ot.get("interest_tags", []) or []),
                        "path": dst})
    return results


def publish_session(*, base_url, console_key, full, outtake_files,
                    uploader=None, poster=None):
    """Upload each out-take clip file, then POST the catalog entry to the prod
    publish endpoint. Returns the response JSON. uploader/poster injectable."""
    import os as _os
    if uploader is None:
        from dashboard.biofield_portal_publish import upload_asset as uploader
    if poster is None:
        import requests
        poster = requests.post
    outtakes = []
    for f in outtake_files:
        with open(f["path"], "rb") as fh:
            data = fh.read()
        filename = _os.path.basename(f["path"])
        url = uploader(data, filename, base_url=base_url, console_key=console_key)
        outtakes.append({"title": f["title"], "interest_tags": f["interest_tags"],
                         "video_ref": url})
    body = {"type": full["type"], "title": full["title"],
            "description": full.get("description", ""), "video_ref": full["video_ref"],
            "interest_tags": full.get("interest_tags", []),
            "transcript": full.get("transcript", ""), "outtakes": outtakes}
    r = poster(f"{base_url.rstrip('/')}/api/console/community/publish",
               json=body, headers={"X-Console-Key": console_key}, timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"publish failed {r.status_code}")
    return r.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_publish_client.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/community_catalog.py tests/test_community_publish_client.py
git commit -m "feat(community): out-take cutting + publish client"
```

---

### Task 6: Local cataloging app (`community_local_app.py`)

**Files:**
- Create: `community_local_app.py`
- Test: `tests/test_community_local_app.py`

**Interfaces:**
- Consumes: `dashboard/community_catalog.py` (`transcribe`, `suggest_catalog`, `cut_outtakes`, `publish_session`).
- Produces: a standalone Flask app (`create_app()`), run on Glen's Mac. Routes: `GET /` (the authoring UI), `POST /analyze` (path + rumble_url + type → transcript + suggestions JSON), `POST /publish` (approved full fields + approved out-takes → cut + publish, returns the prod response).

**Design note:** mirror `biofield_local_app.py`'s structure (`create_app()`, `run(host="127.0.0.1", port=…)`, a `--port` arg). The routes are thin glue over Task 4/5 functions; test them with those functions monkeypatched so no real Whisper/ffmpeg/network runs. The HTML UI (served by `GET /`) is a simple single page: a form for the source file path + Rumble URL + type; an Analyze button that shows the suggested title/tags/out-take ranges as editable fields with accept checkboxes; a Publish button that sends the approved set to `POST /publish`. No em dashes, no ALL CAPS in copy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_community_local_app.py
from unittest import mock
import community_local_app as cla


def _client():
    app = cla.create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_analyze_returns_transcript_and_suggestions():
    c = _client()
    with mock.patch.object(cla, "transcribe",
                           return_value={"text": "hello", "segments": []}), \
         mock.patch.object(cla, "suggest_catalog",
                           return_value={"title": "T", "interest_tags": ["a"],
                                         "outtakes": [{"start": 1, "end": 3,
                                                       "title": "clip", "reason": "r"}]}):
        r = c.post("/analyze", json={"path": "/tmp/x.mp4",
                                     "rumble_url": "https://rumble.com/v-1",
                                     "type": "coaching_replay"})
    d = r.get_json()
    assert d["suggestions"]["title"] == "T"
    assert d["transcript"] == "hello"


def test_publish_cuts_and_publishes():
    c = _client()
    with mock.patch.object(cla, "cut_outtakes",
                           return_value=[{"title": "clip", "interest_tags": [], "path": "/tmp/o0.mp4"}]) as cut, \
         mock.patch.object(cla, "publish_session",
                           return_value={"ok": True, "content_id": 9, "outtakes": 1}) as pub:
        r = c.post("/publish", json={
            "path": "/tmp/x.mp4",
            "full": {"type": "coaching_replay", "title": "T", "description": "d",
                     "video_ref": "https://rumble.com/v-1", "interest_tags": ["a"],
                     "transcript": "hello"},
            "outtakes": [{"start": 1, "end": 3, "title": "clip", "interest_tags": []}]})
    assert r.get_json()["content_id"] == 9
    assert cut.called and pub.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_community_local_app.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# community_local_app.py
"""Local Community cataloging tool — runs on Glen's Mac, NOT the deployed app.

  python3 community_local_app.py --port 8012
  open http://127.0.0.1:8012

Point it at a source recording file + the Rumble unlisted URL of the full
published replay. It runs Whisper + an LLM to suggest a title, interest tags,
and out-take moments; you approve; it cuts the out-takes with ffmpeg, uploads
them, and publishes the catalog entry to prod. Mirrors biofield_local_app.py."""

import argparse
import os
import tempfile

from flask import Flask, request, jsonify

from dashboard.community_catalog import (transcribe, suggest_catalog,
                                         cut_outtakes, publish_session)

_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Community cataloging</title></head><body style="font-family:system-ui;max-width:820px;margin:2rem auto">
<h1>Community cataloging</h1>
<p>Point at a recording file and paste the Rumble unlisted link of the full replay.</p>
<label>Source file path <input id=path size=60></label><br>
<label>Rumble URL <input id=rumble size=60></label><br>
<label>Type
 <select id=type><option value=coaching_replay>Coaching replay</option>
 <option value=course_session>Course session</option></select></label><br>
<button onclick=analyze()>Analyze</button>
<div id=out></div>
<!-- BEGIN community local script -->
<script>
async function analyze(){
  const b={path:path.value,rumble_url:rumble.value,type:type.value};
  const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const d=await r.json(); render(d);
}
function render(d){
  const s=d.suggestions||{};
  out.innerHTML='<h2>Suggested</h2>';
  const t=document.createElement('div');
  t.innerHTML='Title <input id=title size=60>';
  out.appendChild(t); document.getElementById('title').value=s.title||'';
  const tags=document.createElement('div');
  tags.innerHTML='Tags (comma) <input id=tags size=60>';
  out.appendChild(tags); document.getElementById('tags').value=(s.interest_tags||[]).join(', ');
  out.appendChild(document.createElement('hr'));
  window._outtakes=(s.outtakes||[]);
  (s.outtakes||[]).forEach((o,i)=>{
    const d2=document.createElement('div');
    const cap=document.createTextNode(' ['+o.start+'-'+o.end+'] '+(o.reason||''));
    const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.id='ot'+i;
    const ti=document.createElement('input');ti.id='ott'+i;ti.size=40;ti.value=o.title||'';
    d2.appendChild(cb);d2.appendChild(ti);d2.appendChild(cap);out.appendChild(d2);
  });
  const pb=document.createElement('button');pb.textContent='Publish';pb.onclick=publish;out.appendChild(pb);
  window._ctx={path:path.value,rumble:rumble.value,type:type.value,transcript:d.transcript||''};
}
async function publish(){
  const outs=(window._outtakes||[]).map((o,i)=>({start:o.start,end:o.end,
     title:(document.getElementById('ott'+i)||{}).value||o.title,
     interest_tags:[], _on:(document.getElementById('ot'+i)||{}).checked}))
     .filter(o=>o._on).map(({_on,...r})=>r);
  const tags=document.getElementById('tags').value.split(',').map(x=>x.trim()).filter(Boolean);
  const full={type:window._ctx.type,title:document.getElementById('title').value,
     description:'',video_ref:window._ctx.rumble,interest_tags:tags,transcript:window._ctx.transcript};
  const r=await fetch('/publish',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({path:window._ctx.path,full:full,outtakes:outs})});
  const d=await r.json();out.innerHTML='<p>Published. content_id='+(d.content_id||'?')+', out-takes='+(d.outtakes||0)+'</p>';
}
</script>
<!-- END community local script -->
</body></html>"""


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return _PAGE

    @app.route("/analyze", methods=["POST"])
    def analyze():
        b = request.get_json(force=True) or {}
        tr = transcribe(b["path"])
        sug = suggest_catalog(tr["text"])
        return jsonify({"transcript": tr["text"], "suggestions": sug})

    @app.route("/publish", methods=["POST"])
    def publish():
        b = request.get_json(force=True) or {}
        base = os.environ["PUBLIC_BASE_URL"]; key = os.environ["CONSOLE_SECRET"]
        with tempfile.TemporaryDirectory() as wd:
            files = cut_outtakes(b["path"], b.get("outtakes", []), workdir=wd)
            resp = publish_session(base_url=base, console_key=key,
                                   full=b["full"], outtake_files=files)
        return jsonify(resp)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_community_local_app.py -q`
Expected: PASS (2 passed). Then verify the page JS parses: `node --check <(python3 -c "import community_local_app as m,re; print(re.search(r'<script>(.*)</script>', m._PAGE, re.S).group(1))")`.

- [ ] **Step 5: Commit**

```bash
git add community_local_app.py tests/test_community_local_app.py
git commit -m "feat(community): local cataloging app (Mac authoring UI)"
```

---

## Definition of Done

- Prod: a `community_content` store, a `/community` library page + `/api/community/library` tier-gated API (free members never receive a full Rumble `video_ref`), and a `CONSOLE_SECRET`-gated `/api/console/community/publish` endpoint (idempotent on `video_ref`).
- Local: `community_local_app.py` on the Mac transcribes a recording, suggests title/tags/out-takes, cuts approved out-takes with ffmpeg, uploads them via the existing asset mechanism, and publishes the catalog entry.
- All new tests pass; EVOX/consult/triage/masterclass/onboarding untouched.
- No new required prod env; Rumble for full videos, existing `/portal-asset/upload` for out-take clips.

## Deferred (not in this plan)

- Layer B (react/like/block, comments), Layer C (AI curation feed + opt-in matchmaking, honoring the journal privacy line).
- Pipeline automation: Zoom cloud-recording auto-fetch (needs the Zoom S2S app re-enabled), YouTube back-catalog bulk import, scheduling.
- Member-featured out-takes + consent/approval flow.
- Leak-proof full-replay paywalling (self-hosted token-streamed video) if unlisted-Rumble leakage becomes a real problem.
- A console "add content" form on an existing console page (the local tool is the ingest path for slice 1; a manual console form can be a later convenience).
