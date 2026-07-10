# Species + the animal greeting — Implementation Plan (Slice 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An animal's portal greets "Give our Aloha to Sasha," not "Aloha Sasha." Behind a flag, dark until flipped alongside `DEPENDENT_TOS_ENABLED`.

**Architecture:** Mirrors Slice 1 — local source → push → prod table → member-aware payload → flag-gated render. Species lives on the E4L web portal; `fetch-e4l-details.py` scrapes it into `e4l_clients`; a pusher mirrors it into a prod `client_species` table; the portal payload gains `is_animal`/`animal_name`; the greeting reads them.

**Tech Stack:** Python 3, Flask, sqlite3, vanilla JS, pytest, headless Chrome, Playwright (the E4L scrape, operator step).

## Global Constraints

- Flag `ANIMAL_GREETING_ENABLED`, default OFF. Off → payload never gains `is_animal`/`animal_name`, and the greeting is byte-identical to today.
- `is_animal = bool(species) and species.strip().lower() != "human"`. `Cat`/`Dog`/`Horse` and any operator-typed value are animals; blank or `Human` is not. Case-insensitive.
- The greeting uses `animal_name`, never the account name. Fall back to the first name if `animal_name` is blank, so an animal never renders "Give our Aloha to ".
- Member-aware: computed from `email_for_reports` (already re-pointed by `?member=`), so a caregiver on the pet's tab gets the animal greeting. Never from the token holder's own email.
- Best-effort: any exception building the block returns nothing and never breaks the portal load.
- The pusher and all reads of `e4l.db` open it READ-ONLY (`file:...?mode=ro`).
- No email is sent anywhere in this slice.

## Facts (verified 2026-07-10)

- `e4l_clients` (live) has no `species`, `animal_name`, or `detail_scraped` column.
- `02 Skills/fetch-e4l-details.py` scrapes `SpeciesID` (dropdown text) + `AnimalName` per client from the E4L View/Edit page, writes `species`, scopes by `WHERE detail_scraped = 0`.
- `static/client-portal.html:650`: `<div class="hello">Aloha ${esc(first)}</div>`; `first` is the first token of `d.name`; `d.name` = `_member_name or portal.get("name")` (member-aware).
- Sync endpoints to mirror: `api_console_client_scans_sync` (app.py:10768), `api_console_scan_recommendations_sync` (10820), `api_console_scan_recommendations_read` (10870).
- Pusher to mirror: `02 Skills/e4l-scan-recommendations-push.py`.

## File Structure

| file | responsibility |
|---|---|
| `dashboard/client_species.py` (create, deploy-chat) | pure sqlite store: table + upsert + get + `is_animal` |
| `app.py` (modify, deploy-chat) | `/api/console/client-species/sync`, `GET /api/console/client-species`, `POST` override; `_client_species_for()`; payload keys |
| `static/client-portal.html` (modify, deploy-chat) | the flag-gated greeting |
| `tests/test_client_species*.py` (create, deploy-chat) | store, endpoints, payload |
| `02 Skills/setup-e4l-db.py` (modify, vault) | ALTER `e4l_clients` for species/animal_name/detail_scraped |
| `02 Skills/e4l-species-push.py` (create, vault) | read e4l.db RO, POST species+animal_name |
| `02 Skills/e4l-daily-watch.sh` (modify, vault) | call the pusher |

Tasks 1–5 land in deploy-chat and must merge+deploy before the vault pusher (Task 6) can POST. Task 7 is operational (controller): the scrape + backfill + flag.

---

### Task 1: the `client_species` store

**Files:** Create `dashboard/client_species.py`; Test `tests/test_client_species.py`

**Interfaces:**
- `init_table(cx) -> None`
- `upsert(cx, email, species, animal_name) -> None` — idempotent on `email`.
- `get(cx, email) -> dict | None` — `{"species","animal_name","is_animal"}`, or `None` if absent.
- `is_animal(species) -> bool` — `bool(species) and species.strip().lower() != "human"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client_species.py
"""Which portal clients are animals, and by what name to greet them.

Species comes from E4L (Human/Cat/Dog/Horse) or an operator-typed value for any other
mammal. is_animal = set and not Human. The greeting uses animal_name (Sasha), never the
account name.
"""
import sqlite3

import pytest

from dashboard import client_species as cs


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    cs.init_table(con)
    yield con
    con.close()


@pytest.mark.parametrize("species,expected", [
    ("Cat", True), ("Dog", True), ("Horse", True), ("Rabbit", True),
    ("Human", False), ("human", False), ("  HUMAN ", False), ("", False), (None, False),
])
def test_is_animal(species, expected):
    assert cs.is_animal(species) is expected


def test_upsert_and_get(cx):
    cs.upsert(cx, "care@example.com", "Cat", "Sasha")
    r = cs.get(cx, "care@example.com")
    assert r == {"species": "Cat", "animal_name": "Sasha", "is_animal": True}


def test_get_absent_is_none(cx):
    assert cs.get(cx, "nobody@example.com") is None


def test_upsert_is_idempotent_and_updates(cx):
    cs.upsert(cx, "e@x.com", "Cat", "Sasha")
    cs.upsert(cx, "e@x.com", "Rabbit", "Thumper")     # operator override wins
    assert cx.execute("SELECT COUNT(*) FROM client_species").fetchone()[0] == 1
    assert cs.get(cx, "e@x.com")["species"] == "Rabbit"


def test_email_is_normalised(cx):
    cs.upsert(cx, "  Care@Example.COM ", "Dog", "Rex")
    assert cs.get(cx, "care@example.com")["animal_name"] == "Rex"


def test_a_human_row_is_stored_but_not_an_animal(cx):
    cs.upsert(cx, "person@example.com", "Human", "")
    assert cs.get(cx, "person@example.com")["is_animal"] is False
```

- [ ] **Step 2: Run and watch them fail**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_client_species.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name 'client_species'`

- [ ] **Step 3: Implement**

```python
# dashboard/client_species.py
"""Per-client species + animal name, mirrored from the local e4l.db into prod so the
portal can greet an animal correctly. Pure sqlite; no Flask, no network.

Source: the E4L scrape (Human/Cat/Dog/Horse + AnimalName), or an operator override for
any other mammal E4L cannot represent. is_animal = set and not Human — so a new species
needs no code change.
"""
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def is_animal(species):
    return bool(species) and (species or "").strip().lower() != "human"


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_species (
            email        TEXT PRIMARY KEY,
            species      TEXT,
            animal_name  TEXT,
            synced_at    TEXT
        )
    """)
    cx.commit()


def upsert(cx, email, species, animal_name):
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT INTO client_species (email, species, animal_name, synced_at) "
        "VALUES (?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
        "species=excluded.species, animal_name=excluded.animal_name, synced_at=excluded.synced_at",
        (e, (species or "").strip(), (animal_name or "").strip(), _now()))
    cx.commit()


def get(cx, email):
    row = cx.execute("SELECT species, animal_name FROM client_species WHERE email=?",
                     (_norm(email),)).fetchone()
    if not row:
        return None
    species, animal_name = row[0], row[1]
    return {"species": species, "animal_name": animal_name, "is_animal": is_animal(species)}
```

- [ ] **Step 4: Green.** Expected `13 passed`.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat
git add dashboard/client_species.py tests/test_client_species.py
git commit -m "feat(portal): client_species store — who is an animal, and by what name"
```

---

### Task 2: the console endpoints (sync, read, override)

**Files:** Modify `app.py` (beside `api_console_scan_recommendations_read`, ~10870); Test `tests/test_client_species_api.py`

**Interfaces:**
- `POST /api/console/client-species/sync` — body `{"batch":[{"email","species","animal_name"}]}` → `{"ok","count"}`. Console-gated, `_db_lock`, a bad row is skipped.
- `GET /api/console/client-species` — no email → `{"ok","total","animals"}`; `?email=` adds `{"email","species","animal_name","is_animal"}`.
- `POST /api/console/client-species` — operator override, one `{"email","species","animal_name"}` → upsert → `{"ok","email","is_animal"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client_species_api.py
"""Console sync + read + operator override for client species."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_species as cs

HDRS = {"X-Console-Key": "testkey"}
BATCH = [{"email": "care@example.com", "species": "Cat", "animal_name": "Sasha"},
         {"email": "person@example.com", "species": "Human", "animal_name": ""}]


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return app.app.test_client()


def _row(tmp_db, email):
    with sqlite3.connect(tmp_db) as cx:
        cs.init_table(cx)
        return cs.get(cx, email)


def test_sync_requires_the_key(client):
    assert client.post("/api/console/client-species/sync", json={"batch": BATCH}).status_code == 401


def test_sync_writes_rows(client, tmp_db):
    r = client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    assert r.status_code == 200 and r.get_json()["count"] == 2
    assert _row(tmp_db, "care@example.com")["is_animal"] is True
    assert _row(tmp_db, "person@example.com")["is_animal"] is False


def test_sync_missing_batch_is_400(client):
    assert client.post("/api/console/client-species/sync", headers=HDRS, json={}).status_code == 400


def test_sync_skips_a_bad_row(client, tmp_db):
    bad = {"batch": ["nope", {"email": "care@example.com", "species": "Cat", "animal_name": "Sasha"}]}
    r = client.post("/api/console/client-species/sync", headers=HDRS, json=bad)
    assert r.status_code == 200 and r.get_json()["count"] == 1


def test_read_no_email_returns_corpus_counts_only(client):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    b = client.get("/api/console/client-species", headers=HDRS).get_json()
    assert set(b) == {"ok", "total", "animals"}
    assert b["total"] == 2 and b["animals"] == 1


def test_read_with_email(client):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    b = client.get("/api/console/client-species?email=care@example.com", headers=HDRS).get_json()
    assert b["is_animal"] is True and b["animal_name"] == "Sasha"


def test_override_upserts_and_wins(client, tmp_db):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    r = client.post("/api/console/client-species", headers=HDRS,
                    json={"email": "care@example.com", "species": "Rabbit", "animal_name": "Thumper"})
    assert r.status_code == 200 and r.get_json()["is_animal"] is True
    assert _row(tmp_db, "care@example.com")["species"] == "Rabbit"


def test_override_requires_the_key(client):
    assert client.post("/api/console/client-species",
                       json={"email": "x@y.com", "species": "Dog", "animal_name": "Rex"}).status_code == 401
```

- [ ] **Step 2: Run and watch them fail** (404 on the routes)

- [ ] **Step 3: Implement** — insert into `app.py` beside `api_console_scan_recommendations_read`:

```python
@app.route("/api/console/client-species/sync", methods=["POST"])
def api_console_client_species_sync():
    """Owner sync: upsert each client's species + animal name (from the local E4L scrape,
    since prod can't read e4l.db). Sibling of client-scans/sync. Sends nothing."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_species as _cs
    batch = (request.get_json(silent=True) or {}).get("batch")
    if not isinstance(batch, list):
        return jsonify({"error": "batch (list) required"}), 400
    count = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cs.init_table(cx)
        for it in batch:
            if not isinstance(it, dict) or not (it.get("email") or "").strip():
                continue
            try:
                _cs.upsert(cx, it.get("email"), it.get("species"), it.get("animal_name"))
                count += 1
            except Exception as _e:
                print(f"[client-species-sync] skipped: {_e!r}", flush=True)
    return jsonify({"ok": True, "count": count})


@app.route("/api/console/client-species", methods=["GET"])
def api_console_client_species_read():
    """Owner: confirm what the scrape stored. No email → corpus counts (no client data)."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_species as _cs
    email = (request.args.get("email") or "").strip().lower()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_table(cx)
        total = cx.execute("SELECT COUNT(*) FROM client_species").fetchone()[0]
        animals = cx.execute("SELECT COUNT(*) FROM client_species "
                             "WHERE species IS NOT NULL AND lower(trim(species))<>'' "
                             "AND lower(trim(species))<>'human'").fetchone()[0]
        out = {"ok": True, "total": total, "animals": animals}
        if not email:
            return jsonify(out)
        rec = _cs.get(cx, email) or {"species": "", "animal_name": "", "is_animal": False}
    out.update({"email": email, **rec})
    return jsonify(out)


@app.route("/api/console/client-species", methods=["POST"])
def api_console_client_species_override():
    """Owner override — the 'Other' path. Set species (free text) + animal name for one
    client, for a mammal E4L cannot represent, or to correct a scrape."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import client_species as _cs
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cs.init_table(cx)
        _cs.upsert(cx, email, body.get("species"), body.get("animal_name"))
    return jsonify({"ok": True, "email": email, "is_animal": _cs.is_animal(body.get("species"))})
```

- [ ] **Step 4: Green.** Expected `8 passed`.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat
git add app.py tests/test_client_species_api.py
git commit -m "feat(portal): console sync/read/override for client species"
```

---

### Task 3: the payload

**Files:** Modify `app.py`; Test `tests/test_client_species_payload.py`

**Interfaces:**
- `_animal_greeting_enabled() -> bool`
- `_client_species_for(email) -> dict | None` — `{"is_animal","animal_name"}` when flag on and the client is an animal; `None` otherwise.
- `payload["is_animal"]`, `payload["animal_name"]` set from `email_for_reports` when present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client_species_payload.py
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_species as cs


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        cs.init_table(cx)
        cs.upsert(cx, "care@example.com", "Cat", "Sasha")
        cs.upsert(cx, "person@example.com", "Human", "")
    return app


def test_flag_off_returns_none(app_db, monkeypatch):
    monkeypatch.delenv("ANIMAL_GREETING_ENABLED", raising=False)
    assert app_db._client_species_for("care@example.com") is None


def test_flag_on_animal(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    b = app_db._client_species_for("care@example.com")
    assert b == {"is_animal": True, "animal_name": "Sasha"}


def test_flag_on_human_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    assert app_db._client_species_for("person@example.com") is None


def test_flag_on_unknown_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    assert app_db._client_species_for("stranger@example.com") is None


def test_broken_lookup_never_raises(app_db, monkeypatch):
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    monkeypatch.setattr(cs, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    assert app_db._client_species_for("care@example.com") is None
```

- [ ] **Step 2: Run and watch them fail** (`AttributeError: ... '_client_species_for'`)

- [ ] **Step 3: Implement** — beside `_scan_recommendations_enabled` in `app.py`:

```python
def _animal_greeting_enabled():
    """Animal greeting ('Give our Aloha to Sasha'). Default OFF — when off the payload
    never gains is_animal/animal_name and the greeting is byte-identical. Flip alongside
    DEPENDENT_TOS_ENABLED."""
    return (os.environ.get("ANIMAL_GREETING_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _client_species_for(email):
    """{"is_animal": True, "animal_name": ...} when the flag is on AND this client is an
    animal; None otherwise (flag off, human, unknown, or any error)."""
    if not _animal_greeting_enabled() or not email:
        return None
    try:
        from dashboard import client_species as _cs
        with sqlite3.connect(LOG_DB) as cx:
            _cs.init_table(cx)
            rec = _cs.get(cx, email)
        if rec and rec["is_animal"]:
            return {"is_animal": True, "animal_name": rec["animal_name"]}
    except Exception as _e:
        print(f"[client-species] {_e!r}", flush=True)
    return None
```

Then wire into `api_client_portal`, immediately before `return jsonify(payload)`, using the member-aware `email_for_reports`:

```python
    try:
        _sp = _client_species_for(email_for_reports)
        if _sp:
            payload["is_animal"] = _sp["is_animal"]
            payload["animal_name"] = _sp["animal_name"]
    except Exception as _e:
        print(f"[client-species/payload] {_e!r}", flush=True)
```

- [ ] **Step 4: Green.** Expected `5 passed`.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat
git add app.py tests/test_client_species_payload.py
git commit -m "feat(portal): is_animal/animal_name in the portal payload, member-aware, flag-gated"
```

---

### Task 4: the greeting

**Files:** Modify `static/client-portal.html`. Markup — render-verified, not unit-tested.

- [ ] **Step 1: Edit the hero greeting** (`static/client-portal.html:650`). Replace:

```javascript
      <div class="hello">Aloha ${esc(first)}</div>
```

with:

```javascript
      <div class="hello">${d.is_animal
          ? 'Give our Aloha to ' + esc(d.animal_name || first)
          : 'Aloha ' + esc(first)}</div>
```

When the flag is off, `d.is_animal` is absent (falsy), so this renders `Aloha ${esc(first)}` exactly as today.

- [ ] **Step 2: Syntax-check the inline JS**

```bash
cd ~/deploy-chat
python3 - <<'PY' > /tmp/portal.js
import re
src = open("static/client-portal.html").read()
print('\n;\n'.join(re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', src, re.S)))
PY
node --check /tmp/portal.js && echo "JS PARSES OK"
```

`node --check` must pass — a syntax error breaks the portal for every client.

- [ ] **Step 3: Commit**

```bash
cd ~/deploy-chat
git add static/client-portal.html
git commit -m "feat(portal): animal portals greet 'Give our Aloha to <name>'"
```

---

### Task 5: regression, render-verify, PR (controller)

- [ ] **Step 1: Full-suite regression vs origin/main** — ANSI-stripped, `test_journey_assets.py` ignored, `test_portal_concierge_eval.py::test_grounding_and_style_pass_rate` deselected on BOTH sides. `comm -23 branch base` must be empty.

- [ ] **Step 2: Render-verify both flag states** against a real animal (Sasha), with the branch running locally and a proxy injecting `is_animal`/`animal_name` (flag on) or nothing (flag off). Strip `<script>`/`<style>` before grepping the DOM. Assert:
  - flag off → hero reads "Aloha …" (byte-identical);
  - flag on → hero reads "Give our Aloha to Sasha".

- [ ] **Step 3: Push and open the PR.** Body states: flag default OFF, ships dark; flip alongside `DEPENDENT_TOS_ENABLED`; the vault pusher (Task 6) and scrape (Task 7) follow after merge+deploy.

---

### Task 6: the vault side — schema + pusher (separate vault worktree)

**Do not start until Tasks 1–5 are merged and deployed** (the pusher needs the endpoint).
Work in a vault worktree; the vault is `~/AI-Training`.

**Files:**
- Modify: `02 Skills/setup-e4l-db.py` (idempotent ALTERs)
- Create: `02 Skills/e4l-species-push.py`
- Modify: `02 Skills/e4l-daily-watch.sh`
- Test: `02 Skills/tests/test_species_push.py`

**Interfaces:**
- `setup-e4l-db.py` adds `species TEXT`, `animal_name TEXT`, `detail_scraped INTEGER DEFAULT 0` to `e4l_clients`.
- `build_payload(e4l_db_path) -> [{"email","species","animal_name"}]` for clients whose `species` is populated and whose email is non-empty.

- [ ] **Step 1: Schema — idempotent ALTERs in `setup-e4l-db.py`**

After the `e4l_clients` CREATE, add (each guarded, so re-running is a no-op):

```python
    for _col, _decl in (("species", "TEXT"), ("animal_name", "TEXT"),
                        ("detail_scraped", "INTEGER DEFAULT 0")):
        try:
            cx.execute(f"ALTER TABLE e4l_clients ADD COLUMN {_col} {_decl}")
        except Exception:
            pass   # already present
```

Run `python3 "02 Skills/setup-e4l-db.py"` once against the live `e4l.db` and confirm the three columns now exist (`PRAGMA table_info`).

- [ ] **Step 2: The pusher — TDD**

Write `02 Skills/tests/test_species_push.py` first. Copy the structure of the existing
`02 Skills/tests/test_scan_recommendations_push.py` verbatim — same `importlib.util.spec_from_file_location`
loader, same tmp `e4l.db` fixture seeding `e4l_clients` — and adapt to species. It must contain, with full
bodies:
- `test_payload_carries_species_and_animal_name`: seed a `Cat`/`Sasha` client, assert `build_payload` returns
  `[{"email","species":"Cat","animal_name":"Sasha"}]`.
- `test_a_client_with_no_species_is_skipped`: a client with `species=''` is absent from the payload.
- `test_a_client_with_no_email_is_skipped`: a client with `email=''` is absent.
- `test_the_pusher_never_needs_write_access`: monkeypatch the module's `sqlite3.connect` to capture the call
  args, assert the connection string is a `file:...?mode=ro` URI with `uri=True` (NOT a chmod test — that
  passes with or without `mode=ro`, per the Slice-1 lesson).

Then create `02 Skills/e4l-species-push.py`, mirroring `e4l-scan-recommendations-push.py`:
- `build_payload(db)`: `SELECT lower(trim(email)), species, animal_name FROM e4l_clients WHERE trim(coalesce(species,''))<>'' AND trim(coalesce(email,''))<>''`, opened `file:...?mode=ro`.
- `main()`: `--dry`/`--db`/`--batch`; POST `{"batch": [...]}` to `/api/console/client-species/sync`; `CONSOLE_SECRET` from env; catch `HTTPError` then `URLError` then timeout (the Slice-1 lesson).

The read-only test must capture the `sqlite3.connect` args and assert the `?mode=ro` URI — a chmod test passes with or without it (the Slice-1 lesson).

- [ ] **Step 3: Wire into `e4l-daily-watch.sh`** beneath the recommendations push:

```bash
$DOPPLER $PYTHON -u "02 Skills/e4l-species-push.py" || echo "species push exited non-zero; continuing"
```

Verify `bash -n "02 Skills/e4l-daily-watch.sh"`. Do not execute the watch.

- [ ] **Step 4: Commit and open a vault PR.**

---

### Task 7: operational — scrape, backfill, verify, flag (controller)

**After Task 6 merges.** These need `E4L_USERNAME`/`E4L_PASSWORD` and touch the live db + prod.

- [ ] **Step 1: Scope the scrape to the 162.** Mark all `detail_scraped=1`, then set `=0` for clients with a pushed scan:

```bash
cd ~/AI-Training && python3 - <<'PY'
import sqlite3
cx = sqlite3.connect("e4l.db")
cx.execute("UPDATE e4l_clients SET detail_scraped=1")
cx.execute("""UPDATE e4l_clients SET detail_scraped=0 WHERE client_id IN
              (SELECT DISTINCT client_id FROM e4l_scans)""")
n = cx.execute("SELECT COUNT(*) FROM e4l_clients WHERE detail_scraped=0").fetchone()[0]
cx.commit(); print("to scrape:", n)
PY
```

- [ ] **Step 2: Back up e4l.db, then run the scrape.**

```bash
cd ~/AI-Training && cp e4l.db "e4l.db.bak-species-$(date +%Y%m%d-%H%M%S)"
doppler run -p remedy-match -c prd -- python3 "02 Skills/fetch-e4l-details.py"
```

Confirm species populated: `SELECT species, COUNT(*) FROM e4l_clients WHERE detail_scraped=1 GROUP BY 1`. Sasha (client 332311) should read `Cat` with `animal_name` "Sasha".

- [ ] **Step 3: Confirm the endpoint is deployed** — `curl -s -o /dev/null -w "%{http_code}" https://illtowell.com/api/console/client-species` must be **401** (not 404).

- [ ] **Step 4: Dry-run then push.**

```bash
cd ~/AI-Training && python3 "02 Skills/e4l-species-push.py" --dry
doppler run -p remedy-match -c prd -- python3 "02 Skills/e4l-species-push.py"
```

- [ ] **Step 5: Verify in prod, no client data.**

```bash
KEY=$(doppler secrets get CONSOLE_SECRET -p remedy-match -c prd --plain)
curl -s -H "X-Console-Key: $KEY" "https://illtowell.com/api/console/client-species" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['total'],'total |',d['animals'],'animals')"
```

- [ ] **Step 6: Flip the flag — in Doppler, not the Render API** — alongside `DEPENDENT_TOS_ENABLED` when Glen is ready:

```bash
doppler secrets set ANIMAL_GREETING_ENABLED=1 -p remedy-match -c prd
```

- [ ] **Step 7: Render-verify on production** — Sasha's portal greets "Give our Aloha to Sasha"; a human's is unchanged.

---

## Non-goals

- The infoceutical-only recommendation rule (Slice 3).
- Scraping beyond the 162 portal clients.
- A client-facing species picker.
- Any change to `_portal_biofield_unlocked` or the paywall.
