# Scan recommendations → production — Implementation Plan (Slice 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get every client's E4L scan recommendations into production, so a later slice can render them on the portal. Nothing client-facing changes.

**Architecture:** Production cannot read `e4l.db` (it lives on Glen's Mac). The existing pattern is: a local pusher reads `e4l.db` and POSTs to a console-gated sync endpoint, which upserts into a prod table. `client_scans` + `POST /api/console/client-scans/sync` + `02 Skills/e4l-scan-manifest-push.py` already do this for the bare scan-date manifest. This slice adds an exact sibling for the recommendations themselves.

**Tech Stack:** Python 3, Flask, sqlite3, `urllib`, pytest. Two repos: `deploy-chat` (table + endpoint) and the vault `~/AI-Training` (pusher + cron wiring).

## Global Constraints

- `section` values are exactly the strings `"Infoceuticals"` and `"miHealth Functions"`. They come from `e4l_scan_results.section_context`, which Slice 0 populated from the scan PDF's own headings. Never re-derive them from `protocol_days`.
- `priority_rank` is one document-order sequence per scan, starting at 1. `BFA`, when present, is always rank 1. Never renumber per section.
- **A scan's rows are replaced ATOMICALLY** (delete + insert, one transaction), keyed on `(email, scan_id, priority_rank)`.
  An empty or all-invalid item list DELETES NOTHING and returns 0.
  *(Supersedes the original `(email, scan_id, item_code)` key: scan 542814 legitimately lists `ER1`, `ER10`, `ER36`,
  `ER62`, `MR3`, `MR4` twice each — the source PDF prints them twice — and that key silently dropped 6 rows while
  reporting the full count. `(scan_id, priority_rank)` has zero collisions across all 5,925 source rows.)*
- **This slice sends no email and renders nothing.** No notification code, no portal payload key, no feature flag. A later slice reads the table.
- Emails are stored lowercased and trimmed, matching `client_scans`.
- Scans whose client has no email are skipped, matching `e4l-scan-manifest-push.py`. There are currently 11 such rows.
- The console endpoint is gated by `_portal_console_ok()` and writes under `_db_lock`, like every other console write.
- Never write to `~/AI-Training/e4l.db`. The pusher opens it **read-only** (`file:...?mode=ro`).

## Facts measured against the live data (2026-07-09, after Slice 0's backfill)

- `e4l_scan_results` holds 5,925 rows across 571 scans; 161 are `BFA`.
- Joined to `e4l_scans` → `e4l_clients` on a non-empty email: **5,914 rows, 162 distinct emails, 570 distinct scans**. 11 rows belong to a client with no email.
- **2,567** of those rows are `section_context = 'Infoceuticals'`; the rest are `miHealth Functions`.
- Every row resolves a label from `e4l_items` (zero unknowns) — because Slice 0 seeded `BFA`.
- Full payload as JSON is **0.62 MB**, so batching by client is comfortable.
- Zero orphan result rows (every `scan_id` has an `e4l_scans` row).

## File Structure

| file | responsibility |
|---|---|
| `dashboard/scan_recommendations.py` (create, deploy-chat) | pure sqlite store: DDL + idempotent upsert + read helpers. No Flask. |
| `app.py` (modify, deploy-chat) | one console-gated endpoint, `POST /api/console/scan-recommendations/sync`. |
| `tests/test_scan_recommendations.py` (create, deploy-chat) | store unit tests. |
| `tests/test_scan_recommendations_api.py` (create, deploy-chat) | endpoint tests. |
| `02 Skills/e4l-scan-recommendations-push.py` (create, vault) | reads `e4l.db` read-only, POSTs batches. |
| `02 Skills/e4l-daily-watch.sh` (modify, vault) | call the pusher beside the manifest push. |

Tasks 1–3 land in `deploy-chat` and must be merged and deployed before Task 4's pusher can reach the endpoint. Task 5 is an operational backfill, run by the controller.

---

### Task 1: the store

**Files:**
- Create: `dashboard/scan_recommendations.py`
- Test: `tests/test_scan_recommendations.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `init_table(cx) -> None`
  - `replace_scan(cx, email: str, scan_id: str, scan_date: str, items: list[dict]) -> int`
    where each item is `{"item_code", "priority_rank", "protocol_days", "section", "category", "label"}`.
    Replaces the scan's rows atomically. Returns rows inserted; equals `len(for_scan(...))` afterwards.
    An empty/all-invalid `items` returns 0 and deletes nothing.
  - `for_scan(cx, email: str, scan_id: str) -> list[dict]` — all rows, ordered by `priority_rank`.
  - `infoceuticals_for_scan(cx, email: str, scan_id: str) -> list[dict]` — only `section='Infoceuticals'`, ordered by `priority_rank`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_recommendations.py
"""The scan's own recommendations, mirrored into prod so the portal can show them.

Keyed on (email, scan_id, item_code): a re-push UPDATEs, never duplicates. `section`
is carried verbatim from e4l_scan_results.section_context — the scan PDF's own
headings — so "the five infoceuticals" stays a query, not a protocol_days heuristic.
"""
import sqlite3

import pytest

from dashboard import scan_recommendations as sr

EMAIL = "caregiver@example.com"
SCAN = "1037250"
DATE = "2026-07-02"

ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
    {"item_code": "ER2", "priority_rank": 3, "protocol_days": 2,
     "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
]


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    sr.init_table(con)
    yield con
    con.close()


def test_upsert_writes_every_item(cx):
    assert sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS) == 3
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_rows_come_back_in_rank_order(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, list(reversed(ITEMS)))
    assert [r["item_code"] for r in sr.for_scan(cx, EMAIL, SCAN)] == ["BFA", "ED6", "ER2"]


def test_a_repush_updates_and_never_duplicates(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_a_repush_applies_corrected_values(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    fixed = [dict(ITEMS[0], label="Big Field Aligner (BFA)", priority_rank=1)]
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, fixed)
    row = [r for r in sr.for_scan(cx, EMAIL, SCAN) if r["item_code"] == "BFA"][0]
    assert row["label"] == "Big Field Aligner (BFA)"


def test_infoceuticals_excludes_mihealth(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    got = [r["item_code"] for r in sr.infoceuticals_for_scan(cx, EMAIL, SCAN)]
    assert got == ["BFA", "ED6"]


def test_bfa_is_rank_one_and_leads_the_infoceuticals(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    first = sr.infoceuticals_for_scan(cx, EMAIL, SCAN)[0]
    assert first["item_code"] == "BFA" and first["priority_rank"] == 1


def test_email_is_normalised(cx):
    sr.upsert_recommendations(cx, "  CareGiver@Example.COM ", SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_two_scans_for_one_client_do_not_collide(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.upsert_recommendations(cx, EMAIL, "999", "2026-06-13", ITEMS[:1])
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3
    assert len(sr.for_scan(cx, EMAIL, "999")) == 1


def test_an_item_missing_its_code_is_skipped_not_stored_blank(cx):
    n = sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, [{"priority_rank": 1}])
    assert n == 0
    assert sr.for_scan(cx, EMAIL, SCAN) == []


def test_a_scan_with_no_items_writes_nothing(cx):
    assert sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, []) == 0


def test_a_blank_email_writes_nothing(cx):
    assert sr.upsert_recommendations(cx, "", SCAN, DATE, ITEMS) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_scan_recommendations.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'scan_recommendations' from 'dashboard'`

- [ ] **Step 3: Write the store**

```python
# dashboard/scan_recommendations.py
"""A scan's own recommendations, mirrored from the local e4l.db into prod.

Production cannot read e4l.db, so `02 Skills/e4l-scan-recommendations-push.py` POSTs
these rows to a console-gated endpoint. Pure sqlite: no Flask, no network.

`section` is carried verbatim from e4l_scan_results.section_context — the scan PDF's
own "INFOCEUTICALS" / "MIHEALTH FUNCTIONS" headings. It is NOT re-derived from
protocol_days, which only correlates. "The five infoceuticals" is therefore a query.

Keyed on (email, scan_id, item_code): a re-push UPDATEs. The pusher is idempotent and
runs daily, so duplicate rows would compound silently.
"""
import sqlite3
from datetime import datetime, timezone

SECTION_INFOCEUTICAL = "Infoceuticals"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_recommendations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT NOT NULL,
            scan_id       TEXT NOT NULL,
            scan_date     TEXT,
            item_code     TEXT NOT NULL,
            priority_rank INTEGER,
            protocol_days INTEGER,
            section       TEXT,
            category      TEXT,
            label         TEXT,
            synced_at     TEXT,
            UNIQUE(email, scan_id, item_code)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_email ON scan_recommendations(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_scan ON scan_recommendations(email, scan_id)")
    cx.commit()


def upsert_recommendations(cx, email, scan_id, scan_date, items):
    """Write one scan's items. Returns rows written. An item with no item_code is
    skipped rather than stored blank — a blank code would join to no e4l_item and
    render as an empty remedy on the portal."""
    e, sid = _norm(email), str(scan_id or "").strip()
    if not e or not sid:
        return 0
    written = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        code = (it.get("item_code") or "").strip()
        if not code:
            continue
        cx.execute(
            "INSERT INTO scan_recommendations "
            "(email, scan_id, scan_date, item_code, priority_rank, protocol_days, "
            " section, category, label, synced_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(email, scan_id, item_code) DO UPDATE SET "
            "scan_date=excluded.scan_date, priority_rank=excluded.priority_rank, "
            "protocol_days=excluded.protocol_days, section=excluded.section, "
            "category=excluded.category, label=excluded.label, synced_at=excluded.synced_at",
            (e, sid, (scan_date or "").strip(), code, it.get("priority_rank"),
             it.get("protocol_days"), it.get("section"), it.get("category"),
             it.get("label"), _now()))
        written += 1
    cx.commit()
    return written


def for_scan(cx, email, scan_id):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_id=? "
        "ORDER BY priority_rank", (_norm(email), str(scan_id or "").strip())).fetchall()
    return [dict(r) for r in rows]


def infoceuticals_for_scan(cx, email, scan_id):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_id=? AND section=? "
        "ORDER BY priority_rank",
        (_norm(email), str(scan_id or "").strip(), SECTION_INFOCEUTICAL)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_scan_recommendations.py -q -p no:cacheprovider`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat
git add dashboard/scan_recommendations.py tests/test_scan_recommendations.py
git commit -m "feat(e4l): scan_recommendations store, keyed on (email, scan_id, item_code)"
```

---

### Task 2: the console sync endpoint

**Files:**
- Modify: `app.py` — add the route immediately after `api_console_client_scans_sync`
- Test: `tests/test_scan_recommendations_api.py`

**Interfaces:**
- Consumes: `scan_recommendations.init_table`, `.replace_scan`, `.for_scan` from Task 1.
- Produces: `POST /api/console/scan-recommendations/sync`.
  Body: `{"batch": [{"email": str, "scans": [{"scan_id": str, "scan_date": str, "items": [ ... ]}]}]}`
  Response: `{"ok": true, "clients": int, "scans": int, "rows": int}`.
  401 without the console key. 400 when `batch` is absent or not a list.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_recommendations_api.py
"""Console sync for a scan's recommendations. Prod cannot read e4l.db, so the local
pusher POSTs here. Mirrors /api/console/client-scans/sync."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import scan_recommendations as sr

HDRS = {"X-Console-Key": "testkey"}
EMAIL = "caregiver@example.com"

BATCH = [{"email": EMAIL, "scans": [{
    "scan_id": "1037250", "scan_date": "2026-07-02", "items": [
        {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
         "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
        {"item_code": "ER2", "priority_rank": 2, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
    ]}]}]


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


def _rows(tmp_db, scan_id="1037250"):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        sr.init_table(cx)
        return sr.for_scan(cx, EMAIL, scan_id)


def test_sync_requires_the_console_key(client):
    assert client.post("/api/console/scan-recommendations/sync", json={"batch": BATCH}).status_code == 401


def test_sync_writes_the_rows(client, tmp_db):
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["rows"] == 2 and body["clients"] == 1 and body["scans"] == 1
    assert [x["item_code"] for x in _rows(tmp_db)] == ["BFA", "ER2"]


def test_section_survives_the_round_trip(client, tmp_db):
    client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    got = {x["item_code"]: x["section"] for x in _rows(tmp_db)}
    assert got == {"BFA": "Infoceuticals", "ER2": "miHealth Functions"}


def test_a_repush_is_idempotent(client, tmp_db):
    for _ in range(2):
        assert client.post("/api/console/scan-recommendations/sync", headers=HDRS,
                           json={"batch": BATCH}).status_code == 200
    assert len(_rows(tmp_db)) == 2


def test_a_missing_batch_is_rejected(client):
    assert client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={}).status_code == 400


def test_a_bad_item_does_not_abort_the_whole_batch(client, tmp_db):
    bad = [{"email": EMAIL, "scans": [{"scan_id": "1037250", "scan_date": "2026-07-02",
                                       "items": ["not-a-dict", BATCH[0]["scans"][0]["items"][0]]}]}]
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": bad})
    assert r.status_code == 200
    assert [x["item_code"] for x in _rows(tmp_db)] == ["BFA"]


def test_a_client_with_a_blank_email_is_skipped_not_fatal(client, tmp_db):
    mixed = [{"email": "", "scans": BATCH[0]["scans"]}, BATCH[0]]
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": mixed})
    assert r.status_code == 200
    assert r.get_json()["rows"] == 2
    assert len(_rows(tmp_db)) == 2


def test_this_endpoint_sends_no_email(client, tmp_db, monkeypatch):
    """Slice 1 renders nothing and notifies nobody. A future slice reads the table."""
    sent = []
    for name in ("_send_reveal_link", "_notify_client_of_reply"):
        if hasattr(_app(), name):
            monkeypatch.setattr(_app(), name, lambda *a, **k: sent.append(name))
    client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    assert sent == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_scan_recommendations_api.py -q -p no:cacheprovider`
Expected: FAIL with 404 on every POST (route does not exist).

- [ ] **Step 3: Add the endpoint**

Insert into `app.py` directly beneath `api_console_client_scans_sync`:

```python
@app.route("/api/console/scan-recommendations/sync", methods=["POST"])
def api_console_scan_recommendations_sync():
    """Owner sync: upsert each client's per-scan E4L recommendations into
    scan_recommendations (populated by the local e4l-scan-recommendations-push, since
    prod can't read e4l.db). Sibling of /api/console/client-scans/sync.

    Sends NOTHING. Slice 1 stores; a later slice renders. A bad client or a bad item is
    skipped rather than aborting a 162-client batch."""
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import scan_recommendations as _sr
    data = request.get_json(silent=True) or {}
    batch = data.get("batch")
    if not isinstance(batch, list):
        return jsonify({"error": "batch (list) required"}), 400
    clients = scans = rows = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _sr.init_table(cx)
        for it in batch:
            if not isinstance(it, dict):
                continue
            email = it.get("email")
            wrote_for_client = False
            for sc in (it.get("scans") or []):
                if not isinstance(sc, dict):
                    continue
                try:
                    n = _sr.replace_scan(
                        cx, email, sc.get("scan_id"), sc.get("scan_date"), sc.get("items") or [])
                except Exception as _e:
                    print(f"[scan-recs-sync] skipped bad scan: {_e!r}", flush=True)
                    continue
                if n:
                    rows += n
                    scans += 1
                    wrote_for_client = True
            if wrote_for_client:
                clients += 1
    return jsonify({"ok": True, "clients": clients, "scans": scans, "rows": rows})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_scan_recommendations_api.py tests/test_scan_recommendations.py -q -p no:cacheprovider`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat
git add app.py tests/test_scan_recommendations_api.py
git commit -m "feat(e4l): console sync endpoint for a scan's recommendations"
```

---

### Task 3: full-suite regression check, then PR

**Files:** none changed.

- [ ] **Step 1: Capture the baseline on origin/main**

```bash
cd ~/deploy-chat && git worktree add -f --detach /tmp/wt-slice1-base origin/main
cd /tmp/wt-slice1-base && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_journey_assets.py 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "^FAILED tests/" | sed 's/ - .*//' | sort -u > /tmp/slice1-base-fails.txt
wc -l /tmp/slice1-base-fails.txt
```

`tests/test_journey_assets.py` is excluded because it fails to import (`PIL` missing) and aborts collection. That is pre-existing.

- [ ] **Step 2: Run the same suite on the branch**

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=$HOME/deploy-chat ~/.venvs/deploy-chat311/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/test_journey_assets.py 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "^FAILED tests/" | sed 's/ - .*//' | sort -u > /tmp/slice1-branch-fails.txt
comm -23 /tmp/slice1-branch-fails.txt /tmp/slice1-base-fails.txt
```

Note: `grep -E "^FAILED"` will NOT match pytest's colourised output. The `sed` above strips ANSI first — do not remove it.

Expected: the `comm` output is EMPTY (zero regressions). If it is not, stop and fix.

- [ ] **Step 3: Remove the baseline worktree**

```bash
cd ~/deploy-chat && git worktree remove --force /tmp/wt-slice1-base
```

- [ ] **Step 4: Push the branch and open the PR**

```bash
cd ~/deploy-chat
git push -u origin sess/slice1-scan-recommendations
cat > /tmp/slice1-pr.md <<'BODY'
Slice 1 of the scan-recommendations feature: get every client's E4L scan recommendations into production.

Prod cannot read `e4l.db`, so a local pusher POSTs to a console-gated endpoint, exactly as `client_scans` + `/api/console/client-scans/sync` already do for the bare scan-date manifest. This adds the sibling for the recommendations themselves.

- `dashboard/scan_recommendations.py` — pure sqlite store, keyed on `(email, scan_id, item_code)` so the daily push UPDATEs rather than duplicating.
- `POST /api/console/scan-recommendations/sync` — console-key gated, `_db_lock`, a bad client or item is skipped rather than aborting a 162-client batch.

`section` is carried verbatim from `e4l_scan_results.section_context` (the scan PDF's own `INFOCEUTICALS` / `MIHEALTH FUNCTIONS` headings, populated by Slice 0), never re-derived from `protocol_days`.

**This slice sends no email, renders nothing, and introduces no flag.** `api_client_portal` is untouched; the table is write-only until Slice 2. There is a test named `test_this_endpoint_sends_no_email` asserting exactly that.

Full suite: zero regressions against `origin/main`.
BODY
gh pr create --base main --head sess/slice1-scan-recommendations \
  --title "Scan recommendations: store + console sync endpoint" --body-file /tmp/slice1-pr.md
```

---

### Task 4: the local pusher (vault repo)

**Do not start until Tasks 1–3 are merged and deployed**, or the pusher has no endpoint to POST to.

**Files:**
- Create: `02 Skills/e4l-scan-recommendations-push.py`
- Modify: `02 Skills/e4l-daily-watch.sh`

**Interfaces:**
- Consumes: `POST /api/console/scan-recommendations/sync` from Task 2.
- Produces: `build_payload(e4l_db_path) -> list[dict]` shaped `[{"email", "scans":[{"scan_id","scan_date","items":[...]}]}]`.

- [ ] **Step 1: Write the failing test**

```python
# 02 Skills/tests/test_scan_recommendations_push.py
import importlib.util
import sqlite3
from pathlib import Path


def _load(stem):
    p = Path(__file__).resolve().parent.parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PUSH = _load("e4l-scan-recommendations-push")


def _db(tmp_path):
    p = tmp_path / "e4l.db"
    cx = sqlite3.connect(p)
    cx.executescript("""
        CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, email TEXT);
        CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);
        CREATE TABLE e4l_items(code TEXT PRIMARY KEY, category TEXT, name TEXT);
        CREATE TABLE e4l_scan_results(id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER,
            item_code TEXT, priority_rank INTEGER, protocol_days INTEGER, section_context TEXT);
        INSERT INTO e4l_clients VALUES (1, 'Care@Example.com'), (2, '');
        INSERT INTO e4l_scans VALUES (10, 1, '2026-07-02'), (20, 2, '2026-07-03');
        INSERT INTO e4l_items VALUES ('BFA','BFA','Big Field Aligner'), ('ER2','ER','Large Intestine');
        INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank,protocol_days,section_context)
            VALUES (10,'BFA',1,15,'Infoceuticals'), (10,'ER2',2,2,'miHealth Functions'),
                   (20,'BFA',1,15,'Infoceuticals');
    """)
    cx.commit(); cx.close()
    return str(p)


def test_payload_groups_by_client_and_scan(tmp_path):
    out = PUSH.build_payload(_db(tmp_path))
    assert len(out) == 1                       # the blank-email client is skipped
    assert out[0]["email"] == "care@example.com"
    assert len(out[0]["scans"]) == 1
    assert out[0]["scans"][0]["scan_id"] == "10"


def test_items_carry_section_rank_and_label(tmp_path):
    items = PUSH.build_payload(_db(tmp_path))[0]["scans"][0]["items"]
    assert [i["item_code"] for i in items] == ["BFA", "ER2"]
    assert items[0]["section"] == "Infoceuticals" and items[0]["priority_rank"] == 1
    assert items[0]["label"] == "Big Field Aligner"
    assert items[1]["section"] == "miHealth Functions"


def test_a_client_with_no_email_is_skipped(tmp_path):
    assert all(p["email"] for p in PUSH.build_payload(_db(tmp_path)))


def test_the_pusher_never_needs_write_access_to_e4l_db(tmp_path):
    """The pusher must open e4l.db read-only. Proven by making the FILE unwritable:
    a connection that asked for write access would raise. Asserting "rows unchanged
    after a read" would pass whether or not the code opened read-only."""
    import os
    path = _db(tmp_path)
    os.chmod(path, 0o444)
    try:
        out = PUSH.build_payload(path)          # must not raise
    finally:
        os.chmod(path, 0o644)
    assert out and out[0]["scans"][0]["items"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd ~/AI-Training && python3 -m pytest "02 Skills/tests/test_scan_recommendations_push.py" -q`
Expected: FAIL — the module file does not exist.

- [ ] **Step 3: Write the pusher**

```python
#!/usr/bin/env python3
"""Push every client's per-scan E4L recommendations from the local e4l.db to prod
scan_recommendations (prod can't read e4l.db). Idempotent — the endpoint upserts on
(email, scan_id, item_code). Sends no email; a later slice renders the rows.

Mirrors e4l-scan-manifest-push.py. Auth: CONSOLE_SECRET from Doppler remedy-match/prd.
Usage: [--dry] [--db PATH] [--batch N]
"""
import argparse, json, os, sqlite3, sys, urllib.error, urllib.request

BASE = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
DEFAULT_DB = os.path.expanduser("~/AI-Training/e4l.db")


def build_payload(e4l_db_path):
    """[{email, scans:[{scan_id, scan_date, items:[{item_code, priority_rank,
    protocol_days, section, category, label}]}]}] — clients with no email are skipped.

    `section` is e4l_scan_results.section_context verbatim ('Infoceuticals' /
    'miHealth Functions'), never re-derived from protocol_days.
    """
    cx = sqlite3.connect(f"file:{e4l_db_path}?mode=ro", uri=True)
    try:
        rows = cx.execute(
            "SELECT lower(trim(cl.email)) email, s.scan_id, s.scan_date, r.item_code, "
            "       r.priority_rank, r.protocol_days, r.section_context, i.category, i.name "
            "FROM e4l_scan_results r "
            "JOIN e4l_scans s ON s.scan_id = r.scan_id "
            "JOIN e4l_clients cl ON cl.client_id = s.client_id "
            "LEFT JOIN e4l_items i ON i.code = r.item_code "
            "WHERE cl.email IS NOT NULL AND trim(cl.email) <> '' "
            "ORDER BY email, s.scan_date DESC, r.priority_rank ASC").fetchall()
    finally:
        cx.close()

    by_client = {}
    for email, sid, sdate, code, rank, days, section, cat, label in rows:
        scans = by_client.setdefault(email, {})
        sc = scans.setdefault(str(sid), {"scan_id": str(sid), "scan_date": sdate, "items": []})
        sc["items"].append({"item_code": code, "priority_rank": rank, "protocol_days": days,
                            "section": section, "category": cat, "label": label})
    return [{"email": e, "scans": list(scans.values())} for e, scans in by_client.items()]


def _post(secret, batch):
    req = urllib.request.Request(
        f"{BASE}/api/console/scan-recommendations/sync", method="POST",
        data=json.dumps({"batch": batch}).encode(),
        headers={"Content-Type": "application/json", "X-Console-Key": secret})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()

    payload = build_payload(a.db)
    n_rows = sum(len(s["items"]) for c in payload for s in c["scans"])
    n_scans = sum(len(c["scans"]) for c in payload)
    print(f"clients={len(payload)} scans={n_scans} rows={n_rows}")
    if a.dry:
        print("(dry run — nothing sent)")
        return 0

    secret = os.environ.get("CONSOLE_SECRET")
    if not secret:
        print("CONSOLE_SECRET missing (run under: doppler run -p remedy-match -c prd --)")
        return 2

    sent = 0
    for i in range(0, len(payload), a.batch):
        chunk = payload[i:i + a.batch]
        try:
            res = _post(secret, chunk)
        except urllib.error.HTTPError as e:
            print(f"  ! batch {i // a.batch + 1} failed: {e.code} {e.read()[:200]!r}")
            return 1
        sent += res.get("rows", 0)
        print(f"  batch {i // a.batch + 1}: rows={res.get('rows')} scans={res.get('scans')}")
    print(f"Done. rows upserted: {sent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests, then a dry run**

```bash
cd ~/AI-Training
python3 -m pytest "02 Skills/tests/test_scan_recommendations_push.py" -q     # expect 4 passed
python3 "02 Skills/e4l-scan-recommendations-push.py" --dry
```

The dry run must print `clients=162 scans=570 rows=5914` against the live `e4l.db`.

- [ ] **Step 5: Wire it into the daily watch**

In `02 Skills/e4l-daily-watch.sh`, directly beneath the existing manifest-push line:

```bash
$DOPPLER $PYTHON -u "02 Skills/e4l-scan-recommendations-push.py" || echo "scan-recommendations push exited non-zero; continuing"
```

Verify with `bash -n "02 Skills/e4l-daily-watch.sh"` (syntax only; do not execute the watch).

- [ ] **Step 6: Commit and open a vault PR**

```bash
cd ~/AI-Training
git add "02 Skills/e4l-scan-recommendations-push.py" "02 Skills/tests/test_scan_recommendations_push.py" "02 Skills/e4l-daily-watch.sh"
git commit -m "feat(e4l): push per-scan recommendations to prod scan_recommendations"
```

---

### Task 5: the production backfill (controller runs this)

**Files:** none.

- [ ] **Step 1: Verify the endpoint is deployed before pushing anything**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://illtowell.com/api/console/scan-recommendations/sync
```

Expected: **401** (route exists, wants a key). **404 means the deploy has not landed — stop.**

- [ ] **Step 2: Dry run against the live e4l.db**

```bash
cd ~/AI-Training && python3 "02 Skills/e4l-scan-recommendations-push.py" --dry
```

Expected: `clients=162 scans=570 rows=5914`.

- [ ] **Step 3: Push for real**

```bash
cd ~/AI-Training && doppler run -p remedy-match -c prd -- python3 "02 Skills/e4l-scan-recommendations-push.py"
```

- [ ] **Step 4: Verify in production**

```bash
KEY=$(doppler secrets get CONSOLE_SECRET -p remedy-match -c prd --plain)
curl -s -H "X-Console-Key: $KEY" -X POST -H "Content-Type: application/json" \
  -d '{"batch":[]}' https://illtowell.com/api/console/scan-recommendations/sync
```

An empty batch returns `{"ok":true,"clients":0,"scans":0,"rows":0}` and proves the route is live without writing anything. Then confirm the real counts by re-running Step 3 — a second push must report the same `rows` (upsert), and must not multiply them.

Acceptance: the second push reports `rows=5914` again, not `11828`. That is the idempotency proof, run against production.

Note the stored total will be **5,914**, matching the reported rows exactly, because `replace_scan` preserves the 6
duplicate-`item_code` rows in scan 542814 rather than collapsing them.

---

## What this plan deliberately does not do

- **No portal payload, no rendering, no flag.** `api_client_portal` is untouched. The table is write-only until Slice 2.
- **No email.** There is no notification path here, which is why the `test_this_endpoint_sends_no_email` guard exists.
- **No `band`/colour column.** Slice 0 established that colour is not extractable from `pdftotext`; `section` replaces it. The design doc's Slice 1 says `band` — it is superseded by this plan.
- **No change to `client_scans`** or the manifest push. They stay the bare date manifest.
