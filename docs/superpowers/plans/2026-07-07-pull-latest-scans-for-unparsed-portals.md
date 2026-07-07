# Pull Latest Scans — `--latest-only` flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--latest-only` flag to `02 Skills/scrape-e4l-http.py` so a `--clients` batch pulls exactly one PDF per client (their most-recent scan), enabling the "pull the 56 unparsed portal clients" runbook.

**Architecture:** One small change to the vault script `scrape-e4l-http.py`: a pure `_latest_only(scans)` helper (+ `_parse_date`) and a `--latest-only` arg that slices each client's scan list to the newest before the download loop. The runbook (scrape → parse → backfill) is operational and NOT part of this plan.

**Tech Stack:** Python; pytest. This is a VAULT file (`~/AI-Training/02 Skills/…`), edited directly (the vault is not a worktree) and committed to the vault git repo (`~/AI-Training`). The test loads the module via importlib and exercises only the pure helper — plain `python3 -m pytest`, no network.

## Global Constraints

- **Vault paths** (NOT the deploy-chat worktree): modify `~/AI-Training/02 Skills/scrape-e4l-http.py`; test in `~/AI-Training/02 Skills/tests/`.
- **Pure + testable:** `_latest_only`/`_parse_date` use only stdlib (`datetime`); no network, no import-time side effects (module already loads cleanly).
- **Behavior:** `--latest-only` reduces each client's scan list to a single most-recent scan (newest by parsed date, tie-broken by numeric scan id); default (flag absent) is unchanged.

---

### Task 1: `--latest-only` flag on `scrape-e4l-http.py`

**Files:**
- Modify: `~/AI-Training/02 Skills/scrape-e4l-http.py`
- Test: `~/AI-Training/02 Skills/tests/test_e4l_scrape_latest_only.py` (new)

**Interfaces:**
- Produces: `_parse_date(s) -> datetime.date`, `_latest_only(scans) -> list` (0/1 elem), and a `--latest-only` CLI flag that applies `_latest_only` per client before the download loop.

- [ ] **Step 1: Write the failing test**

Create `~/AI-Training/02 Skills/tests/test_e4l_scrape_latest_only.py`:

```python
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)


def _load():
    spec = importlib.util.spec_from_file_location(
        "scrape_e4l_http", os.path.join(SKILLS, "scrape-e4l-http.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mod = _load()


def test_latest_only_picks_newest_by_date():
    scans = [{"id": "100", "date": "5/1/2025"},
             {"id": "101", "date": "6/15/2026"},
             {"id": "99", "date": "1/1/2024"}]
    assert mod._latest_only(scans) == [{"id": "101", "date": "6/15/2026"}]


def test_latest_only_id_fallback_on_unknown_dates():
    scans = [{"id": "100", "date": "unknown"},
             {"id": "250", "date": "unknown"},
             {"id": "7", "date": "unknown"}]
    assert mod._latest_only(scans) == [{"id": "250", "date": "unknown"}]


def test_latest_only_mixed_iso_and_slash():
    scans = [{"id": "1", "date": "2026-06-20"}, {"id": "2", "date": "6/25/2026"}]
    assert mod._latest_only(scans) == [{"id": "2", "date": "6/25/2026"}]


def test_latest_only_empty():
    assert mod._latest_only([]) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest "$HOME/AI-Training/02 Skills/tests/test_e4l_scrape_latest_only.py" -q`
Expected: FAIL — `module 'scrape_e4l_http' has no attribute '_latest_only'`.

- [ ] **Step 3: Add `import datetime` + the two helpers**

In `~/AI-Training/02 Skills/scrape-e4l-http.py`, add `import datetime` to the top import block (next to `import argparse`). Then add these two functions immediately above `def _parse_clients_arg(` :

```python
def _parse_date(s):
    """Parse an E4L scan-list date ('M/D/YYYY' or 'YYYY-MM-DD') to a date. Returns
    date.min for 'unknown'/unparseable so a dated scan always sorts as newer."""
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return datetime.date.min


def _latest_only(scans):
    """Reduce a scan list [{id, date}, ...] to just the most-recent scan (a 0- or
    1-element list): newest by parsed date, tie-broken by numeric scan id (ids are
    monotonic, so this is robust even when every date is 'unknown')."""
    if not scans:
        return []
    return [max(scans, key=lambda sc: (_parse_date(sc.get("date")),
                                       int(sc.get("id") or 0)))]
```

- [ ] **Step 4: Add the `--latest-only` arg and apply it in the loop**

In `main()`'s argparse block (next to the `--all` argument), add:

```python
    p.add_argument("--latest-only", action="store_true",
                   help="download only each client's single most-recent scan")
```

In the per-client loop, immediately before the `if not args.all:` line (right after the
`scans = get_client_scans(...)` try/except that assigns `scans`), add:

```python
        if args.latest_only:
            scans = _latest_only(scans)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest "$HOME/AI-Training/02 Skills/tests/test_e4l_scrape_latest_only.py" -q`
Expected: PASS — 4 passed.

- [ ] **Step 6: Commit to the vault repo**

The vault (`~/AI-Training`) is a normal git repo (auto-snapshotted hourly); make a discrete commit so the change is traceable rather than folded into a timestamped snapshot. Stage ONLY these two paths:

```bash
cd "$HOME/AI-Training"
git add "02 Skills/scrape-e4l-http.py" "02 Skills/tests/test_e4l_scrape_latest_only.py"
git commit -m "feat(e4l): scrape-e4l-http --latest-only (one PDF per client)"
```

---

## Notes for the reviewer / executor

- Sanity-check the CLI parses: `python3 "$HOME/AI-Training/02 Skills/scrape-e4l-http.py" --clients 1,2 --latest-only --dry-run` should print the dry-run line and exit without downloading (dry-run returns before the loop, so `--latest-only` is inert there — that's fine; it only proves argparse accepts the flag).
- The operational runbook (derive 56 client_ids → scrape `--clients … --latest-only` → parse → backfill `--apply` → verify) is executed separately, with the user's go, after this flag ships. It is NOT part of this plan.
