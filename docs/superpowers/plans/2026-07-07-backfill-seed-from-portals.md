# Backfill Seed-From-Portals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broaden `scripts/backfill_portal_findings.py` to seed from the portal list (covers the 119 portals with a scan here, not 3), with per-client resilience and a `--limit` for batched applies.

**Architecture:** Change the local driver only. Seed `candidate_emails` from the tokened portal list (drop the local-intake-DB read). Rename `plan_backfill`'s seed param and make it skip (not abort) on a per-candidate error. Add `--limit` and a `_post` guard in `main()`.

**Tech Stack:** Python; pytest. The driver test only module-loads the script and exercises the pure `plan_backfill` — plain `python3 -m pytest`, no Doppler.

## Global Constraints

- **One file** for the code change: `scripts/backfill_portal_findings.py` (+ its test file). No endpoint, no prod-app, no schema change.
- **Guards unchanged and still enforced:** findings-only, existing-portal-only (server-side 404), no email, no create. Seeding from portals means candidates are all existing portals; no-scan candidates skip with "no findings computed".
- **Identity-safe:** findings are computed for the portal's OWN email (`findings_for_scan_date(portal_email, scan_date)`); a client whose scans are under a different email computes `[]` and is skipped.
- **Dry-run by default;** `--apply` required to write. Idempotent.
- **Resilience:** a per-candidate error (planning) or a `_post` error (apply) is logged and skipped, never fatal.

---

### Task 1: Seed the backfill driver from portals (+ resilience, --limit)

**Files:**
- Modify: `scripts/backfill_portal_findings.py`
- Test: `tests/test_backfill_driver.py`

**Interfaces:**
- Produces: `plan_backfill(portal_emails, candidate_emails, report_dates_of, findings_of)` — same return `(patches, skips)`; a candidate whose `report_dates_of`/`findings_of` raises is skipped with `reason` starting `"error:"`. `main()` seeds `candidate_emails` from tokened portals, honors `--limit N`, and guards each `_post`.

- [ ] **Step 1: Update the tests (rename seed param + add an error-resilience test)**

In `tests/test_backfill_driver.py`, rename `intake_emails=` to `candidate_emails=` in all four existing tests, and append one new test:

```python
def test_candidate_error_is_skipped_not_fatal():
    def boom_findings(e, d):
        if e == "boom@p.com":
            raise RuntimeError("network blip")
        return _F
    patches, skips = mod.plan_backfill(
        portal_emails={"boom@p.com", "ok@p.com"},
        candidate_emails=["boom@p.com", "ok@p.com"],
        report_dates_of=lambda e: [],
        findings_of=boom_findings)
    # boom is skipped with an error reason; ok is still patched (one blip is not fatal)
    assert {"email": "ok@p.com", "scan_date": None, "findings": _F} in patches
    assert any(s["email"] == "boom@p.com" and s["reason"].startswith("error:") for s in skips)
```

The four renamed calls become (only the keyword changes):

```python
    patches, skips = mod.plan_backfill(
        portal_emails={"has@p.com"},
        candidate_emails=["missing@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: _F)
```
…and the same `intake_emails=` → `candidate_emails=` rename in `test_matched_with_report_dates_patches_each_date`, `test_matched_no_report_dates_patches_portal_record`, and `test_matched_but_no_findings_is_skipped` (nothing else in those tests changes).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_backfill_driver.py -q`
Expected: FAIL — `plan_backfill` still has the `intake_emails` keyword (TypeError: unexpected keyword argument 'candidate_emails'), and the new error test would fail (no try/except yet).

- [ ] **Step 3: Rename + harden `plan_backfill`**

Replace `plan_backfill` (lines 35-59) with:

```python
def plan_backfill(portal_emails, candidate_emails, report_dates_of, findings_of):
    """Pure planner. Returns (patches, skips).
    patches: [{email, scan_date (None=portal record), findings}]. skips: [{email, reason}].
    A candidate whose report_dates_of/findings_of raises is skipped (reason "error: ...")
    so one client's network blip can't abort the whole run."""
    patches, skips = [], []
    for email in candidate_emails:
        e = (email or "").strip().lower()
        if e not in portal_emails:
            skips.append({"email": e, "reason": "no existing portal (would create/dup)"})
            continue
        try:
            dates = report_dates_of(e) or []
            entries = []
            if dates:
                for d in dates:
                    f = findings_of(e, d) or []
                    if f:
                        entries.append({"email": e, "scan_date": d, "findings": f})
            else:
                f = findings_of(e, None) or []
                if f:
                    entries.append({"email": e, "scan_date": None, "findings": f})
        except Exception as ex:
            skips.append({"email": e, "reason": f"error: {ex}"})
            continue
        if not entries:
            skips.append({"email": e, "reason": "no findings computed"})
            continue
        patches.extend(entries)
    return patches, skips
```

- [ ] **Step 4: Seed from portals + `--limit` + `_post` guard in `main()`; drop the intake read**

Four edits in the same file:

(a) Remove the now-unused `import sqlite3` (line 16) and the `INTAKE_DB` definition (lines 25-26).

(b) Update the module docstring's Env/behavior lines to drop `BIOFIELD_DB` and say it seeds from the portal list. Replace lines 2-11 with:

```python
"""Backfill content.findings on portals published before findings were baked in at
publish time (#661). Seeds from the tokened portal list and patches each portal whose
scan (in this Mac's e4l.db) yields findings, via the findings-only endpoint. No email
is ever sent; no portal is ever created; only content.findings is written. A candidate
with no matching scan is skipped. Dry-run by default; --apply executes.

Env: CONSOLE_SECRET (console key), PORTAL_PUBLISH_BASE_URL or --base (prod base),
E4L_DB (defaults to ~/AI-Training/e4l.db via dashboard.biofield_e4l).

Run:  python3 scripts/backfill_portal_findings.py                 # dry-run, all portals
      python3 scripts/backfill_portal_findings.py --apply --limit 10   # first 10, for real
"""
```

(c) In `main()`, add the `--limit` arg and replace the intake-DB seed (lines 92-96) with a portal seed. The arg block becomes:

```python
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    ap.add_argument("--base", default=os.environ.get("PORTAL_PUBLISH_BASE_URL", ""))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of portals processed (for a batched first run)")
```

and the seed (replacing the `cx = sqlite3.connect(...)` block, lines 92-96):

```python
    # Seed from the portal list: every tokened portal is a candidate. A candidate whose
    # scan yields no findings (no e4l scan under that email) is skipped by the planner.
    candidate_emails = sorted(portal_emails)
    if args.limit:
        candidate_emails = candidate_emails[:args.limit]
```

(d) Update the `plan_backfill(...)` call (line 126) to pass `candidate_emails`, and guard the `_post` in the apply loop (lines 134-139):

```python
    patches, skips = plan_backfill(portal_emails, candidate_emails, report_dates_of, findings_of)
```

```python
        if args.apply:
            body = {"email": p["email"], "findings": p["findings"]}
            if p["scan_date"]:
                body["scan_date"] = p["scan_date"]
            try:
                res = _post(f"{base}/api/console/portal/backfill-findings", key, body)
                print(f"      -> {res}")
            except Exception as ex:
                print(f"      -> ERROR (skipped, re-runnable): {ex}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_backfill_driver.py -q`
Expected: PASS — 5 passed (4 renamed + the new error-resilience test).

- [ ] **Step 6: Commit**

Stage ONLY these two paths (never `git add -A`/`.`):

```bash
git add scripts/backfill_portal_findings.py tests/test_backfill_driver.py
git commit -m "feat(portal): backfill driver seeds from portals (covers all with a scan)"
```

---

## Notes for the reviewer / executor

- `main()`'s HTTP behavior (portal seed, `--limit`, `_post` guard) is not unit-tested (it's I/O); it is verified by the live dry-run after merge (`--apply --limit 10` first, then the rest).
- The `plan_backfill` "no existing portal" skip now never fires (the seed is the portal list) but is kept as defense-in-depth — the endpoint also 404s a non-existent portal server-side.
- `sqlite3` and `BIOFIELD_DB` are gone from the driver; it no longer touches the local intake DB.
