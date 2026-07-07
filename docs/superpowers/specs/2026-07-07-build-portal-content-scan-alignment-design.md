# Fix build_portal_content scan alignment (findings_for_scan_date)

**Date:** 2026-07-07
**Follow-up to:** #661 (baked findings at publish) — closes the latent misalignment its final-review sibling (#664) surfaced.
**Scope:** one function + two existing tests updated. No prod-app change, no schema change, no new endpoint.

## Problem

`dashboard/biofield_portal_publish.py` `build_portal_content` bakes the scan's findings
into published portal content via `scan_context(email, scan_date)`, passing the report's
`scan_date` as the `today` arg on the belief that it aligns findings to *that* scan.

It does not. `scan_context` resolves the scan via `_latest_scan(cx, email)`, which
`ORDER BY scan_date DESC LIMIT 1` and **ignores the date entirely** — `today` only feeds a
freshness message (see [[reference_scan_context_latest_only]]). So `build_portal_content`
always bakes the client's **latest** scan's findings, regardless of which report it is
publishing.

- Common case (Glen authors + publishes the current/newest scan): latest == the report's
  scan → correct.
- Re-publishing an older scan while a newer one exists: the older report gets the newer
  scan's findings → wrong findings on a client-facing report.

#664 added `dashboard/biofield_e4l.findings_for_scan_date(email, scan_date)` (date-specific,
merged-identity aware) and used it in the backfill driver. This slice makes
`build_portal_content` use it too, so publishing is correct at the source.

## Decision (locked)

**Strict, no fallback.** When the report's `scan_date` has no exactly-matching E4L scan,
findings are `[]` (never wrong findings from a different scan). The authored intake's date
equals its E4L scan date in practice (the #664 backfill confirmed exact-match resolves), so
this is the normal path; empty chips are safer than wrong chips.

## Design

In `build_portal_content`:

- Rename the injectable keyword param `scan_context=None` → `findings_provider=None`. The
  provider signature is `(email, scan_date) -> list[finding dicts]` (each `{rank, code, name,
  description, category, group}`, like `scan_context()['findings']`). When `None`, lazily
  import `dashboard.biofield_e4l.findings_for_scan_date`.
- Replace `raw = scan_context(email, scan_date).get("findings") or []` with
  `raw = findings_provider(email, scan_date) or []` (the helper returns the list directly —
  no `.get("findings")`).
- Everything else unchanged: the `if email and scan_date:` guard, the `try/except → []`
  none-raising contract, and the trim to `{code, name, description, rank}`.

Sketch of the changed block:

```python
    findings = []
    if email and scan_date:
        try:
            _fp = findings_provider
            if _fp is None:
                from dashboard.biofield_e4l import findings_for_scan_date as _fp
            raw = _fp(email, scan_date) or []
            findings = [{"code": f.get("code", ""), "name": f.get("name", ""),
                         "description": f.get("description", ""), "rank": f.get("rank")}
                        for f in raw]
        except Exception:
            findings = []
```

Update the module comment above the block to state it aligns to the exact scan_date via
`findings_for_scan_date` (drop the incorrect "passing scan_date as today aligns" wording).

## Test updates (existing #661 tests)

In `tests/test_biofield_portal_publish_build.py`:

- `test_build_populates_findings_from_scan`: inject `findings_provider=fake` where `fake`
  returns the findings **list directly** (not `{"findings": [...]}`), e.g.
  `def fake(email, today): captured["args"] = (email, today); return [ {driver}, {blank} ]`.
  Keep the assertions: 2 trimmed findings, description preserved on the driver, blank kept,
  and `captured["args"] == (email, scan_date)` (alignment still verified).
- `test_build_findings_empty_when_scan_context_raises` → rename intent to
  `test_build_findings_empty_when_provider_raises`: inject `findings_provider=boom`;
  assert `content["findings"] == []`.

No new test is required — `findings_for_scan_date`'s own date-specificity is already covered
by `tests/test_findings_for_scan_date.py` (#664).

## Out of scope

- Any change to `scan_context` / `_latest_scan` (other callers rely on latest-scan behavior).
- The portal render path, the chip UI, or the backfill (all shipped).

## Verification

- `python3 -m pytest tests/test_biofield_portal_publish_build.py -q` → all pass (the two
  updated tests + the three pre-existing `build` tests). Plain pytest — the file imports only
  `dashboard` modules, no Doppler.
- Live is implicit: the next real publish of the current scan is unchanged; a re-publish of an
  older scan now bakes that scan's findings. No separate live step needed.
