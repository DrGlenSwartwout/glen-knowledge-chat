# Portal — populate `findings` in published biofield content

**Date:** 2026-07-07
**Surface:** publish path (`dashboard/biofield_portal_publish.py`), consumed by the local intake tool.
**Scope:** one function + its unit tests. No prod app change, no schema change, no new endpoint.

## Problem

The client portal's stress-pattern chips (and the tap-to-reveal detail shipped in
PR #659) render from `d.findings`. But every published portal stores
`findings: []`, so on real portals the chips never appear:

- `dashboard/biofield_portal_publish.py` → `build_portal_content()` hardcodes
  `"findings": []` (line ~146). This is the single builder of portal biofield
  content; both publish routes (`/api/console/biofield-portal`,
  `/admin/portal/upsert`) receive the content it produces.
- Prod's `api_client_portal` never merges live scan findings (`scan_context` runs
  only in the local `:8011` intake tool, not in `app.py`).
- The reveal-draft flow (`/api/e4l/reveal-draft`) writes to the *separate* console
  `biofield_reveals` store, not the client portal.

Verified live on the Donna 329741 record: her portal payload returns
`findings: 0`. The UI (PR #659) is deployed and correct; nothing feeds it.

## Goal

Bake the scan's findings (with `e4l_description`) into the published portal content
at build time, so `client_findings` fills and the chips render live.

## Decisions (locked)

- **All findings** (infoceuticals + ER/MR stresses). Described ones become tappable
  chips; blank-description ones (ER/Nutrition/Environmental) render as plain chips.
  Full pattern list, matching the shipped UI.
- **Forward-only.** The fix populates findings on the next publish/re-publish of
  each client. Existing portals stay chip-less until re-published. No backfill
  sweep in this slice.

## Design

`build_portal_content(cx, test_id, *, special_price_cents, catalog=None,
audio_url=None, report_pdf_url=None)` runs locally during publish, where
`~/AI-Training/e4l.db` is readable. It already resolves the client `email` (from
the authored report's `client`) and the `scan_date` (`rep.get("date")`).

**Change:** replace `"findings": []` with findings pulled from the scan.

- Add an injectable keyword param `scan_context=None`. When `None`, lazily import
  the real `dashboard.biofield_e4l.scan_context`. (Injectable so the pure unit
  tests never depend on the machine's `e4l.db` — mirrors the module's existing
  injectable `http_post` on `publish_to_portal`.)
- Call `scan_context(email, scan_date)` — passing the published `scan_date` as the
  `today` argument returns *that* scan (the most recent as-of that date), aligning
  findings to the exact report being published rather than "latest."
- Trim each finding to the four fields the portal consumes and stores:
  `{"code", "name", "description", "rank"}` (drop `category`/`group` — the portal
  ignores them). `scan_context` already caps each group at 12.
- Never raise: if `email` or `scan_date` is blank, or `scan_context` raises/returns
  no scan, `findings` stays `[]`. The module's "none-raising builder" contract holds.

Sketch:

```python
def build_portal_content(cx, test_id, *, special_price_cents, catalog=None,
                         audio_url=None, report_pdf_url=None, scan_context=None):
    ...
    email = (client.get("email") or "").strip().lower()   # already derived below for return
    scan_date = rep.get("date") or ""
    _scan_ctx = scan_context
    if _scan_ctx is None:
        from dashboard.biofield_e4l import scan_context as _scan_ctx
    findings = []
    if email and scan_date:
        try:
            raw = _scan_ctx(email, scan_date).get("findings") or []
            findings = [{"code": f.get("code", ""), "name": f.get("name", ""),
                         "description": f.get("description", ""), "rank": f.get("rank")}
                        for f in raw]
        except Exception:
            findings = []
    content = {
        ...
        "findings": findings,
        ...
    }
```

`email` is currently computed only in the return dict (line ~154); the change hoists
that computation up so both the `scan_context` call and the return reuse it.

## Data / PHI

No new PHI category crosses to prod. Finding *names* are the same scan results
already surfaced as the healing-path layers; `e4l_description` is generic catalog
text ("The Source Driver bioenergetically supports…"), not client-specific. This
stays within the module's "only the finished portal payload crosses to prod"
contract.

## Out of scope

- Backfilling existing published portals (forward-only, per decision).
- Any change to `app.py`, the portal render path, or the chip UI (shipped in #659).
- The reveal-draft / `biofield_reveals` console store.

## Verification

- **Unit (pytest, deterministic via injected `scan_context`):** in
  `tests/test_biofield_portal_publish_build.py`:
  1. Injected `scan_context` returns two findings (one described driver, one
     blank-description ER) → `content["findings"]` is a 2-item list, each trimmed to
     exactly `{code, name, description, rank}`; the driver keeps its description, the
     ER has `""`.
  2. `scan_context` is called with `(email, scan_date)` — capture and assert the
     args (scan-date alignment).
  3. Injected `scan_context` that raises → `content["findings"] == []` (never
     raises).
  4. Blank `scan_date` → `scan_context` not called, `findings == []`.
- **Regression:** existing `test_biofield_portal_publish_build.py` tests still pass
  (they don't assert on findings; the real `scan_context` returns `[]` when
  `e4l.db` is absent, so they stay deterministic).
- **Live (manual, optional, Glen from the intake tool):** re-publish one real
  client, then render their portal — the chips now appear. Not automated here
  because it needs the local intake flow + a real client.
