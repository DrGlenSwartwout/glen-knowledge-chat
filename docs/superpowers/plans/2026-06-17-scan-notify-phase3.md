# Scan Notification — Phase 3 (live-unfold UX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a client opens a fresh analysis, it **unfolds in front of them** — their stress patterns appear first, then the healing-path layers assemble one by one — instead of just popping in.

**Architecture:** The client content endpoint starts returning the readable `findings` (stress patterns, minus clinician-only notes). The portal plays a one-time staged reveal the first time a given scan's analysis is shown (a `localStorage` guard per scan date); returning visits render statically. Pure UX on top of Phases 1–2 (the cold "Preparing…"→ready transition already lands here).

**Tech Stack:** Flask (one endpoint field), JS/CSS (`static/client-portal.html`). Spec: `docs/superpowers/specs/2026-06-17-scan-notify-ondemand-unfold-design.md` (component 5, the unfold). Replaces the basic "Preparing…" hand-off from Phase 2 with the polished reveal.

---

## Repo
DEPLOY-CHAT worktree `/tmp/wt-deploy-chat-5326cc61` branch `sess/5326cc61-phase3` (all tasks, one PR). Suite: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`. Baseline 1775 passed, 2 skipped.

---

## Task 1: surface `findings` on the client content endpoint

**Files:** Modify `app.py` (`api_client_portal`, ~7238); Test `tests/test_client_portal_routes.py`

The client unfold needs the stress patterns. `content.findings` already exists (the synthesis carries it); the client endpoint just doesn't return it. Return it **without** `clinical_notes` (clinician-only), whenever an analysis exists.

- [ ] **Step 1: Failing test**

```python
def test_content_endpoint_returns_findings_without_clinical_notes(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    tok = _seed_portal(appmod, "fd@y.com", "FD", {"layers": []})
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    R.upsert_report(cx, "fd@y.com", datetime.date.today().isoformat(), "s1",
        {"layers": [{"n": 1, "title": "T", "remedy": "R", "patterns": ["ED4"]}],
         "findings": [{"code": "ED4", "name": "Nerve Driver",
                       "description": "supports nerve impulses", "rank": 1,
                       "clinical_notes": "SECRET clinician note"}]}, "interested")
    cx.close()
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["findings"][0]["name"] == "Nerve Driver"
    assert j["findings"][0]["description"] == "supports nerve impulses"
    assert "clinical_notes" not in j["findings"][0]      # stripped for clients


def test_content_endpoint_findings_empty_when_none(client):
    c, appmod = client
    tok = _seed_portal(appmod, "nf@y.com", "NF", {"layers": [{"n": 1, "title": "X"}]})
    j = c.get(f"/api/portal/{tok}").get_json()
    assert j["findings"] == []
```

- [ ] **Step 2: Run → FAIL** (`-m pytest tests/test_client_portal_routes.py -q -k findings_without_clinical or findings_empty`)

- [ ] **Step 3: Implement** — in `api_client_portal`, before the final `return jsonify({...})`, build a client-safe findings list from `bf_content`:

```python
    client_findings = [{"code": f.get("code", ""), "name": f.get("name", ""),
                        "description": f.get("description", ""), "rank": f.get("rank")}
                       for f in (bf_content.get("findings") or [])]
```
  and add `"findings": client_findings,` to the returned `jsonify({...})` dict.

- [ ] **Step 4: Run → PASS.** **Step 5: FULL suite** (existing content-endpoint tests still pass — additive). **Step 6: Commit** (`-m "portal: surface stress-pattern findings (no clinical notes) on content endpoint"`)

---

## Task 2: staged live-unfold — `static/client-portal.html`

**Files:** Modify `static/client-portal.html`; no unit test (node --check + suite + manual)

The first time a given scan's analysis renders (not `pending`, not previously unfolded on this device), play the reveal; otherwise render statically. The data is all present after Task 1 (`d.findings` + the biofield `layers`).

- [ ] **Step 1: CSS** — before `</style>`, add reveal animation classes:

```css
  .reveal{opacity:0;transform:translateY(8px);animation:rvl .5s ease forwards}
  @keyframes rvl{to{opacity:1;transform:none}}
  .pat-chip{display:inline-block;margin:3px 5px 3px 0;padding:4px 10px;border-radius:14px;
    background:var(--brand-soft,rgba(61,138,82,.15));border:1px solid var(--line,#21472d);font-size:.85rem}
  @media (prefers-reduced-motion: reduce){.reveal{animation:none;opacity:1;transform:none}}
```
(Use the page's real brand variable names — confirm `--brand`/`--line`/`--cream` etc. by reading the existing styles; substitute if different.)

- [ ] **Step 2: Unfold sequencer.** Add JS that, given the biofield block + `d.findings`, decides whether to animate:
  - A guard key per scan: `const ukey = "rvl:" + (bf.scan_date || "current");` and `const firstTime = !localStorage.getItem(ukey);`.
  - When rendering a NON-pending, NON-confirmed-returning analysis for the **first time**, render in stages:
    1. **Patterns first** — a "Your scan identified these stress patterns" heading + the `d.findings` names as `.pat-chip` elements, each with the `.reveal` class and a staggered `animation-delay` (e.g. `i*120ms`).
    2. **Layers assemble** — after the patterns (a short delay), reveal the layer cards one at a time (each `.reveal` with `animation-delay` stepped ~400ms), each showing title → meaning → its stresses → the remedy (blurred, "being confirmed by Dr. Glen" — exactly the existing blurred render).
    3. After the sequence, `localStorage.setItem(ukey, "1")`.
  - When `!firstTime` (returning) OR `confirmed`: render statically (no `.reveal`/delays) — the existing render path.
  - Implement the staging with CSS `animation-delay` on `.reveal` elements (simplest — one render pass, the browser staggers them), not a chain of setTimeouts. Compute each element's delay from its index.
- [ ] **Step 3: Cold hand-off.** Phase 2's pending→ready poll calls `load()`, which re-renders; on the render where status flips from `pending` to a real analysis, `firstTime` is true → the unfold plays. Confirm the pending branch from Phase 2 still posts the process-request + polls, and that clearing the poll timer (Phase 2) happens before/at the unfold render.
- [ ] **Step 4: Verify.** Extract the main `<script>`, `node --check`. Run the full Python suite (served-page test 200). Brand: no emojis, "Order" not "Reorder", CSS-only animation, honors `prefers-reduced-motion`.
- [ ] **Step 5: Commit** (`-m "portal: live-unfold — patterns then layers assemble on first view"`)

---

## Task 3: PR
- [ ] Full suite green. Push `sess/5326cc61-phase3`; open a PR (base main) "Scan notification Phase 3: live-unfold (patterns → layers reveal)". Body: notes the content endpoint now returns client-safe `findings`; the unfold plays once per scan (localStorage), statically thereafter; honors reduced-motion.
- [ ] **Manual (after deploy):** a fresh analysis (engaged pre-process, or a cold poll completing) plays the staged reveal — patterns chips appear, then layers assemble; reload the same scan → renders statically (no re-animation); confirmed scans render full + static.

---

## Self-Review notes
- **Spec coverage (Phase 3):** patterns-first then layers-assemble reveal (T2); the cold "Preparing…"→unfold hand-off (T2 step 3, on top of Phase 2); findings available to the client (T1). Remedies stay blurred pending Glen's confirm (unchanged — the unfold only animates the patterns + interpretation, exactly per the approved design).
- **Type consistency:** client `findings` = `{code,name,description,rank}` (no `clinical_notes`); the unfold guard key `rvl:<scan_date>`; `.reveal` + staggered `animation-delay`.
- **Verify during impl:** the page's real brand CSS variable names (T1 not affected; T2 CSS); that `d.findings` is read where the biofield render has the content payload; the Phase 2 `pending` branch + poll-timer clear interplay (don't double-animate); confirm `prefers-reduced-motion` disables it.
- **No data/PHI concern:** findings are stress patterns (not remedies); showing them to the client is the approved "patterns + interpretation" reveal, and `clinical_notes` is stripped.
