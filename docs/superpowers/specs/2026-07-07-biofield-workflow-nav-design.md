# Biofield Workflow Navigation — Design

**Date:** 2026-07-07
**Status:** Design approved (Glen approved "both" parts)
**Problem:** The clinical workflow is the Match pillar (Biofield / Reveals / Intake / Tags), but the Intake and Tags tabs bounce to the local :8011 tool, which carries no pillar nav. Its only exit is a single "← Console" link that dumps the user at Overview, losing both the pillar context and the client being worked on. To mark a client consult-ready (the next step after authoring their intake) the user has to re-navigate to Biofield and re-type the email.

## Goal

Make the two local :8011 pages (Biofield Intake authoring `/author/<id>`, and Tags `/clinical-tags`) flow with the clinical workflow: (1) a client-carrying "Mark consult-ready →" next-step button on the Intake page, and (2) a Match sub-tab strip on both local pages so no biofield page dead-ends at Overview.

## Key facts (verified)

- The prod Biofield page (`console-biofield-portal.html`) ALREADY handles `?email=`: on load it pre-fills the `#email` input and calls `loadExisting()` (line 225), so a deep-link pre-selects the client and the consult-ready buttons act on it. **No change to the prod page is needed.**
- The `/author/<test_id>` page already has the client email in scope (`c_email`, biofield_local_app.py ~line 422).
- The local app knows the prod console base: `PUBLIC_BASE_URL` (default `https://illtowell.com`), and the console key `CONSOLE_SECRET`.
- The Match sub-tabs and their hrefs are defined in prod `static/op-nav.js`: Biofield `/console/biofield-portal`, Reveals `/console/biofield-reveals`, Intake `/console/biofield-intake`, Tags `/console/clinical-tags`.

## Design

### Part 1 — Match sub-tab strip on the local pages

A small horizontal strip rendered at the top of both local pages: **Biofield · Reveals · Intake · Tags**, each linking to `{PUBLIC_BASE_URL}/console/{page}?key={CONSOLE_SECRET}`. The current page's tab is highlighted (Intake on `/author`, Tags on `/clinical-tags`). Reuses the prod console cookie/key so the links land authed. This replaces the dead-end "← Console → Overview" flow with direct movement between clinical steps.

The four hrefs are static (mirror op-nav.js). Rendered by one shared helper so both pages stay in sync. The strip is plain text links (no op-nav.js dependency — the local pages deliberately do not load the prod nav bundle).

### Part 2 — Client-carrying consult-ready next-step button (Intake page only)

On `/author/<id>`, a prominent **"Mark consult-ready →"** button linking to
`{PUBLIC_BASE_URL}/console/biofield-portal?email={c_email}&key={CONSOLE_SECRET}`.
Because the Biofield page pre-loads `?email=`, the client is already selected on arrival and marking them ready is one click, no re-typing. Placed near the page header, visually distinct from the tab strip (it is the primary workflow action, not just navigation).

If `c_email` is blank for a session (no client on the report), omit the button (the tab strip still gives a path to Biofield).

## Units

| Unit | Change |
|------|--------|
| `biofield_local_app.py` | A `_workflow_nav(active, client_email="")` helper returning the strip HTML (+ the consult-ready button when `client_email` is set). Injected into the `/author` page render (Intake active, with `c_email`) and available to the Tags page. |
| `dashboard/clinical_tags_console.py` | Render the same strip (Tags active, no button) at the top of its page, next to / replacing the existing "← Business OS" link. |

Keep the existing "← Business OS" link too (or fold it into the strip) — the strip is additive; the point is the workflow tabs are now present.

## Error handling / edge cases

- Blank `CONSOLE_SECRET` (local dev): links render without `?key=` (the prod cookie still auths an interactive session); do not crash.
- Blank `c_email`: omit the consult-ready button.
- HTML-escape `c_email` in the href (`urllib.parse.quote` for the query value).

## Copy

- Tab labels: Biofield, Reveals, Intake, Tags (match op-nav). Button: "Mark consult-ready →". No em dashes, no ALL-CAPS.

## Out of scope

- No change to the prod Biofield page (already deep-link-ready).
- No general biofield-journey stepper yet (this pattern generalizes to it later).
- No change to op-nav.js or the prod console pages.

## Go-live

Both files are local-only (run on :8011). After merge, Glen pulls `~/deploy-chat` and restarts the local server (`launchctl kickstart -k gui/$(id -u)/com.glen.biofield-local-server`) since the :8011 app has no hot-reload. Then render-verify: on `/author/<id>` the strip + consult-ready button show and land on the Biofield page with the client pre-selected; on `/clinical-tags` the strip shows with Tags highlighted.
