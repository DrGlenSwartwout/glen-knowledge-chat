# Spec: Biofield editor — stress-pattern findings + unbalanced workspace

**Date:** 2026-06-17
**Status:** Approved (design) — pending implementation plan
**Related:** [[project_e4l_scan_ingestion]] · the synthesis-fidelity work (e4l_synthesis patterns-retention) · the biofield editor (`console-biofield-portal.html`) · [[project_ascension_pricing_model]] (the "purple Infoceutical patterns as stresses" idea — client-facing display is OUT of scope here).

---

## Goal

Show Dr. Glen the actual E4L stress-pattern findings in the biofield editor while he reviews a draft, so he can assess each remedy match and balance any stresses the synthesis left uncovered. Each layer shows the stresses it balances; a bottom panel shows the **unbalanced** stresses with an action to **add a layer** that balances some/all of them.

## Scope

In scope: carrying the full scan-findings detail into portal content (synthesis, vault) + the editor display/interaction (deploy-chat `console-biofield-portal.html`). Out of scope: showing stress patterns to *clients* in the portal (separate decision); the blur-reveal machine (unchanged); any change to remedy/dosing logic.

## Data model

The synthesis already retains each layer's `patterns` (scan `item_code`s). Add to the portal content a top-level **`findings`** list — the full top-priority scan patterns with display detail:

```
content.findings = [ { code, name, description, rank }, ... ]   # name = e4l_items.full_name; description = e4l_description; rank = priority_rank
```

- A finding is **balanced** if its `code` appears in some layer's `patterns`; **unbalanced** otherwise. The editor derives this live (no stored flag) so editing layers updates the panel in real time.
- `clinical_notes`, when present, rides along in the finding (shown in the expanded detail).
- Back-compat: content without `findings` (legacy/pre-this-feature reports) → no findings shown, no unbalanced panel; the editor behaves exactly as today.

## Components

### 1. Synthesis carries findings — `e4l_synthesis.to_portal_content` + `e4l-portal-import.py` (vault)
The importer already has the full `patterns` from `pull_patterns` (code, `full_name`, `e4l_description`, `priority_rank`, `clinical_notes`, …). It passes that detail to `to_portal_content` (new optional `findings=` arg, default None → omit); `to_portal_content` maps it to `content["findings"] = [{code, name, description, rank, clinical_notes}]` (top-N, ordered by rank). Per-layer `patterns` codes are unchanged. Pure + unit-tested.

### 2. Per-layer balanced stresses — `console-biofield-portal.html`
Under each layer card, render the stresses that layer balances: for each code in the layer's `patterns`, look it up in `content.findings` and show a **compact line** `[CODE] Full Name · #rank`. Clicking a line **expands** the plain-English description (+ clinical notes when present). Read-only. If a layer has no findings detail (legacy), show nothing.

### 3. Unbalanced stresses panel — `console-biofield-portal.html`
A panel **below the layers** listing every finding whose code is in **no** layer's `patterns`, same compact-line + expand format, each with a checkbox. Header shows the count (e.g. "Unbalanced stresses (3)"). Hidden entirely when there are no findings or none are unbalanced.

### 4. "Add layer to balance" action — `console-biofield-portal.html`
A button on the unbalanced panel: with one or more findings checked, **"Add a layer for selected"** creates a **new layer pre-loaded with those `patterns`** (its balanced-stresses list shows them) and an empty remedy/dosing for Glen to fill. The selected findings immediately leave the unbalanced panel (now balanced by the new layer). "Select all" convenience included. Glen sets the remedy on the new layer like any other.

### 5. Persist findings through edit — `console-biofield-portal.html`
`content.findings` is preserved through load → edit → publish (stashed like `patterns` already is), so the findings + balanced/unbalanced state survive a republish and remain for the next review.

## Data flow
scan → synthesis writes `content.findings` (+ per-layer `patterns`) → editor load → per-layer cards show their balanced stresses; bottom panel shows unbalanced → Glen expands descriptions to assess, edits remedies, and "adds a layer" to balance leftovers → publish (findings + patterns retained).

## Error handling
- No `findings` in content → no per-layer stresses, no unbalanced panel (legacy-safe, exactly today's behavior).
- A layer `patterns` code with no matching finding → that code is simply skipped in the display (no crash).
- "Add a layer" with nothing selected → no-op with a hint.
- Findings are read-only scan facts; only remedy/dosing/layer-membership are editable.

## Testing
- **Synthesis (vault, unit):** `to_portal_content(..., findings=...)` writes `content.findings` with `{code,name,description,rank,clinical_notes}` ordered by rank; omitted when `findings` not passed; per-layer `patterns` unchanged.
- **Editor (JS check + manual):** per-layer stresses render from `findings` by code + expand; unbalanced = findings not in any layer's `patterns`, count correct; "Add a layer for selected" creates a layer with those `patterns` and removes them from the panel; legacy content (no `findings`) shows neither section; findings survive load→publish. Full Python suite stays green (served-page test).

## Definition of done
The editor shows, per layer, the stress findings that layer balances (compact + expandable description); a bottom panel lists the unbalanced stresses with a count and per-item expand; Glen can select unbalanced stresses and add a layer that balances them (which moves them out of the panel); findings persist through publish; reports without a findings list render exactly as today. Synthesis unit tests green; full suite green.
