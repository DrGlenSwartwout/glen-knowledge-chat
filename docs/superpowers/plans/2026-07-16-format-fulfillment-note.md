# Record packaging format as a fulfillment note (one-time + autoship)

**Date:** 2026-07-16 · **Repo:** deploy-chat · **Status:** implemented

## Goal
The customer's chosen packaging format (`_FORMATS`: `bottle` | `larger` | `refill`) was collected on the buy page but **discarded** — `begin_checkout` parsed it and never used it, and `/reorder/subscribe` never saw it. Record a **non-default** format choice as a fulfillment note on the order line, for **both** the one-time and autoship flows (Glen: option A).

## Design — one seam
Every flow (one-time `begin_checkout`, subscribe first order, and the renewal charge cron) prices through **`_price_cart`'s cart loop**. So `format` is threaded onto the cart item and the label is resolved + appended there once; renewals inherit it for free (the stored subscription `items_json` carries the per-item `format` opaquely).

- `_FORMAT_LABELS` (`app.py`, after `_FORMATS`): `{id: label}` excluding `bottle` (the default), so only a non-standard choice becomes a note.
- `_price_cart` loop: `_fmt_label = _FORMAT_LABELS.get(c.get("format"))`; `_disp_name = f'{name} — {label}'` when present, else plain name. Applied to `items_rec[].name` (fulfillment kanban, `console-orders.html` renders `name`) and `qbo_lines[].description` (invoice/Sales Receipt). **`qbo_lines[].name` stays the plain product name** so QBO item mapping is unaffected. Packing keys off `slug`/`bottle_type` and repertoire off `slug`, so decorating the display name is safe.
- `begin_checkout`: the inline cart item becomes `{"slug", "qty", "format": fmt}` (fmt already parsed).
- Frontend `static/begin-buy.html`: `placeSubscription` items become `[{slug, qty, format: chosenFormat}]` (one-time `placeOrder` already sends `format`, which `begin_checkout` threads).

## Scope / non-goals
- Only `begin_checkout` (begin-buy one-time) and `reorder_subscribe` (+ its renewal cron) — the two flows the buy page uses. `reorder_checkout` (the separate reorder page) is out of scope.
- Format applies to single FF SKUs (the only products with a format picker); bundles have no picker → `chosenFormat` defaults to `bottle` → no note. No SKU/price change (option A, not C).

## Tests
`tests/test_format_fulfillment_note.py` (Doppler): refill/larger decorate both `items_rec[].name` and `qbo_lines[].description`; QBO line `name` stays plain; `bottle`/unset → undecorated. 4 pass. `node --check` clean on the inline script.

## Rollout
deploy-chat merge=deploy. After deploy: place a one-time FF order choosing "refill" and confirm the note lands on the order line (console kanban) and the invoice description.
