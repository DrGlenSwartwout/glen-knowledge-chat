# Phase 3 Task 1 Report: Order Linkables in briefing_links.py

## Status
**DONE**

## Commit
`d0f0c6e` — feat: order linkables in briefing_links registry

## Test Summary
18/18 pass (15 original + 4 new order tests).

---

## RED → GREEN Sequence

### RED (before implementation)
```
$ cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q
3 failed, 15 passed

FAILURES:
- test_order_url_encodes_id: AttributeError — no order_url function
- test_build_linkables_mints_order_links: KeyError 'ref' — orders not processed
- test_build_linkables_mixed_person_invoice_order_share_counter: KeyError 'ref' — order not in counter
- test_build_linkables_orders_error_block_is_safe: PASS (error dict skipped gracefully)
```

### GREEN (after implementation)
```
$ cd /tmp/wt-deploy-chat-e0f30eb0 && python3 -m pytest tests/test_briefing_links.py -q
18 passed
```

---

## Implementation Summary

### Changes to `dashboard/briefing_links.py`

1. **`order_url(order_id) -> str`**
   - Returns `/console/orders?order=<urlencoded-id>` (no console key; appended client-side)

2. **`_iter_order_records(snapshot)`**
   - Yields `(record_dict, display, order_id)` from `snapshot["orders"]` list
   - Graceful error handling: `isinstance(orders, list)` guard skips error dict
   - Display priority: `name` → `email` → `"Order #<id>"` fallback

3. **`build_linkables(snapshot)` — Added order pass**
   - Order pass runs AFTER invoice pass (preserves people→invoice→order order)
   - Shared counter, dedup by url (r1, r2, r3, …)
   - Mint with fallback: `display or ("Order #" + oid)`

### Changes to `tests/test_briefing_links.py`

Appended 4 new test cases:
- `test_order_url_encodes_id`: URL encoding verification
- `test_build_linkables_mints_order_links`: Order record stamping with display fallback
- `test_build_linkables_orders_error_block_is_safe`: Error dict handling (no crash)
- `test_build_linkables_mixed_person_invoice_order_share_counter`: Counter increment (r1→r2→r3)

---

## Test Matrix

| Test | Result |
|------|--------|
| 15 original tests (people + invoices) | ✓ PASS (unchanged) |
| test_order_url_encodes_id | ✓ PASS |
| test_build_linkables_mints_order_links | ✓ PASS |
| test_build_linkables_orders_error_block_is_safe | ✓ PASS |
| test_build_linkables_mixed_person_invoice_order_share_counter | ✓ PASS |
| **Total** | **18/18 PASS** |

---

## Design Notes

1. **No console key in URL**: Bare `/console/orders?order=<id>` per spec (client-side append)
2. **Error-block guard**: `isinstance(orders, list)` prevents crash when API returns `{"_error": "..."}`
3. **Display fallback**: `name` or `email` (if empty) or `"Order #<id>"` as final fallback
4. **Shared counter**: All three types (person, invoice, order) use same ref namespace
5. **Dedup by URL**: Identical URLs reuse refs (prevents duplicate entries)

## Concerns
None. Implementation matches brief exactly, all tests pass, existing code unaffected.

## P3 final-review fix wave

**Changes made:**

1. **FIX 1 (prompt)** — `dashboard/briefing_runner.py` SLUG_PROMPTS["clients-pipeline"]: added `orders` to the "Cover ONLY … from the snapshot's …" block list (was `inbox`, `gohighlevel`, `scoreapp`; now includes `orders`), resolving the tension with the orders directive appended later in the same prompt.

2. **FIX 2 (prompt)** — Same prompt: rewrote `"(status new with pay_status unpaid, or cart/proposed)"` → `"(status \`new\` or \`proposed\` with pay_status unpaid)"`. Removes the non-existent `cart` status; keeps the unpaid-vs-unshipped distinction and the orders-are-not-QBO-AR clause intact.

3. **FIX 3 (test, discriminating)** — `tests/test_briefing_runner_links.py` `test_record_links_example_covers_orders`: replaced vacuous body (matched on snapshot JSON + stale `(ref:` check) with empty-snapshot assertion `assert "order to act on" in prompt` — pins the literal RECORD LINKS instruction text so removing the example would fail the test.

4. **FIX 4 (test, load-bearing)** — `tests/test_attention_orders.py` `_seed`: added `("Deliv Dev", "dev@x.com", "delivered", "paid", 700)` row; expected statuses remain `["new", "new", "packed", "proposed"]` — `delivered` is now load-bearing in the terminal-status exclusion.

**Test command:**
```
cd /tmp/wt-deploy-chat-e0f30eb0 && PINECONE_API_KEY=dummy python3 -m pytest tests/test_briefing_runner_links.py tests/test_attention_orders.py -q
```

**Result:** 11 passed, 1 warning in 0.35s
