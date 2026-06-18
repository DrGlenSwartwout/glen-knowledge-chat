# Task 5 Fix Report — Comparison Section Body Bug

## Problem

`begin_product_page_data` in `app.py` built the `comparison` section with `"body": {}`.
The real archetype data (`_SALES_ARCHETYPES`: `{columns, rows, excipient_callout}`) was
only emitted at a separate top-level `"comparison"` key in the JSON response.

The JS renderer `renderComparisonBody(body, mironAssets)` in `static/begin-product.html`
receives `sec.body` via `buildSectionBody`. Because `sec.body` was `{}`, the guard:

```js
if (!body || !Array.isArray(body.columns) || !Array.isArray(body.rows)){
  return wrap;
}
```

short-circuited immediately — so the comparison table, excipient callout, and Miron
rotator never rendered.

## Changes

### app.py

1. Changed comparison section body from `{}` to `_SALES_ARCHETYPES` (no dict duplication):
   ```python
   {"id": "comparison", "title": "How it compares", "default_open": False, "body": _SALES_ARCHETYPES},
   ```
2. Removed the now-redundant top-level `"comparison": _SALES_ARCHETYPES` key from the
   `jsonify(...)` response. The `"miron_assets"` top-level key was kept (the JS reads it
   from `data.miron_assets` and passes it as the second argument to `renderComparisonBody`).

### static/begin-product.html

No changes required. Line 494 already routes the comparison case:
```js
case 'comparison': return renderComparisonBody(sec.body, mironAssets);
```
With `sec.body` now carrying the archetype dict, the guard no longer short-circuits and
the table, callout, and rotator all render.

### tests/test_sales_pages_phase1.py

Updated `test_product_page_data_shape` to read comparison data from the section body
instead of the (now removed) top-level `data["comparison"]` key:
```python
comp = next(s for s in data["sections"] if s["id"] == "comparison")["body"]
rows = {r["label"] for r in comp["rows"]}
assert "Packaging" in rows and "Microplastic exposure" in rows
assert "stearates" in comp["excipient_callout"].lower()
assert len(comp["columns"]) == 3
```

## Test Output

```
============================= test session starts ==============================
collected 7 items

tests/test_sales_pages_phase1.py::test_product_url_built_when_flag_on PASSED
tests/test_sales_pages_phase1.py::test_product_url_empty_when_flag_off PASSED
tests/test_sales_pages_phase1.py::test_product_page_200_known_slug PASSED
tests/test_sales_pages_phase1.py::test_product_page_404_unknown_slug PASSED
tests/test_sales_pages_phase1.py::test_product_page_data_shape PASSED
tests/test_sales_pages_phase1.py::test_match_to_product_page_roundtrip PASSED
tests/test_sales_pages_phase1.py::test_section_pref_accepts_new_section_ids PASSED

======================== 7 passed, 1 warning in 11.74s =========================
```
