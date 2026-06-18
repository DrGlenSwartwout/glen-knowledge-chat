# Miron Story Caption — Implementation Report

## Summary

Added an editable "why violet glass" heritage caption beneath the Miron bottle-science rotator on the new-format product page.

## Files Changed

### `data/miron-assets.json`
Added a top-level `"story"` key containing two paragraph strings describing the historical and cultural significance of violet glass (Ancient Egypt / Cleopatra + Eastern Yin/Yang tradition). Existing `"assets"` array untouched. Valid JSON confirmed.

### `app.py`
In `begin_product_page_data`, added `"miron_story": _MIRON_ASSETS.get("story", [])` alongside the existing `"miron_assets"` field in the JSON response. Uses `.get` with `[]` default so the endpoint remains safe if the key is ever absent.

### `static/begin-product.html`
- Added CSS rule `.sp-miron-story` (muted 13px text, 1.6 line-height, 10px top margin) in the Miron CSS block.
- `renderComparisonBody(body, mironAssets, mironStory)` — added `mironStory` param; after the rotator is appended, iterates `mironStory` and renders each non-empty string as a `<p class="sp-miron-story">` using `textContent` (never innerHTML). Renders nothing if array is empty.
- `buildSectionBody(sec, ctaUrl, mironAssets, mironStory)` — added param, passes through to `renderComparisonBody`.
- `buildAccordion(sec, openSet, ctaUrl, mironAssets, mironStory)` — added param, passes through to `buildSectionBody`.
- `renderProduct(data)` — reads `var mironStory = Array.isArray(data.miron_story) ? data.miron_story : [];` and passes to `buildAccordion`.

All existing params left in their original order; `mironStory` appended at the end throughout.

### `tests/test_sales_pages_phase1.py`
In `test_product_page_data_shape`, added:
```python
assert isinstance(data["miron_story"], list)
assert any("violet glass" in s.lower() for s in data["miron_story"])
```

## Test Results

All 7 tests in `tests/test_sales_pages_phase1.py` passed (0 failures, 1 pre-existing gevent monkey-patch warning unrelated to this change).

## Concerns

None. Implementation matches the spec exactly.
