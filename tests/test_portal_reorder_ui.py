# tests/test_portal_reorder_ui.py
"""Task 6: portal reorder module UI (static/client-portal.html). Pure markup/
JS-source assertions — no app import, no network — mirroring the pattern in
test_portal_finder_markup.py. Guards against three concrete regressions:
  1. the rendered JS referencing a payload field Task 5's builder doesn't emit
     (see app.py::_portal_reorder_module and tests/test_portal_reorder_module.py
     for the ground-truth shape),
  2. the Reorder button losing its one-shot double-fire latch,
  3. locked-row copy drifting into backward/loss framing ("you overpaid").
"""
import pathlib
import re

HTML = pathlib.Path("static/client-portal.html").read_text()

# Fields Task 5's _portal_reorder_module() actually emits (app.py) — the
# single source of truth this test pins the UI against.
REORDER_FIELDS = {"slug", "name", "qty", "regular_cents", "your_cents",
                   "is_member_price", "in_repertoire", "channel", "is_reorder"}
LOCKED_FIELDS = {"slug", "name", "regular_cents", "tier"}
UPSELL_FIELDS = {"reorders_30d", "spend_30d_cents", "member_would_pay_cents",
                  "savings_cents", "net_after_fee_cents", "already_member"}


def _render_fn_source():
    m = re.search(r"function render\(d, v\)\{[\s\S]*?\n\}\n", HTML)
    assert m, "render(d, v) function not found in client-portal.html"
    return m.group(0)


def test_reorder_module_present_and_guarded():
    src = _render_fn_source()
    assert "Array.isArray(d.reorder) && d.reorder.length" in src
    assert "Array.isArray(d.locked_rows) && d.locked_rows.length" in src
    assert "d.membership_upsell &&" in src


def test_reorder_row_only_references_real_payload_fields():
    src = _render_fn_source()
    block = src[src.index('Array.isArray(d.reorder) && d.reorder.length'):
                src.index('Array.isArray(d.locked_rows)')]
    used = set(re.findall(r"\bit\.(\w+)", block))
    assert used <= REORDER_FIELDS, f"unknown reorder field(s) referenced: {used - REORDER_FIELDS}"
    # sanity: it should actually be exercising the fields that matter for pricing
    assert {"regular_cents", "your_cents", "is_member_price"} <= used


def test_reorder_row_labels_provenance_and_reserves_reorder_word():
    """Glen 2026-07-11 relabel (labels only): every purchase carries a provenance
    label by channel ('Ordered on your portal' vs the storefront label), and the
    word 'Reorder' is reserved for a true reorder (is_reorder) — a first-time
    purchase gets a neutral CTA instead."""
    src = _render_fn_source()
    block = src[src.index('Array.isArray(d.reorder) && d.reorder.length'):
                src.index('Array.isArray(d.locked_rows)')]
    assert "it.channel" in block
    assert "it.is_reorder" in block
    assert "Ordered on your portal" in block
    # storefront provenance label present (its own label, not the portal one)
    assert "remedymatch.com" in block.lower()
    # clinic/other channels (fmp, dispensary, ...) get their own label too
    assert "Ordered with Dr. Glen" in block
    # 'Reorder' still offered for true reorders
    assert "Reorder" in block


def test_locked_rows_only_references_real_fields_and_is_forward_framed():
    src = _render_fn_source()
    block = src[src.index('Array.isArray(d.locked_rows)'):
                src.index('d.membership_upsell &&')]
    used = set(re.findall(r"\bit\.(\w+)", block))
    assert used <= LOCKED_FIELDS, f"unknown locked_rows field(s) referenced: {used - LOCKED_FIELDS}"
    assert "tier" in used
    low = block.lower()
    assert "overpa" not in low  # never "you overpaid" or similar backward framing
    assert "unlock" in low


def test_membership_upsell_only_references_real_fields_and_hides_for_members():
    src = _render_fn_source()
    idx = src.index('d.membership_upsell &&')
    block = src[idx: idx + 1800]
    used = set(re.findall(r"\bmu\.(\w+)", block))
    assert used <= UPSELL_FIELDS, f"unknown membership_upsell field(s) referenced: {used - UPSELL_FIELDS}"
    assert {"savings_cents", "reorders_30d", "net_after_fee_cents"} <= used
    # gated: hidden entirely when already_member is true
    guard_line = src[idx: idx + 120]
    assert "already_member" in guard_line


def test_reorder_button_has_one_shot_latch():
    m = re.search(r"async function reorderItem\(btn\)\{[\s\S]*?\n\}\n", HTML)
    assert m, "reorderItem(btn) not found"
    fn = m.group(0)
    # Latch must fire synchronously, before the first await, so a double-click
    # or a slow network can't double-fire the checkout call.
    guard_idx = fn.index("if(btn.disabled) return;")
    disable_idx = fn.index("btn.disabled = true;")
    first_await_idx = fn.index("await ")
    assert guard_idx < disable_idx < first_await_idx


def test_reorder_uses_existing_checkout_endpoint_not_a_new_one():
    m = re.search(r"async function reorderItem\(btn\)\{[\s\S]*?\n\}\n", HTML)
    fn = m.group(0)
    assert "/api/portal/${encodeURIComponent(token)}/checkout" in fn
    assert "stripe_url" in fn  # redirects to Stripe's hosted confirm page, never charges inline
