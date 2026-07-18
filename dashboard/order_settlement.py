"""Single per-kind settlement definition for a PAID checkout order, called by
BOTH the /begin/checkout-return redirect and the Stripe webhook book-back, so a
closed browser tab settles exactly what the redirect settles (no drift).

This module owns ONLY the dispatch. The actual side-effect functions live in
app.py (they need LOG_DB, Stripe, etc.) and are injected as `deps` -- mirroring
how qbo_heal takes find_receipt/book/stamp. It deliberately does NOT touch
mark-paid / receipt-booking / PI-stamp; those already fire correctly in each
caller. Every dep is idempotent per order_ref, so calling from both paths and
re-running is safe.
"""

# Kinds that earn loyalty points + referral credit the common way. VERIFIED
# against the redirect's shared gate (all of retail/reorder/portal-reorder/
# subscribe/client) plus biofield's own block -- so all six. client is INCLUDED
# on purpose: today it settles common (global-scope) points/referral AND its own
# dispensary-scope points; that pre-existing double-scope is preserved.
# Membership/subscription-product kinds are handled by their own _fulfill_*
# webhook fulfillers, not here.
_COMMON_POINTS_KINDS = {"retail", "reorder", "portal-reorder", "subscribe", "client", "biofield"}


def settle_paid_order_effects(*, kind, order, md, pi_id, sid, deps):
    """Run every per-kind side-effect for a paid order, idempotently, best-effort
    per effect. Returns {"kind", "settled": [names], "skipped": [names]}."""
    order_ref = (md or {}).get("invoice_id") or ""
    settled, skipped = [], []

    def _do(name, fn):
        try:
            fn()
            settled.append(name)
        except Exception as e:  # best-effort: one bad settler never aborts the rest
            print(f"[settlement] {name} failed kind={kind} ref={order_ref!r}: {e!r}", flush=True)
            skipped.append(name)

    if kind in _COMMON_POINTS_KINDS and order:
        _do("points", lambda: deps.settle_points(order, order_ref))
        _do("referral", lambda: deps.settle_referral(order, order_ref))

    if kind == "subscribe":
        _do("subscription", lambda: deps.ensure_subscription(md, pi_id))

    # Group-bundle grant is a RETAIL effect, not a subscribe one: grant_group_months
    # is stamped only on kind=="retail" program orders (see
    # _stripe_checkout_url_for_retail in app.py), never on subscribe. Dispatch by
    # the metadata's presence, kind-agnostic, so both the redirect and the webhook
    # (which route purely through this orchestrator) grant it identically. The
    # dep (_grant_group_bundle) retains its own GROUP_BUNDLE_ENABLED/pi_id gate.
    if int((md or {}).get("grant_group_months") or 0) > 0:
        _do("group_bundle", lambda: deps.grant_group_bundle(md, pi_id))

    if kind == "client":
        _do("client", lambda: deps.settle_client(md))
    elif kind == "biofield":
        _do("biofield", lambda: deps.settle_biofield(md, sid))

    # Membership-line grant: a checkout order of ANY kind may carry a membership
    # line (kind-agnostic, like group_bundle -- gated here on the line's presence,
    # not on kind). Reuse membership_products' pure detector as the single source
    # of truth (no app/LOG_DB dependency), so this only fires for orders that
    # actually bought a membership. The dep is idempotent per order_ref; both the
    # redirect and the webhook route through here, so a closed tab still delivers.
    if order:
        from . import membership_products as _mp
        if _mp.cart_has_membership_tier(order.get("items") or []):
            _do("membership_line", lambda: deps.grant_membership_line(order))

    return {"kind": kind, "settled": settled, "skipped": skipped}
