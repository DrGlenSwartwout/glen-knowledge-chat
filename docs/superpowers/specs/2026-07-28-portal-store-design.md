# Portal Store: making myhealingoasis ordering the primary store

**Date:** 2026-07-28
**Owner:** Dr. Glen Swartwout
**Status:** Approved design, ready for an implementation plan

---

## 1. Objective

Make ordering inside the myhealingoasis.com portal the primary way clients buy, replacing
what the remedymatch.com GrooveKart storefront does today. remedymatch.com stays live as a
backup process. Nothing is retired or redirected.

The gap being closed is browse and cart. Members today can reorder what they already own and
can follow a product link to a single-item checkout, but they cannot browse the catalog or
build a multi-item order.

## 2. What already exists (verified in the codebase, 2026-07-28)

The commerce engine is built and live:

- `_price_cart()` prices a cart through `dashboard/pricing.py` with membership, volume,
  coupons, points redemption and ship credit.
- `_shipping_for_cart()` prices shipping from the bottle/box matrix; `dashboard/tax.py`
  handles Hawaii GET.
- `_checkout_cart()` ingests the order, mints the Stripe URL and books the line-faithful
  QBO Sales Receipt via `/begin/checkout-return` on payment.
- `/reorder` gives magic-link auth, an item list scoped to prior purchases, quantity editing
  and checkout.
- `/begin/product/<slug>` is the in-funnel product page; `begin_checkout(slug)` is a
  single-item checkout with an inline opt-in gate returning `need_optin`.
- Practitioner personal, dropship and storefront carts, wholesale and invoices all exist.
- `dashboard/order_destination.py` already refuses to send anyone to a GrooveKart URL,
  because that URL is absent on 669 of 966 sellable products and drops the client out of
  the funnel and out of their courtesy pricing.

What does not exist:

- A persistent cart. `reorder.html` holds items in a page-local JS object (`rows = {}`).
- Any catalog taxonomy. `data/products.json` is 1,085 entries with `name`, `price_cents`,
  `bottle_type`, `description`, and no category, collection or tag field. GrooveKart
  supplied whatever browse structure existed.
- Any product search. There is no product-search endpoint in the app.

Condition structure does exist: `data/condition_programs_seed.json` holds condition to
ranked-remedy programs with labels, doses, alternates and consult flags, wired through
`dashboard/condition_programs.py` and live in the portal behind `_support_programs_enabled()`.

## 3. Decisions taken during brainstorming

1. The gap to close is browse and cart inside the portal, not cold-traffic guest checkout.
2. Free Tier-1 membership is the access route to the store. Browse lives in the portal.
3. A visitor arriving on a public product page adds to an anonymous cart with no friction.
   Membership is created at checkout, and the anonymous cart is merged in.
4. The membership created is the same Tier-1 membership `/begin` creates, not a lighter one.
5. Browse spine: condition-led primary, personalized landing, Body Map as a second way in,
   plus keyword search over name, product-page content, ingredients and tags.
6. Aisle scope for this build: own-catalog formulations and essences. The 44 Oasis tools stay
   a roadmap and are handled next. Fullscript stays a separate channel.
7. Fullscript items are ordered and fulfilled through Fullscript, never added to the cart,
   and are excluded from search. They appear only where curated: condition programs and the
   Oasis surfaces.
8. Pricing policy is unchanged by this build. `same_sku` per-line quantity tiers stay open to
   everyone; `program_total` mix and match stays gated on paid membership; `open_total` stays
   off.
9. Sequencing: cart first, browse second, search quality third. First live exposure is a
   single flagged condition aisle.

## 4. Architecture and data model

The cart is a server-side object, not browser state.

**`carts`**: one open cart per owner. Owner is either a cart token (anonymous) or an email
(member). Status moves `open` to `ordered` when checkout succeeds, carrying the
`checkout_ref` so an order can be traced back to the cart that produced it.

**`cart_items`**: `cart_id`, `slug`, `qty`, `format` (bottle, larger or refill, the same three
the product pages use), `added_at`, and the surface the item came from (browse, product page,
program, reorder, wishlist). Unique on (`cart_id`, `slug`, `format`) so a repeat add
increments rather than duplicates.

No price is ever stored on a cart row.

Three new modules, none of them importing Flask, so each is testable alone:

**`dashboard/cart_store.py`** is persistence only: get or create, add, set quantity, remove,
list, merge, mark ordered. It knows nothing about prices.

**`dashboard/shop_catalog.py`** decides what is in the aisle and how it groups. Reads
`data/products.json`, excludes `inactive` and `info_only` entries, separates the essence
family so it does not swamp browse, exposes browse, by-condition (reading
`dashboard/condition_programs.py`) and lookup. It knows nothing about carts.

**`dashboard/shop_search.py`** owns the keyword index over name, description, generated
product-page sections (`dashboard/product_page_sections.py`) and ingredients
(`data/fmp-ingredient-content.json`, `ingredients_source`), plus the GrooveKart tags once
extracted. Separate from the catalog because it has a build step and its own freshness
problem.

Routes in `app.py` are thin: cart CRUD, shop browse, shop search, the shop page, and the
portal cart panel.

`_checkout_cart()` is not rewritten. The only change is that it receives items read from
`cart_store` instead of a list posted from a page.

**Cross-host bridge.** The anonymous cart cookie lives on illtowell.com, where product pages
are served. The moment the merge attaches an email, the cart is member-owned and readable
from myhealingoasis.com by session. No cookie is shared between the two hosts, and
`PUBLIC_BASE_URL` and `PORTAL_BASE_URL` stay distinct as they must.

**Flags.** `PORTAL_CART_ENABLED` and `PORTAL_SHOP_ENABLED`. Both ship dark, and the portal
payload is byte-identical when off.

## 5. Browse surfaces

**Personalized landing.** A member opening Shop sees their own things first: their support
program if they have one, their ranked My Remedies list, and replenish-due items from order
history. A brand new free member has none of these, so the landing degrades to condition
aisles, search and a small top-products row. That degraded state is designed first, because
every new member arrives in it.

**Condition aisles (primary spine).** Each `condition_programs_seed.json` entry becomes a
shoppable program: label, ranked remedies in order, doses where given, alternates shown as
alternates rather than as separate products, and Add the program alongside Add one item. The
`consult_recommended` flag and staging language travel into the shop and are never stripped
for a cleaner card, since that flag is what keeps a shopping surface from reading as a
prescription. `broad_benefit_slugs` remains a small set that may be suggested broadly, not a
licence to attach everything to every condition.

**Keyword search.** Over name, description, product-page sections and ingredients, plus
GrooveKart tags once extracted. Ranking order: name, then tag, then ingredient, then body
text, so a search for magnesium surfaces the magnesium formulations before every formula
containing a trace of it. Own-catalog only; Fullscript is excluded.

**Body Map as second way in.** System to conditions to program, riding the condition spine.
There is no system-to-product mapping in the data and none is invented here. Clicking a
system lands on the conditions within it and their programs.

**Essences** form their own family in the aisle rather than being interleaved. Twenty four
animal essences sorted next to formulations would bury the formulations, and someone shopping
for essences is on a different errand.

## 6. Cart and merge flow

**Adding.** The first add creates a cart and sets an opaque `rm_cart` cookie on illtowell.com.
No email, no membership, no friction.

**Checkout, step one, identity.** With no member session, checkout collects name, email and
Terms agreement, creating the same Tier-1 membership `/begin` creates. On success the member
and their portal exist, and `cart_store.merge()` moves the anonymous cart onto the email.

**Merge rule.** If that email already has an open cart, from another device or an earlier
visit, quantities take the higher of the two rather than the sum. Adding the same bottle on a
phone and then a laptop is far more often one intent repeated than an order for two, and the
failure mode of summing is a customer charged double.

**Checkout, step two, money.** Unchanged. Address, then `_price_cart` recomputes from scratch,
then the existing Stripe path, then `/begin/checkout-return` books the QBO receipt on payment.

**Afterward.** The cart is marked ordered and the member lands in their portal, where the cart
tile is now theirs on every device.

## 7. Pricing rules on the store surfaces

The anonymous cart and the free-member cart price identically: list, minus `same_sku` per-line
quantity tiers, which are open to everyone. The merge therefore moves no numbers, and there is
no "identify to unlock savings" framing, which would be false for a free member. Identity is
asked for because ordering requires a name, an email and Terms, and because it gives the buyer
a portal.

Shipping and Hawaii GET need an address, so they read as calculated at checkout until one
exists.

**Paid-membership line at the cart.** Shown only when the mix-and-match saving on that specific
cart is real and nonzero, computed as `program_total_pct(total_months, settings,
program_member=True)`, which needs no repertoire. One line, linking to `/membership`. No
comparison panel, and nothing on product pages. A cart of essences or other non-FF items shows
no saving and must not be made to look as though it would.

## 8. Failure modes

- **Stale prices** cannot happen: carts store no prices and `_price_cart` recomputes at
  checkout.
- **Item goes away between add and pay.** Every cart read revalidates against `shop_catalog`
  and marks the row unavailable rather than silently dropping it, the same shape as the
  `available` flag `reorder.html` uses. Checkout refuses while an unavailable row is present.
- **Empty priced cart and non-US ship-to** already raise `CheckoutError` and keep that
  behavior.
- **Double submit.** Merge is idempotent and keyed on the cart token.
- **Concurrent adds from two tabs** are safe via the uniqueness constraint.
- **Stripe inactive.** `_STRIPE_ACTIVE` false currently yields an empty URL. The store must
  fail visibly rather than confirm an order with no way to pay.
- **Cleared cookie** loses an anonymous cart with no recovery. Accepted, and an argument for
  merging at checkout, after which the cart is durable and cross-device.
- **Postgres.** New tables use `RETURNING id`, never `cur.lastrowid`, which raises on the
  adapter. Column types are declared so the runtime init path does not leave a SQLite-shaped
  column behind.

## 9. Verification

**Unit.** The merge matrix (no member cart, existing member cart, colliding slug and format,
empty anonymous cart); catalog exclusion rules pinned to the repo catalog file rather than
`$DATA_DIR`, which strips `products.json` under the full suite; search ranking against a
fixture.

**Integration.** A seeded anonymous cart driven through identity, merge and pricing to a
mocked Stripe.

**Flag-off.** Portal payload byte-identical with both flags off, the same pattern
`support_programs` uses.

**Live.** A browser render verify on a real portal, then one small real order through the
flagged condition aisle before the aisle list widens.

Do not run the bare full suite locally; it sends real email. Use the CI gate.

## 10. Sequencing

**Slice 1, the cart.** `carts` and `cart_items`, `cart_store`, cart API, anonymous cookie cart,
merge at checkout minting Tier-1 membership, Add to cart on `/begin/product/<slug>`, and the
Cart tile plus cart panel in the portal hub. Worth shipping alone: today someone following a
product link can only buy that one item in that one checkout.

**Slice 2, browse.** `shop_catalog`, personalized landing, condition aisles, Body Map entry,
and the Shop tile and panel. First live exposure is a single flagged condition aisle
(glaucoma is the natural candidate, since the seed already carries three glaucoma programs),
so the first real money through the new path is observable and small. Then the aisle list
widens.

**Slice 3, search.** `shop_search`, the index build, and extracting the hidden tags out of
GrooveKart.

**Afterward, the GrooveKart fallbacks get repointed** at the new store: the affiliate "Shop for
Remedies" offer, `store_homepage` in the alias config, and the search-by-name URL at
`app.py:5478`. GrooveKart keeps running and keeps accepting orders through
`/webhook/groovekart`. It simply stops being where our own systems send people.

## 11. Out of scope

- Retiring or redirecting remedymatch.com. It stays as the backup process.
- The 44 Oasis tools. None of them exists in `data/products.json`, so none has a price, a
  `bottle_type` or a shipping profile, and none can pass through `_price_cart`. Deciding which
  are actually fulfilled, including the three Mithreal silver garments which are Glen's own
  product line, is the next piece of work and needs its own spec.
- Fullscript in the cart or in search.
- Any change to `same_sku` pricing policy.
- Abandoned-cart email. Possible once a cart is merged and has an email and a portal link, but
  deliberately not smuggled into this build.
- A direct body-system to product mapping, which would be new curation work.

## 12. Open items

- The GrooveKart hidden tags have to be extracted before search reaches parity with the
  remedymatch.com search box. Until then search covers name, description, page sections and
  ingredients only.
- Bundles and digital goods are in the aisle if they price cleanly through `_price_cart`; each
  needs a check during Slice 2 rather than an assumption here.
