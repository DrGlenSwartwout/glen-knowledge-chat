# Console pricing + rewards settings editor

Makes the pricing-engine and rewards "go-live tunables" editable in the Console,
persisted to `pricing-settings.json`, with **live-reload** so an edit takes effect on the
next order without a redeploy.

## What's editable

| Setting | File key | Meaning | Default |
|---|---|---|---|
| Wholesale / discount floor | `discount_floor_pct` | every % discount clamps up to `list * this` (= wholesale) | 0.57 |
| Points floor | `points_floor_pct` | points clamp up to `list * this` (points can go deeper than the discount floor) | 0.43 |
| Points earned | `points_earn_pct` | earn this fraction of full-price spend, as redemption-value cents | 0.05 |
| Point redemption value | `points_redeem_per_point_cents` | cents per point (5 = 20 points per $1) | 5 |
| Subscribe tiers | `subscribe_tiers` | escalating % by completed-order count (1st, 2nd, 3rd+) | [5, 10, 15] |
| Cadences | `cadences` | offered subscription cadences, in months | [1, 2, 3] |
| Volume curve | `volume_anchors` | `[total_months, pct_off]` knots, ascending; linear-interpolated, flat beyond the last | [[1,0],[3,14],[6,29],[12,43]] |
| Referral reward | `rewards.referral_reward_pct` | fraction of a referred full-price sale credited to the referrer | 0.05 |
| Cash-out review threshold | `rewards.cash_out_threshold_cents` | balance (in cents) that raises a cash-out review todo | 10000 ($100) |
| Cash-out face value | `rewards.cash_out_face_pct` | fraction of accrued balance actually paid out on cash-out | 0.70 |
| Referral reward by certification | `rewards.referral_cert_anchors` | `[modules_completed, pct]` knots, ascending; interpolated, flat beyond the last. Applies ONLY when the referrer is a practitioner (interpolated by their completed modules); a non-practitioner referrer stays at the base `referral_reward_pct`. Gated by `REWARDS_TIERS_ENABLED`. Curve pct is whole percentage points. | [[0,5],[6,10],[12,15]] |

Per-SKU floor / MAP overrides are **not** here: they live on the product record
(`wholesale_cents`, `sku_discount_floor_pct`, `sku_points_floor_pct`) read by
`pricing.unit_floor_cents`, and the practitioner MAP lives in `practitioner_settings`.

## File shape and representations

`pricing-settings.json` stores values exactly as the engine consumes them:

- fractions (`*_pct` except volume anchors) are decimals in `[0, 1]` (0.57 = 57%);
- `volume_anchors`' second element is whole **percentage points** (0–100);
- everything `*_cents` is integer cents.

The editor page shows fractions as percent and `cash_out_threshold_cents` as dollars; the
conversion lives only in the page JS. The file and the API speak fractions/cents/whole-percent.

```json
{
  "discount_floor_pct": 0.57,
  "points_floor_pct": 0.43,
  "points_earn_pct": 0.05,
  "points_redeem_per_point_cents": 5,
  "subscribe_tiers": [5, 10, 15],
  "cadences": [1, 2, 3],
  "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
  "rewards": {
    "referral_reward_pct": 0.05,
    "cash_out_threshold_cents": 10000,
    "cash_out_face_pct": 0.70
  }
}
```

## Persistence + live-reload

- The file lives on the **env-`DATA_DIR` persistent disk** (the same root as `LOG_DB` /
  `_CLIPS_DIR`), not the repo's read-only `data/` baseline, so console saves survive
  redeploys.
- `app.py` reads it through `_pricing_settings()` (and `_rewards_settings()` for the nested
  `rewards`), which re-reads the file only when its mtime changes. A save busts the cache,
  so the next order prices with the new values. No redeploy needed.
- If the file is absent (never saved) or unreadable, the accessor returns `{}` and the
  engine falls back to the built-in `pricing.DEFAULTS` / `rewards.DEFAULTS`. The shipped
  values therefore equal the defaults until the first Save.

## API

`GET/POST /api/console/pricing-settings` (CONSOLE_SECRET-gated via `X-Console-Key` header or
`?key=`):

- **GET** → `{"saved": <raw file or {}>, "effective": <defaults merged with saved>, "defaults": <built-in>}`.
- **POST** → body in the file shape (a partial is allowed). Validated; on success the file is
  written atomically (tempfile + `os.replace`), the cache is busted, and it returns
  `{"saved", "effective"}`. On a validation failure it returns `400 {"errors": [...]}` and
  writes nothing.

Validation: fractions in `[0, 1]`; `points_redeem_per_point_cents` integer ≥ 1;
`cash_out_threshold_cents` integer ≥ 0; `volume_anchors` ascending `[months≥1, pct 0–100]`
pairs; `points_floor_pct ≤ discount_floor_pct`. Unknown keys are dropped.

## Page

`/console/pricing-settings` (`static/console-pricing-settings.html`) — console-key gated,
op-nav bar, sections for Discounts & points, Volume curve, Subscriptions, Rewards &
referrals, and a Preview panel. Preview posts a sample cart to the public
`/api/pricing/preview` and reads the **live** engine, so save first, then preview to see the
new pricing.
