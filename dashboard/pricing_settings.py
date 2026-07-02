# dashboard/pricing_settings.py
"""Shape, validation, and effective-merge for the console-editable pricing + rewards
settings (persisted to DATA_DIR/pricing-settings.json). Pure: no Flask, no file IO."""
from dashboard import pricing as _pricing
from dashboard import rewards as _rewards

_PRICING_FRACTIONS = ("discount_floor_pct", "points_floor_pct", "points_earn_pct")
_REWARDS_FRACTIONS = ("referral_reward_pct", "cash_out_face_pct")


def defaults_view():
    """Built-in defaults in the file's shape (rewards nested)."""
    d = dict(_pricing.DEFAULTS)
    d["volume_anchors"] = [list(a) for a in _pricing.DEFAULTS["volume_anchors"]]
    d["subscribe_tiers"] = list(_pricing.DEFAULTS["subscribe_tiers"])
    d["cadences"] = list(_pricing.DEFAULTS["cadences"])
    d["discounts"] = {k: {"enabled": bool(v["enabled"]),
                          "anchors": [list(a) for a in v["anchors"]]}
                      for k, v in _pricing.DEFAULTS["discounts"].items()}
    d["rewards"] = dict(_rewards.DEFAULTS)
    if isinstance(d["rewards"].get("referral_cert_anchors"), list):
        d["rewards"]["referral_cert_anchors"] = [list(a) for a in d["rewards"]["referral_cert_anchors"]]
    return d


def effective(raw):
    """Merge raw overrides over the defaults, returning the file-shaped effective view.
    The nested 'rewards' overrides are kept out of the pricing merge."""
    raw = raw or {}
    pricing_over = {k: v for k, v in raw.items() if k != "rewards"}
    eff = _pricing.load_settings(pricing_over)
    eff["rewards"] = _rewards.load_settings(raw.get("rewards") or {})
    if "discounts" not in raw or not isinstance(raw.get("discounts"), dict):
        base = _pricing.DEFAULTS["discounts"]
        # Only an explicit saved override for the legacy top-level volume_anchors
        # backfills open_total's ramp; absent that, open_total keeps its own
        # (disabled, zero) code default -- eff["volume_anchors"] is always present
        # post-merge (load_settings fills it from DEFAULTS), so it can't be used
        # to distinguish "no override" from "explicit override".
        legacy = raw.get("volume_anchors") or base["open_total"]["anchors"]
        eff["discounts"] = {
            "same_sku":      {"enabled": bool(base["same_sku"]["enabled"]),
                              "anchors": [list(a) for a in base["same_sku"]["anchors"]]},
            "program_total": {"enabled": bool(base["program_total"]["enabled"]),
                              "anchors": [list(a) for a in base["program_total"]["anchors"]]},
            "open_total":    {"enabled": False, "anchors": [list(a) for a in legacy]},
        }
    return eff


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_fraction(name, v, errors):
    if not _is_number(v) or not (0.0 <= float(v) <= 1.0):
        errors.append(f"{name} must be a number between 0 and 1")
        return None
    return float(v)


def _validate_anchors(anchors, min_x):
    """Validate + normalize an ascending [x, pct] anchor table: x integer >= min_x, strictly
    ascending; pct a number in [0, 100]. Returns the normalized list, or None if invalid."""
    if not (isinstance(anchors, list) and len(anchors) >= 1):
        return None
    norm, last = [], None
    for pair in anchors:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                and _is_number(pair[0]) and _is_number(pair[1])):
            return None
        x, p = int(pair[0]), float(pair[1])
        if x < min_x or not (0.0 <= p <= 100.0):
            return None
        if last is not None and x <= last:
            return None
        last = x
        norm.append([x, int(p) if float(p).is_integer() else p])
    return norm


def validate(payload):
    """Validate a (possibly partial) settings payload. Returns (clean, errors).
    clean contains only known, well-formed keys in the file shape; errors is a list
    of human-readable messages (non-empty => caller should reject)."""
    payload = payload or {}
    clean, errors = {}, []

    for name in _PRICING_FRACTIONS:
        if name in payload:
            v = _check_fraction(name, payload[name], errors)
            if v is not None:
                clean[name] = v

    if "points_redeem_per_point_cents" in payload:
        v = payload["points_redeem_per_point_cents"]
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            clean["points_redeem_per_point_cents"] = v
        else:
            errors.append("points_redeem_per_point_cents must be an integer >= 1")

    for name in ("subscribe_tiers", "cadences"):
        if name in payload:
            v = payload[name]
            if (isinstance(v, list) and v
                    and all(_is_number(x) and x >= 0 for x in v)):
                clean[name] = [int(x) if float(x).is_integer() else float(x) for x in v]
            else:
                errors.append(f"{name} must be a non-empty list of non-negative numbers")

    if "volume_anchors" in payload:
        norm = _validate_anchors(payload["volume_anchors"], 1)
        if norm is not None:
            clean["volume_anchors"] = norm
        else:
            errors.append("volume_anchors must be ascending [months>=1, pct 0-100] pairs")

    if "discounts" in payload:
        dblock = payload["discounts"] or {}
        dclean = {}
        for tname in ("same_sku", "program_total", "open_total"):
            if tname not in dblock:
                continue
            t = dblock[tname] or {}
            en = t.get("enabled")
            if not isinstance(en, bool):
                errors.append(f"discounts.{tname}.enabled must be true or false")
                continue
            norm = _validate_anchors(t.get("anchors"), 1)
            if norm is None:
                errors.append(f"discounts.{tname}.anchors must be ascending [qty>=1, pct 0-100] pairs")
                continue
            dclean[tname] = {"enabled": en, "anchors": norm}
        if dclean:
            clean["discounts"] = dclean

    if "rewards" in payload:
        rwd = payload["rewards"] or {}
        rclean = {}
        for name in _REWARDS_FRACTIONS:
            if name in rwd:
                v = _check_fraction(name, rwd[name], errors)
                if v is not None:
                    rclean[name] = v
        if "cash_out_threshold_cents" in rwd:
            v = rwd["cash_out_threshold_cents"]
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                rclean["cash_out_threshold_cents"] = v
            else:
                errors.append("cash_out_threshold_cents must be an integer >= 0")
        if "referral_cert_anchors" in rwd:
            norm = _validate_anchors(rwd["referral_cert_anchors"], 0)
            if norm is not None:
                rclean["referral_cert_anchors"] = norm
            else:
                errors.append("referral_cert_anchors must be ascending [modules>=0, pct 0-100] pairs")
        clean["rewards"] = rclean

    df = clean.get("discount_floor_pct", _pricing.DEFAULTS["discount_floor_pct"])
    pf = clean.get("points_floor_pct", _pricing.DEFAULTS["points_floor_pct"])
    if _is_number(df) and _is_number(pf) and pf > df:
        errors.append("points_floor_pct must be <= discount_floor_pct")

    return clean, errors
