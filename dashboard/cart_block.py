"""Portal view block for the persistent cart. {"enabled": False} when the flag is
off. Guarded so a failing source degrades to a zero count rather than raising into
the portal payload."""
from dashboard import cart_store as _cs


def build_block(cx, email, enabled) -> dict:
    if not enabled:
        return {"enabled": False}
    try:
        _cs.init_cart_tables(cx)
        token = _cs.open_token_for_email(cx, email)
        count = sum(i["qty"] for i in _cs.items(cx, token)) if token else 0
    except Exception:
        count = 0
    return {"enabled": True, "count": count}
