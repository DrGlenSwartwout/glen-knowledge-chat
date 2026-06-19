import datetime
import secrets
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS referral_codes ("
               "email TEXT PRIMARY KEY, code TEXT UNIQUE, created_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS referral_redemptions ("
               "referee_email TEXT PRIMARY KEY, code TEXT, owner_email TEXT, "
               "order_ref TEXT, created_at TEXT)")
    cx.commit()


def get_or_create_code(cx, email):
    init_tables(cx)
    e = _norm(email)
    row = cx.execute("SELECT code FROM referral_codes WHERE email=?", (e,)).fetchone()
    if row:
        return row[0]
    for _ in range(10):
        code = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:8].upper()
        try:
            cx.execute("INSERT INTO referral_codes (email, code, created_at) VALUES (?,?,?)",
                       (e, code, _now()))
            cx.commit()
            return code
        except sqlite3.IntegrityError:  # UNIQUE collision on code, retry
            continue
    raise RuntimeError("could not mint a unique referral code")


def owner_of(cx, code):
    init_tables(cx)
    row = cx.execute("SELECT email FROM referral_codes WHERE code=?", (code,)).fetchone()
    return row[0] if row else None


def has_redeemed(cx, referee_email):
    init_tables(cx)
    return cx.execute("SELECT 1 FROM referral_redemptions WHERE referee_email=?",
                      (_norm(referee_email),)).fetchone() is not None


def resolve(cx, code, referee_email, *, pct):
    init_tables(cx)
    owner = owner_of(cx, code)
    ref = _norm(referee_email)
    if not owner or owner == ref or has_redeemed(cx, ref):
        return None
    return {"owner_email": owner, "coupon_pct": int(pct)}


def record_redemption(cx, code, owner_email, referee_email, order_ref):
    init_tables(cx)
    cur = cx.execute(
        "INSERT OR IGNORE INTO referral_redemptions (referee_email, code, owner_email, order_ref, created_at) "
        "VALUES (?,?,?,?,?)",
        (_norm(referee_email), code, _norm(owner_email), order_ref or "", _now()))
    cx.commit()
    return cur.rowcount > 0
