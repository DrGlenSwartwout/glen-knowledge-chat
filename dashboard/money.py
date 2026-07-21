"""Money widgets — PB sessions, Authorize.net (GrooveKart + GHL), Wise, QuickBooks."""

import os
import threading
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .cache import cached, last_success
from dashboard import db

# ── Credentials (from Render env vars) ────────────────────────────────────────
PB_CLIENT_ID     = os.environ.get("PRACTICE_BETTER_CLIENT_ID", "")
PB_CLIENT_SECRET = os.environ.get("PRACTICE_BETTER_CLIENT_SECRET", "")
AN_LOGIN         = os.environ.get("AUTHNET_API_LOGIN_ID", "")
AN_KEY           = os.environ.get("AUTHNET_TRANSACTION_KEY", "")
WISE_TOKEN       = os.environ.get("WISE_API_TOKEN", "")
WISE_PROFILE_ID  = int(os.environ.get("WISE_PROFILE_ID", "50305528"))
QB_CLIENT_ID     = os.environ.get("QUICKBOOKS_PROD_CLIENT_ID", "")
QB_CLIENT_SECRET = os.environ.get("QUICKBOOKS_PROD_CLIENT_SECRET", "")
QB_REALM_ID      = os.environ.get("QUICKBOOKS_PROD_REALM_ID", "")
QB_REFRESH_TOKEN = os.environ.get("QUICKBOOKS_PROD_REFRESH_TOKEN", "")

QB_TOKEN_URL    = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
QB_BASE_URL     = f"https://quickbooks.api.intuit.com/v3/company/{QB_REALM_ID}"

# Legacy file cache path, kept read-only for one-time seeding during cutover.
# The refresh token itself now lives in the oauth_tokens DB row "qbo_refresh"
# (see _qb_rt_read/_qb_rt_write_at below) so it survives across disk-less
# multi-instance web workers. Retire this file once one successful prod
# refresh has written the DB row.
QB_RT_CACHE     = os.environ.get("QB_RT_CACHE_PATH", "/data/qb_refresh_token")

_rt_lock = threading.Lock()
_QB_RT_NAME = "qbo_refresh"


def _qb_default_db_path() -> str:
    # Mirror app.LOG_DB / gmail_token.default_db_path: DATA_DIR persistent disk
    # on Render (/data), repo root in local dev.
    root = os.environ.get("DATA_DIR") or str(Path(__file__).resolve().parent.parent)
    return str(Path(root) / "chat_log.db")


def _qb_rt_read(db_path: str):
    try:
        with db.connect(db_path, timeout=10) as cx:
            row = cx.execute(
                "SELECT token_json FROM oauth_tokens WHERE name=?", (_QB_RT_NAME,)
            ).fetchone()
    except db.OperationalError as e:
        # Table not yet created (fresh DB): SQLite "no such table" /
        # Postgres UndefinedTable. Treat as "no token", not an error.
        if "no such table" in str(e).lower() or "does not exist" in str(e).lower():
            return None
        raise
    return (row[0].strip() or None) if row and row[0] else None


def _qb_rt_write_at(db_path: str, token: str) -> None:
    """Upsert the RT row. Assumes the caller holds the appropriate lock
    (in-process on SQLite, the FOR UPDATE row lock on Postgres)."""
    if not token:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with db.connect(db_path, timeout=10) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS oauth_tokens (name TEXT PRIMARY KEY, "
            "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        cx.execute(
            "INSERT INTO oauth_tokens (name, token_json, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET token_json=excluded.token_json, "
            "updated_at=excluded.updated_at",
            (_QB_RT_NAME, token, ts),
        )
        cx.commit()


def _qb_rt_seed_from_legacy_file():
    """One-time: if the DB row is empty but the legacy /data file exists,
    return its contents so the already-rotated on-disk RT is not lost at cutover."""
    try:
        with open(QB_RT_CACHE) as f:
            return f.read().strip() or None
    except (FileNotFoundError, OSError):
        return None


# ── PB ────────────────────────────────────────────────────────────────────────
def pb_token():
    r = requests.post("https://api.practicebetter.io/oauth2/token",
                      data={"grant_type": "client_credentials",
                            "client_id": PB_CLIENT_ID,
                            "client_secret": PB_CLIENT_SECRET},
                      timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


def pb_get(token, path, params=None):
    r = requests.get(f"https://api.practicebetter.io{path}",
                     headers={"Authorization": f"Bearer {token}"},
                     params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


@cached("money.pb")
def pb_data(days=30):
    token = pb_token()
    invoices = pb_get(token, "/consultant/payments/invoices").get("items", [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    collected = outstanding = 0.0
    recent = []
    for inv in invoices:
        date_str = inv.get("invoiceDate", "")
        try:
            inv_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            inv_date = datetime.min.replace(tzinfo=timezone.utc)
        total = inv.get("total", {}).get("amount", 0) / 100
        paid  = inv.get("amountPaid", {}).get("amount", 0) / 100
        due   = inv.get("amountDue", {}).get("amount", 0) / 100
        if due > 0:
            outstanding += due
        if inv_date >= cutoff:
            collected += paid
            client = inv.get("clientRecord", {}).get("profile", {})
            recent.append({
                "date": date_str[:10],
                "name": f"{client.get('firstName','')} {client.get('lastName','')}".strip(),
                "email": client.get("email", ""),
                "amount": total, "paid": paid, "due": due,
                "invoice": inv.get("invoiceNumber", "—"),
            })
    return {"collected": collected, "outstanding": outstanding,
            "invoices": recent, "last_success": last_success("money.pb")}


# ── Authorize.net ─────────────────────────────────────────────────────────────
def an_post(payload):
    r = requests.post("https://api.authorize.net/xml/v1/request.api",
                      json=payload, timeout=15)
    r.raise_for_status()
    text = r.text.lstrip("﻿")  # AN returns BOM
    import json as _json
    return _json.loads(text)


@cached("money.an")
def an_data(days=30):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    payload = {"getSettledBatchListRequest": {
        "merchantAuthentication": {"name": AN_LOGIN, "transactionKey": AN_KEY},
        "includeStatistics": True,
        "firstSettlementDate": start.strftime("%Y-%m-%dT00:00:00Z"),
        "lastSettlementDate":  end.strftime("%Y-%m-%dT23:59:59Z"),
    }}
    data = an_post(payload).get("batchList", [])
    total = refunds = 0.0
    count = 0
    by_date = []
    for batch in data:
        stats = batch.get("statistics", [])
        b_charged  = sum(s.get("chargeAmount", 0)  for s in stats)
        b_refunded = sum(s.get("refundAmount", 0)  for s in stats)
        b_count    = sum(s.get("chargeCount", 0)   for s in stats)
        total += b_charged; refunds += b_refunded; count += b_count
        by_date.append({"date": batch.get("settlementTimeLocal", "")[:10],
                        "amount": b_charged, "refunds": b_refunded, "count": b_count})
    return {"total": total, "refunds": refunds, "net": total - refunds,
            "count": count, "batches": by_date,
            "last_success": last_success("money.an")}


# ── Wise ──────────────────────────────────────────────────────────────────────
@cached("money.wise")
def wise_data(days=30):
    h = {"Authorization": f"Bearer {WISE_TOKEN}"}
    bals = requests.get(
        f"https://api.wise.com/v4/profiles/{WISE_PROFILE_ID}/balances?types=STANDARD",
        headers=h, timeout=15).json()
    balances = [{"currency": b["currency"], "amount": b["amount"]["value"]} for b in bals]
    return {"balances": balances, "last_success": last_success("money.wise")}


# ── QuickBooks ────────────────────────────────────────────────────────────────
def qb_refresh():
    """Refresh the QB access token. The refresh token rotates on every use
    (Intuit invalidates the one just sent), so read->POST->persist must be
    serialized: a SELECT..FOR UPDATE row lock on Postgres (correct across
    instances), an in-process lock on SQLite (single instance by construction).
    Seed order for the RT: persisted DB row -> legacy /data file -> env var."""
    import base64
    auth = base64.b64encode(f"{QB_CLIENT_ID}:{QB_CLIENT_SECRET}".encode()).decode()
    db_path = _qb_default_db_path()

    def _do_refresh(rt):
        r = requests.post(QB_TOKEN_URL,
                          headers={"Authorization": f"Basic {auth}",
                                   "Accept": "application/json",
                                   "Content-Type": "application/x-www-form-urlencoded"},
                          data={"grant_type": "refresh_token", "refresh_token": rt},
                          timeout=15)
        r.raise_for_status()
        payload = r.json()
        return payload.get("refresh_token"), payload["access_token"]

    def _candidates(persisted):
        cands = []
        if persisted:
            cands.append(("db", persisted))
        seed = _qb_rt_seed_from_legacy_file()
        if seed and seed != persisted:
            cands.append(("file", seed))
        if QB_REFRESH_TOKEN and QB_REFRESH_TOKEN not in (persisted, ""):
            cands.append(("env", QB_REFRESH_TOKEN))
        return cands

    last_err = None

    if db.backend() == "postgres":
        # Hold the row lock across the whole read->POST->write critical section.
        with db.connect(db_path, timeout=15) as cx:
            cx.execute(
                "CREATE TABLE IF NOT EXISTS oauth_tokens (name TEXT PRIMARY KEY, "
                "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            row = cx.execute(
                "SELECT token_json FROM oauth_tokens WHERE name=? FOR UPDATE",
                (_QB_RT_NAME,),
            ).fetchone()
            persisted = (row[0].strip() or None) if row and row[0] else None
            for source, rt in _candidates(persisted):
                try:
                    new_rt, access = _do_refresh(rt)
                    if new_rt:
                        cx.execute(
                            "INSERT INTO oauth_tokens (name, token_json, updated_at) "
                            "VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET "
                            "token_json=excluded.token_json, updated_at=excluded.updated_at",
                            (_QB_RT_NAME, new_rt, datetime.now(timezone.utc).isoformat()),
                        )
                    cx.commit()  # releases the FOR UPDATE lock
                    return access
                except requests.HTTPError as e:
                    last_err = e
                    print(f"[qb] refresh from {source} failed: {e}")
                    continue
            cx.commit()
    else:
        # SQLite single instance: in-process lock; do NOT hold a DB write txn
        # across the network POST (SQLite write locks are database-wide).
        with _rt_lock:
            persisted = _qb_rt_read(db_path)
            for source, rt in _candidates(persisted):
                try:
                    new_rt, access = _do_refresh(rt)
                    if new_rt:
                        _qb_rt_write_at(db_path, new_rt)
                    return access
                except requests.HTTPError as e:
                    last_err = e
                    print(f"[qb] refresh from {source} failed: {e}")
                    continue

    if last_err:
        raise last_err
    raise RuntimeError("No QB refresh token available (DB empty + file/env unset)")


def qb_get(access_token, path, params=None):
    r = requests.get(f"{QB_BASE_URL}{path}",
                     headers={"Authorization": f"Bearer {access_token}",
                              "Accept": "application/json"},
                     params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


@cached("money.qb_banks")
def qb_banks():
    tok = qb_refresh()
    rs = qb_get(tok, "/query",
                {"query": "SELECT * FROM Account WHERE AccountType = 'Bank'"})
    accounts = rs.get("QueryResponse", {}).get("Account", [])
    return {"accounts": [{"name": a.get("Name"),
                          "balance": a.get("CurrentBalance", 0),
                          "currency": a.get("CurrencyRef", {}).get("value", "USD")}
                         for a in accounts],
            "last_success": last_success("money.qb_banks")}


# ── Aggregate endpoints ───────────────────────────────────────────────────────
def today_summary():
    """Today's incoming across all sources."""
    today = datetime.now(timezone.utc).date().isoformat()
    pb = pb_data(days=1)
    an = an_data(days=1)
    wise = wise_data()
    return {
        "pb_today": sum(i["paid"] for i in pb["invoices"] if i["date"] == today),
        "an_today": sum(b["amount"] for b in an["batches"] if b["date"] == today),
        "wise_balances": wise["balances"],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def week_summary():
    pb = pb_data(days=7)
    an = an_data(days=7)
    return {
        "pb_collected": pb["collected"],
        "pb_outstanding": pb["outstanding"],
        "an_net": an["net"],
        "an_count": an["count"],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
