"""Money widgets — PB sessions, Authorize.net (GrooveKart + GHL), Wise, QuickBooks."""

import os
import requests
from datetime import datetime, timedelta, timezone

from .cache import cached, last_success

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

QB_TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
QB_BASE_URL  = f"https://quickbooks.api.intuit.com/v3/company/{QB_REALM_ID}"


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
    """Refresh and persist QB token (Render env var update needed manually if expired)."""
    import base64
    auth = base64.b64encode(f"{QB_CLIENT_ID}:{QB_CLIENT_SECRET}".encode()).decode()
    r = requests.post(QB_TOKEN_URL,
                      headers={"Authorization": f"Basic {auth}",
                               "Accept": "application/json",
                               "Content-Type": "application/x-www-form-urlencoded"},
                      data={"grant_type": "refresh_token",
                            "refresh_token": QB_REFRESH_TOKEN},
                      timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


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
