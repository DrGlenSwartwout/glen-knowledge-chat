"""QuickBooks Online write layer for invoicing.

Reuses the proven auth in dashboard/money.py (qb_refresh + /data refresh-token
rotation, production base URL, realm from env). Adds the create/POST side:
customers, items, invoices, and the hosted payment link.

QBO note: an invoice line REQUIRES a SalesItemLineDetail.ItemRef to an existing
Item, so we resolve/create a Service item (which needs an income account).
"""

import requests
from . import money

QB_BASE_URL = money.QB_BASE_URL
MINOR = "75"  # QBO API minor version (required for writes)


def _post(path, body):
    tok = money.qb_refresh()
    r = requests.post(f"{QB_BASE_URL}{path}",
                      headers={"Authorization": f"Bearer {tok}",
                               "Accept": "application/json",
                               "Content-Type": "application/json"},
                      params={"minorversion": MINOR},
                      json=body, timeout=25)
    if not r.ok:
        # surface Intuit's fault detail — it's far more useful than a bare 400
        raise requests.HTTPError(f"{r.status_code} {r.text[:600]}", response=r)
    return r.json()


def _query(q):
    tok = money.qb_refresh()
    return money.qb_get(tok, "/query", {"query": q, "minorversion": MINOR})


def _esc(s):
    return (s or "").replace("'", "''")


# ── customers ────────────────────────────────────────────────────────────────
def find_or_create_customer(email, name=""):
    email = (email or "").strip()
    if email:
        rs = _query(f"SELECT * FROM Customer WHERE PrimaryEmailAddr = '{_esc(email)}'")
        arr = rs.get("QueryResponse", {}).get("Customer", [])
        if arr:
            return arr[0]
    disp = (name or email or "RemedyMatch Customer").strip()
    # DisplayName must be unique in QBO; fall back to email-qualified if needed
    body = {"DisplayName": disp}
    if email:
        body["PrimaryEmailAddr"] = {"Address": email}
    if name:
        parts = name.split(None, 1)
        body["GivenName"] = parts[0]
        if len(parts) > 1:
            body["FamilyName"] = parts[1]
    try:
        return _post("/customer", body).get("Customer")
    except requests.HTTPError:
        body["DisplayName"] = f"{disp} ({email})" if email else disp
        return _post("/customer", body).get("Customer")


# ── items + income account ─────────────────────────────────────────────────────
def list_items():
    rs = _query("SELECT * FROM Item")
    return rs.get("QueryResponse", {}).get("Item", [])


def _first_income_account_id():
    rs = _query("SELECT * FROM Account WHERE AccountType = 'Income'")
    accts = rs.get("QueryResponse", {}).get("Account", [])
    return accts[0]["Id"] if accts else None


def find_or_create_item(name, price=None, income_account_id=None):
    """Find a Service/Inventory item by name, else create a Service item.
    Returns the Item dict. Creating needs an income account."""
    rs = _query(f"SELECT * FROM Item WHERE Name = '{_esc(name)}'")
    arr = rs.get("QueryResponse", {}).get("Item", [])
    if arr:
        return arr[0]
    inc = income_account_id or _first_income_account_id()
    if not inc:
        raise RuntimeError("No income account found to create a QBO item")
    body = {"Name": name[:100], "Type": "Service",
            "IncomeAccountRef": {"value": inc}}
    if price is not None:
        body["UnitPrice"] = round(float(price), 2)
    return _post("/item", body).get("Item")


# ── invoices ──────────────────────────────────────────────────────────────────
def create_invoice(customer, lines, *, allow_online_pay=False, email_to=None):
    """lines: [{name, amount(unit $), qty, description?, item_id?}].
    customer: a QBO Customer dict (from find_or_create_customer)."""
    inv_lines = []
    for ln in lines:
        qty = int(ln.get("qty", 1) or 1)
        unit = round(float(ln["amount"]), 2)
        item_id = ln.get("item_id")
        if not item_id:
            item = find_or_create_item(ln.get("name", "RemedyMatch Product"), unit)
            item_id = item["Id"]
        inv_lines.append({
            "DetailType": "SalesItemLineDetail",
            "Amount": round(unit * qty, 2),
            "Description": ln.get("description") or ln.get("name", ""),
            "SalesItemLineDetail": {"ItemRef": {"value": item_id},
                                    "Qty": qty, "UnitPrice": unit},
        })
    body = {"CustomerRef": {"value": customer["Id"]}, "Line": inv_lines}
    if email_to:
        body["BillEmail"] = {"Address": email_to}
    if allow_online_pay:
        body["AllowOnlineCreditCardPayment"] = True
        body["AllowOnlineACHPayment"] = True
    return _post("/invoice", body).get("Invoice")


def get_invoice_pay_link(invoice):
    """Shareable hosted-payment link. Present (InvoiceLink) only when the invoice
    was created with online payment enabled AND QuickBooks Payments is active."""
    return (invoice or {}).get("InvoiceLink") or ""


def void_invoice(invoice_id, sync_token):
    """Void a test invoice (keeps the number, zeroes it). Used in verification."""
    return _post("/invoice?operation=void",
                 {"Id": str(invoice_id), "SyncToken": str(sync_token)})


# ── recurring transactions (subscriptions / memberships) ──────────────────────
def create_recurring_invoice(customer, *, item_name, amount, day_of_month,
                             start_date, interval="Monthly", num_interval=1,
                             template_name=None, email_to=None,
                             allow_online_pay=False, description=None):
    """Create a QBO RecurringTransaction (Invoice template) that auto-generates the
    invoice each cycle.

    amount: unit price ($). day_of_month: 1-31. start_date: 'YYYY-MM-DD'.
    RecurType 'Automated' = QBO auto-creates (and, per the company's email
    preference, emails) the invoice each cycle. Auto-CHARGING a card additionally
    needs QuickBooks Payments active + the member enrolled in Autopay / card-on-file;
    until then the member pays each emailed invoice via Zelle/Wise.

    Returns the created template's Invoice dict (carries Id + SyncToken)."""
    amt = round(float(amount), 2)
    item = find_or_create_item(item_name, amt)
    line = {
        "DetailType": "SalesItemLineDetail",
        "Amount": amt,
        "Description": description or item_name,
        "SalesItemLineDetail": {"ItemRef": {"value": item["Id"]},
                                "Qty": 1, "UnitPrice": amt},
    }
    name = (template_name or f"{item_name} - {customer.get('DisplayName', 'member')}")[:100]
    inv = {
        "Line": [line],
        "CustomerRef": {"value": customer["Id"]},
        "RecurringInfo": {
            "Name": name,
            "RecurType": "Automated",
            "Active": True,
            "ScheduleInfo": {
                "IntervalType": interval,
                "NumInterval": int(num_interval),
                "DayOfMonth": int(day_of_month),
                "StartDate": start_date,
            },
        },
    }
    if email_to:
        inv["BillEmail"] = {"Address": email_to}
    if allow_online_pay:
        inv["AllowOnlineCreditCardPayment"] = True
        inv["AllowOnlineACHPayment"] = True
    out = _post("/recurringtransaction", {"RecurringTransaction": {"Invoice": inv}})
    return (out.get("RecurringTransaction") or {}).get("Invoice") or out.get("Invoice")


def list_recurring():
    rs = _query("SELECT * FROM RecurringTransaction")
    arr = rs.get("QueryResponse", {}).get("RecurringTransaction", [])
    # each item nests the underlying txn (e.g. {"Invoice": {...}}); normalize
    out = []
    for it in arr:
        ent = it.get("Invoice") or it.get("RecurringTransaction", {}).get("Invoice") or it
        out.append(ent)
    return out


def set_recurring_active(rt_invoice_id, sync_token, active):
    """Activate/deactivate a recurring template (sparse update on RecurringInfo.Active)."""
    body = {"RecurringTransaction": {"Invoice": {
        "Id": str(rt_invoice_id), "SyncToken": str(sync_token), "sparse": True,
        "RecurringInfo": {"Active": bool(active)}}}}
    return _post("/recurringtransaction", body)


def delete_recurring(rt_invoice_id, sync_token):
    """Delete a recurring template (used to clean up the guarded prod test)."""
    return _post("/recurringtransaction?operation=delete",
                 {"RecurringTransaction": {"Invoice": {
                     "Id": str(rt_invoice_id), "SyncToken": str(sync_token)}}})
