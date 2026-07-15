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


def _first_bank_account_id():
    """Return the Id of the first QBO Bank account (DepositToAccountRef source)."""
    rs = _query("SELECT * FROM Account WHERE AccountType = 'Bank'")
    accts = rs.get("QueryResponse", {}).get("Account", [])
    return accts[0]["Id"] if accts else None


def create_refund_receipt(customer_id, amount, *, item_id=None,
                          bank_account_id=None, description="Refund"):
    """Issue a QBO RefundReceipt (records a money-out customer refund). `amount`
    is dollars (float). `bank_account_id` is the DepositToAccountRef -- the account
    the refund comes OUT of (defaults to the first Bank account). Returns the
    RefundReceipt dict. Mirrors the other write functions: a single _post call."""
    amt = round(float(amount), 2)
    if amt <= 0:
        raise ValueError("refund amount must be positive")
    if not item_id:
        item_id = find_or_create_item("Refund", amt)["Id"]
    if not bank_account_id:
        bank_account_id = _first_bank_account_id()
    if not bank_account_id:
        raise RuntimeError("no QBO bank account found for DepositToAccountRef")
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "DepositToAccountRef": {"value": str(bank_account_id)},
        "Line": [{
            "DetailType": "SalesItemLineDetail",
            "Amount": amt,
            "Description": description,
            "SalesItemLineDetail": {
                "ItemRef": {"value": str(item_id)},
                "Qty": 1,
                "UnitPrice": amt,
            },
        }],
    }
    return _post("/refundreceipt", body).get("RefundReceipt")


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
def _discount_line(discount_cents):
    """A single fixed-amount (not percent) QBO DiscountLineDetail, or None."""
    if not discount_cents or int(discount_cents) <= 0:
        return None
    return {
        "DetailType": "DiscountLineDetail",
        "Amount": round(int(discount_cents) / 100.0, 2),
        "DiscountLineDetail": {"PercentBased": False},
    }


def _build_invoice_lines(lines, discount_cents=0):
    """Pure: build the QBO Line array from RESOLVED lines (each must carry item_id),
    appending one fixed-amount DiscountLineDetail when discount_cents > 0. No I/O —
    item resolution happens before this is called."""
    inv_lines = []
    for ln in lines:
        qty = int(ln.get("qty", 1) or 1)
        unit = round(float(ln["amount"]), 2)
        inv_lines.append({
            "DetailType": "SalesItemLineDetail",
            "Amount": round(unit * qty, 2),
            "Description": ln.get("description") or ln.get("name", ""),
            "SalesItemLineDetail": {"ItemRef": {"value": str(ln["item_id"])},
                                    "Qty": qty, "UnitPrice": unit},
        })
    disc = _discount_line(discount_cents)
    if disc:
        inv_lines.append(disc)
    return inv_lines


def create_invoice(customer, lines, *, allow_online_pay=False, email_to=None,
                   discount_cents=0, tax_cents=0):
    """lines: [{name, amount(unit $), qty, description?, item_id?}].
    customer: a QBO Customer dict (from find_or_create_customer).
    discount_cents: optional fixed-amount discount (e.g. Wellness Credit redeemed).
    tax_cents: app-computed GET to stamp on the invoice (overrides QBO's AST engine
    via TxnTaxDetail.TotalTax); 0 = no tax line. See dashboard/tax.py."""
    resolved = []
    for ln in lines:
        item_id = ln.get("item_id")
        if not item_id:
            unit = round(float(ln["amount"]), 2)
            item_id = find_or_create_item(ln.get("name", "RemedyMatch Product"), unit)["Id"]
        resolved.append({**ln, "item_id": item_id})
    body = {"CustomerRef": {"value": customer["Id"]},
            "Line": _build_invoice_lines(resolved, discount_cents)}
    if email_to:
        body["BillEmail"] = {"Address": email_to}
    if allow_online_pay:
        body["AllowOnlineCreditCardPayment"] = True
        body["AllowOnlineACHPayment"] = True
    if tax_cents and int(tax_cents) > 0:
        # App-computed override: QBO honors an explicit TotalTax instead of its
        # automated calculation. TaxExcluded → line amounts are pre-tax.
        body["TxnTaxDetail"] = {"TotalTax": round(int(tax_cents) / 100.0, 2)}
        body["GlobalTaxCalculation"] = "TaxExcluded"
    return _post("/invoice", body).get("Invoice")


def get_invoice(invoice_id):
    """Read a single invoice by Id (for the SyncToken + current lines)."""
    tok = money.qb_refresh()
    return money.qb_get(tok, f"/invoice/{invoice_id}", {"minorversion": MINOR}).get("Invoice")


def add_invoice_line(invoice_id, *, name, amount, qty=1, item_id=None, description=None):
    """Append a product line to an existing (unpaid) invoice and return the updated
    invoice (new TotalAmt + SyncToken). Used by the post-buy concierge so a member's
    add-ons land on the SAME invoice they're about to pay. QBO Line is replaced wholesale
    on update, so we resend the existing item lines + the new one (QBO recomputes totals)."""
    inv = get_invoice(invoice_id)
    if not inv:
        raise RuntimeError(f"invoice {invoice_id} not found")
    unit = round(float(amount), 2)
    if not item_id:
        item_id = find_or_create_item(name, unit)["Id"]
    qty = int(qty or 1)
    keep = [l for l in inv.get("Line", []) if l.get("DetailType") == "SalesItemLineDetail"]
    keep.append({
        "DetailType": "SalesItemLineDetail",
        "Amount": round(unit * qty, 2),
        "Description": description or name,
        "SalesItemLineDetail": {"ItemRef": {"value": str(item_id)}, "Qty": qty, "UnitPrice": unit},
    })
    body = {"Id": str(invoice_id), "SyncToken": inv["SyncToken"], "sparse": True, "Line": keep}
    return _post("/invoice", body).get("Invoice")


def apply_invoice_discount(invoice_id, discount_cents):
    """Set a single fixed-amount discount line on an existing unpaid invoice,
    preserving its item lines (QBO replaces Line wholesale on update). Used by the
    wholesale checkout after Wellness Credit is redeemed, so the discount equals the
    amount actually committed to the ledger. Returns the updated invoice."""
    inv = get_invoice(invoice_id)
    if not inv:
        raise RuntimeError(f"invoice {invoice_id} not found")
    lines = [l for l in inv.get("Line", []) if l.get("DetailType") == "SalesItemLineDetail"]
    disc = _discount_line(discount_cents)
    if disc:
        lines.append(disc)
    body = {"Id": str(invoice_id), "SyncToken": inv["SyncToken"], "sparse": True, "Line": lines}
    return _post("/invoice", body).get("Invoice")


def replace_invoice_lines(invoice_id, lines, *, discount_cents=0, tax_cents=0):
    """Replace ALL of an unpaid invoice's lines wholesale (QBO replaces Line on a
    sparse update) with a new set — used by the console invoice editor. `lines` is the
    same shape create_invoice takes: [{name, amount(unit $), qty, description?, item_id?}]
    (include a shipping line if any). item_ids are resolved/created like create_invoice;
    one fixed-amount discount line is appended when discount_cents > 0. tax_cents stamps
    an app-computed GET override (0 = none). Returns the updated invoice (new TotalAmt +
    SyncToken). Raises if the invoice is missing or QBO rejects the update (e.g. paid)."""
    inv = get_invoice(invoice_id)
    if not inv:
        raise RuntimeError(f"invoice {invoice_id} not found")
    resolved = []
    for ln in lines:
        item_id = ln.get("item_id")
        if not item_id:
            unit = round(float(ln["amount"]), 2)
            item_id = find_or_create_item(ln.get("name", "RemedyMatch Product"), unit)["Id"]
        resolved.append({**ln, "item_id": item_id})
    body = {"Id": str(invoice_id), "SyncToken": inv["SyncToken"], "sparse": True,
            "Line": _build_invoice_lines(resolved, discount_cents)}
    if tax_cents and int(tax_cents) > 0:
        body["TxnTaxDetail"] = {"TotalTax": round(int(tax_cents) / 100.0, 2)}
        body["GlobalTaxCalculation"] = "TaxExcluded"
    return _post("/invoice", body).get("Invoice")


def record_payment(customer_id, amount_cents, invoice_id, method=None):
    """Record a QBO Payment applied to an invoice. Idempotent: skips when the invoice
    balance is already ≤ 0 (so a re-hit of the return URL won't double-pay). `method`
    (optional) is recorded as a free-text memo (PrivateNote) so split payments by
    different methods are distinguishable."""
    inv = get_invoice(invoice_id)
    if not inv:
        raise RuntimeError(f"invoice {invoice_id} not found")
    try:
        balance = float(inv.get("Balance", inv.get("TotalAmt", 0)) or 0)
    except Exception:
        balance = 0.0
    if balance <= 0:
        return inv   # already paid
    amt = round(int(amount_cents) / 100.0, 2)
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "TotalAmt": amt,
        "Line": [{"Amount": amt,
                  "LinkedTxn": [{"TxnId": str(invoice_id), "TxnType": "Invoice"}]}],
    }
    if method:
        body["PrivateNote"] = "Console payment — method: " + str(method)
    return _post("/payment", body).get("Payment")


def void_payment(qbo_txn_id):
    """Delete a QBO Payment (used when a console payment row is voided). QBO
    'delete' requires the entity's CURRENT SyncToken (optimistic concurrency),
    so fetch the Payment first — mirrors get_invoice's fetch-then-act pattern.
    Raises if the payment can't be fetched or the delete is rejected."""
    tok = money.qb_refresh()
    payment = money.qb_get(tok, f"/payment/{qbo_txn_id}",
                           {"minorversion": MINOR}).get("Payment")
    if not payment:
        raise RuntimeError(f"payment {qbo_txn_id} not found")
    _post("/payment?operation=delete",
         {"Id": str(qbo_txn_id), "SyncToken": str(payment["SyncToken"])})


def void_refund(qbo_txn_id):
    """Delete a QBO RefundReceipt (used when a console refund row is voided). A
    refund's qbo_txn_id is a RefundReceipt Id, not a Payment Id — this targets
    the correct endpoint. QBO 'delete' requires the entity's CURRENT SyncToken
    (optimistic concurrency), so fetch the RefundReceipt first — mirrors
    void_payment's fetch-then-act pattern. Raises if the refund receipt can't
    be fetched or the delete is rejected."""
    tok = money.qb_refresh()
    refund = money.qb_get(tok, f"/refundreceipt/{qbo_txn_id}",
                          {"minorversion": MINOR}).get("RefundReceipt")
    if not refund:
        raise RuntimeError(f"refund receipt {qbo_txn_id} not found")
    _post("/refundreceipt?operation=delete",
         {"Id": str(qbo_txn_id), "SyncToken": str(refund["SyncToken"])})


def record_refund(customer_id, amount_cents, invoice_id, method=None):
    """Record money-out against the customer for a refund on an invoice, as a QBO
    RefundReceipt. Amount in cents. `method` is memoed for traceability."""
    amt = round(int(amount_cents) / 100.0, 2)
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "TotalAmt": amt,
        "Line": [{"Amount": amt, "DetailType": "SalesItemLineDetail",
                  "SalesItemLineDetail": {}}],
        "PrivateNote": "Console refund — invoice " + str(invoice_id)
                       + (" — method: " + str(method) if method else ""),
    }
    return _post("/refundreceipt", body).get("RefundReceipt")


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
    # QBO recurring DayOfMonth supports only 1-28 (so every month has that day);
    # the 29th-31st clamp to 28.
    dom = max(1, min(int(day_of_month), 28))
    name = (template_name or f"{item_name} - {customer.get('DisplayName', 'member')}")[:100]
    inv = {
        "Line": [line],
        "CustomerRef": {"value": customer["Id"]},
        "RecurringInfo": {
            "Name": name,
            "RecurType": "Automated",   # Automated = QBO auto-creates each cycle
            "Active": True,
            "ScheduleInfo": {
                # Intuit's ScheduleInfo expects these numeric fields as STRINGS
                "IntervalType": interval,
                "NumInterval": str(int(num_interval)),
                "DayOfMonth": str(dom),
                "StartDate": start_date,
            },
        },
    }
    if email_to:
        inv["BillEmail"] = {"Address": email_to}
    if allow_online_pay:
        inv["AllowOnlineCreditCardPayment"] = True
        inv["AllowOnlineACHPayment"] = True
    # The create request posts the entity keyed by its type ("Invoice"); the outer
    # {"RecurringTransaction": ...} wrapper is response-only (it would 2120 here).
    out = _post("/recurringtransaction", {"Invoice": inv})
    return out.get("Invoice") or (out.get("RecurringTransaction") or {}).get("Invoice")


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
    body = {"Invoice": {
        "Id": str(rt_invoice_id), "SyncToken": str(sync_token), "sparse": True,
        "RecurringInfo": {"Active": bool(active)}}}
    return _post("/recurringtransaction", body)


def delete_recurring(rt_invoice_id, sync_token):
    """Delete a recurring template (used to clean up the guarded prod test). The entity
    is keyed by its type ("Invoice"); no outer RecurringTransaction wrapper."""
    return _post("/recurringtransaction?operation=delete",
                 {"Invoice": {"Id": str(rt_invoice_id),
                              "SyncToken": str(sync_token or "0")}})
