"""Harvest a buyer's contact from the order emails in the connected mailbox.

Pure parsing + precision gate; no Gmail/GHL calls here (those are injected by the
watcher). Always reads the CUSTOMER block, never the merchant block
(Healing Oasis / 351 Wailuku Drive / support@remedymatch.com / (808) 217-9647).
"""
from __future__ import annotations
import re
from typing import Callable, Optional

_MERCHANT_EMAIL = "support@remedymatch.com"


def _norm_name(s: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", "", (s or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_source(sender: str) -> Optional[str]:
    s = (sender or "").lower()
    if "eprocessingnetwork.com" in s:
        return "eprocessing"
    if "mail.authorize.net" in s:
        return "authorizenet"
    if "support@remedymatch.com" in s:
        return "neworder"
    if "drglenswartwout@gmail.com" in s:
        return "invoice"
    return None


def _clean(v: Optional[str]) -> Optional[str]:
    v = (v or "").strip()
    return v or None


def parse_order_email(source: str, body: str) -> Optional[dict]:
    body = body or ""
    name = email = phone = None
    products: list[str] = []

    if source == "eprocessing":
        # Customer labels are "Name:" and "E-Mail:" (hyphen). The merchant block
        # uses "Email:" (no hyphen) = support@remedymatch.com — never matched here.
        m = re.search(r"^\s*Name:\s*(.+?)\s*$", body, re.M)
        name = _clean(m.group(1)) if m else None
        m = re.search(r"E-Mail:\s*([^\s<>]+@[^\s<>]+)", body)
        email = _clean(m.group(1)) if m else None

    elif source == "neworder":
        m = re.search(r"customer:\s*(.+?)\s*\(([^)]+@[^)]+)\)", body, re.I)
        if m:
            name = _clean(m.group(1))
            email = _clean(m.group(2))
        products = [re.sub(r"<[^>]+>", "", t).strip()
                    for t in re.findall(r'/remedies/[^"]+">([^<]+)</a>', body)]

    elif source == "authorizenet":
        # Scope to the customer block only — a merchant block earlier in the
        # body (Healing Oasis / support@remedymatch.com / (808) 217-9647) can
        # carry its own "Phone:"/"Email:" labels and must never be matched.
        marker = "Customer Information"
        idx = body.find(marker)
        customer_block = body[idx + len(marker):] if idx != -1 else body
        f = re.search(r"First Name:\s*(.+?)\s*$", customer_block, re.M)
        l = re.search(r"Last Name:\s*(.+?)\s*$", customer_block, re.M)
        if f or l:
            name = _clean(" ".join(x.group(1).strip() for x in (f, l) if x))
        m = re.search(r"\bEmail:\s*([^\s<>]+@[^\s<>]+)", customer_block)
        email = _clean(m.group(1)) if m else None
        m = re.search(r"Phone:\s*([0-9()+\-.\s]{7,})", customer_block)
        phone = _clean(m.group(1)) if m else None

    elif source == "invoice":
        m = re.search(r"^To:\s*(.*?)\s*<([^>]+@[^>]+)>", body, re.M)
        if m:
            name = _clean(m.group(1))
            email = _clean(m.group(2))
        else:
            m = re.search(r"^To:\s*([^\s<>]+@[^\s<>]+)", body, re.M)
            email = _clean(m.group(1)) if m else None
    else:
        return None

    if email and email.lower() == _MERCHANT_EMAIL:
        email = None
    return {"source": source, "name": name, "email": email,
            "phone": phone, "products": products}
