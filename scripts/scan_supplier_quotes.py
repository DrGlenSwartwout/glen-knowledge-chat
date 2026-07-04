#!/usr/bin/env python3
"""Scan Glen's inbox for supplier quotes → stage into supplier_quotes (review queue).
IMAP read (reuses the email-bounce-scan pattern) + Haiku tool-use extraction. Dry-run default."""
import argparse, imaplib, email, os, re, sqlite3, sys
from email.header import decode_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import sourcing as sc  # noqa: E402

_MODEL = "claude-haiku-4-5-20251001"
_KW = re.compile(r"\b(quote|price|\$|/kg|/lb|per kg|moq|minimum order|lead time|cost|coa|c of a)\b", re.I)

_TOOL = {
    "name": "record_quote",
    "description": "Record a supplier price quote extracted from a sourcing email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_supplier_quote": {"type": "boolean"},
            "supplier_name": {"type": "string"}, "ingredient_name": {"type": "string"},
            "price": {"type": "number"}, "price_unit": {"type": "string"},
            "currency": {"type": "string"},
            "moq": {"type": "number"}, "moq_unit": {"type": "string"},
            "lead_time_days": {"type": "integer"}, "confidence": {"type": "number"},
        },
        "required": ["is_supplier_quote"],
    },
}


def looks_like_quote(subject: str, body: str) -> bool:
    return bool(_KW.search((subject or "") + " " + (body or "")))


def _client():
    """Lazy import so unit tests that mock the module don't pull anthropic at import."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def extract_quote(subject, body, client=None):
    client = client or _client()
    msg = client.messages.create(
        model=_MODEL, max_tokens=600, tools=[_TOOL], tool_choice={"type": "tool", "name": "record_quote"},
        messages=[{"role": "user", "content":
                   f"Extract the supplier price quote from this email. If it is not a supplier price "
                   f"quote, set is_supplier_quote=false.\n\nSubject: {subject}\n\n{body[:6000]}"}])
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            data = block.input
            return data if data.get("is_supplier_quote") else None
    return None


def _to_stage_row(msg_id, from_email, subject, data, has_attachments=0, received_at=None):
    return {"gmail_msg_id": msg_id, "from_email": from_email, "subject": subject,
            "received_at": received_at,
            "raw_snippet": (subject or "")[:200], "has_attachments": has_attachments,
            "supplier_name": data.get("supplier_name"), "ingredient_name": data.get("ingredient_name"),
            "price": data.get("price"), "price_unit": data.get("price_unit"), "currency": data.get("currency"),
            "moq": data.get("moq"), "moq_unit": data.get("moq_unit"),
            "lead_time_days": data.get("lead_time_days"), "confidence": data.get("confidence")}


def _received_iso(date_hdr):
    """RFC-2822 Date header → 'YYYY-MM-DD HH:MM:SS' (UTC-ish) for a sortable received_at; None on failure."""
    if not date_hdr:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_hdr).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _decode(s):
    if not s:
        return ""
    out = []
    for part, enc in decode_header(s):
        out.append(part.decode(enc or "utf-8", "replace") if isinstance(part, bytes) else part)
    return "".join(out)


def _body_text(m):
    if m.is_multipart():
        for p in m.walk():
            if p.get_content_type() == "text/plain":
                try:
                    return p.get_payload(decode=True).decode(p.get_content_charset() or "utf-8", "replace")
                except Exception:
                    pass
        return ""
    try:
        return m.get_payload(decode=True).decode(m.get_content_charset() or "utf-8", "replace")
    except Exception:
        return ""


def scan(write=False, days=14, db_path=None, imap=None, client=None, max_messages=None) -> dict:
    user = os.environ.get("GMAIL_DRGLEN_USER", "drglenswartwout@gmail.com")
    pw = os.environ.get("GMAIL_DRGLEN_APP_PASSWORD", "")
    own = imap is None
    if own:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(user, pw)
    staged = scanned = 0
    try:
        imap.select("INBOX")
        import datetime
        since = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%d-%b-%Y")
        typ, data = imap.search(None, "SINCE", since)
        ids = data[0].split() if data and data[0] else []
        # Bound the work: a full RFC822 fetch per message is the slow part, so cap to
        # the most-recent N. The daily cron uses a short window; a wider manual backfill
        # can raise the cap. None = unbounded.
        if max_messages and len(ids) > max_messages:
            ids = ids[-max_messages:]
        cx = sqlite3.connect(db_path or sc._default_db_path()); cx.row_factory = sqlite3.Row
        try:
            sc.init_sourcing_schema(cx)
            existing = {r["gmail_msg_id"] for r in cx.execute("SELECT gmail_msg_id FROM supplier_quotes WHERE gmail_msg_id IS NOT NULL")}
            for num in ids:
                typ, md = imap.fetch(num, "(RFC822)")
                m = email.message_from_bytes(md[0][1])
                msg_id = m.get("Message-ID") or num.decode()
                if msg_id in existing:
                    continue
                subject = _decode(m.get("Subject")); body = _body_text(m)
                scanned += 1
                if not looks_like_quote(subject, body):
                    continue
                data2 = extract_quote(subject, body, client=client)
                if not data2:
                    continue
                row = _to_stage_row(msg_id, _decode(m.get("From")), subject, data2,
                                    has_attachments=1 if m.is_multipart() and any(p.get_filename() for p in m.walk()) else 0,
                                    received_at=_received_iso(m.get("Date")))
                n = sc.stage_quotes(cx, [row]); staged += n
            if write:
                cx.commit()
            else:
                cx.rollback()
        finally:
            cx.close()
    finally:
        if own:
            imap.logout()
    return {"scanned": scanned, "staged": staged, "mode": "write" if write else "dry_run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    print(scan(write=args.write, days=args.days))


if __name__ == "__main__":
    main()
