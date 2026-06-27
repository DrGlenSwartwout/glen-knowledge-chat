"""Read recent inbox emails FROM known clients, for the testimonial-invite scan.

Privacy: metadata (From/Subject) + Gmail's short snippet only — never the full body, never
stored. Returns {client_email: text} for the classifier; only a short positive quote is ever
persisted (in the candidate), and only from emails whose sender is a known client.
"""
import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def _hdr(msg, name):
    for h in ((msg.get("payload") or {}).get("headers") or []):
        if (h.get("name") or "").lower() == name.lower():
            return h.get("value") or ""
    return ""


def _from_email(value):
    m = _EMAIL_RE.search(value or "")
    return m.group(0).lower() if m else ""


def recent_client_messages(known_emails, *, days=90, limit=200, service=None):
    """{client_email: text} for inbox messages from known clients in the window.
    Best-effort: returns {} on any Gmail/auth error."""
    known = {(e or "").strip().lower() for e in (known_emails or []) if e}
    if not known:
        return {}
    try:
        if service is None:
            from dashboard import inbox as _inbox
            service = _inbox._get_gmail_service()
        listing = service.users().messages().list(
            userId="me", q=f"in:inbox newer_than:{int(days)}d", maxResults=int(limit)).execute()
        out = {}
        for m in (listing.get("messages") or []):
            try:
                msg = service.users().messages().get(
                    userId="me", id=m["id"], format="metadata",
                    metadataHeaders=["From", "Subject"]).execute()
            except Exception:
                continue
            frm = _from_email(_hdr(msg, "From"))
            if frm not in known:
                continue
            text = (_hdr(msg, "Subject") + ". " + (msg.get("snippet") or "")).strip()
            if text:
                out[frm] = (out.get(frm, "") + "\n" + text).strip()[:2000]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[gmail_feedback] read failed: {e!r}", flush=True)
        return {}
