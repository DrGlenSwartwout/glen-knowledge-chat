"""Cert cohort portal notifications (member-comms subsystem A, step 3).

Two one-to-one emails that drive members to their portal:
- feedback_ready: auto-fired when a cohort submission is approved/rejected.
- assignment_notice: fired per-student by the console "notify cohort" button.

Both reuse the practitioner magic-link sign-in + the generic inbox sender, and link to the
portal (plus, for the assignment, the direct one-click record link). Best-effort: never raise.
"""
import os

from dashboard import inbox as _inbox
from dashboard import practitioner_portal as _pp


def _base_url() -> str:
    return (os.environ.get("PUBLIC_BASE_URL") or "https://illtowell.com").rstrip("/")


def _strip(s: str) -> str:
    """Keep Glen's no-em-dash voice without importing app helpers."""
    return (s or "").replace(" — ", " - ").replace("—", "-")


def _portal_magic_url(pid, email) -> str:
    tok = _pp.create_magic_link_token(pid, email)
    return f"{_base_url()}/practitioner/login-verify?token={tok}"


def _resolve_pid(practitioner_id, email):
    pid = int(practitioner_id or 0)
    if pid > 0:
        return pid
    try:
        got = _pp.id_for_email(email)
        return int(got) if got else 0
    except Exception:
        return 0


def send_feedback_ready(email, name, outcome, *, practitioner_id=0, send=None) -> bool:
    """'Your feedback is ready' — fired automatically on review. outcome: approved | refine."""
    send = send or _inbox.send_email
    email = (email or "").strip()
    if not email:
        return False
    try:
        url = _portal_magic_url(_resolve_pid(practitioner_id, email), email)
        if outcome == "approved":
            subject = "Your Level 1 assignment is approved"
            lead = "Your reflection video has been reviewed and your Level 1 assignment is approved."
        else:
            subject = "Feedback on your Level 1 video"
            lead = ("Thank you for your reflection video. We've added some feedback to help you "
                    "refine it.")
        body = (f"Hi {name or 'there'},\n\n{lead}\n\n"
                f"Open your portal to see your outcome and quality scores:\n{url}\n\n"
                f"In wellness,\nDr. Glen\n")
        send(email, subject, _strip(body), from_name="Dr. Glen Swartwout")
        return True
    except Exception as e:  # noqa: BLE001 - notifications never block moderation
        print(f"[cert_notify] feedback-ready failed for {email!r}: {e!r}", flush=True)
        return False


def send_assignment_notice(email, name, record_url, *, practitioner_id=0, send=None) -> bool:
    """'New assignment' — portal link + the direct one-click record link."""
    send = send or _inbox.send_email
    email = (email or "").strip()
    if not email:
        return False
    try:
        portal_url = _portal_magic_url(_resolve_pid(practitioner_id, email), email)
        subject = "A short Level 1 assignment - share your story on video"
        body = (f"Hi {name or 'there'},\n\n"
                f"You have a new Level 1 assignment: record a short video about something you've "
                f"learned in the course or a result you've experienced.\n\n"
                f"Record it now (one click):\n{record_url}\n\n"
                f"Or open your portal to see the full assignment:\n{portal_url}\n\n"
                f"In wellness,\nDr. Glen\n")
        send(email, subject, _strip(body), from_name="Dr. Glen Swartwout")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[cert_notify] assignment-notice failed for {email!r}: {e!r}", flush=True)
        return False
