"""Guard: customer-facing URLs must not hardcode the onrender backend host.

illtowell.com is the public funnel domain. Customer-facing links (share links,
affiliate recruit links, emailed URLs) must derive from PUBLIC_BASE_URL so they
read illtowell.com, not glen-knowledge-chat.onrender.com. A small allow-list
covers the legitimate internal references to Render's own service URL (its
self-call hostname fallbacks and the local-token-mint dev-ops hint), which are
NOT customer-facing.
"""
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"

# Substrings that legitimately reference the Render backend host. Each line in
# app.py mentioning onrender.com must contain one of these, or it's a regression.
_ALLOWED_LINE_MARKERS = (
    "RENDER_EXTERNAL_URL",        # Render self-call URL fallback
    "RENDER_EXTERNAL_HOSTNAME",   # Render self-call hostname fallback
    "or set ",                    # token-mint dev-ops hint (points devs at the live backend)
)


def test_public_base_url_defaults_to_illtowell():
    src = APP.read_text()
    assert 'os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com")' in src


def test_no_customer_facing_onrender_hardcode():
    offenders = []
    for i, line in enumerate(APP.read_text().splitlines(), 1):
        if "onrender.com" not in line:
            continue
        if any(m in line for m in _ALLOWED_LINE_MARKERS):
            continue
        offenders.append(f"{i}: {line.strip()}")
    assert not offenders, (
        "Customer-facing onrender.com hardcode(s) found; derive from "
        "PUBLIC_BASE_URL instead:\n" + "\n".join(offenders))
