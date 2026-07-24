"""Console sub-pages must share the main console's stored key, and must not
render a blank/empty state when an API call fails.

Background: /console/household read its key ONLY from its own ?key= URL param.
Opened from an already-unlocked console (no ?key=), every call 401'd and the page
rendered an EMPTY members table — indistinguishable from "no members" — so adds
looked like they silently did nothing when in fact they were never authorized.
"""

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"

# Pages that authenticate with the console key and must fall back to the shared
# localStorage key the main console (console.html saveKey) writes.
KEY_FALLBACK_PAGES = [
    "console-household.html",
    "console-members.html",
    "console-studio-credits.html",
]


def test_console_pages_fall_back_to_stored_console_key():
    for name in KEY_FALLBACK_PAGES:
        src = (STATIC / name).read_text()
        assert 'localStorage.getItem("console_key")' in src, (
            f"{name} does not fall back to the shared console_key; opened without "
            f"?key= it will silently 401"
        )
        # still accepts a scriptable ?key=
        assert 'get("key")' in src, f"{name} dropped ?key= support"


def test_household_page_surfaces_auth_failure_instead_of_empty_table():
    src = (STATIC / "console-household.html").read_text()
    # load() must bail out on a failed response rather than rendering the table
    assert "if(!r.ok)" in src
    assert "Unauthorized" in src
    # the write paths must report failure too, not silently re-load
    assert "Add failed" in src
    assert "Remove failed" in src
