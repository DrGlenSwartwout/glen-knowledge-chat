"""Static assertions on the order builder's pickup controls. No browser needed:
these pin the wiring that the design depends on."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SRC = (repo_root / "static" / "order-new.html").read_text()


def test_has_a_per_client_default_checkbox():
    assert 'id="pickup-default"' in SRC
    assert "Pickup by default for this client" in SRC


def test_the_default_checkbox_saves_on_its_own():
    """It POSTs immediately, not as part of the order payload."""
    assert "savePickupDefault()" in SRC
    assert "/api/console/client-prefs" in SRC


def test_pickup_is_disabled_when_nothing_is_shippable():
    assert "function syncShippingUI()" in SRC
    assert "No shipping — nothing physical in this order" in SRC


def test_edit_mode_lets_the_order_channel_win():
    """On edit, the stored order's channel decides Pickup — not the client default."""
    # This line is pre-existing code OUTSIDE loadPickupDefault: on edit-load, the
    # stored order's channel prefills Pickup directly. Kept here for context, but
    # it doesn't prove the guard below — that's a separate assertion.
    assert 'if (o.channel==="pickup") $("pickup").checked = true;' in SRC

    start = SRC.index("async function loadPickupDefault(")
    end = SRC.index("\n}", start)
    body = SRC[start:end]
    # The guard has to live INSIDE loadPickupDefault, not merely appear
    # somewhere in the file (a bare "!EDIT_OID in SRC" would pass even if the
    # guard sat on an unrelated line, or in a different function entirely).
    assert "!EDIT_OID" in body
    checked_lines = [ln for ln in body.splitlines() if '$("pickup").checked = true' in ln]
    assert checked_lines, "expected a line inside loadPickupDefault that sets Pickup checked"
    assert all("!EDIT_OID" in ln for ln in checked_lines), (
        "the !EDIT_OID guard must sit on the same line as the assignment it "
        "gates, or an edit could silently re-adopt the client's stored default"
    )


def test_typing_an_email_also_refreshes_the_pickup_default():
    """Finding 1: staff who type/paste an email directly (bypassing the
    cust-search autocomplete -> pickPerson) must still see the pickup-default
    row and get that client's stored preference fetched. If the c-email input
    handler is ever reverted to `refreshPreview` alone, this must fail."""
    assert '$("c-email").addEventListener("input", onEmailInput);' in SRC
    assert '$("c-email").addEventListener("input", refreshPreview);' not in SRC

    start = SRC.index("function onEmailInput(")
    end = SRC.index("\n}", start)
    body = SRC[start:end]
    assert "refreshPreview()" in body, "must keep re-pricing as email changes"
    assert "loadPickupDefault(" in body, "must also refresh the pickup-default UI"
    # An email is typed one character at a time; debounce like cust-search does,
    # rather than firing a client-prefs request per keystroke.
    assert "setTimeout(" in body
