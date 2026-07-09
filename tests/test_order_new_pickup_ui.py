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
    assert 'if (o.channel==="pickup") $("pickup").checked = true;' in SRC
    assert "!EDIT_OID" in SRC   # the guard inside loadPickupDefault
