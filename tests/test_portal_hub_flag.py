import inspect
from dashboard import portal_view


def test_get_portal_view_accepts_hub_enabled():
    sig = inspect.signature(portal_view.get_portal_view)
    assert "hub_enabled" in sig.parameters
    assert sig.parameters["hub_enabled"].default is False
