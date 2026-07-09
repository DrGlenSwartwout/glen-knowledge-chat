"""order-new.html must (a) pre-check pickup from the client's flag on CREATE,
(b) clear it when a non-pickup client is picked, (c) never do either in EDIT mode."""
import pathlib
import re

HTML = pathlib.Path("static/order-new.html").read_text()


def test_pick_person_sets_checkbox_both_ways():
    """`= !!p.pickup_default` — never `if (x) ... = true`, which cannot clear."""
    assert re.search(r'\$\("pickup"\)\.checked\s*=\s*!!p\.pickup_default', HTML)


def test_pick_person_guards_edit_mode():
    """Edit mode prefills from the ORDER's channel, never the client's flag."""
    m = re.search(r"function pickPerson\(p\)\{(.*?)\n\}", HTML, re.S)
    assert m, "pickPerson not found"
    body = m.group(1)
    assert "pickup_default" in body
    assert "EDIT_OID" in body, "pickPerson must not touch pickup in edit mode"


def test_always_picks_up_toggle_exists_and_posts():
    assert 'id="pickup-default"' in HTML
    assert "/api/console/customers/pickup" in HTML
