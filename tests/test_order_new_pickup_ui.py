"""order-new.html must (a) pre-check pickup from the client's flag on CREATE,
(b) clear it when a non-pickup client is picked, (c) never do either in EDIT mode."""
import pathlib
import re

HTML = pathlib.Path("static/order-new.html").read_text()


def test_pick_person_sets_checkbox_both_ways():
    """`= !!p.pickup_default` — never `if (x) ... = true`, which cannot clear."""
    assert re.search(r'\$\("pickup"\)\.checked\s*=\s*!!p\.pickup_default', HTML)


def test_pick_person_guards_edit_mode():
    """Edit mode prefills from the ORDER's channel, never the client's flag.

    The guard `if (!EDIT_OID)` must directly prefix the assignment
    $("pickup").checked = !!p.pickup_default; without it, edit mode
    would re-latch the channel (PR #734) by re-resolving the client flag.
    """
    # Require the exact structural relationship: if (!EDIT_OID) directly
    # guards the assignment to $("pickup").checked. Tolerates whitespace.
    pattern = r'if\s*\(\s*!EDIT_OID\s*\)\s*\$\("pickup"\)\.checked\s*=\s*!!p\.pickup_default'
    assert re.search(pattern, HTML), (
        "pickPerson must guard $('pickup').checked = !!p.pickup_default "
        "with if (!EDIT_OID) — edit mode never re-checks the client flag"
    )


def test_always_picks_up_toggle_exists_and_posts():
    assert 'id="pickup-default"' in HTML
    assert "/api/console/customers/pickup" in HTML
