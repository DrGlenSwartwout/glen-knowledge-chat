"""Edit Invoice exposes an explicit, confirmed delete control per remedy."""
from pathlib import Path


SRC = (Path(__file__).resolve().parent.parent / "static" / "order-new.html").read_text()


def test_edit_invoice_has_explicit_delete_button_per_line():
    assert "${EDIT_OID?'Delete':'Remove'}" in SRC
    assert 'class="mini danger"' in SRC
    assert 'onclick="rmLine(${i})"' in SRC


def test_edit_invoice_delete_requires_confirmation():
    assert 'if (EDIT_OID && !confirm("Delete "' in SRC
    assert "LINES.splice(i,1)" in SRC
