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


def test_edit_invoice_has_print_download_pdf_button():
    assert 'id="print-inv-btn"' in SRC
    assert "Print / Download PDF" in SRC
    assert '"/api/console/order/"+Number(EDIT_OID)+"/invoice-link"' in SRC
    assert '$("print-inv-btn").style.display = "inline-block"' in SRC


def test_line_items_table_contains_delete_column_without_overlay():
    assert '<div class="lines-scroll">' in SRC
    assert ".lines-scroll { width:100%; overflow-x:auto; overflow-y:hidden;" in SRC
    assert "table.lines { width:100%; min-width:1000px;" in SRC
