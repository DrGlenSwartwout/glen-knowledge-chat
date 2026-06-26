import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="playwright not installed")

from dashboard.biofield_report_pdf import report_pdf_bytes, save_report_pdf

HTML = "<!doctype html><html><body><h1>Hello PDF</h1></body></html>"

def test_returns_real_pdf_bytes():
    data = report_pdf_bytes(HTML)
    assert isinstance(data, (bytes, bytearray))
    assert data[:5] == b"%PDF-"          # real PDF magic
    assert len(data) > 800

def test_save_writes_file(tmp_path):
    out = str(tmp_path / "r.pdf")
    p = save_report_pdf(HTML, out)
    assert p == out
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"
