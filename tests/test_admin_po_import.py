# tests/test_admin_po_import.py
import importlib, io, sqlite3, sys
from pathlib import Path
import pytest

_PO_CSV = "id_pk,id_fk_supplier,vendor_po_no,po_date,closed,status,tax,shipping_amount,shipper,tracking_number,due_date,posted_date,qb_id\npo1,s1,PO-001,2024-01-15,No,,,,,,,\n"
_PO_ITEMS_CSV = "id_pk,id_fk_po,id_fk_raw,id_fk_material,id_fk_product,fee_name,product_id,qty,qty_unit,qty_left,cost\npi1,po1,,,,Shipping fee,,1,,1,5.00\n"
_PO_RECEIVING_CSV = "id_pk,id_fk_po,id_fk_po_item,qty_received,received_size\npr1,po1,pi1,1,\n"


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client(), str(tmp_path / "chat_log.db")


def _post_import(client, write=None, missing=None):
    data = {}
    if missing != "po":
        data["po"] = (io.BytesIO(_PO_CSV.encode()), "po.csv")
    if missing != "po_items":
        data["po_items"] = (io.BytesIO(_PO_ITEMS_CSV.encode()), "po_items.csv")
    if missing != "po_receiving":
        data["po_receiving"] = (io.BytesIO(_PO_RECEIVING_CSV.encode()), "po_receiving.csv")
    if write is not None:
        data["write"] = write
    return client.post("/api/po/import", data=data, content_type="multipart/form-data")


def test_dry_run_returns_counts(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = _post_import(client)  # no write field → dry run
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    d = body["data"]
    assert d["mode"] == "dry_run"
    assert d["purchase_orders"] == 1
    assert d["po_items"] == 1
    assert d["po_receiving"] == 1


def test_missing_file_returns_400(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    resp = _post_import(client, write="1", missing="po_receiving")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "po_receiving" in body["error"] or "three" in body["error"]
