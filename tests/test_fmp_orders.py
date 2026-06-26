import sqlite3
from dashboard import fmp_orders as fo


def _cx():
    cx = sqlite3.connect(":memory:")
    fo.ensure_tables(cx)
    return cx


def _seed(cx):
    cx.executemany("INSERT INTO fmp_clients (id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business) VALUES (?,?,?,?,?,?,?,?)", [
        ("c1", "JoAnn", "Cuddigan", "Sun Star Organics", "joann@sunstarorganics.com", "", "808-555", ""),
        ("c2", "Pam", "Schreur", "", "iamsure@att.net", "", "", ""),
    ])
    cx.executemany("INSERT INTO fmp_invoices (id_pk,id_fk_client,invoice_date,status,subtotal,total,shipping,outstanding) VALUES (?,?,?,?,?,?,?,?)", [
        ("i1", "c1", "2026-04-01", "Closed", "100.00", "113.00", "13.00", "0.00"),
        ("i2", "c1", "2026-01-15", "Closed", "50.00", "50.00", "0.00", "0.00"),
    ])
    cx.executemany("INSERT INTO fmp_invoice_items (id_pk,id_fk_invoice,id_fk_product,description,qty,price,ext_price) VALUES (?,?,?,?,?,?,?)", [
        ("it1", "i1", "p1", "Lens-Zyme", "2", "35.00", "70.00"),
        ("it2", "i1", "p2", "Lipid Cleanse", "1", "30.00", "30.00"),
        ("it3", "i2", "p1", "Lens-Zyme", "1", "50.00", "50.00"),
    ])
    cx.executemany("INSERT INTO fmp_client_addresses (id_pk,id_fk_client,type,street,city,province,postal_code,country) VALUES (?,?,?,?,?,?,?,?)", [
        ("a1", "c1", "", "123 Farm Rd", "Asheville", "NC", "28801", "USA"),
        ("a2", "c1", "", "old 9 Other St", "Portland", "OR", "97214", "USA"),
    ])
    cx.commit()


def test_lookup_by_client_id_orders_newest_first_items_grouped():
    cx = _cx(); _seed(cx)
    res = fo.client_order_history(cx, client_id="c1")
    assert len(res) == 1
    c = res[0]
    assert c["client"]["email"] == "joann@sunstarorganics.com"
    assert [o["id"] for o in c["orders"]] == ["i1", "i2"]      # newest first
    assert len(c["orders"][0]["items"]) == 2                   # i1 has 2 lines
    assert {i["description"] for i in c["orders"][0]["items"]} == {"Lens-Zyme", "Lipid Cleanse"}
    assert len(c["addresses"]) == 2                            # both on file


def test_lookup_by_email_case_insensitive():
    cx = _cx(); _seed(cx)
    res = fo.client_order_history(cx, email="JoAnn@SunStarOrganics.com")
    assert len(res) == 1 and res[0]["client"]["id"] == "c1"


def test_lookup_by_name_like_matches_company_and_person():
    cx = _cx(); _seed(cx)
    assert {r["client"]["id"] for r in fo.client_order_history(cx, name="cuddigan")} == {"c1"}
    assert {r["client"]["id"] for r in fo.client_order_history(cx, name="sun star")} == {"c1"}
    assert fo.client_order_history(cx, name="zzz") == []


def test_build_projection_from_csv(tmp_path):
    d = tmp_path
    (d / "clients.csv").write_text("id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business\nc1,Jo,Cud,SSO,jo@x.com,,,\n")
    (d / "invoices.csv").write_text("id_pk,id_fk_client,invoice_date,closed,zc_invoice_subtotal,zc_invoice_total,shipping_fee,zc_overdue_balance\ni1,c1,2026-04-01,Active,100,113,13,0\n")
    (d / "invoice_items.csv").write_text("id_pk,id_fk_invoice,id_fk_product,description,qty,price,zc_ext_price\nit1,i1,p1,Lens-Zyme,2,35,70\n")
    (d / "clients_address.csv").write_text("id_pk,id_fk_client,type,address_street,address_city,address_province,address_postal_code,address_country\na1,c1,,123 Farm Rd,Asheville,NC,28801,USA\n")
    cx = sqlite3.connect(":memory:")
    counts = fo.build_projection_from_csv(cx, d)
    assert counts == {"clients": 1, "invoices": 1, "items": 1, "addresses": 1}
    res = fo.client_order_history(cx, client_id="c1")
    assert res[0]["orders"][0]["total"] == "113"
    assert res[0]["orders"][0]["items"][0]["ext_price"] == "70"
    assert res[0]["addresses"][0]["city"] == "Asheville"


def test_to_payload_roundtrips():
    cx = _cx(); _seed(cx)
    pay = fo.to_payload(cx)
    assert set(pay) == {"clients", "invoices", "items", "addresses"}
    cx2 = sqlite3.connect(":memory:"); fo.ensure_tables(cx2)
    fo.ingest_payload(cx2, pay)
    assert fo.client_order_history(cx2, client_id="c1")[0]["client"]["id"] == "c1"


def test_invoice_dates_normalized_and_sorted_across_years(tmp_path):
    d = tmp_path
    (d / "clients.csv").write_text("id_pk,name_first,name_last,company,email,phone_res,phone_cell,phone_business\nc1,Jo,Cud,,jo@x.com,,,\n")
    # M/D/YYYY across a year boundary: text-sort would put 12/1/2025 after 1/5/2026 (wrong)
    (d / "invoices.csv").write_text(
        "id_pk,id_fk_client,invoice_date,closed,zc_invoice_subtotal,zc_invoice_total,shipping_fee,zc_overdue_balance\n"
        "iA,c1,1/5/2026,Active,10,10,0,0\n"
        "iB,c1,12/1/2025,Active,20,20,0,0\n")
    (d / "invoice_items.csv").write_text("id_pk,id_fk_invoice,id_fk_product,description,qty,price,zc_ext_price\n")
    (d / "clients_address.csv").write_text("id_pk,id_fk_client,type,address_street,address_city,address_province,address_postal_code,address_country\n")
    cx = sqlite3.connect(":memory:")
    fo.build_projection_from_csv(cx, d)
    orders = fo.client_order_history(cx, client_id="c1")[0]["orders"]
    assert [o["date"] for o in orders] == ["2026-01-05", "2025-12-01"]  # ISO + newest-first


def test_none_raising_on_empty():
    cx = _cx()
    assert fo.client_order_history(cx, name="anything") == []
