import pathlib

STATIC = pathlib.Path(__file__).resolve().parents[1] / "static"


def test_op_nav_has_client_subentry():
    js = (STATIC / "op-nav.js").read_text()
    assert '/console/client' in js
    assert 'id:"client"' in js or "id:'client'" in js


def test_crm_links_to_hub():
    html = (STATIC / "console-crm.html").read_text()
    assert "/console/client" in html
