import pytest
app_mod = pytest.importorskip("app")


@pytest.fixture
def client():
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _preview(client, lines, email=""):
    r = client.post("/api/orders/price-preview",
                    json={"email": email, "lines": lines},
                    headers={"X-Console-Key": app_mod.CONSOLE_SECRET})
    assert r.status_code == 200, r.data
    return r.get_json()


def test_membership_line_flips_products_to_member(client):
    # A non-member cart of 6 different FFs prices at LIST without a membership line...
    ffs = [{"slug": s, "qty": 1} for s in
           ["paracleanse", "nerve-repair", "neuroceramides",
            "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]]
    base = _preview(client, ffs, email="nonmember-test@example.com")
    ff_lines = [l for l in base["lines"] if l["slug"] != "membership:month"]
    assert all(l["effective_unit_cents"] == l["list_cents"] for l in ff_lines)

    # ...but adding the membership line flips them to member pricing (savings > 0)
    withmem = _preview(client, ffs + [{"slug": "membership:month", "qty": 1}],
                       email="nonmember-test@example.com")
    ff_lines2 = [l for l in withmem["lines"] if l["slug"] != "membership:month"]
    assert all(l["effective_unit_cents"] < l["list_cents"] for l in ff_lines2)
    # the membership line itself is priced at the tier price and is not FF
    mem = [l for l in withmem["lines"] if l["slug"] == "membership:month"][0]
    assert mem["effective_unit_cents"] == 9900
    assert mem["is_ff"] is False
    assert mem["savings_cents"] == 0
    # subtotal includes the $99 membership on top of the (now discounted) products
    assert withmem["subtotal_cents"] == sum(l["line_cents"] for l in withmem["lines"])


def test_price_inhouse_invoice_membership_line_flips_products_to_member():
    # Direct coverage of _price_inhouse_invoice (the real order-creation/edit path,
    # not just the preview endpoint) — a membership line in the cart should price
    # itself at the tier's fixed price AND flip the accompanying FFs to member rate.
    ffs = [{"slug": s, "qty": 1} for s in
           ["paracleanse", "nerve-repair", "neuroceramides",
            "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]]
    lines = ffs + [{"slug": "membership:month", "qty": 1}]
    priced = app_mod._price_inhouse_invoice(
        lines, email="nonmember-direct-test@example.com", pickup=True, ship=None)
    assert priced is not None

    items_rec = priced["items_rec"]
    mem_recs = [r for r in items_rec if r["slug"] == "membership:month"]
    assert len(mem_recs) == 1
    mem = mem_recs[0]
    assert mem["kind"] == "membership"
    assert mem["line_cents"] == 9900
    assert mem["unit_cents"] == 9900

    ff_recs = [r for r in items_rec if r["slug"] != "membership:month"]
    assert len(ff_recs) == 6
    for r in ff_recs:
        assert r["unit_cents"] < 6997, r

    assert priced["subtotal_cents"] == sum(r["line_cents"] for r in items_rec)
