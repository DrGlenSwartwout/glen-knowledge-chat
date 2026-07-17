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
