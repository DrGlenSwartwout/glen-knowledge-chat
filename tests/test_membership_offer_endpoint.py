# tests/test_membership_offer_endpoint.py
import pytest
app_mod = pytest.importorskip("app")


@pytest.fixture
def client():
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def test_offer_math_for_six_ffs(client):
    ffs = [{"slug": s, "qty": 1} for s in
           ["paracleanse", "nerve-repair", "neuroceramides",
            "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]]
    r = client.post("/api/orders/membership-offer",
                    json={"email": "nonmember-test@example.com", "tier": "month", "lines": ffs},
                    headers={"X-Console-Key": app_mod.CONSOLE_SECRET})
    assert r.status_code == 200, r.data
    j = r.get_json()
    assert j["gross_cents"] == 9900
    assert j["savings_cents"] > 0                       # membership unlocks FF savings
    assert j["net_add_cents"] == max(0, 9900 - j["savings_cents"])
    assert j["net_savings_cents"] == j["savings_cents"]
    assert "month" in j["offered_tiers"]
