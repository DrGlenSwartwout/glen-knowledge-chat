"""The public finder must never expose a certification-track practitioner's raw
email/phone in the search payload (contact is via the inquiry form). Website and
cert level stay; non-cert rows are unchanged."""
import pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod.pf_geocode, "geocode_place",
                        lambda place, country=None: (30.0, -97.0))

    rows = [
        {"id": "1", "name": "Cert Student", "tier": "panel_in_cert",
         "email": "cert@personal.com", "phone": "555-111-2222",
         "website": "https://linkedin.com/in/certstudent",
         "modules_completed": 0, "city": "Austin", "state": "TX"},
        {"id": "2", "name": "Certified Pro", "tier": "panel_certified",
         "email": "pro@personal.com", "phone": "555-333-4444",
         "website": "", "modules_completed": 7, "city": "Mesa", "state": "AZ"},
        {"id": "3", "name": "Directory Org", "tier": "org_member",
         "email": "org@clinic.com", "phone": "555-999-0000",
         "website": "https://clinic.com", "modules_completed": None,
         "city": "Austin", "state": "TX"},
    ]
    monkeypatch.setattr(appmod.pf_db, "run_search", lambda **kw: [dict(r) for r in rows])

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_cert_rows_redact_email_and_phone_keep_website_and_level(client):
    r = client.get("/api/practitioner-finder/search?location=Austin&country=US")
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.get_json()["practitioners"]}

    cert = by_name["Cert Student"]
    assert cert["email"] is None and cert["phone"] is None      # redacted
    assert cert["website"] == "https://linkedin.com/in/certstudent"  # kept
    assert cert["modules_completed"] == 0                        # level kept

    certified = by_name["Certified Pro"]
    assert certified["email"] is None and certified["phone"] is None
    assert certified["modules_completed"] == 7

    # non-cert directory rows are untouched
    org = by_name["Directory Org"]
    assert org["email"] == "org@clinic.com"
    assert org["phone"] == "555-999-0000"
