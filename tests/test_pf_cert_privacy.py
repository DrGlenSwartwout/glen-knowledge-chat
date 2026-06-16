"""The public finder hides a certification-track practitioner's raw email/phone
by default (contact is via the inquiry form). A practitioner who opts in
(show_contact=true) has email/phone published. Website + cert level always stay;
non-cert rows are unchanged. The internal show_contact flag is stripped from the
payload."""
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
         "modules_completed": 0, "show_contact": False,
         "city": "Austin", "state": "TX"},
        {"id": "2", "name": "Certified Pro", "tier": "panel_certified",
         "email": "pro@personal.com", "phone": "555-333-4444",
         "website": "", "modules_completed": 7, "show_contact": True,
         "city": "Mesa", "state": "AZ"},
        {"id": "3", "name": "Directory Org", "tier": "org_member",
         "email": "org@clinic.com", "phone": "555-999-0000",
         "website": "https://clinic.com", "modules_completed": None,
         "city": "Austin", "state": "TX"},
    ]
    monkeypatch.setattr(appmod.pf_db, "run_search", lambda **kw: [dict(r) for r in rows])

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_cert_row_without_optin_redacts_keeps_website_and_level(client):
    r = client.get("/api/practitioner-finder/search?location=Austin&country=US")
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.get_json()["practitioners"]}

    cert = by_name["Cert Student"]
    assert cert["email"] is None and cert["phone"] is None      # redacted (no opt-in)
    assert cert["website"] == "https://linkedin.com/in/certstudent"  # kept
    assert cert["modules_completed"] == 0                        # level kept
    assert "show_contact" not in cert                            # internal flag stripped


def test_cert_row_with_optin_shows_contact(client):
    r = client.get("/api/practitioner-finder/search?location=Mesa&country=US")
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.get_json()["practitioners"]}

    certified = by_name["Certified Pro"]
    assert certified["email"] == "pro@personal.com"             # opted in -> shown
    assert certified["phone"] == "555-333-4444"
    assert certified["modules_completed"] == 7
    assert "show_contact" not in certified                      # internal flag stripped


def test_non_cert_row_always_shows_contact(client):
    r = client.get("/api/practitioner-finder/search?location=Austin&country=US")
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.get_json()["practitioners"]}

    org = by_name["Directory Org"]
    assert org["email"] == "org@clinic.com"
    assert org["phone"] == "555-999-0000"
    assert "show_contact" not in org
