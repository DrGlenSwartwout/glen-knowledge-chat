import types
from dashboard import inbox
from dashboard import gmail_token as gt

def test_inbox_service_uses_durable_loader(monkeypatch):
    fake_creds = object()
    loaded = gt.LoadedGmail(creds=fake_creds, source="db",
                            original_json="{}", name="inbox_gmail")
    seen = {}

    def _fake_load(db_path, name="inbox_gmail", scopes=None):
        seen["name"] = name
        return loaded

    monkeypatch.setattr(gt, "load_gmail_credentials", _fake_load)
    monkeypatch.setattr(inbox, "_build_service",
                        lambda creds: types.SimpleNamespace(creds=creds))
    svc = inbox._get_gmail_service()
    assert seen["name"] == "inbox_gmail"
    assert svc.creds is fake_creds
