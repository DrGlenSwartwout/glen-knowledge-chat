"""Task 4: _support_program_for applies the modifier resolver (base ±
modifiers) rather than the program's raw authored items."""
import sqlite3
from dashboard import condition_programs as cp


def test_support_program_for_applies_diagnosis_implied(monkeypatch, tmp_path):
    import app
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_table(cx)
        cp.upsert(cx, "dry-amd", "Dry AMD", False,
            [{"slug": "wholomega", "name": "WholOmega"}],
            [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
              "client_default": True,
              "items": [{"slug": "lipid-zyme", "name": "Lipid Zyme"}]}])
    monkeypatch.setattr(app, "_client_condition_for", lambda e: "dry-amd")
    monkeypatch.setattr(app, "order_destination",
                        type("D", (), {"destination_for": staticmethod(lambda s: "/x")}))
    sp = app._support_program_for("client@example.com")
    assert [i["name"] for i in sp["items"]] == ["WholOmega", "Lipid Zyme"]
