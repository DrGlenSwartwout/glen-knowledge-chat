"""Token-gated triage submit + prefill route: POST seeds condition programs
via condition_triage.seed_from_triage; GET returns stored answers for prefill."""
import importlib
import sqlite3
import sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email):
    from dashboard import client_portal as cp, condition_programs as cprog
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cp.init_client_portal_table(cx)
    cprog.init_table(cx)
    cprog.upsert(cx, "glaucoma-elevated-iop", "Glaucoma - Elevated IOP", False,
                 [{"slug": "neuroprotect", "name": "Neuroprotect"}])
    cprog.upsert(cx, "glaucoma-normal-iop", "Glaucoma - Normal IOP", False,
                 [{"slug": "eye-calm", "name": "Eye Calm"}])
    tok = cp.ensure_token(cx, email, "T")
    cx.commit()
    return tok


def test_post_triage_seeds_programs(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-a@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "glaucoma", "iop_od": 25})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["programs"] == ["glaucoma-elevated-iop"]
    assert body["seeded"] > 0


def test_get_triage_returns_stored_answers(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-b@x.com")
    appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "glaucoma", "iop_od": 25})
    r = appmod.app.test_client().get(f"/api/portal/{tok}/triage?condition=glaucoma")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    triage = body["triage"]
    assert triage["iop_od"] == "25"
    assert triage["resolved_programs"] == ["glaucoma-elevated-iop"]


def test_post_triage_vision_improvement_self_inits_condition_programs(tmp_path, monkeypatch):
    """vision-improvement is a brand-new condition_programs row added after
    prod's once-ever seed already fired, so seed_if_empty alone can never
    reach it there -- only ensure_program (called from
    _init_support_programs_tables) can. This proves the triage route calls
    that init itself, rather than relying on some other route having run
    first: we seed ONLY client_portal (no condition_programs setup at all,
    not even init_table) and still expect the program to be ensured and its
    remedies recorded."""
    from dashboard import client_portal as cp
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, "triage-vi@x.com", "T")
    cx.commit()

    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "vision-improvement"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["programs"] == ["vision-improvement"]
    assert body["seeded"] > 0


def test_starter_remedies_batch_saves_all_sections(tmp_path, monkeypatch):
    from dashboard import portal_health_history as hh, portal_extended_history as eh
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "starter-batch@x.com")

    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/starter-remedies",
        json={
            "conditions": [
                {"condition": "glaucoma", "iop_od": 25},
                {"condition": "vision-improvement"},
            ],
            "products": {
                "supplements_yes": True,
                "supplements_text": "Brand X Product Y",
            },
            "extended": {
                "surgeries_yes": True,
                "surgeries_text": "Appendectomy",
            },
        },
    )

    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(appmod.LOG_DB)
    assert hh.get(cx, "starter-batch@x.com")["supplements_text"] == "Brand X Product Y"
    assert eh.get(cx, "starter-batch@x.com")["answers"]["surgeries_text"] == "Appendectomy"
    rows = cx.execute(
        "SELECT condition FROM condition_triage WHERE email=? ORDER BY condition",
        ("starter-batch@x.com",),
    ).fetchall()
    cx.close()
    assert rows == [("glaucoma",), ("vision-improvement",)]


def test_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().post(
        "/api/portal/nope/triage", json={"condition": "glaucoma"}).status_code == 404
    assert appmod.app.test_client().get(
        "/api/portal/nope/triage").status_code == 404


def test_post_triage_wet_amd_reports_consult_recommended_true(tmp_path, monkeypatch):
    from dashboard import condition_programs as cprog
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-wet@x.com")
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cprog.upsert(cx, "wet-amd", "Wet AMD", True,
                 [{"slug": "angiogenx", "name": "AngiogenX"}])
    cx.commit()
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "macular", "amd_type": "wet"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["programs"] == ["wet-amd"]
    assert body["consult_recommended"] is True


def test_post_triage_dry_amd_reports_consult_recommended_false(tmp_path, monkeypatch):
    from dashboard import condition_programs as cprog
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-dry@x.com")
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cprog.upsert(cx, "dry-amd", "Dry AMD", False,
                 [{"slug": "wholomega", "name": "WholOmega"}])
    cx.commit()
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "macular", "amd_type": "dry"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["programs"] == ["dry-amd"]
    assert body["consult_recommended"] is False


def test_post_triage_glaucoma_reports_consult_recommended_false(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-glauc@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "glaucoma", "iop_od": 25})
    assert r.status_code == 200
    assert r.get_json()["consult_recommended"] is False


def test_post_triage_cataract_whitelist_keys_reach_the_resolver(tmp_path, monkeypatch):
    """Every new cataract/macular answer key must survive the route's POST
    answer whitelist -- if one is missing, the triage silently mis-resolves
    in production despite green unit tests on resolve_programs directly."""
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-cat@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "cataract", "cataract_type": "psc", "age": 60})
    assert r.status_code == 200
    body = r.get_json()
    # age=60 (>50) + told-PSC -> BOTH programs per the decision table -- this
    # only happens if "age" (and cataract_type) actually reached resolve_programs.
    assert body["programs"] == ["psc-cataract", "senile-cataract"]


def test_post_triage_cataract_not_sure_risk_flags_reach_the_resolver(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-cat2@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "cataract", "age": 60, "steroids": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["programs"] == ["psc-cataract", "senile-cataract"]


def test_post_triage_macular_whitelist_keys_reach_the_resolver(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-mac@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "macular", "injections": True})
    assert r.status_code == 200
    assert r.get_json()["programs"] == ["wet-amd"]


def test_post_triage_yellow_vision_whitelist_key_reaches_the_resolver(tmp_path, monkeypatch):
    """yellow_vision must survive the whitelist so brunescent threads through
    to seeding -- proven indirectly via the seeded-count including lens-zyme."""
    from dashboard import condition_programs as cprog
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-yv@x.com")
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cprog.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
                 [{"slug": "golden-book", "name": "Golden Book"}],
                 [{"when": "brunescent", "action": "add", "source": "client-reported",
                   "client_default": False,
                   "items": [{"slug": "lens-zyme", "name": "Lens-Zyme Brunescence Buster"}]}])
    cx.commit()
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage",
        json={"condition": "cataract", "cataract_type": "senile", "age": 70,
              "yellow_vision": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["seeded"] == 2  # golden-book + lens-zyme
