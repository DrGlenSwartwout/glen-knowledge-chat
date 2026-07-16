from dashboard.condition_programs import resolve_program_items

def _prog():
    return {"items": [{"slug": "base1", "name": "Base One"}],
            "modifiers": [{"when": "scar", "action": "add", "source": "diagnosis-implied",
                           "client_default": True, "composer_default": False,
                           "items": [{"slug": "scar-solve", "name": "Scar Solve"}]}]}

def _slugs(rows): return {r["slug"] for r in rows}

def test_client_ignores_composer_default():
    # client_default True => client still gets it, composer_default is irrelevant here
    assert "scar-solve" in _slugs(resolve_program_items(_prog(), audience="client"))

def test_practitioner_honors_composer_default_false():
    assert "scar-solve" not in _slugs(resolve_program_items(_prog(), audience="practitioner"))

def test_practitioner_falls_back_to_client_default_when_absent():
    p = _prog(); del p["modifiers"][0]["composer_default"]
    assert "scar-solve" in _slugs(resolve_program_items(p, audience="practitioner"))
