import sys
from pathlib import Path

from dashboard import briefing_runner as br


def _repo():
    return Path(__file__).resolve().parent.parent


def test_prompt_includes_ref_citation_instruction():
    snap = {"inbox": {"oldest": [{"from": "jane@x.com", "age_days": 5}]}}
    from dashboard import briefing_links as bl
    bl.build_linkables(snap)  # stamps ref so the snapshot shows it
    prompt = br._build_user_prompt(snap, "clients-pipeline")
    assert "ref" in prompt
    assert "(ref:" in prompt              # shows the markdown-link form
    assert "never write a real" in prompt.lower()
    assert '"ref": "r1"' in prompt        # the stamped ref is visible to the LLM


def test_regenerate_all_persists_links():
    # source-assert: the runner builds a registry and writes it per slug
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "build_linkables" in src
    assert "write_links" in src


def test_pb_data_surfaces_client_email():
    # source-assert: pb_data's recent entry includes an email field
    src = (_repo() / "dashboard" / "money.py").read_text()
    assert 'client.get("email"' in src
