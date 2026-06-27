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


def test_money_snapshot_includes_qbo_ar():
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "import finance as _finance" in src
    assert '"qbo_ar"' in src
    assert "_finance.open_invoices" in src


def test_money_prompt_uses_qbo_ar_for_receivables():
    from dashboard import briefing_runner as br
    p = br.SLUG_PROMPTS["money-cash"]
    assert "qbo_ar" in p                      # AR comes from the QBO block
    assert "practice_better" in p             # PB still named, as separate activity


def test_record_links_instruction_covers_invoices():
    from dashboard import briefing_runner as br
    # _build_user_prompt embeds the RECORD LINKS rule; it must invite linking invoices, not just people
    snap = {"money": {"qbo_ar": [{"id": "501", "doc": "1024", "customer": "Acme Co",
                                  "balance": 5000.0, "days_overdue": 32}]}}
    from dashboard import briefing_links as bl
    bl.build_linkables(snap)
    prompt = br._build_user_prompt(snap, "money-cash")
    assert "invoice" in prompt.lower()
    assert "(ref:" in prompt  # the markdown-link form is still shown


def test_snapshot_includes_orders():
    src = (_repo() / "dashboard" / "briefing_runner.py").read_text()
    assert "import orders as _orders" in src
    assert '"orders"' in src
    assert "_orders.attention_orders" in src


def test_clients_prompt_calls_out_orders():
    from dashboard import briefing_runner as br
    p = br.SLUG_PROMPTS["clients-pipeline"]
    assert "orders" in p.lower()


def test_record_links_example_covers_orders():
    from dashboard import briefing_runner as br
    snap = {"orders": [{"id": 7, "name": "Carol", "status": "new", "pay_status": "unpaid"}]}
    from dashboard import briefing_links as bl
    bl.build_linkables(snap)
    prompt = br._build_user_prompt(snap, "clients-pipeline")
    assert "order" in prompt.lower()
    assert "(ref:" in prompt
