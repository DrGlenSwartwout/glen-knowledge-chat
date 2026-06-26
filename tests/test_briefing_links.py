from dashboard import briefing_links as bl


def test_person_url_encodes_email():
    assert bl.person_url("jane@x.com") == "/console/crm?email=jane%40x.com"


def test_build_linkables_stamps_inbox_senders():
    snap = {"inbox": {"oldest": [
        {"subject": "Re: order", "from": "jane@x.com", "age_days": 5},
        {"subject": "hi", "from": "bob@y.com", "age_days": 2},
    ]}}
    reg = bl.build_linkables(snap)
    refs = [r["ref"] for r in snap["inbox"]["oldest"]]
    assert refs == ["r1", "r2"]
    assert reg["r1"] == {"type": "person", "display": "jane@x.com",
                         "url": "/console/crm?email=jane%40x.com"}


def test_build_linkables_stamps_pb_invoice_clients_with_email():
    snap = {"money": {"practice_better": {"invoices": [
        {"name": "Jane Doe", "email": "jane@x.com", "invoice": "INV-1", "due": 50},
    ]}}}
    reg = bl.build_linkables(snap)
    ref = snap["money"]["practice_better"]["invoices"][0]["ref"]
    assert reg[ref]["display"] == "Jane Doe"
    assert reg[ref]["url"] == "/console/crm?email=jane%40x.com"


def test_build_linkables_dedupes_same_person_across_blocks():
    snap = {
        "inbox": {"oldest": [{"from": "jane@x.com", "age_days": 5}]},
        "money": {"practice_better": {"invoices": [
            {"name": "Jane Doe", "email": "jane@x.com", "due": 50}]}},
    }
    reg = bl.build_linkables(snap)
    assert snap["inbox"]["oldest"][0]["ref"] == "r1"
    assert snap["money"]["practice_better"]["invoices"][0]["ref"] == "r1"
    assert list(reg.keys()) == ["r1"]


def test_build_linkables_skips_records_without_email():
    snap = {"money": {"practice_better": {"invoices": [
        {"name": "No Email Client", "invoice": "INV-9", "due": 99},
    ]}}}
    reg = bl.build_linkables(snap)
    assert "ref" not in snap["money"]["practice_better"]["invoices"][0]
    assert reg == {}


def test_build_linkables_survives_error_blocks():
    snap = {"inbox": {"_error": "inbox: TimeoutError: boom"},
            "money": {"practice_better": {"_error": "pb_data: KeyError"}}}
    assert bl.build_linkables(snap) == {}  # no crash, no refs


def test_build_linkables_parses_from_header_with_display_name():
    snap = {"inbox": {"oldest": [
        {"subject": "Re: order", "from": "Jane Doe <Jane@X.com>", "age_days": 5},
    ]}}
    reg = bl.build_linkables(snap)
    ref = snap["inbox"]["oldest"][0]["ref"]
    assert reg[ref]["url"] == "/console/crm?email=jane%40x.com"
    assert reg[ref]["display"] == "Jane Doe"


def test_build_linkables_dedupes_case_insensitively():
    snap = {"inbox": {"oldest": [
        {"from": "Jane@X.com", "age_days": 5},
        {"from": "jane@x.com", "age_days": 2},
    ]}}
    reg = bl.build_linkables(snap)
    assert [r.get("ref") for r in snap["inbox"]["oldest"]] == ["r1", "r1"]
    assert list(reg.keys()) == ["r1"]


def test_build_linkables_excludes_self_and_automated_senders():
    # Real inbox senders include the connected mailbox itself and bounce/no-reply
    # system addresses; none of those are linkable clients.
    snap = {"inbox": {"oldest": [
        {"from": "Glen Swartwout <drglenswartwout@gmail.com>", "age_days": 9},
        {"from": "Mail Delivery Subsystem <mailer-daemon@googlemail.com>", "age_days": 7},
        {"from": "no-reply@stripe.com", "age_days": 4},
        {"from": "Real Client <client@example.com>", "age_days": 3},
    ]}}
    reg = bl.build_linkables(snap)
    oldest = snap["inbox"]["oldest"]
    assert "ref" not in oldest[0]           # self (connected mailbox)
    assert "ref" not in oldest[1]           # mailer-daemon
    assert "ref" not in oldest[2]           # no-reply@
    assert oldest[3]["ref"] == "r1"         # the only real client links
    assert list(reg.keys()) == ["r1"]
    assert reg["r1"]["url"] == "/console/crm?email=client%40example.com"


def test_is_person_email_filters():
    assert bl._is_person_email("client@example.com") is True
    assert bl._is_person_email("drglenswartwout@gmail.com") is False   # default self
    assert bl._is_person_email("mailer-daemon@googlemail.com") is False
    assert bl._is_person_email("postmaster@x.com") is False
    assert bl._is_person_email("noreply@x.com") is False
    assert bl._is_person_email("no-reply@x.com") is False
    assert bl._is_person_email("notifications@x.com") is False
    assert bl._is_person_email("") is False
    assert bl._is_person_email("not-an-email") is False
