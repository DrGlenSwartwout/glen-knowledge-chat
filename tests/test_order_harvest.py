from dashboard.order_harvest import detect_source, parse_order_email, _norm_name

def test_detect_source():
    assert detect_source("Transactions@prod.eprocessingnetwork.com") == "eprocessing"
    assert detect_source("noreply@mail.authorize.net") == "authorizenet"
    assert detect_source("support@remedymatch.com") == "neworder"
    assert detect_source("Glen Swartwout <drglenswartwout@gmail.com>") == "invoice"
    assert detect_source("noreply-ecns@usps.com") is None

# eProcessing: merchant "Email:" block first, then customer "Name:"/"E-Mail:"
EPROC = """This message is to confirm that a transaction has been processed
Remedy Match LLC may be contacted at:
   Address: 351 Wailuku Drive
     Phone: (808) 217-9647
     Email: support@remedymatch.com
Order information is as follows:
    Invoice: 480
    Name: Jane Buyer
    E-Mail: jane.buyer@example.com
    Card Type: AX
"""

def test_parse_eprocessing_customer_block_not_merchant():
    r = parse_order_email("eprocessing", EPROC)
    assert r["name"] == "Jane Buyer"
    assert r["email"] == "jane.buyer@example.com"   # customer, NOT support@remedymatch.com

# remedymatch New-order (storefront): customer line + product remedy links
NEWORDER = """<p>New order : #1042</p>
<p>customer: Sam Storefront (sam.storefront@example.com)</p>
<p>Delivery address: Sam Storefront, 5 Main St, Reno, NV 89501</p>
<table><tr><td><a href="/remedies/204-ocuheal-eye-drops">OcuHeal Eye Drops</a></td><td>2</td></tr></table>
"""

def test_parse_neworder_customer_and_products():
    r = parse_order_email("neworder", NEWORDER)
    assert r["name"] == "Sam Storefront"
    assert r["email"] == "sam.storefront@example.com"
    assert r["products"] == ["OcuHeal Eye Drops"]

# Authorize.net Merchant Email Receipt: merchant block, then customer fields
AUTHNET = """Merchant: Remedy Match LLC
support@remedymatch.com
Customer Information
First Name: Carl
Last Name: Client
Email: carl.client@example.com
Phone: 555-222-3333
"""

def test_parse_authorizenet():
    r = parse_order_email("authorizenet", AUTHNET)
    assert r["name"] == "Carl Client"
    assert r["email"] == "carl.client@example.com"
    assert r["phone"] == "555-222-3333"

# Authorize.net receipt where the merchant block ALSO carries Phone:/Email:
# labels ahead of "Customer Information" — must not leak into the customer record.
AUTHNET_WITH_MERCHANT = """Merchant: Remedy Match LLC
Phone: (808) 217-9647
Email: support@remedymatch.com
Customer Information
First Name: Carl
Last Name: Client
Email: carl.client@example.com
Phone: 555-222-3333
"""

def test_parse_authorizenet_ignores_merchant_labels():
    r = parse_order_email("authorizenet", AUTHNET_WITH_MERCHANT)
    assert r["name"] == "Carl Client"
    assert r["email"] == "carl.client@example.com"
    assert r["phone"] == "555-222-3333"
    assert "(808) 217-9647" not in (r["phone"] or "")
    assert r["email"] != "support@remedymatch.com"

def test_parse_invoice_uses_to_header():
    body = "To: Deb Buyer <deb.buyer@example.com>\nSubject: Your Remedy Match invoice INH-77\n"
    r = parse_order_email("invoice", body)
    assert r["email"] == "deb.buyer@example.com"
    assert r["name"] == "Deb Buyer"

def test_parse_unknown_source_returns_none():
    assert parse_order_email("mystery", "whatever") is None

def test_norm_name():
    assert _norm_name("J. Morris  Williams") == "j morris williams"


from dashboard.order_harvest import harvest_buyer

def _search_returning(msgs):
    return lambda query: msgs

def test_harvest_single_exact_match_hits():
    msgs = [{"sender": "Transactions@prod.eprocessingnetwork.com",
             "body": "Name: J Morris Williams\nE-Mail: jmw@example.com\n"}]
    r = harvest_buyer(_search_returning(msgs), "J Morris Williams")
    assert r["email"] == "jmw@example.com"
    assert r["first"] == "J" and r["last"] == "Morris Williams"
    assert r["source"] == "eprocessing"

def test_harvest_name_mismatch_returns_none():
    msgs = [{"sender": "Transactions@prod.eprocessingnetwork.com",
             "body": "Name: Someone Else\nE-Mail: else@example.com\n"}]
    assert harvest_buyer(_search_returning(msgs), "J Morris Williams") is None

def test_harvest_two_distinct_emails_returns_none():
    msgs = [
        {"sender": "Transactions@prod.eprocessingnetwork.com",
         "body": "Name: Pat Lee\nE-Mail: pat1@example.com\n"},
        {"sender": "support@remedymatch.com",
         "body": "customer: Pat Lee (pat2@example.com)\n"},
    ]
    assert harvest_buyer(_search_returning(msgs), "Pat Lee") is None

def test_harvest_same_email_two_sources_prefers_neworder():
    msgs = [
        {"sender": "Transactions@prod.eprocessingnetwork.com",
         "body": "Name: Pat Lee\nE-Mail: pat@example.com\n"},
        {"sender": "support@remedymatch.com",
         "body": 'customer: Pat Lee (pat@example.com)\n<a href="/remedies/1-x">X</a>'},
    ]
    r = harvest_buyer(_search_returning(msgs), "Pat Lee")
    assert r["email"] == "pat@example.com"
    assert r["source"] == "neworder"          # storefront wins → enables onboarding
    assert r["products"] == ["X"]

def test_harvest_merchant_only_returns_none():
    msgs = [{"sender": "noreply-ecns@usps.com", "body": "irrelevant"}]
    assert harvest_buyer(_search_returning(msgs), "Anyone") is None
