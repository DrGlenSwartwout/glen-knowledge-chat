# tests/test_course_entitlements.py
import sqlite3
import pytest
from dashboard import course_entitlements as ce


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    ce.init_course_entitlements_table(c)
    yield c
    c.close()


def test_unknown_email_is_zero(cx):
    assert ce.paid_level_for(cx, "nobody@example.com") == 0


def test_grant_cert_is_lifetime(cx):
    ce.grant_cert(cx, "A@Example.com", source="stripe", stripe_ref="cs_1")
    # normalized email, far-future now still active (lifetime)
    assert ce.paid_level_for(cx, "a@example.com", now=9_999_999_999) == 2


def test_grant_membership_active_until_expiry(cx):
    ce.grant_membership(cx, "m@x.com", until_epoch=1000.0, source="stripe", stripe_ref="sub_1")
    assert ce.paid_level_for(cx, "m@x.com", now=500.0) == 2
    assert ce.paid_level_for(cx, "m@x.com", now=1500.0) == 0


def test_membership_extend_never_shortens(cx):
    ce.grant_membership(cx, "m@x.com", until_epoch=2000.0, source="stripe", stripe_ref="sub_1")
    ce.grant_membership(cx, "m@x.com", until_epoch=1000.0, source="stripe", stripe_ref="sub_1")  # older
    assert ce.paid_level_for(cx, "m@x.com", now=1500.0) == 2  # still covered by 2000


def test_idempotent_double_grant_same_ref(cx):
    ce.grant_cert(cx, "a@x.com", source="stripe", stripe_ref="cs_9")
    ce.grant_cert(cx, "a@x.com", source="stripe", stripe_ref="cs_9")
    n = cx.execute("SELECT COUNT(*) FROM course_entitlements WHERE stripe_ref='cs_9'").fetchone()[0]
    assert n == 1


def test_expire_membership_relocks(cx):
    ce.grant_membership(cx, "m@x.com", until_epoch=9_999_999_999.0, source="stripe", stripe_ref="sub_7")
    assert ce.paid_level_for(cx, "m@x.com", now=1.0) == 2
    ce.expire_membership(cx, stripe_ref="sub_7")
    assert ce.paid_level_for(cx, "m@x.com", now=1.0) == 0


def test_manual_membership_no_ref(cx):
    ce.grant_membership(cx, "c@x.com", until_epoch=1000.0, source="manual")
    assert ce.paid_level_for(cx, "c@x.com", now=500.0) == 2


def test_paid_level_never_raises_on_broken_cx():
    class Boom:
        def execute(self, *a, **k):
            raise RuntimeError("db down")
    assert ce.paid_level_for(Boom(), "a@x.com") == 0
