import sys
from pathlib import Path
from unittest.mock import patch

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_backfill_tags_named_inner_circle():
    """The backfill should call ghl_tag_contact + tag with
    beta-personal-email for each named inner-circle email."""
    inner_circle = [
        ("Glen Swartwout", "this.elf@gmail.com"),
        ("Rae Luscombe", "suerae1111@gmail.com"),
    ]
    with patch("scripts.backfill_beta_cohort.ghl_tag_contact") as m_tag:
        from scripts.backfill_beta_cohort import tag_inner_circle
        tag_inner_circle(inner_circle)
        assert m_tag.call_count == 2
        for call in m_tag.call_args_list:
            email, tag = call.args
            assert tag == "beta-personal-email"


def test_backfill_skips_placeholder_emails():
    """Emails containing '<' (placeholder syntax) should be skipped."""
    with patch("scripts.backfill_beta_cohort.ghl_tag_contact") as m_tag:
        from scripts.backfill_beta_cohort import tag_inner_circle
        tag_inner_circle([
            ("Real Person", "real@example.com"),
            ("Placeholder", "<email-tbd>"),
            ("No Email",    ""),
        ])
        assert m_tag.call_count == 1  # only the real one
