import os
from tests.courses_fixture import write_sample_course
from dashboard import courses_lint as cl


def test_clean_course_has_no_errors(tmp_path):
    write_sample_course(str(tmp_path))
    assert cl.lint_courses(root=str(tmp_path)) == []


def test_bad_access_value_is_flagged(tmp_path):
    write_sample_course(str(tmp_path))
    p = os.path.join(str(tmp_path), "ash-intro", "01-intro", "02-welcome.md")
    with open(p, "w") as f:
        f.write("---\ntitle: X\naccess: premium\ndownloads: []\n---\n<p>body</p>\n")
    errs = cl.lint_courses(root=str(tmp_path))
    assert any("access" in e and "premium" in e for e in errs)


def test_missing_body_is_flagged(tmp_path):
    write_sample_course(str(tmp_path))
    p = os.path.join(str(tmp_path), "ash-intro", "01-intro", "01-out-takes.md")
    with open(p, "w") as f:
        f.write("---\ntitle: X\naccess: public\ndownloads: []\n---\n   \n")
    errs = cl.lint_courses(root=str(tmp_path))
    assert any("missing body" in e for e in errs)


def test_disallowed_iframe_host_is_flagged(tmp_path):
    write_sample_course(str(tmp_path))
    p = os.path.join(str(tmp_path), "ash-intro", "01-intro", "01-out-takes.md")
    with open(p, "w") as f:
        f.write(
            "---\ntitle: X\naccess: public\ndownloads: []\n---\n"
            '<iframe src="https://evil.com/steal"></iframe>\n<p>body</p>\n'
        )
    errs = cl.lint_courses(root=str(tmp_path))
    assert any("disallowed host" in e and "evil.com" in e for e in errs)


def test_missing_lesson_file_is_flagged(tmp_path):
    write_sample_course(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), "ash-intro", "01-intro", "02-welcome.md"))
    errs = cl.lint_courses(root=str(tmp_path))
    assert any("02-welcome" in e for e in errs)
