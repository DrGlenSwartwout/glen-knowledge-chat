from tests.courses_fixture import write_sample_course
from dashboard import courses_content as cc


def test_load_course_parses_structure(tmp_path):
    write_sample_course(str(tmp_path))
    course = cc.load_course("ash-intro", root=str(tmp_path))
    assert course.title == "ASH Intro"
    assert len(course.modules) == 1
    lessons = course.modules[0].lessons
    assert [l.slug for l in lessons] == ["01-out-takes", "02-welcome"]
    assert lessons[0].access == "public"
    assert lessons[0].rumble_id == ""  # optional/ignored in the Stage 1.5 model
    assert "rumble.com/embed/v1abcd" in lessons[0].body_md
    assert lessons[1].access == "member"
    assert "Welcome transcript" in lessons[1].body_md
    assert "youtube.com/embed/v2efgh" in lessons[1].body_md


def test_list_courses_finds_course_dirs(tmp_path):
    write_sample_course(str(tmp_path))
    slugs = [c.slug for c in cc.list_courses(root=str(tmp_path))]
    assert slugs == ["ash-intro"]


def test_render_body_outputs_html():
    assert "<p>" in cc.render_body("hello **world**")
