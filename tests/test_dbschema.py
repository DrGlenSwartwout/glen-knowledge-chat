from dashboard.dbschema import schema_for_path

def test_basename_to_schema():
    assert schema_for_path("/data/chat_log.db") == "chat_log"
    assert schema_for_path("e4l.db") == "e4l"

def test_sanitizes_and_lowercases():
    assert schema_for_path("/x/My-DB.sqlite") == "my_db"

def test_memory_and_empty_default_public():
    assert schema_for_path(":memory:") == "public"
    assert schema_for_path("") == "public"
