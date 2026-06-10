from fakedet_system.memory import ConversationStore


def test_history_roundtrip():
    s = ConversationStore(":memory:")
    s.add_turn("sess", "user", "hello", "user_guide")
    s.add_turn("sess", "assistant", "hi there", "user_guide")
    s.add_turn("other", "user", "different session")

    turns = s.recent_turns("sess", 8)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "hello"
    assert turns[1]["intent"] == "user_guide"
    assert s.recent_turns("sess", 0) == []


def test_recent_turns_is_oldest_first_and_capped():
    s = ConversationStore(":memory:")
    for i in range(10):
        s.add_turn("s", "user", f"m{i}")
    turns = s.recent_turns("s", 3)
    assert [t["content"] for t in turns] == ["m7", "m8", "m9"]


def test_core_memory_upsert_and_internal_keys_hidden():
    s = ConversationStore(":memory:")
    s.set_core_memory("s", "name", "Duong")
    s.set_core_memory("s", "name", "Duong V.")
    s.set_last_image("s", "data:image/png;base64,AAAA")

    assert s.get_core_memory("s")["name"] == "Duong V."
    # __-prefixed blobs must never leak into a prompt string
    fmt = s.format_core_memory("s")
    assert "name: Duong V." in fmt
    assert "AAAA" not in fmt
    assert s.get_last_image("s") == "data:image/png;base64,AAAA"
    assert s.get_last_image("missing") == ""
