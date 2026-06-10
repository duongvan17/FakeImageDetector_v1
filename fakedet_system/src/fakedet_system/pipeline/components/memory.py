
from __future__ import annotations

from haystack import component

from ...memory import ConversationStore


@component
class MemoryReader:
    def __init__(self, store: ConversationStore, history_turns: int) -> None:
        self._store = store
        self._n = history_turns

    @component.output_types(history=list, core_memory=str, last_image=str)
    def run(self, session_id: str) -> dict:
        return {
            "history": self._store.recent_turns(session_id, self._n),
            "core_memory": self._store.format_core_memory(session_id),
            "last_image": self._store.get_last_image(session_id),
        }


@component
class MemoryWriter:
    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    @component.output_types(reply=str)
    def run(
        self,
        session_id: str,
        user_text: str,
        reply: str,
        intent: str,
        image_data_uri: str,
    ) -> dict:
        if user_text:
            self._store.add_turn(session_id, "user", user_text, intent)
        self._store.add_turn(session_id, "assistant", reply, intent)
        if image_data_uri:
            self._store.set_last_image(session_id, image_data_uri)
        return {"reply": reply}
