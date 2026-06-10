"""FakeDet thesis system.

A *linear* Haystack pipeline (explicitly NOT agentic) wrapped around the
trained FakeDet VLM:

    InputParser -> IntentClassifier -> MemoryReader -> PromptBuilder
                 -> FakeDetClient -> MemoryWriter

Design constraints (thesis):
  * Intent classification is delegated to an EXTERNAL LLM API (Gemini by
    default, any OpenAI-compatible endpoint otherwise). This package never
    runs an LLM itself.
  * The *final answer always comes from the trained FakeDet model*, reached
    over HTTP (the model runs on a cloud GPU; this package runs on the
    local 4 GB box).
  * Conversation history + core memory are persisted in SQLite.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
