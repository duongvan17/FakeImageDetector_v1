"""Assemble the linear FakeDet pipeline.

A Haystack ``Pipeline`` (typed DAG) whose data flow is a single straight
line — explicitly NOT agentic, no tool-choosing loop:

    input_parser -> intent -> memory_reader -> prompt_builder
                 -> responder -> memory_writer -> (reply)

``input_parser`` fans its outputs out to the stages that need them; the
side edges (session_id / user_text / intent / image) feed ``memory_writer``
so the finished turn is persisted. They do not change execution order.
"""
from __future__ import annotations

import os
from functools import lru_cache

from haystack import Pipeline

from ..config import Settings, get_settings
from ..memory import ConversationStore
from .components import (
    InputParser,
    IntentClassifier,
    MemoryReader,
    MemoryWriter,
    PromptBuilder,
    Responder,
    FeedbackIngester,
    FeedbackRAGStore,
)


def _init_tracing() -> None:
    """Wire Langfuse tracing if its keys are configured (env-driven).
    Every Haystack component run becomes a span. ``langfuse-haystack`` v1
    auto-reads ``LANGFUSE_HOST`` / ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
    and speaks the Langfuse v2 server API. No-op when keys are absent
    (tests / offline runs unaffected)."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return
    try:
        from haystack import tracing
        from haystack.dataclasses import ChatMessage
        from haystack.tracing import utils as tracing_utils
        from haystack_integrations.tracing.langfuse import LangfuseTracer
        from haystack_integrations.tracing.langfuse import tracer as lf_tracer_mod
        from langfuse import Langfuse

        # Haystack reads HAYSTACK_CONTENT_TRACING_ENABLED at IMPORT time —
        # by now haystack is already imported. Flip the flag directly so
        # spans include component inputs/outputs (not just names + timing).
        tracing.tracer.is_content_tracing_enabled = True

        # Bug in langfuse-haystack 1.x: ``set_content_tag`` assumes every
        # dict with a ``"messages"`` key holds Haystack ``ChatMessage``
        # objects and unconditionally calls ``.to_openai_dict_format()``.
        # Our pipeline passes plain dicts (PromptBuilder.plan["messages"]
        # is an OpenAI-format list of dicts) → AttributeError. Patch to
        # mirror the safer "replies" branch: serialise raw when items
        # aren't ChatMessage.
        def _strip_images(items):
            out = []
            for m in items:
                if getattr(m, "content", None) is not None:
                    # m is ChatMessage
                    continue # handled by to_openai_dict_format below
                if isinstance(m, dict) and isinstance(m.get("content"), list):
                    new_content = []
                    for p in m["content"]:
                        if isinstance(p, dict) and p.get("type") == "image_url":
                            new_content.append({"type": "image_url", "image_url": {"url": "[IMAGE OMITTED FOR LANGFUSE]"}})
                        else:
                            new_content.append(p)
                    out.append({**m, "content": new_content})
                else:
                    out.append(m)
            return out

        def _safe_set_content_tag(self, key, value):
            if not lf_tracer_mod.proxy_tracer.is_content_tracing_enabled:
                return
            if key.endswith(".input") and isinstance(value, dict) and "messages" in value:
                items = value["messages"]
                if all(isinstance(m, ChatMessage) for m in items):
                    payload = [m.to_openai_dict_format() for m in items]
                else:
                    payload = _strip_images(items)
                self._span.update(input=payload)
            elif key.endswith(".output") and isinstance(value, dict) and "replies" in value:
                items = value["replies"]
                if all(isinstance(r, ChatMessage) for r in items):
                    payload = [m.to_openai_dict_format() for m in items]
                else:
                    payload = _strip_images(items)
                self._span.update(output=payload)
            elif key.endswith(".input"):
                self._span.update(input=tracing_utils.coerce_tag_value(value))
            elif key.endswith(".output"):
                self._span.update(output=tracing_utils.coerce_tag_value(value))
            self._data[key] = value

        lf_tracer_mod.LangfuseSpan.set_content_tag = _safe_set_content_tag

        # The Langfuse client reads keys/host from env when args are omitted.
        client = Langfuse()
        tracing.enable_tracing(LangfuseTracer(tracer=client, name="fakedet"))
    except Exception:  # noqa: BLE001 — tracing is best-effort
        pass


def build_pipeline(settings: Settings, store: ConversationStore, rag_store: FeedbackRAGStore) -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("input_parser", InputParser())
    pipe.add_component("intent", IntentClassifier(settings))
    pipe.add_component(
        "memory_reader", MemoryReader(store, settings.memory_history_turns)
    )
    pipe.add_component("prompt_builder", PromptBuilder())
    pipe.add_component("feedback_ingester", FeedbackIngester(settings, rag_store))
    pipe.add_component("responder", Responder(settings, rag_store))
    pipe.add_component("memory_writer", MemoryWriter(store))

    # input_parser fans out
    pipe.connect("input_parser.user_text", "intent.user_text")
    pipe.connect("input_parser.has_image", "intent.has_image")
    pipe.connect("input_parser.user_text", "prompt_builder.user_text")
    pipe.connect("input_parser.has_image", "prompt_builder.has_image")
    pipe.connect("input_parser.image_data_uri", "prompt_builder.image_data_uri")
    pipe.connect("input_parser.session_id", "memory_reader.session_id")
    pipe.connect("input_parser.session_id", "memory_writer.session_id")
    pipe.connect("input_parser.user_text", "memory_writer.user_text")
    pipe.connect("input_parser.image_data_uri", "memory_writer.image_data_uri")

    # intent -> prompt + recorded with the turn
    pipe.connect("intent.intent", "prompt_builder.intent")
    pipe.connect("intent.intent", "memory_writer.intent")

    # memory context -> prompt
    pipe.connect("memory_reader.history", "prompt_builder.history")
    pipe.connect("memory_reader.core_memory", "prompt_builder.core_memory")
    pipe.connect("memory_reader.last_image", "prompt_builder.last_image")

    # plan -> feedback_ingester -> responder -> persist
    pipe.connect("prompt_builder.plan", "feedback_ingester.plan")
    pipe.connect("feedback_ingester.plan", "responder.plan")
    pipe.connect("responder.reply", "memory_writer.reply")

    return pipe


@lru_cache(maxsize=1)
def _shared() -> tuple[Pipeline, ConversationStore]:
    _init_tracing()
    settings = get_settings()
    store = ConversationStore(settings.memory_db_path)
    rag_store = FeedbackRAGStore(settings.chroma_db_path)
    return build_pipeline(settings, store, rag_store), store


def run_chat(
    messages: list[dict],
    session_id: str = "default",
    backend_url: str = "",
    backend_model: str = "",
) -> dict:
    """Run one turn.

    ``backend_url`` and ``backend_model`` are optional per-request overrides
    (the serve layer resolves them from the requested model id, so the same
    pipeline can talk to HF Space *and* Colab, and pick v1 or v2 classifier,
    without restarting).
    Returns ``{"reply": str, "intent": str}``."""
    pipe, _ = _shared()
    result = pipe.run(
        {
            "input_parser": {"messages": messages, "session_id": session_id},
            "feedback_ingester": {
                "backend_url": backend_url,
                "backend_model": backend_model,
            },
            "responder": {
                "backend_url": backend_url,
                "backend_model": backend_model,
            },
        },
        include_outputs_from={"intent"},
    )
    return {
        "reply": result["memory_writer"]["reply"],
        "intent": result.get("intent", {}).get("intent", "unknown"),
    }
