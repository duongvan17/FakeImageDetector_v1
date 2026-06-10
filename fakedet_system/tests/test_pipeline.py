"""End-to-end wiring. External API + model backend mocked. Proves the four
intents route to the right responder, memory persists (incl. last image),
and follow-ups re-feed that image to the big external model."""
import httpx
import respx

from fakedet_system.config import Settings
from fakedet_system.memory import ConversationStore
from fakedet_system.pipeline.build import build_pipeline

_PNG = "data:image/png;base64,iVBORw0KGgo="
_GEMINI = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


def _settings(with_key: bool = False):
    return Settings(
        intent_api_key="k" if with_key else "",
        model_backend_url="http://model.test/v1",
        memory_history_turns=8,
    )


def _backend(reply: str):
    return respx.post("http://model.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": reply}}]
        })
    )


def _gemini(reply: str):
    return respx.post(_GEMINI).mock(return_value=httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": reply}]}}]
    }))


@respx.mock
def test_image_intent_goes_to_trained_model_and_image_remembered():
    route = _backend("🟥 FAKE — deepfake. Evidence: warped ear")
    store = ConversationStore(":memory:")
    pipe = build_pipeline(_settings(), store)

    result = pipe.run({"input_parser": {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "real or fake?"},
            {"type": "image_url", "image_url": {"url": _PNG}},
        ]}],
        "session_id": "img",
    }}, include_outputs_from={"intent"})

    assert result["intent"]["intent"] == "image_deepfake"
    assert _PNG.encode() in route.calls.last.request.content
    assert "FAKE" in result["memory_writer"]["reply"]
    # the image is kept for later "explain it more" follow-ups
    assert store.get_last_image("img") == _PNG


@respx.mock
def test_out_of_scope_is_static_no_calls():
    backend = _backend("should not be called")
    store = ConversationStore(":memory:")
    pipe = build_pipeline(_settings(), store)

    result = pipe.run({"input_parser": {
        "messages": [{"role": "user", "content": "thời tiết hôm nay thế nào?"}],
        "session_id": "oos",
    }}, include_outputs_from={"intent"})

    assert result["intent"]["intent"] == "misleading"
    assert "ngoài phạm vi" in result["memory_writer"]["reply"]
    assert not backend.called  # no model, no API for out-of-scope


@respx.mock
def test_user_guide_answered_by_external_model():
    _gemini("Attach an image to get a REAL/FAKE verdict.")
    store = ConversationStore(":memory:")
    pipe = build_pipeline(_settings(with_key=True), store)

    result = pipe.run({"input_parser": {
        "messages": [{"role": "user", "content": "hệ thống này dùng sao?"}],
        "session_id": "g",
    }}, include_outputs_from={"intent"})

    assert result["intent"]["intent"] == "user_guide"
    assert "REAL/FAKE" in result["memory_writer"]["reply"]


@respx.mock
def test_followup_reanalyses_previous_image_via_external_model():
    store = ConversationStore(":memory:")

    # turn 1: an image is detected by the trained model
    with respx.mock:
        _backend("🟥 FAKE — deepfake")
        build_pipeline(_settings(), store).run({"input_parser": {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _PNG}},
            ]}],
            "session_id": "f",
        }})
    assert store.get_last_image("f") == _PNG

    # turn 2: "why was it fake / look closer" -> external VLM with the image
    g = _gemini("The ear is warped and lighting is inconsistent.")
    result = build_pipeline(_settings(with_key=True), store).run(
        {"input_parser": {
            "messages": [{"role": "user", "content": "tại sao ảnh trước là giả?"}],
            "session_id": "f",
        }}, include_outputs_from={"intent"})

    assert result["intent"]["intent"] == "text_to_detect"
    body = g.calls.last.request.content.decode()
    assert "inline_data" in body          # the previous image was re-fed
    assert "warped" in result["memory_writer"]["reply"]


@respx.mock
def test_gemini_route_degrades_to_trained_model_when_no_key():
    # no external key -> text_to_detect should fall back to the model backend
    route = _backend("fallback answer from trained LLM")
    store = ConversationStore(":memory:")
    pipe = build_pipeline(_settings(with_key=False), store)

    result = pipe.run({"input_parser": {
        "messages": [{"role": "user", "content": "phân tích kĩ hơn giúp tôi"}],
        "session_id": "d",
    }}, include_outputs_from={"intent"})

    assert result["intent"]["intent"] == "text_to_detect"
    assert route.called
    assert result["memory_writer"]["reply"] == "fallback answer from trained LLM"
