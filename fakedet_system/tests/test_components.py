import httpx
import respx

from fakedet_system.config import (
    INTENT_IMAGE_DEEPFAKE,
    INTENT_MISLEADING,
    INTENT_TEXT_TO_DETECT,
    INTENT_USER_GUIDE,
    Settings,
)
from fakedet_system.pipeline.components import (
    InputParser,
    IntentClassifier,
    PromptBuilder,
)

_PNG = "data:image/png;base64,iVBORw0KGgo="
_GEMINI = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


# --------------------------------------------------------------- InputParser
def test_input_parser_text_only():
    out = InputParser().run(
        messages=[{"role": "user", "content": "is this true?"}],
        session_id="s1",
    )
    assert out["user_text"] == "is this true?"
    assert out["has_image"] is False
    assert out["session_id"] == "s1"


def test_input_parser_multimodal():
    out = InputParser().run(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "check this"},
            {"type": "image_url", "image_url": {"url": _PNG}},
        ],
    }])
    assert out["has_image"] is True
    assert out["image_data_uri"] == _PNG
    assert out["user_text"] == "check this"


# ----------------------------------------------------------- IntentClassifier
def test_intent_image_short_circuits_without_api():
    out = IntentClassifier(Settings(intent_api_key="")).run("", True)
    assert out == {"intent": INTENT_IMAGE_DEEPFAKE, "intent_source": "rule"}


def test_intent_heuristic_routes_out_of_scope_to_misleading():
    clf = IntentClassifier(Settings(intent_api_key=""))
    assert clf.run("how do I use this?", False)["intent"] == INTENT_USER_GUIDE
    assert clf.run("tại sao ảnh trước là giả", False)["intent"] == INTENT_TEXT_TO_DETECT
    # off-topic must NOT be mistaken for a detection request
    assert clf.run("thời tiết hôm nay thế nào", False)["intent"] == INTENT_MISLEADING
    assert clf.run("viết hộ tôi đoạn code python", False)["intent"] == INTENT_MISLEADING


@respx.mock
def test_intent_calls_external_and_normalises():
    route = respx.post(_GEMINI).mock(return_value=httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": "misleading\n"}]}}]
    }))
    out = IntentClassifier(Settings(intent_api_key="k")).run("anything", False)
    assert route.called
    assert out == {"intent": INTENT_MISLEADING, "intent_source": "gemini"}


@respx.mock
def test_intent_falls_back_when_api_errors():
    respx.post(_GEMINI).mock(return_value=httpx.Response(500))
    out = IntentClassifier(Settings(intent_api_key="k")).run("how to use", False)
    assert out["intent_source"] == "heuristic"
    assert out["intent"] == INTENT_USER_GUIDE


# ------------------------------------------------------------- PromptBuilder
def _pb(**kw):
    base = dict(
        user_text="", intent=INTENT_MISLEADING, has_image=False,
        image_data_uri="", history=[], core_memory="", last_image="",
    )
    base.update(kw)
    return PromptBuilder().run(**base)["plan"]


def test_plan_image_route_model_with_image():
    plan = _pb(intent=INTENT_IMAGE_DEEPFAKE, has_image=True, image_data_uri=_PNG)
    assert plan["route"] == "model"
    assert plan["messages"][0]["content"][1]["image_url"]["url"] == _PNG


def test_plan_misleading_is_static_notice():
    plan = _pb(intent=INTENT_MISLEADING, user_text="thời tiết?")
    assert plan["route"] == "static"
    assert "ngoài phạm vi" in plan["static_reply"]


def test_plan_user_guide_route_gemini_help_prompt():
    plan = _pb(intent=INTENT_USER_GUIDE, user_text="dùng sao?")
    assert plan["route"] == "gemini"
    assert "help assistant" in plan["system"]
    assert plan["image"] == ""
    assert plan["messages"][-1] == {"role": "user", "content": "dùng sao?"}


def test_plan_text_to_detect_reuses_last_image_and_context():
    plan = _pb(
        intent=INTENT_TEXT_TO_DETECT, user_text="tại sao giả?",
        history=[{"role": "assistant", "content": "FAKE: blurry edges"}],
        core_memory="name: Duong", last_image=_PNG,
    )
    assert plan["route"] == "gemini"
    assert plan["image"] == _PNG
    assert "name: Duong" in plan["system"]
    assert {"role": "assistant", "content": "FAKE: blurry edges"} in plan["messages"]
    assert plan["messages"][-1] == {"role": "user", "content": "tại sao giả?"}
