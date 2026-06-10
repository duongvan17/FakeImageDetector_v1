"""Environment-driven configuration for the FakeDet system.

Everything is read from environment variables (a local ``.env`` is loaded
automatically if present — see ``.env.example``). Nothing here imports
torch or the model; the trained model is reached over HTTP via
``model_backend_url``.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

# Intent taxonomy (fixed, 4 labels):
#
#   image_deepfake  - an image is attached -> the trained model decides
#                     real/fake + evidence. (decided locally, no API call)
#   text_to_detect  - in-scope follow-up / contextual question: "why was
#                     the previous image fake", "analyse it more closely",
#                     "how many images did I ask about". Answered with
#                     context/memory by the big external VLM (Gemini),
#                     optionally re-looking at the previous image.
#   user_guide      - questions about THIS system / how to use it.
#   misleading      - OUT OF SCOPE / misleading / unrelated (e.g. "what's
#                     the weather"): answered with a fixed scope notice.
INTENT_IMAGE_DEEPFAKE = "image_deepfake"
INTENT_TEXT_TO_DETECT = "text_to_detect"
INTENT_USER_GUIDE = "user_guide"
INTENT_MISLEADING = "misleading"
INTENT_USER_FEEDBACK = "user_feedback"

# Labels the external classifier chooses among when there is NO image.
TEXT_INTENTS = (INTENT_TEXT_TO_DETECT, INTENT_USER_GUIDE, INTENT_MISLEADING, INTENT_USER_FEEDBACK)
ALL_INTENTS = (INTENT_IMAGE_DEEPFAKE, *TEXT_INTENTS)


class Settings(BaseModel):
    # ---- external big LLM/VLM (Gemini default; any OpenAI-compatible too).
    #      Used BOTH for intent classification AND for answering the
    #      text_to_detect / user_guide intents (the "ăn gian" big model).
    intent_provider: str = "gemini"          # "gemini" | "openai"
    intent_api_key: str = ""
    intent_model: str = "gemini-2.0-flash"
    intent_base_url: str = "https://api.openai.com/v1"  # only if provider=openai
    intent_timeout_s: float = 20.0
    external_max_tokens: int = 512           # answer length for gemini route

    # ---- trained FakeDet model backend (fakedet_vlm.serve on a cloud GPU)
    # Two slots so the user can keep BOTH backends online and pick per chat:
    # `model_backend_url` = HF Space (slow, 24/7); `colab_backend_url` =
    # Colab T4 (fast, ephemeral, set only when the notebook is running).
    # Each appears as a separate model ID in /v1/models — open-webui lists
    # them in the dropdown and routing happens per request.
    model_backend_url: str = "http://localhost:8000/v1"
    colab_backend_url: str = ""
    model_backend_key: str = "sk-fakedet"
    model_name: str = "fakedet-vlm"
    model_name_v2: str = "fakedet-vlm-v2"
    model_timeout_s: float = 120.0

    # ---- memory (SQLite, lives on the local box)
    memory_db_path: str = "./runs/fakedet_memory.sqlite"
    memory_history_turns: int = 8
    
    # ---- rag
    chroma_db_path: str = "./runs/chroma"

    # ---- serving
    served_model_id: str = "fakedet-system"

    @property
    def external_enabled(self) -> bool:
        """The external API is used only when a key is configured. Without
        it, intent falls back to a keyword heuristic and the gemini-route
        answers fall back to the trained model's text path."""
        return bool(self.intent_api_key)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()  # no-op if there is no .env
    return Settings(
        intent_provider=_env("FAKEDET_INTENT_PROVIDER", "gemini"),
        intent_api_key=_env("FAKEDET_INTENT_API_KEY", ""),
        intent_model=_env("FAKEDET_INTENT_MODEL", "gemini-2.0-flash"),
        intent_base_url=_env("FAKEDET_INTENT_BASE_URL", "https://api.openai.com/v1"),
        intent_timeout_s=float(_env("FAKEDET_INTENT_TIMEOUT_S", "20")),
        external_max_tokens=int(_env("FAKEDET_EXTERNAL_MAX_TOKENS", "512")),
        model_backend_url=_env("FAKEDET_BACKEND_URL", "http://localhost:8000/v1"),
        colab_backend_url=_env("FAKEDET_BACKEND_URL_COLAB", ""),
        model_backend_key=_env("FAKEDET_BACKEND_KEY", "sk-fakedet"),
        model_name=_env("FAKEDET_MODEL_NAME", "fakedet-vlm"),
        model_name_v2=_env("FAKEDET_MODEL_NAME_V2", "fakedet-vlm-v2"),
        model_timeout_s=float(_env("FAKEDET_BACKEND_TIMEOUT_S", "120")),
        memory_db_path=_env("FAKEDET_MEMORY_DB", "./runs/fakedet_memory.sqlite"),
        memory_history_turns=int(_env("FAKEDET_MEMORY_TURNS", "8")),
        served_model_id=_env("FAKEDET_SYSTEM_MODEL_ID", "fakedet-system"),
    )
