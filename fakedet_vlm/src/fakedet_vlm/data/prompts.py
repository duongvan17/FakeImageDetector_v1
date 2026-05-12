"""Prompt templates for FakeDet VLM (Qwen2.5 chat format)."""
from __future__ import annotations

IMAGE_PLACEHOLDER = "<image>"

SYSTEM_PROMPT = (
    "You are an expert in image forensics and deepfake detection. "
    "Given an image, decide whether it is real or AI-generated/manipulated, "
    "and explain the visual artifacts that justify your decision."
)

USER_QUESTION = (
    f"{IMAGE_PLACEHOLDER}\n"
    "Is this image authentic or a deepfake? "
    "If it is a deepfake, list the specific artifacts you observe."
)


def build_chat_prompt(
    user_question: str = USER_QUESTION,
    system_prompt: str = SYSTEM_PROMPT,
    assistant_response: str | None = None,
) -> tuple[str, str]:
    """Return ``(prompt_prefix, response_suffix)``.

    ``prompt_prefix`` is everything up to (and including) ``<|im_start|>assistant\\n``.
    ``response_suffix`` is the assistant text plus ``<|im_end|>\\n``. Splitting
    the prompt this way lets us tokenize the two halves separately and build a
    label tensor with -100 over the prompt — no fragile substring matching.
    """
    prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    suffix = ""
    if assistant_response is not None:
        suffix = f"{assistant_response}<|im_end|>\n"
    return prefix, suffix


def format_assistant_response(label: int, clue: str | None) -> str:
    """Build the target assistant text from FakeClue ``(label, clue)``."""
    if label == 0:
        return "This image is authentic. No deepfake manipulation detected."
    clue = (clue or "").strip()
    if not clue:
        clue = (
            "Detected manipulation artifacts consistent with GAN-based "
            "synthesis or face-swapping techniques."
        )
    return f"This image is a deepfake. Evidence: {clue}"
