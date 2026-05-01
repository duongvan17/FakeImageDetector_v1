"""Dataset smoke test using a stubbed tokenizer (no LLM weights needed)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from fakedet_vlm.data import FakeClueDataset, IMAGE_PLACEHOLDER  # noqa: E402


def _make_tokenizer():
    # Use the actual Qwen2.5 tokenizer if cached locally; skip otherwise.
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen2.5 tokenizer not cached locally")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if IMAGE_PLACEHOLDER not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    return tok


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()


def test_dataset_emits_correct_image_token_count(tmp_path):
    tok = _make_tokenizer()
    img_path = tmp_path / "x.png"
    img_path.write_bytes(_png_bytes())

    rec = {"image": "x.png", "label": 1, "clue": "Unnatural skin texture"}
    ds = FakeClueDataset(
        records=[rec],
        images_dir=tmp_path,
        tokenizer=tok,
        num_visual_tokens=196,
    )
    item = ds[0]

    image_token_id = tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    assert (item["input_ids"] == image_token_id).sum().item() == 196
    assert item["pixel_values"].shape == (3, 224, 224)
    # Labels are -100 over the prompt and real ids over the response.
    assert (item["labels"] == -100).sum().item() > 0
    assert (item["labels"] != -100).sum().item() > 0
    # Image tokens must fall entirely within the masked-out prompt region.
    image_positions = (item["input_ids"] == image_token_id).nonzero().squeeze(-1)
    assert (item["labels"][image_positions] == -100).all()
