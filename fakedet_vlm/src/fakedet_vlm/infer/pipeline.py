"""Single-image inference for a trained FakeDet VLM."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from fakedet_vlm.data.prompts import IMAGE_PLACEHOLDER, build_chat_prompt
from fakedet_vlm.models import FakeDetVLM
from fakedet_vlm.models.vit_loader import CLIP_MEAN, CLIP_STD


class DeepfakeDetector:
    def __init__(
        self,
        llm_name: str,
        vision_checkpoint: str | Path,
        adapter_dir: str | Path | None = None,
        projector_path: str | Path | None = None,
        num_visual_tokens: int = 196,
        image_size: int = 224,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.num_visual_tokens = num_visual_tokens
        self.image_size = image_size

        self.model = FakeDetVLM(
            llm_name=llm_name,
            vision_checkpoint=vision_checkpoint,
            num_visual_tokens=num_visual_tokens,
            image_size=image_size,
            load_in_4bit=True,
        )

        if adapter_dir and Path(adapter_dir).exists():
            from peft import PeftModel
            self.model.llm = PeftModel.from_pretrained(self.model.llm, str(adapter_dir))

        if projector_path and Path(projector_path).exists():
            sd = torch.load(str(projector_path), map_location="cpu")
            self.model.projector.load_state_dict(sd)

        self.model.eval()

        # Move vision tower + projector to the same device as the LLM. The
        # 4-bit LLM is placed via device_map; the rest defaults to CPU.
        if device == "cuda" and torch.cuda.is_available():
            self.model.vision_tower = self.model.vision_tower.to(device)
            self.model.projector = self.model.projector.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
        ])

    @torch.no_grad()
    def detect(self, image_path: str | Path, max_new_tokens: int = 128) -> dict:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)

        prefix, _ = build_chat_prompt(assistant_response=None)
        prefix = prefix.replace(
            IMAGE_PLACEHOLDER,
            IMAGE_PLACEHOLDER * self.num_visual_tokens,
            1,
        )
        ids = self.model.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        input_ids = ids["input_ids"].to(self.device)
        attention_mask = ids["attention_mask"].to(self.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        # ``inputs_embeds`` generation returns only the newly generated ids.
        text = self.model.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        is_fake = ("deepfake" in text.lower()) or ("fake" in text.lower() and "authentic" not in text.lower())
        return {
            "image_path": str(image_path),
            "classification": "Fake" if is_fake else "Real",
            "response": text,
        }
