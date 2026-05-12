"""Stage 1: projector alignment.

Frozen: ViT, LLM. Trainable: projector MLP only.
Goal: align vision tokens with the LLM's input embedding distribution.

Usage:
  python -m fakedet_vlm.train.stage1 \
    --base configs/base.yaml --stage configs/stage1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import TrainingArguments

from fakedet_vlm.data import FakeClueDataset, VLMCollator
from fakedet_vlm.models import FakeDetVLM
from fakedet_vlm.utils import cleanup_memory

from .common import VLMTrainer, load_config, torch_dtype_from_str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="configs/base.yaml")
    ap.add_argument("--stage", type=str, default="configs/stage1.yaml")
    args = ap.parse_args()

    cfg = load_config(args.stage, args.base)
    m_cfg, d_cfg, t_cfg, s_cfg = cfg["model"], cfg["data"], cfg["train"], cfg["system"]

    torch.manual_seed(s_cfg["seed"])
    output_dir = Path(s_cfg["output_root"]) / cfg["output_subdir"]

    # 1) Model
    model = FakeDetVLM(
        llm_name=m_cfg["llm_name"],
        vision_checkpoint=m_cfg["vision_checkpoint"],
        vision_dim=m_cfg["vision_dim"],
        num_visual_tokens=m_cfg["num_visual_tokens"],
        image_size=m_cfg["image_size"],
        load_in_4bit=m_cfg["load_in_4bit"],
        torch_dtype=torch_dtype_from_str(m_cfg["torch_dtype"]),
    )
    model.freeze_vision()
    model.freeze_llm()
    model.unfreeze_projector()

    # Gradient checkpointing is enabled by HF Trainer when training_args
    # ``gradient_checkpointing=True``. Our VLM forwards the call to the LLM
    # and also turns on ``enable_input_require_grads`` so gradients flow
    # back to the projector through the frozen embedding table.

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage1] trainable params = {n_train/1e6:.2f}M (projector only)")

    # 2) Data
    train_ds = FakeClueDataset.from_json(
        d_cfg["train_json"],
        d_cfg["images_dir"],
        tokenizer=model.tokenizer,
        num_visual_tokens=m_cfg["num_visual_tokens"],
        image_size=m_cfg["image_size"],
        max_length=d_cfg["max_length"],
    )
    val_ds = FakeClueDataset.from_json(
        d_cfg["val_json"],
        d_cfg["images_dir"],
        tokenizer=model.tokenizer,
        num_visual_tokens=m_cfg["num_visual_tokens"],
        image_size=m_cfg["image_size"],
        max_length=d_cfg["max_length"],
    )
    collator = VLMCollator(pad_token_id=model.tokenizer.pad_token_id)

    # 3) TrainingArguments
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        warmup_ratio=t_cfg["warmup_ratio"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        optim=t_cfg["optim"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=t_cfg["report_to"],
        remove_unused_columns=t_cfg["remove_unused_columns"],
        seed=s_cfg["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = VLMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    # Persist projector explicitly under the stage output root.
    proj_path = output_dir / "projector.pt"
    torch.save(model.projector.state_dict(), proj_path)
    print(f"[stage1] saved projector → {proj_path}")
    cleanup_memory()


if __name__ == "__main__":
    main()
