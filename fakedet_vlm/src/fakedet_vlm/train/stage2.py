"""Stage 2: SFT with LoRA on the LLM + projector continuation.

Frozen: ViT.
Trainable: projector (continuing from stage1) + LLM via LoRA.
Two parameter groups with separate learning rates (projector_lr vs lora_lr).

Usage:
  python -m fakedet_vlm.train.stage2 \
    --base configs/base.yaml --stage configs/stage2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback, TrainingArguments

from fakedet_vlm.data import DeepfakeAugment, FakeClueDataset, VLMCollator
from fakedet_vlm.models import FakeDetVLM
from fakedet_vlm.utils import cleanup_memory

from .common import VLMTrainer, load_config, torch_dtype_from_str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="configs/base.yaml")
    ap.add_argument("--stage", type=str, default="configs/stage2.yaml")
    args = ap.parse_args()

    cfg = load_config(args.stage, args.base)
    m_cfg, d_cfg, t_cfg, s_cfg = cfg["model"], cfg["data"], cfg["train"], cfg["system"]
    lora_cfg = cfg["lora"]

    torch.manual_seed(s_cfg["seed"])
    output_dir = Path(s_cfg["output_root"]) / cfg["output_subdir"]

    # 1) Build base VLM
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

    # 2) Load stage 1 projector if present
    stage1_proj = cfg.get("stage1_projector")
    if stage1_proj and Path(stage1_proj).exists():
        sd = torch.load(stage1_proj, map_location="cpu")
        missing, unexpected = model.projector.load_state_dict(sd, strict=False)
        print(f"[stage2] loaded projector from {stage1_proj} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print(f"[stage2] no stage1 projector found at {stage1_proj!r} — using fresh init")
    model.unfreeze_projector()

    # 3) Apply LoRA to the (4-bit) LLM
    model.llm = prepare_model_for_kbit_training(
        model.llm,
        use_gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
    )
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model.llm = get_peft_model(model.llm, lora)
    model.llm.print_trainable_parameters()

    # 4) Data
    train_ds = FakeClueDataset.from_json(
        d_cfg["train_json"], d_cfg["images_dir"],
        tokenizer=model.tokenizer,
        num_visual_tokens=m_cfg["num_visual_tokens"],
        image_size=m_cfg["image_size"],
        max_length=d_cfg["max_length"],
        augment=DeepfakeAugment(seed=s_cfg["seed"]),
    )
    val_ds = FakeClueDataset.from_json(
        d_cfg["val_json"], d_cfg["images_dir"],
        tokenizer=model.tokenizer,
        num_visual_tokens=m_cfg["num_visual_tokens"],
        image_size=m_cfg["image_size"],
        max_length=d_cfg["max_length"],
        augment=None,
    )
    collator = VLMCollator(pad_token_id=model.tokenizer.pad_token_id)

    # 5) Training args. We pass a placeholder LR; the optimizer is built by
    # ``VLMTrainer`` with two param groups (projector vs LoRA) below.
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["lora_lr"],  # used as base for scheduler
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
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    class TwoLRTrainer(VLMTrainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            from transformers.trainer_pt_utils import get_parameter_names
            decay_params = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_params = [n for n in decay_params if "bias" not in n]

            proj_params, lora_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("projector."):
                    proj_params.append(p)
                else:
                    lora_params.append(p)

            optim_cls, optim_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(
                [
                    {"params": proj_params, "lr": t_cfg["projector_lr"]},
                    {"params": lora_params, "lr": t_cfg["lora_lr"]},
                ],
                **{k: v for k, v in optim_kwargs.items() if k != "lr"},
            )
            return self.optimizer

    trainer = TwoLRTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t_cfg["early_stopping_patience"])],
    )

    trainer.train()

    # Final artefacts
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(final_dir))
    torch.save(model.projector.state_dict(), final_dir / "projector.pt")
    model.tokenizer.save_pretrained(str(final_dir))
    print(f"[stage2] saved LoRA + projector + tokenizer → {final_dir}")
    cleanup_memory()


if __name__ == "__main__":
    main()
