"""Evaluate a trained FakeDet VLM on a JSON validation set.

For each sample:
  1. Loads the image and pre-formatted prompt.
  2. Generates a response with greedy decoding.
  3. Parses ``Real / Fake`` from the response.
  4. Records prediction + log-prob proxy for AUC.

Outputs:
  - ``runs/<subdir>/eval_predictions.jsonl`` — one line per sample.
  - ``runs/<subdir>/eval_metrics.json`` — overall + per-category metrics.

Usage:
  python scripts/eval.py \
      --val-json data/val.json --images-dir data/images \
      --adapter-dir runs/stage2_sft/final \
      --projector runs/stage2_sft/final/projector.pt \
      --out runs/stage2_sft/eval
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fakedet_vlm.data.prompts import IMAGE_PLACEHOLDER, build_chat_prompt  # noqa: E402
from fakedet_vlm.models.vit_loader import CLIP_MEAN, CLIP_STD  # noqa: E402


def _build_processor(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])


def _parse_classification(text: str) -> tuple[int, float]:
    """Map free-text response → ``(prediction, fake_score)``.

    ``fake_score`` is a coarse proxy in [0, 1] derived from the language used,
    sufficient for AUC ranking in absence of a calibrated probability.
    """
    t = text.lower()
    fake_words = ("deepfake", "fake", "manipulated", "synthetic", "ai-generated")
    real_words = ("authentic", "real", "genuine", "no manipulation", "no deepfake")
    has_fake = any(w in t for w in fake_words)
    has_real = any(w in t for w in real_words)

    if has_fake and not has_real:
        return 1, 0.9
    if has_real and not has_fake:
        return 0, 0.1
    if has_fake and has_real:
        # Mention of both — trust whichever appears first.
        first_fake = min((t.find(w) for w in fake_words if w in t), default=10**9)
        first_real = min((t.find(w) for w in real_words if w in t), default=10**9)
        return (1, 0.6) if first_fake < first_real else (0, 0.4)
    return 0, 0.5  # ambiguous → conservative real


def _metrics(preds: list[int], targets: list[int], scores: list[float]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    out = {
        "n": len(preds),
        "accuracy": float(accuracy_score(targets, preds)),
    }
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    out.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
    if len(set(targets)) == 2:
        try:
            out["auc"] = float(roc_auc_score(targets, scores))
        except ValueError:
            out["auc"] = None
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--llm-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--vision-checkpoint", default="../clip_model/best_model.pth")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--projector", default=None)
    ap.add_argument("--num-visual-tokens", type=int, default=196)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defer heavy imports so ``--help`` works without transformers/peft.
    from fakedet_vlm.models import FakeDetVLM

    print("[1/4] Loading model ...")
    model = FakeDetVLM(
        llm_name=args.llm_name,
        vision_checkpoint=args.vision_checkpoint,
        num_visual_tokens=args.num_visual_tokens,
        image_size=args.image_size,
        load_in_4bit=(args.device == "cuda"),
    )
    if args.adapter_dir and Path(args.adapter_dir).exists():
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, args.adapter_dir)
    if args.projector and Path(args.projector).exists():
        sd = torch.load(args.projector, map_location="cpu")
        model.projector.load_state_dict(sd)
    model.eval()

    processor = _build_processor(args.image_size)
    images_dir = Path(args.images_dir)

    print("[2/4] Loading val set ...")
    with open(args.val_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    print(f"[3/4] Predicting on {len(records)} samples ...")
    predictions = []
    cat_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {"preds": [], "targets": [], "scores": []}
    )
    overall = {"preds": [], "targets": [], "scores": []}

    prefix_template, _ = build_chat_prompt(assistant_response=None)
    expanded_prefix = prefix_template.replace(
        IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER * args.num_visual_tokens, 1
    )
    enc = model.tokenizer(expanded_prefix, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc["attention_mask"].to(args.device)

    for rec in tqdm(records):
        img = Image.open(images_dir / rec["image"]).convert("RGB")
        pixel_values = processor(img).unsqueeze(0).to(args.device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        text = model.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        pred, score = _parse_classification(text)
        target = int(rec.get("label", 0))
        category = rec.get("category", "unknown")

        predictions.append({
            "image": rec["image"],
            "category": category,
            "target": target,
            "pred": pred,
            "score": score,
            "response": text,
        })
        overall["preds"].append(pred); overall["targets"].append(target); overall["scores"].append(score)
        cat_acc[category]["preds"].append(pred)
        cat_acc[category]["targets"].append(target)
        cat_acc[category]["scores"].append(score)

    print("[4/4] Computing metrics ...")
    metrics = {"overall": _metrics(overall["preds"], overall["targets"], overall["scores"])}
    metrics["per_category"] = {
        cat: _metrics(d["preds"], d["targets"], d["scores"])
        for cat, d in sorted(cat_acc.items())
    }

    with open(out_dir / "eval_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    o = metrics["overall"]
    print(f"\n[Overall] n={o['n']} acc={o['accuracy']:.3f} "
          f"P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} "
          f"AUC={o.get('auc', float('nan')):.3f}")
    print("[Per-category]")
    for cat, m in metrics["per_category"].items():
        print(f"  {cat:<16s} n={m['n']:>5d}  acc={m['accuracy']:.3f}  F1={m['f1']:.3f}")

    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
