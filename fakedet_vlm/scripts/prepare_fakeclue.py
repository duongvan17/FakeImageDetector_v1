"""Download FakeClue from HuggingFace and write LLaVA-style JSON splits.

Output layout:
  data/
    train.json
    val.json
    images/
      train_000000.jpg
      ...

Each JSON record:
  {"image": "train_000123.jpg", "label": 1, "clue": "...",
   "conversations": [{"from":"human","value":"<image>\\n..."},
                     {"from":"gpt","value":"This image is a deepfake. Evidence: ..."}]}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fakedet_vlm.data.prompts import IMAGE_PLACEHOLDER, USER_QUESTION, format_assistant_response  # noqa: E402


def _print_split_stats(name: str, records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        return
    real = sum(1 for r in records if r["label"] == 0)
    fake = n - real
    by_cat = Counter(r.get("category", "unknown") for r in records)
    print(f"\n[{name}] total={n}  real={real} ({real/n:.1%})  fake={fake} ({fake/n:.1%})")
    if by_cat and set(by_cat) != {"unknown"}:
        print("  per-category:")
        for cat, count in by_cat.most_common():
            print(f"    {cat:<16s} {count:>6d} ({count/n:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = all; useful for fast smoke runs")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    out_root = Path(args.out)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Downloading lingcco/FakeClue ...")
    ds = load_dataset("lingcco/FakeClue", trust_remote_code=True)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    indices = list(range(len(split)))
    random.Random(args.seed).shuffle(indices)
    if args.max_samples:
        indices = indices[: args.max_samples]

    n_train = int(len(indices) * args.train_ratio)
    train_idx = set(indices[:n_train])

    train_records, val_records = [], []
    print(f"[2/3] Writing {len(indices)} samples ...")
    for i in tqdm(indices):
        item = split[int(i)]
        img: Image.Image = item["image"].convert("RGB")
        label = int(item.get("label", 0))
        clue = item.get("clue", "") or ""
        category = item.get("category", item.get("class", "unknown"))

        is_train = i in train_idx
        prefix = "train" if is_train else "val"
        fname = f"{prefix}_{i:06d}.jpg"
        img.save(images_dir / fname, quality=95)

        rec = {
            "image": fname,
            "label": label,
            "category": category,
            "clue": clue,
            "conversations": [
                {"from": "human", "value": USER_QUESTION},
                {"from": "gpt", "value": format_assistant_response(label, clue)},
            ],
        }
        (train_records if is_train else val_records).append(rec)

    print(f"[3/3] Train={len(train_records)} Val={len(val_records)}")
    _print_split_stats("train", train_records)
    _print_split_stats("val", val_records)
    with open(out_root / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False)
    with open(out_root / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False)
    print(f"Saved JSON to {out_root}/. Images at {images_dir}/")
    _ = IMAGE_PLACEHOLDER  # silence import-only lint


if __name__ == "__main__":
    main()
