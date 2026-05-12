"""Download FakeClue from HuggingFace Hub + extract + convert to local format.

The ``lingcco/FakeClue`` repo on HF Hub has a non-standard layout that breaks
``datasets.load_dataset(...)``:

    data_json/train.json   (~104K records, list of dicts)
    data_json/test.json
    train.zip              (image archive, ~25 GB)
    test.zip

This script downloads the JSON + zip directly via ``hf_hub_download``,
extracts images, and writes our local ``train.json`` / ``val.json`` referring
to the extracted image paths.

Record schema in source:
  - image: relative path like "ff++/fake/Deepfakes/c23/frames/071_054/160.png"
  - label: 0 = fake, 1 = real         (NOTE: opposite of the common convention)
  - cate: category (e.g. "deepfake", "human", "scenery", ...)
  - width, height
  - conversations: already in LLaVA format with <image> placeholder

We:
  - Pass conversations through verbatim — our ``FakeClueDataset`` uses them.
  - Re-label to our convention (1 = fake, 0 = real) so the eval pipeline
    matches what the model is trained to say.
  - Carve a val split from FakeClue's train (default 90/10), or use
    FakeClue's test split as val via ``--use-test-as-val``.

Usage:
  python scripts/prepare_fakeclue.py --out data --train-ratio 0.9
  python scripts/prepare_fakeclue.py --out data --max-samples 200   # smoke
  python scripts/prepare_fakeclue.py --out data --use-test-as-val
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    raise SystemExit("pip install huggingface_hub") from e

REPO_ID = "lingcco/FakeClue"


def _print_split_stats(name: str, records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        return
    fake = sum(1 for r in records if r["label"] == 1)
    real = n - fake
    by_cat = Counter(r.get("category", "unknown") for r in records)
    print(f"\n[{name}] total={n}  real={real} ({real/n:.1%})  fake={fake} ({fake/n:.1%})")
    if by_cat:
        print("  per-category:")
        for cat, count in by_cat.most_common():
            print(f"    {cat:<16s} {count:>6d} ({count/n:.1%})")


def _normalize_records(src_records: list[dict], image_subset: set[str] | None) -> list[dict]:
    """Map source schema → our schema, flipping label semantics."""
    out = []
    for rec in src_records:
        image_path = rec.get("image")
        if not image_path:
            continue
        if image_subset is not None and image_path not in image_subset:
            continue
        src_label = int(rec.get("label", 0))
        # Source: 0=fake, 1=real.  Our pipeline: 1=fake, 0=real.
        our_label = 1 - src_label
        out.append({
            "image": image_path,
            "label": our_label,
            "category": rec.get("cate", "unknown"),
            "conversations": rec["conversations"],
        })
    return out


def _extract_zip(zip_path: Path, target_dir: Path) -> set[str]:
    """Extract all files; return the set of relative file paths extracted."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in tqdm(zf.infolist(), desc=f"extract {zip_path.name}"):
            if info.is_dir():
                continue
            zf.extract(info, target_dir)
            extracted.add(info.filename)
    return extracted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--train-ratio", type=float, default=0.9,
                    help="If --use-test-as-val is OFF, slice train.json into "
                         "train/val by this ratio.")
    ap.add_argument("--use-test-as-val", action="store_true",
                    help="Use FakeClue's test split as our val (instead of "
                         "carving val from train).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = all. Useful for smoke testing.")
    ap.add_argument("--keep-zip-cache", action="store_true",
                    help="Do not delete the downloaded zip files from the HF "
                         "cache after extraction. Saves disk for re-runs.")
    args = ap.parse_args()

    out_root = Path(args.out)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading annotation JSON from HF Hub")
    train_json_path = Path(hf_hub_download(REPO_ID, "data_json/train.json", repo_type="dataset"))
    print(f"      {train_json_path}")
    if args.use_test_as_val:
        test_json_path = Path(hf_hub_download(REPO_ID, "data_json/test.json", repo_type="dataset"))
        print(f"      {test_json_path}")
    else:
        test_json_path = None

    print("\n[2/4] Downloading zip archives from HF Hub (large)")
    train_zip_path = Path(hf_hub_download(REPO_ID, "train.zip", repo_type="dataset"))
    print(f"      {train_zip_path}  ({train_zip_path.stat().st_size / 1e9:.1f} GB)")
    if args.use_test_as_val:
        test_zip_path = Path(hf_hub_download(REPO_ID, "test.zip", repo_type="dataset"))
        print(f"      {test_zip_path}  ({test_zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        test_zip_path = None

    print("\n[3/4] Extracting zips into data/images/")
    extracted_train = _extract_zip(train_zip_path, images_dir)
    if test_zip_path:
        extracted_test = _extract_zip(test_zip_path, images_dir)
    else:
        extracted_test = set()

    if not args.keep_zip_cache:
        print("\n  Deleting cached zips to free disk ...")
        try:
            train_zip_path.unlink()
            if test_zip_path:
                test_zip_path.unlink()
        except OSError as e:
            print(f"  warning: could not delete cached zips: {e}")

    print("\n[4/4] Building local annotations")
    with open(train_json_path, "r", encoding="utf-8") as f:
        src_train = json.load(f)
    train_records = _normalize_records(src_train, extracted_train)

    if args.use_test_as_val:
        with open(test_json_path, "r", encoding="utf-8") as f:
            src_test = json.load(f)
        val_records = _normalize_records(src_test, extracted_test)
    else:
        rng = random.Random(args.seed)
        rng.shuffle(train_records)
        n_train = int(len(train_records) * args.train_ratio)
        val_records = train_records[n_train:]
        train_records = train_records[:n_train]

    if args.max_samples:
        train_records = train_records[: args.max_samples]
        val_records = val_records[: max(args.max_samples // 10, 50)]

    with open(out_root / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False)
    with open(out_root / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False)

    print(f"\nSaved to {out_root}/")
    print(f"  train.json: {len(train_records)} records")
    print(f"  val.json:   {len(val_records)} records")
    print(f"  images/:    {len(extracted_train) + len(extracted_test)} files")

    _print_split_stats("train", train_records)
    _print_split_stats("val", val_records)


if __name__ == "__main__":
    main()
