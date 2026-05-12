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


def _detect_top_prefix(names: list[str]) -> str | None:
    """If every file in the zip lives under a single top-level dir like
    ``train/`` or ``images/``, return that dir name (without trailing slash).
    Otherwise return None."""
    tops = set()
    for n in names:
        if "/" in n:
            tops.add(n.split("/", 1)[0])
        else:
            return None  # at least one file at the root → no common prefix
    return tops.pop() if len(tops) == 1 else None


def _extract_zip(zip_path: Path, target_dir: Path, json_image_paths: set[str]) -> set[str]:
    """Extract all files, auto-stripping a common top-level prefix if the
    annotation JSON references paths without it.

    Returns the set of paths *as they appear in the annotation JSON* (so the
    caller can match them directly when normalising records).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        infolist = [i for i in zf.infolist() if not i.is_dir()]
        all_names = [i.filename for i in infolist]

        top_prefix = _detect_top_prefix(all_names)
        strip_prefix = None
        if top_prefix and json_image_paths:
            # If JSON already has the prefix, no need to strip.
            sample = next(iter(json_image_paths))
            if not sample.startswith(top_prefix + "/"):
                strip_prefix = top_prefix
                print(f"  zip top-level dir is {top_prefix!r}, "
                      f"JSON paths don't have it → stripping on extract")

        extracted = set()
        for info in tqdm(infolist, desc=f"extract {zip_path.name}"):
            out_name = info.filename
            if strip_prefix and out_name.startswith(strip_prefix + "/"):
                out_name = out_name[len(strip_prefix) + 1:]
            if not out_name:
                continue
            target = target_dir / out_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.add(out_name)

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

    # Load JSON FIRST so the extractor can compare zip layout vs JSON refs.
    with open(train_json_path, "r", encoding="utf-8") as f:
        src_train = json.load(f)
    train_json_paths = {r["image"] for r in src_train if r.get("image")}
    print(f"      {len(src_train)} train records, {len(train_json_paths)} unique image paths")

    if args.use_test_as_val:
        with open(test_json_path, "r", encoding="utf-8") as f:
            src_test = json.load(f)
        test_json_paths = {r["image"] for r in src_test if r.get("image")}
    else:
        src_test = []
        test_json_paths = set()

    print("\n[3/4] Extracting zips into data/images/")
    extracted_train = _extract_zip(train_zip_path, images_dir, train_json_paths)
    print(f"  extracted {len(extracted_train)} files")
    overlap = len(extracted_train & train_json_paths)
    print(f"  {overlap}/{len(train_json_paths)} JSON paths now resolve to extracted files")
    if overlap == 0:
        # Show a sample mismatch so the user can diagnose quickly.
        print("  ERROR: no overlap between extracted files and JSON references!")
        print(f"  Sample extracted: {sorted(extracted_train)[:3]}")
        print(f"  Sample JSON paths: {sorted(train_json_paths)[:3]}")
        raise SystemExit(1)

    if test_zip_path:
        extracted_test = _extract_zip(test_zip_path, images_dir, test_json_paths)
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
    train_records = _normalize_records(src_train, extracted_train)

    if args.use_test_as_val:
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
