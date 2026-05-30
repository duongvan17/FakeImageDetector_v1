"""Quick LLM comparison for the FakeDet VLM — Stage 1 + Stage 2 on a small
subset, multiple LLM backbones, to justify the Qwen2.5-1.5B choice.

Per LLM:
  1. Build FakeDetVLM with that LLM (frozen ViT + projector + 4-bit LLM)
  2. Stage 1 (projector warmup, 1 epoch on 5k subset, LLM frozen)
  3. Stage 2 (projector + LoRA, 3 epoch on 5k subset)
  4. Record: final eval_loss, wall time, peak VRAM, params

Usage on the pod:
    python scripts/compare_llms.py \\
        --llms Qwen/Qwen2.5-0.5B-Instruct \\
               Qwen/Qwen2.5-1.5B-Instruct \\
               TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --max-train-samples 5000 \\
        --output-root runs/llm_cmp
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def make_subset(src_json: str, dst_json: str, k: int, seed: int = 42) -> int:
    """Random-sample ``k`` records from ``src_json`` into ``dst_json``."""
    records = json.load(open(src_json, encoding="utf-8"))
    rng = random.Random(seed)
    sub = rng.sample(records, min(k, len(records)))
    Path(dst_json).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    return len(sub)


def patch_yaml(src: Path, dst: Path, patches: dict) -> None:
    """Load ``src``, deep-merge ``patches``, write to ``dst``."""
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}

    def merge(d, p):
        for k, v in p.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v

    merge(cfg, patches)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run_one(llm_name: str, args: argparse.Namespace) -> dict:
    slug = llm_name.replace("/", "_").replace("-", "_")
    out_root = Path(args.output_root) / slug
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_src = Path(args.base_config)
    s1_src = Path(args.stage1_config)
    s2_src = Path(args.stage2_config)
    base_dst = cfg_dir / "base.yaml"
    s1_dst = cfg_dir / "stage1.yaml"
    s2_dst = cfg_dir / "stage2.yaml"

    train_sub = f"data/train_{args.max_train_samples}.json"
    if not Path(train_sub).exists():
        n = make_subset(args.train_json, train_sub, args.max_train_samples, args.seed)
        _log(f"made subset {train_sub} ({n} samples)")

    # base.yaml — swap LLM, point at subset + vision ckpt override.
    model_patch = {"llm_name": llm_name}
    if args.vision_checkpoint:
        model_patch["vision_checkpoint"] = args.vision_checkpoint
    patch_yaml(base_src, base_dst, {
        "model": model_patch,
        "data":  {"train_json": train_sub},
    })

    # stage1.yaml — short warmup, save into per-LLM dir.
    patch_yaml(s1_src, s1_dst, {
        "output_subdir": f"llm_cmp/{slug}/stage1",
        "train": {
            "num_train_epochs": args.stage1_epochs,
            "logging_steps": 10,
            "save_steps": 99999,   # skip mid-epoch save (subset is tiny)
            "eval_steps": 99999,
        },
    })

    # stage2.yaml — short LoRA SFT, point at stage 1 projector.
    s1_proj = f"runs/llm_cmp/{slug}/stage1/projector.pt"
    patch_yaml(s2_src, s2_dst, {
        "output_subdir": f"llm_cmp/{slug}/stage2",
        "stage1_projector": s1_proj,
        "train": {
            "num_train_epochs": args.stage2_epochs,
            "logging_steps": 10,
            "save_steps": 99999,
            "eval_steps": 99999,
            "early_stopping_patience": 99,
        },
    })

    metrics: dict = {"llm": llm_name, "slug": slug}

    for name, dst in [("stage1", s1_dst), ("stage2", s2_dst)]:
        _log(f"=== {llm_name} :: {name} ===")
        t0 = time.time()
        cmd = ["python", "-m", f"fakedet_vlm.train.{name}",
               "--base", str(base_dst), "--stage", str(dst)]
        ret = subprocess.run(cmd, check=False)
        dur = time.time() - t0
        metrics[f"{name}_seconds"] = round(dur, 1)
        metrics[f"{name}_exit"] = ret.returncode
        _log(f"{llm_name} {name} done in {dur:.0f}s (exit {ret.returncode})")
        if ret.returncode != 0:
            _log(f"  ABORT {llm_name} — {name} failed")
            break

    # Persist metrics row.
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llms", nargs="+", required=True,
                   help="HF model ids, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-json",   default="data/train.json")
    p.add_argument("--vision-checkpoint", default="",
                   help="override model.vision_checkpoint in base config "
                        "(empty = leave as configured)")
    p.add_argument("--output-root",  default="runs/llm_cmp")
    p.add_argument("--max-train-samples", type=int, default=5000)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument("--base-config",   default="configs/base.yaml")
    p.add_argument("--stage1-config", default="configs/stage1.yaml")
    p.add_argument("--stage2-config", default="configs/stage2.yaml")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = []
    for llm in args.llms:
        summary.append(run_one(llm, args))

    summary_path = Path(args.output_root) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _log(f"SUMMARY → {summary_path}")
    for row in summary:
        _log(f"  {row['llm']:<45}  "
             f"s1={row.get('stage1_seconds','?')}s  "
             f"s2={row.get('stage2_seconds','?')}s  "
             f"exit=({row.get('stage1_exit','?')},{row.get('stage2_exit','?')})")


if __name__ == "__main__":
    main()
