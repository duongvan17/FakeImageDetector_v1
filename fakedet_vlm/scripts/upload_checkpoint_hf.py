"""Upload the local ``clip_model/best_model.pth`` to a private HuggingFace
Hub repository so the cloud GPU can pull it down with one ``hf download``.

Run on your LOCAL machine (where the 1 GB checkpoint lives):

    # 1. Get a Hub write token at https://huggingface.co/settings/tokens
    # 2. Login
    hf auth login                                # paste write-scope token

    # 3. Upload (creates the repo on first run)
    python scripts/upload_checkpoint_hf.py \
        --repo-id <your-username>/fakedet-vit-b16-fakeclue \
        --ckpt ../clip_model/best_model.pth

The repo is created **private** by default. Pass ``--public`` if you want it
public.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="HF repo, e.g. 'duongvan17/fakedet-vit-b16-fakeclue'")
    ap.add_argument("--ckpt", default="../clip_model/best_model.pth",
                    help="Path to the .pth checkpoint")
    ap.add_argument("--public", action="store_true",
                    help="Create a public repo (default: private)")
    ap.add_argument("--commit-message", default="Upload ViT-B/16 FakeClue checkpoint")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    api = HfApi()

    print(f"[1/3] Creating repo {args.repo_id} (private={not args.public}) ...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    print(f"[2/3] Uploading {ckpt_path.name} ({ckpt_path.stat().st_size / 1e6:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="best_model.pth",
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    print(f"[3/3] Done.\n  https://huggingface.co/{args.repo_id}")
    print("\nOn the cloud GPU, pull it back with:")
    print(f"  hf download {args.repo_id} best_model.pth \\")
    print(f"      --local-dir ../clip_model")


if __name__ == "__main__":
    main()
