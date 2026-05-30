"""Train ViT-B/16 (CLIP pretrained) on FakeClue with anti-overfit recipe v2.

Six concrete fixes compared to v1 (best_model.pth):
  1. Differential LR        — head 1e-4, backbone 5e-6 (instead of 1 group @ 2e-5)
  2. JPEG augmentation      — random quality 40-95 → break the "compression
                              quality" shortcut between FakeClue real/fake
  3. RandAugment + erasing  — stronger regularisation, force robust features
  4. Label-smoothed BCE     — calibrated outputs (sigmoid not pinned to {0,1})
  5. Class-balanced BCE     — pos_weight = n_real / n_fake (FakeClue is fake-
                              heavy → v1 had recall=1.0 / precision=0.97)
  6. Cross-domain val every epoch  — save best by MEAN(in-domain, OOD), not
                                     by in-domain only

Backbone: timm `vit_base_patch16_clip_224` (CLIP OpenAI weights). Head is a
fresh `Linear(768, 1)` — NOT continued from best_model.pth.

Usage on RTX 4090 (Vast.ai):
    python train_vision_v2.py \\
        --train-json data/train.json \\
        --val-json   data/val.json \\
        --image-root data/images \\
        --celebdf-json data/celebdf.json --celebdf-root data/celebdf_imgs \\
        --output-dir runs/vision_v2 \\
        --epochs 9 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# CLIP-OpenAI normalization stats (matches CLIP-ViT-B/16 pretraining).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------- dataset
_IMAGE_KEYS = ("image_path", "image", "img_path", "img", "path", "file", "filename")
_LABEL_KEYS = ("label", "y", "target", "is_fake", "fake")


def _first_key(rec: dict, names: tuple[str, ...], where: str):
    for k in names:
        if k in rec:
            return rec[k]
    raise KeyError(f"{where}: no field in {names!r}; record keys = {list(rec.keys())}")


class FakeClueDataset(Dataset):
    """Reads records produced by `scripts/prepare_fakeclue.py`. Tolerates
    several common field names (image_path/image/path, label/target/is_fake)
    because the prep script's schema varies across FakeClue versions.
    Label convention: **1 = fake**."""

    def __init__(self, json_path: str, image_root: str, transform):
        with open(json_path, encoding="utf-8") as f:
            self.records = json.load(f)
        self.image_root = Path(image_root)
        self.transform = transform
        # Probe schema on the first record (fail fast with a clear message).
        if self.records:
            _first_key(self.records[0], _IMAGE_KEYS, "image field")
            _first_key(self.records[0], _LABEL_KEYS, "label field")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img_path = _first_key(r, _IMAGE_KEYS, "image field")
        label = _first_key(r, _LABEL_KEYS, "label field")
        img = Image.open(self.image_root / img_path).convert("RGB")
        return self.transform(img), float(label), r.get("category", "")


# -------------------------------------------------------------- transforms
def train_transform():
    # v2.JPEG requires torchvision ≥ 0.17. PIL→Tensor first so v2 ops compose.
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.JPEG(quality=(40, 95)),        # ★ break the compression-quality shortcut
        v2.GaussianBlur(3, sigma=(0.1, 0.5)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CLIP_MEAN, CLIP_STD),
        v2.RandomErasing(p=0.25),
    ])


def eval_transform():
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# -------------------------------------------------------------- loss / eval
def smooth_bce(logits: torch.Tensor, targets: torch.Tensor,
               pos_weight: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """BCE-with-logits + label smoothing. Smoothing moves the targets from
    {0,1} to {s/2, 1-s/2} → sigmoid output stops collapsing to extremes,
    making the model calibrated rather than 99.x%-confident on everything."""
    soft = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pos_weight)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        all_logits.append(model(x).squeeze(-1))
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    acc = ((probs >= 0.5).float() == labels).float().mean().item()
    auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
    return acc, auc


# ----------------------------------------------------------------- train
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- data ----
    train_ds = FakeClueDataset(args.train_json, args.image_root, train_transform())
    val_ds = FakeClueDataset(args.val_json, args.image_root, eval_transform())
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optional cross-domain val sets (Celeb-DF / DFDC). Same JSON schema.
    cross_dls = {}
    for name, j, r in (
        ("celeb_df", args.celebdf_json, args.celebdf_root),
        ("dfdc",     args.dfdc_json,    args.dfdc_root),
    ):
        if j and Path(j).exists():
            cross_dls[name] = DataLoader(
                FakeClueDataset(j, r, eval_transform()),
                batch_size=args.batch_size * 2, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

    # ---- class-balanced BCE ----
    n_fake = sum(1 for rec in train_ds.records
                 if float(_first_key(rec, _LABEL_KEYS, "label field")) == 1.0)
    n_real = len(train_ds.records) - n_fake
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    print(f"train: {len(train_ds)} samples  fake={n_fake}  real={n_real}  "
          f"pos_weight={pos_weight.item():.3f}")
    for name, dl in cross_dls.items():
        print(f"  cross-domain '{name}': {len(dl.dataset)} samples")

    # ---- model: CLIP pretrained backbone + fresh head ----
    model = timm.create_model(
        "vit_base_patch16_clip_224", pretrained=True, num_classes=1
    ).to(device)

    # ---- differential LR (head fast, backbone slow) ----
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optim = AdamW(
        [
            {"params": head_params,     "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ---- scheduler: 1-epoch linear warmup + cosine to 0 ----
    steps_per_epoch = len(train_dl)
    warmup_steps = steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optim, T_max=total_steps - warmup_steps, eta_min=0.0),
        ],
        milestones=[warmup_steps],
    )

    # ---- AMP for RTX 4090 ----
    use_amp = device == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for step, (x, y, _) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x).squeeze(-1)
                    loss = smooth_bce(logits, y, pos_weight, args.label_smoothing)
                loss.backward()
            else:
                logits = model(x).squeeze(-1)
                loss = smooth_bce(logits, y, pos_weight, args.label_smoothing)
                loss.backward()
            optim.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            if step % 50 == 0:
                lr_h = optim.param_groups[0]["lr"]
                lr_b = optim.param_groups[1]["lr"]
                print(f"[E{epoch}][{step:>4}/{steps_per_epoch}] "
                      f"loss={loss.item():.4f} lr_head={lr_h:.2e} lr_bb={lr_b:.2e}")

        # ---- validation ----
        val_acc, val_auc = evaluate(model, val_dl, device)
        cross_metrics = {name: evaluate(model, dl, device) for name, dl in cross_dls.items()}

        # Score = mean acc across in-domain + cross-domain (save by THIS, not in-domain alone)
        all_accs = [val_acc] + [m[0] for m in cross_metrics.values()]
        score = sum(all_accs) / len(all_accs)

        log = [f"E{epoch}  loss={epoch_loss/n_batches:.4f}",
               f"val={val_acc:.4f}/{val_auc:.4f}"]
        for name, (ca, cu) in cross_metrics.items():
            log.append(f"{name}={ca:.4f}/{cu:.4f}")
        log.append(f"score={score:.4f}")
        print(" | ".join(log))

        # ---- save best (by combined score) ----
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "acc": val_acc,
                "auc": val_auc,
                "cross_domain": {n: {"acc": m[0], "auc": m[1]} for n, m in cross_metrics.items()},
                "combined_score": score,
                "category_names": sorted({
                    r.get("category", "") for r in train_ds.records if r.get("category")
                }),
                "config": vars(args),
            }, out_dir / "best_model_v2.pth")
            print(f"  ✓ saved best (score={best_score:.4f})")

        # ---- always keep last ----
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "acc": val_acc, "auc": val_auc,
        }, out_dir / "last_model_v2.pth")

    print(f"\nDONE. best combined score = {best_score:.4f}")
    print(f"Checkpoint: {out_dir/'best_model_v2.pth'}")


# ------------------------------------------------------------------- cli
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-json",   default="data/train.json")
    p.add_argument("--val-json",     default="data/val.json")
    p.add_argument("--image-root",   default="data/images")
    p.add_argument("--celebdf-json", default="", help="optional cross-domain val")
    p.add_argument("--celebdf-root", default="")
    p.add_argument("--dfdc-json",    default="")
    p.add_argument("--dfdc-root",    default="")
    p.add_argument("--output-dir",   default="runs/vision_v2")
    p.add_argument("--epochs",          type=int,   default=9)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--lr-head",         type=float, default=1e-4)
    p.add_argument("--lr-backbone",     type=float, default=5e-6)
    p.add_argument("--weight-decay",    type=float, default=0.02)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
