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
import sys
import time
from pathlib import Path


# Print to stderr is line-buffered even when stdout is piped to `tee`.
# Use this for "where am I?" markers so they always appear in real-time.
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _fmt_dur(seconds: float) -> str:
    """Format seconds as a compact h/m/s string."""
    s = int(max(0, seconds))
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    if s >= 60:
        return f"{s//60}m{s%60:02d}s"
    return f"{s}s"


_log("importing torch ...")
import torch  # noqa: E402
_log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")

_log("importing timm + torchvision + sklearn + PIL ...")
import timm  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
_log("all imports done")

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
    _log(f"train() entered, args={vars(args)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"device = {device}")
    if device == "cuda":
        _log(f"GPU: {torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- data ----
    _log(f"loading train json: {args.train_json}")
    train_ds = FakeClueDataset(args.train_json, args.image_root, train_transform())
    _log(f"train dataset: {len(train_ds)} samples, image_root={args.image_root}")
    _log(f"loading val json: {args.val_json}")
    val_ds = FakeClueDataset(args.val_json, args.image_root, eval_transform())
    _log(f"val dataset: {len(val_ds)} samples")
    _log(f"building DataLoaders (num_workers={args.num_workers})")
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    _log(f"DataLoaders ready: train batches={len(train_dl)}, val batches={len(val_dl)}")

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
    _log("counting class distribution ...")
    n_fake = sum(1 for rec in train_ds.records
                 if float(_first_key(rec, _LABEL_KEYS, "label field")) == 1.0)
    n_real = len(train_ds.records) - n_fake
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    _log(f"class: fake={n_fake}  real={n_real}  pos_weight={pos_weight.item():.3f}")
    for name, dl in cross_dls.items():
        _log(f"cross-domain '{name}': {len(dl.dataset)} samples")

    # ---- model: CLIP pretrained backbone + fresh head ----
    _log("calling timm.create_model('vit_base_patch16_clip_224', pretrained=True) ...")
    _log("(this downloads ~570MB on first run — may take 1-3 min)")
    t0 = time.time()
    model = timm.create_model(
        "vit_base_patch16_clip_224", pretrained=True, num_classes=1
    )
    _log(f"model built in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    _log(f"moving model to {device} ...")
    t0 = time.time()
    model = model.to(device)
    _log(f"moved to {device} in {time.time()-t0:.1f}s")

    # ---- differential LR (head fast, backbone slow) ----
    _log("building optimizer + scheduler ...")
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    _log(f"head params: {sum(p.numel() for p in head_params)} | "
         f"backbone params: {sum(p.numel() for p in backbone_params)}")
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

    _log(f"scheduler ready: warmup_steps={warmup_steps}, total_steps={total_steps}")
    t_run_start = time.time()

    # ---- AMP for RTX 4090 ----
    use_amp = device == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    _log(f"starting training loop (epochs={args.epochs}, AMP={use_amp})")

    for epoch in range(1, args.epochs + 1):
        _log(f"=== EPOCH {epoch}/{args.epochs} ===")
        model.train()
        epoch_loss, n_batches = 0.0, 0
        _log("requesting first batch from DataLoader ...")
        t_batch = time.time()
        t_epoch_start = time.time()
        t_window = time.time()       # rolling window timer for step throughput
        step_times: list[float] = []  # last N step durations
        WINDOW = 20                   # rolling average over 20 steps

        for step, (x, y, _) in enumerate(train_dl):
            if step == 0:
                _log(f"first batch ready in {time.time()-t_batch:.1f}s, shape={tuple(x.shape)}")
                _log("moving first batch to GPU + first forward (may take 5-10s) ...")
                t_first = time.time()
            t_step = time.time()
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
            step_dur = time.time() - t_step
            step_times.append(step_dur)
            if len(step_times) > WINDOW:
                step_times.pop(0)
            if step == 0:
                _log(f"first forward+backward+step done in {time.time()-t_first:.2f}s, loss={loss.item():.4f}")
            epoch_loss += loss.item()
            n_batches += 1
            # log frequently in epoch 1 to confirm progress; less verbose after
            log_every = 10 if epoch == 1 and step < 100 else 50
            if step % log_every == 0:
                lr_h = optim.param_groups[0]["lr"]
                lr_b = optim.param_groups[1]["lr"]
                avg_step = sum(step_times) / len(step_times)   # seconds compute
                # Wall time per step (includes data load) — what really matters
                # for ETA. step >=1 only (need elapsed since epoch start).
                wall_per_step = (time.time() - t_epoch_start) / (step + 1)
                imgs_per_s = args.batch_size / wall_per_step
                steps_left_epoch = steps_per_epoch - step - 1
                steps_left_total = steps_left_epoch + (args.epochs - epoch) * steps_per_epoch
                eta_sec = steps_left_total * wall_per_step
                # GPU stats — show CURRENT and PEAK VRAM in the window.
                # PyTorch frees activations after backward, so memory_allocated
                # alone hides the real fwd+bwd peak that determines how big a
                # batch you can run. Reset peak each log window for fresh read.
                if device == "cuda":
                    vram_gb = torch.cuda.memory_allocated(0) / 1e9
                    peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
                    vram_tot = torch.cuda.get_device_properties(0).total_memory / 1e9
                    torch.cuda.reset_peak_memory_stats(0)
                    try:
                        util = torch.cuda.utilization(0)
                        gpu_part = (f"  VRAM cur={vram_gb:.1f}GB peak={peak_gb:.1f}/"
                                    f"{vram_tot:.0f}GB util={util}%")
                    except Exception:  # noqa: BLE001
                        gpu_part = (f"  VRAM cur={vram_gb:.1f}GB peak={peak_gb:.1f}/"
                                    f"{vram_tot:.0f}GB")
                else:
                    gpu_part = ""
                _log(f"E{epoch} step {step:>4}/{steps_per_epoch}  "
                     f"loss={loss.item():.4f}  lr_h={lr_h:.1e} lr_bb={lr_b:.1e}  "
                     f"compute={avg_step*1000:.0f}ms  wall={wall_per_step*1000:.0f}ms "
                     f"({imgs_per_s:.0f} img/s){gpu_part}  ETA={_fmt_dur(eta_sec)}")

        epoch_train_dur = time.time() - t_epoch_start

        # ---- validation ----
        _log(f"E{epoch} train done in {_fmt_dur(epoch_train_dur)}, running val ...")
        t_val = time.time()
        val_acc, val_auc = evaluate(model, val_dl, device)
        cross_metrics = {name: evaluate(model, dl, device) for name, dl in cross_dls.items()}
        val_dur = time.time() - t_val

        # Score = mean acc across in-domain + cross-domain (save by THIS, not in-domain alone)
        all_accs = [val_acc] + [m[0] for m in cross_metrics.values()]
        score = sum(all_accs) / len(all_accs)

        log = [f"E{epoch}/{args.epochs}",
               f"loss={epoch_loss/n_batches:.4f}",
               f"val={val_acc:.4f}/{val_auc:.4f}"]
        for name, (ca, cu) in cross_metrics.items():
            log.append(f"{name}={ca:.4f}/{cu:.4f}")
        log.append(f"score={score:.4f}")
        log.append(f"train={_fmt_dur(epoch_train_dur)}")
        log.append(f"val={_fmt_dur(val_dur)}")
        _log("  ".join(log))

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

    _log(f"DONE. total wall time = {_fmt_dur(time.time()-t_run_start)}, "
         f"best combined score = {best_score:.4f}")
    _log(f"Checkpoint: {out_dir/'best_model_v2.pth'}")


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
    _log("__main__ entered, parsing args ...")
    train(parse_args())
