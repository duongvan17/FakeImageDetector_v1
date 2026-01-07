import os
import argparse
import logging
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FakeClueDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = [] # (path, label_idx, category_idx)
        self.targets = [] # label_idx
        self.categories = [] # category_idx
        
        # Define classes
        self.classes = ['real', 'fake']
        self.class_to_idx = {'real': 0, 'fake': 1}
        
        # Find categories
        self.category_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.cat_to_idx = {name: i for i, name in enumerate(self.category_names)}
        
        logger.info(f"Found categories in {split}: {self.category_names}")
        
        # Scan images
        self._scan_dataset()
        
    def _scan_dataset(self):
        count_by_cat_label = Counter()
        
        for cat in self.category_names:
            cat_dir = self.root_dir / cat
            cat_idx = self.cat_to_idx[cat]
            
            for label in self.classes:
                label_dir = cat_dir / label
                if not label_dir.exists():
                    continue
                    
                # Recursive search for images
                # FakeVLM dataset has nested folders like train/doc/fake/magazine_ch_2/xxx.jpg
                images = list(label_dir.rglob("*.jpg")) + list(label_dir.rglob("*.png")) + \
                         list(label_dir.rglob("*.jpeg")) + list(label_dir.rglob("*.JPG"))
                
                label_idx = self.class_to_idx[label]
                
                for img_path in images:
                    self.samples.append((str(img_path), label_idx, cat_idx))
                    self.targets.append(label_idx)
                    self.categories.append(cat_idx)
                
                count_by_cat_label[(cat, label)] = len(images)
                
        logger.info(f"Dataset stats for {self.root_dir.name}:")
        for (cat, label), count in sorted(count_by_cat_label.items()):
            logger.info(f"  {cat}/{label}: {count}")
        logger.info(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cat = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, cat, path
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            # Return a dummy image or handle error appropriately
            # For simplicity, we'll just return the next item (circular)
            return self.__getitem__((idx + 1) % len(self))

def get_balanced_sampler(dataset):
    # Calculate weights for each sample to balance (Category, Label) pairs
    # We want each (cat, label) group to have equal probability of being sampled
    
    # Count samples per (cat, label)
    counts = Counter()
    for i in range(len(dataset)):
        _, label, cat = dataset.samples[i]
        counts[(cat, label)] += 1
        
    # Calculate weight per group
    # weight = 1 / count
    weights_per_group = {}
    for k, v in counts.items():
        weights_per_group[k] = 1.0 / v if v > 0 else 0
        
    # Assign weight to each sample
    sample_weights = []
    for i in range(len(dataset)):
        _, label, cat = dataset.samples[i]
        weight = weights_per_group.get((cat, label), 0)
        sample_weights.append(weight)
        
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, accum_iter=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels, cats, paths) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device).float()
        
        with autocast():
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            # Normalize loss for accumulation
            loss = loss / accum_iter
            
        scaler.scale(loss).backward()
        
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Revert loss normalization for logging
        running_loss += (loss.item() * accum_iter) * images.size(0)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
        
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, use_tta=False):
    """Validation with optional Test-Time Augmentation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-category tracking
    cat_correct = Counter()
    cat_total = Counter()
    
    # For AUC calculation
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for images, labels, cats, paths in pbar:
        images, labels = images.to(device), labels.to(device).float()
        
        with autocast():
            if use_tta:
                # Test-Time Augmentation: average predictions from original + horizontal flip
                outputs1 = model(images).squeeze(-1)
                outputs2 = model(torch.flip(images, dims=[3])).squeeze(-1)  # horizontal flip
                outputs = (outputs1 + outputs2) / 2
            else:
                outputs = model(images).squeeze(-1)
            
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Store for AUC
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        # Per-category accuracy
        for pred, label, cat in zip(preds, labels, cats):
            cat_idx = cat.item()
            if pred == label:
                cat_correct[cat_idx] += 1
            cat_total[cat_idx] += 1
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    
    # Log per-category results (inspired by FakeVLM)
    dataset = loader.dataset
    logger.info(f"\nValidation Results (Epoch {epoch}):")
    logger.info(f"  Overall: Acc={correct/total:.4f}, AUC={auc:.4f}")
    for cat_idx, count in sorted(cat_total.items()):
        cat_name = dataset.category_names[cat_idx]
        cat_acc = cat_correct[cat_idx] / count if count > 0 else 0
        logger.info(f"  Category {cat_name}: Acc={cat_acc:.4f} ({cat_correct[cat_idx]}/{count})")
    
    return running_loss / total, correct / total, auc

# Old validate function removed, replaced by the one above


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='FakeVLM-main/FakeVLM-main/playground/data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Max epochs (early stopping will handle)')
    parser.add_argument('--batch_size', type=int, default=4, help='Small batch size for 4GB GPU')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--img_size', type=int, default=224, help='Image size (224 for ViT Base)')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_clip_224.openai')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model & Normalization Config
    # Try to use CLIP model first (Best for FakeVLM replication)
    # If fails, fallback to ImageNet but MUST switch normalization stats
    
    if 'clip' in args.model_name:
        # CLIP Mean/Std
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        logger.info(f"Using CLIP Normalization for model {args.model_name}")
    else:
        # ImageNet Mean/Std
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        logger.info(f"Using ImageNet Normalization for model {args.model_name}")

    # Data Transforms
    # Training: Light augmentation (learned from FakeVLM - they don't augment, but we can for vision-only)
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        # Light augmentation - careful not to destroy fake artifacts
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.RandomRotation(degrees=3),  # Very slight rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Datasets
    logger.info("Loading datasets...")
    train_dataset = FakeClueDataset(args.data_dir, transform=train_transform, split='train')
    val_dataset = FakeClueDataset(args.data_dir, transform=val_transform, split='test')
    
    # Sampler
    sampler = get_balanced_sampler(train_dataset)
    
    # DataLoaders
    # Increase num_workers for faster data loading (4 is good for most systems)
    # persistent_workers keeps workers alive between epochs (faster!)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=2,  # Reduced from 4 to save memory
        pin_memory=True
        # persistent_workers removed - causes memory leak with grad checkpointing
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,
        shuffle=False, 
        num_workers=2,  # Increased
        pin_memory=True
    )
    
    # Model: Switch to CLIP ViT Base (matches FakeVLM's strong visual backbone)
    logger.info(f"Creating model: {args.model_name}")
    model = timm.create_model(
        args.model_name, 
        pretrained=True, 
        num_classes=1 
    )
    # Enable gradient checkpointing to save VRAM (Critical for 4GB GPU with ViT)
    model.set_grad_checkpointing(enable=True)
    model = model.to(device)
    
    # Loss with Label Smoothing (0.1) to prevent overfitting
    # BCEWithLogitsLoss doesn't support label_smoothing directly in older torch versions, 
    # but we can implement it manually or use a trick if needed. 
    # Actually, recent torch versions support it. Let's check or use a custom one.
    # For safety/compatibility, we'll stick to standard BCE but can implement smoothing if needed.
    # Let's use standard BCE for now to avoid complexity, but add AUC metric.
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer: Tuned betas for ViT/CLIP fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05, betas=(0.9, 0.98))
    
    # Scheduler with Warmup
    warmup_epochs = 1
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    scaler = GradScaler()
    
    best_acc = 0.0
    best_auc = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {args.batch_size} * 8 = {args.batch_size * 8}")
    logger.info(f"Early stopping patience: {args.patience} epochs")
    logger.info(f"Test-Time Augmentation: {args.use_tta}")
    
    for epoch in range(1, args.epochs + 1):
        # Effective batch size = 4 * 8 = 32 (optimal for 4GB GPU)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, accum_iter=8)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, epoch, use_tta=args.use_tta)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint (use AUC as primary metric)
        improved = False
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            improved = True
            patience_counter = 0
            
            save_path = Path(args.output_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'acc': best_acc,
                'auc': best_auc,
                'category_names': train_dataset.category_names,
            }, save_path)
            logger.info(f"[BEST] Saved model to {save_path} (AUC: {best_auc:.4f}, Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{args.patience} epochs")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            logger.info(f"Best AUC: {best_auc:.4f}, Best Acc: {best_acc:.4f}")
            break
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    # Set OOM prevention config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    main()
