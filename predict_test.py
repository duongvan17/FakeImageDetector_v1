import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Scan images
        for cat_dir in self.root_dir.iterdir():
            if not cat_dir.is_dir(): continue
            
            for label in ['real', 'fake']:
                label_dir = cat_dir / label
                if not label_dir.exists(): continue
                
                # Recursive search
                images = list(label_dir.rglob("*.jpg")) + list(label_dir.rglob("*.png")) + \
                         list(label_dir.rglob("*.jpeg")) + list(label_dir.rglob("*.JPG"))
                
                for img_path in images:
                    self.samples.append((str(img_path), label, cat_dir.name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, cat = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, path, label, cat
        except Exception:
            return torch.zeros(3, 336, 336), path, label, cat # Dummy

def predict_all(data_dir, model_path, output_csv='preds.csv', batch_size=16, device='cuda'):
    # Config (MUST MATCH TRAINING!)
    img_size = 224
    model_name = 'vit_base_patch16_clip_224.openai'
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Transform
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    transform = create_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=clip_mean,
        std=clip_std
    )
    
    # Dataset & Loader
    dataset = TestDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print("Model not found! Using random weights.")
        
    model = model.to(device)
    model.eval()
    
    results = []
    correct = 0
    total = 0
    
    print(f"Starting prediction on {len(dataset)} images...")
    
    with torch.no_grad():
        for images, paths, true_labels, cats in tqdm(loader):
            images = images.to(device)
            outputs = model(images).squeeze()
            
            # Handle batch size 1 case where squeeze removes batch dim
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # If batch size 1, probs is scalar
            if probs.ndim == 0:
                probs = [probs.item()]
            
            for i, prob in enumerate(probs):
                pred_label = 'fake' if prob > 0.5 else 'real'
                true_label = true_labels[i]
                
                is_correct = (pred_label == true_label)
                if is_correct: correct += 1
                total += 1
                
                results.append({
                    'filepath': paths[i],
                    'category': cats[i],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'prob_fake': prob,
                    'correct': is_correct
                })
                
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    acc = correct / total if total > 0 else 0
    print(f"\nPrediction complete!")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Saved results to {output_csv}")
    
    # Per category stats
    print("\nPer-category Accuracy:")
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        cat_acc = cat_df['correct'].mean()
        print(f"  {cat}: {cat_acc:.4f} ({len(cat_df)} images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='FakeVLM-main/FakeVLM-main/playground/data/test')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_csv', type=str, default='preds.csv')
    args = parser.parse_args()
    
    predict_all(args.data_dir, args.model_path, args.output_csv)
