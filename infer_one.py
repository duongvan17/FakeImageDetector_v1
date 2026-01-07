import argparse
import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import create_transform
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def predict_one(image_path, model_path, model_name='convnext_tiny.fb_in22k_ft_in1k', img_size=336, device='cuda'):
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    
    # Load checkpoint
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both full checkpoint and state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model path {model_path} not found. Using random weights (for testing only).")
    
    model = model.to(device)
    model.eval()
    
    # Transform (Must match training!)
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    transform = create_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bicubic',
        mean=clip_mean,
        std=clip_std
    )
    
    # Load and Preprocess Image
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return
    
    # Inference
    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        prob = torch.sigmoid(logit).item()
        
    # Result
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob
    
    print(f"\nImage: {image_path}")
    print(f"Prediction: {label}")
    print(f"Probability (Fake): {prob:.4f}")
    print(f"Confidence: {confidence:.4f}")
    
    return label, prob

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to .pth checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    predict_one(args.image_path, args.model_path, device=args.device)
