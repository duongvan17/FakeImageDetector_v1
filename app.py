import os
import io
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import timm
from torchvision import transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None
device = None
transform = None

def load_model():
    """Load the trained model"""
    global model, device, transform
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration (must match training!)
    model_name = 'vit_base_patch16_clip_224.openai'
    img_size = 224
    model_path = 'checkpoints/best_model.pth'
    
    # CLIP normalization (must match training!)
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)
    ])
    
    # Load model
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {model_path}")
        if 'acc' in checkpoint:
            print(f"✓ Model accuracy: {checkpoint['acc']:.4f}")
    else:
        print(f"⚠ Warning: Model checkpoint not found at {model_path}")
        print("Using random weights (predictions will be meaningless)")
    
    model = model.to(device)
    model.eval()
    
    return model, device, transform

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logit = model(input_tensor).squeeze()
            prob_fake = torch.sigmoid(logit).item()
        
        # Determine label and confidence
        if prob_fake > 0.5:
            label = "FAKE"
            confidence = prob_fake * 100
        else:
            label = "REAL"
            confidence = (1 - prob_fake) * 100
        
        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'prob_fake': round(prob_fake * 100, 2),
            'prob_real': round((1 - prob_fake) * 100, 2)
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    load_model()
    print("Model loaded successfully!\n")
    
    # Run Flask app
    print("Starting Flask server...")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
