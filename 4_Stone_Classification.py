import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
from PIL import Image
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class ViTMultiPatchClassifier(nn.Module):  
    def __init__(self, num_classes=4):  
        super().__init__()  
        config = ViTConfig(  
            image_size=304,  
            patch_size=16,  
            num_channels=3,  
            hidden_size=768,  
            num_hidden_layers=12,  
            num_attention_heads=12,  
            intermediate_size=3072,  
            num_labels=num_classes  
        )  
        self.vit = ViTModel(config)  
        self.fc = nn.Linear(config.hidden_size, num_classes)  

    def forward(self, x):  
        out = self.vit(pixel_values=x).last_hidden_state[:, 1:, :]  
        out = self.fc(out)  
        return out  # shape [B, 361, num_classes]   

def parse_args():
    parser = argparse.ArgumentParser(description='ViT Go Board Multi-Patch Prediction Script')
    parser.add_argument('--model_path', type=str, default='./vit_go_model_final_50.pth', help='Model path')
    parser.add_argument('--img_dir', type=str, default='./go_board_cut', help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='./go_board_cut_pred', help='Output NPY directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def load_model(model_path, device):
    try:
        model = ViTMultiPatchClassifier()
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logging.info(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise

def predict_and_save_npy(model, img_path, output_dir, device):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((304, 304)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Predict classes for all patches
            patch_predictions = model(image_tensor)
            
            # Get predicted class for each patch (0, 1, 2, 3)
            patch_classes = patch_predictions.argmax(dim=-1)
            
            # Reshape patch_classes to 19x19 grid
            patch_classes_np = patch_classes.cpu().numpy().reshape(19, 19)

        # Create NPY filename
        npy_filename = os.path.basename(img_path).replace('.jpg', '.npy').replace('.png', '.npy')
        npy_path = os.path.join(output_dir, npy_filename)

        # Save prediction results (classes)
        np.save(npy_path, patch_classes_np)
        
        return True
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return False

def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    try:
        model = load_model(args.model_path, args.device)
        model.to(args.device)
        model.eval()
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return

    # Find all image files
    img_files = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    logging.info(f"Found {len(img_files)} image files")

    # Predict and process
    successful_predictions = 0
    failed_predictions = 0

    for img_file in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(args.img_dir, img_file)
        result = predict_and_save_npy(model, img_path, args.output_dir, args.device)
        
        if result:
            successful_predictions += 1
        else:
            failed_predictions += 1

    logging.info(f"Processing complete: {successful_predictions} successful, {failed_predictions} failed")
    logging.info(f"Prediction results saved to {args.output_dir}")

if __name__ == "__main__":
    main()