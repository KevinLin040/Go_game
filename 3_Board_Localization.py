import os  
import json  
import torch  
import numpy as np  
from PIL import Image, ImageDraw  
from transformers import DetrImageProcessor, DetrForObjectDetection  
from pycocotools.coco import COCO  
import matplotlib.pyplot as plt  
import argparse  
from tqdm import tqdm  
import shutil  

def parse_args():  
    parser = argparse.ArgumentParser(description='DETR Object Detection Prediction Script')  
    parser.add_argument('--model_path', type=str, default='./detr_final_model_w0.1_e100', help='Model path')  
    parser.add_argument('--img_dir', type=str, default='go_images_2fps', help='Test image directory')  
    parser.add_argument('--output_dir', type=str, default='./go_board_cut', help='Output directory')  
    parser.add_argument('--threshold', type=float, default=0.9, help='Detection threshold')  
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')  
    return parser.parse_args()

def load_model(model_path):
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    return processor, model

def predict_and_save_bboxes(model, processor, img_path, output_dir, threshold):
    image = Image.open(img_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**encoding)

    # Post-process prediction results
    target_sizes = torch.tensor([image.size[::-1]])
    processed_outputs = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    # Extract bounding boxes, select the smallest one (if multiple)
    bboxes = []
    if processed_outputs["scores"].numel() > 0:
        for box, score in zip(processed_outputs["boxes"], processed_outputs["scores"]):
            if score >= threshold:
                x1, y1, x2, y2 = box.tolist()
                bboxes.append((x1, y1, x2, y2))

    if bboxes:
        # Select the smallest bounding box
        bboxes = sorted(bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))  # Sort by area
        x1, y1, x2, y2 = bboxes[0]  # Take the smallest bbox
        
        # Crop the bounding box
        cropped_image = image.crop((x1, y1, x2, y2))
        output_image_path = os.path.join(output_dir, os.path.basename(img_path))
        cropped_image.save(output_image_path)

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_path}")
    processor, model = load_model(args.model_path)
    model.to(args.device)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(img_files):
        img_path = os.path.join(args.img_dir, img_file)
        predict_and_save_bboxes(model, processor, img_path, args.output_dir, args.threshold)

    print(f"All images processed and saved to {args.output_dir}.")

if __name__ == "__main__":
    main()