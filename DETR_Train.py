import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from torch.utils.data import Dataset

class CustomTrainer(Trainer):
    def __init__(self, size_loss_weight=1, **kwargs):
        """
        Initialize custom trainer, add size_loss_weight parameter
        """
        self.size_loss_weight = size_loss_weight
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss function, including size-aware loss
        Based on paper: Lsize-aware = Σi L1(bi, b̂i)/area(bi)
        """
        # Original model forward pass
        outputs = model(**inputs)
        
        # Original DETR loss
        loss_dict = outputs.loss_dict
        
        # Get predicted and ground truth boxes
        pred_boxes = outputs.pred_boxes
        labels = inputs["labels"]
        
        # Size-aware Loss
        size_loss = 0
        valid_boxes = 0
        
        # Small constant for numerical stability
        eps = 1e-6
        
        for label in labels:
            if "boxes" in label and label["boxes"].numel() > 0:
                gt_boxes = label["boxes"]
                
                # Calculate box area reciprocal as weight (1/area)
                box_areas = gt_boxes[:, 2] * gt_boxes[:, 3]
                inv_areas = 1.0 / (box_areas + eps)  # Avoid division by zero
                
                # Ensure predicted and ground truth box counts match
                # Note: Ideally use DETR's matching result, simplified here
                num_boxes = min(pred_boxes.flatten(0, 1).shape[0], gt_boxes.shape[0])
                
                # Calculate L1 Loss
                l1_loss = F.l1_loss(
                    pred_boxes.flatten(0, 1)[:num_boxes], 
                    gt_boxes[:num_boxes], 
                    reduction='none'
                )
                
                # Use 1/area as weight, emphasize small target boxes
                weighted_loss = l1_loss * inv_areas[:num_boxes].unsqueeze(1)
                
                size_loss += weighted_loss.mean()
                valid_boxes += 1
        
        # If valid boxes exist, add size loss
        if valid_boxes > 0:
            size_loss /= valid_boxes
            
            # Use size_loss_weight parameter to control weight
            total_loss = outputs.loss + self.size_loss_weight * size_loss
        else:
            total_loss = outputs.loss
        
        return (total_loss, outputs) if return_outputs else total_loss

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DETR Object Detection Training')
    parser.add_argument('--pretrained', type=str, default='facebook/detr-resnet-50', help='Pretrained model')
    parser.add_argument('--train_dir', type=str, default='go8020/train/images', help='Training image directory')
    parser.add_argument('--val_dir', type=str, default='go8020/val/images', help='Validation image directory')
    parser.add_argument('--train_ann', type=str, default='go8020/coco/train.json', help='Training annotation file')
    parser.add_argument('--val_ann', type=str, default='go8020/coco/val.json', help='Validation annotation file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--size_loss_weight', type=float, default=1, help='Size-aware loss weight (λs)')
    return parser.parse_args()

def fix_coco_format(ann_file):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Fix image information
    if isinstance(data['images'], str):
        images_dict = json.loads(data['images'])
        data['images'] = [images_dict[str(i)] for i in range(len(images_dict))]
    
    # Fix annotation information
    if isinstance(data['annotations'], str):
        ann_dict = json.loads(data['annotations'])
        fixed_annotations = []
        for i in range(len(ann_dict)):
            ann = ann_dict[str(i)]
            
            # Fix bbox format
            if isinstance(ann['bbox'], dict):
                ann['bbox'] = [
                    float(ann['bbox'].get('0', 0)),
                    float(ann['bbox'].get('1', 0)),
                    float(ann['bbox'].get('2', 0)),
                    float(ann['bbox'].get('3', 0)),
                ]
            
            # Fix other fields
            ann['id'] = int(ann.get('id', i))
            ann['image_id'] = int(ann.get('image_id', 0))
            ann['category_id'] = int(ann.get('category_id', 1))
            ann['area'] = float(ann.get('area', 0))
            ann['iscrowd'] = int(ann.get('iscrowd', 0))
            
            fixed_annotations.append(ann)
        
        data['annotations'] = fixed_annotations
    
    # Fix category information
    if isinstance(data['categories'], dict):
        cat_dict = data['categories']
        data['categories'] = [cat_dict[str(i)] for i in range(len(cat_dict))]
    
    return data

class CocoDetrDataset(Dataset):
    def __init__(self, img_folder, ann_file, processor):
        self.img_folder = img_folder
        self.processor = processor
        
        # Fix COCO data format
        self.coco_data = fix_coco_format(ann_file)
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Get category information, use default if none
        self.categories = self.coco_data.get('categories', [{'id': 1, 'name': 'default'}])
        
        # Create image_id to annotation mapping
        self.image_to_anns = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
        
        print(f"Loaded {len(self.images)} images and {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image information
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.img_folder, image_info['file_name'])
        
        # Try to open image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Unable to load image {image_path}: {e}")
            image = Image.new('RGB', (640, 480), color='white')
        
        # Process annotations
        annotations = self.image_to_anns.get(image_id, [])
        valid_annotations = []
        for ann in annotations:
            if ann.get("iscrowd", 0) == 0:
                valid_annotations.append({
                    'id': ann['id'],
                    'image_id': image_id,
                    'category_id': ann['category_id'] - 1,  # Convert to 0-based
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': 0
                })
        
        # Prepare target information
        target = {
            "image_id": image_id,
            "annotations": valid_annotations,
            "orig_size": list(image.size)
        }
        
        # Process image and annotations
        encoding = self.processor(
            images=image, 
            annotations=target, 
            return_tensors="pt",
            size={'height': 1024, 'width': 1024}  # Use fixed height and width
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0] if encoding["labels"] else {
                "class_labels": torch.tensor([], dtype=torch.long),
                "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
                "image_id": torch.tensor(image_id),
                "orig_size": torch.tensor(list(image.size))
            }
        }

def collate_fn(batch):
    pixel_values = []
    labels = []
    
    for item in batch:
        pixel_values.append(item["pixel_values"])
        labels.append(item["labels"])
    
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels
    }

def main():
    # Parse command line arguments
    args = parse_args()

    # Select processor
    processor = DetrImageProcessor.from_pretrained(
        args.pretrained, 
        size={'height': 1024, 'width': 1024}  # Use fixed height and width
    )

    # Create datasets
    train_dataset = CocoDetrDataset(args.train_dir, args.train_ann, processor)
    val_dataset = CocoDetrDataset(args.val_dir, args.val_ann, processor)

    # Get number of classes
    num_classes = len(train_dataset.categories)
    print(f"Detected {num_classes} classes")

    # Load model - use ignore_mismatched_sizes=True to handle class count mismatch
    model = DetrForObjectDetection.from_pretrained(
        args.pretrained,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Training arguments - remove custom parameters
    args_training = TrainingArguments(
        output_dir="./detr_output",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",  # Fixed eval_strategy to evaluation_strategy
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
        warmup_steps=100,
        weight_decay=1e-4,
        logging_steps=50,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        dataloader_pin_memory=False
        # Removed size_loss_weight parameter
    )

    # Create CustomTrainer - pass size_loss_weight during initialization
    trainer = CustomTrainer(
        size_loss_weight=args.size_loss_weight,  # Directly pass parameter
        model=model,
        args=args_training,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    # Start training
    print(f"Starting training...size_loss_weight={args.size_loss_weight}")
    trainer.train()

    # Save final model
    final_model_path = f"./detr_final_model_w{args.size_loss_weight}_e{args.epochs}"
    trainer.save_model(final_model_path)
    
    # Explicitly save preprocessor_config
    processor.save_pretrained(final_model_path)
    
    print(f"Model saved to {final_model_path}")
    print(f"Preprocessor configuration saved to {final_model_path}/preprocessor_config.json")

if __name__ == "__main__":
    main()