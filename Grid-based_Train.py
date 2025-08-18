import os, time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import ViTModel, ViTConfig

# Custom Dataset 
class WholeBoardDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.imgs = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.transform = transform

    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.imgs[idx]).convert("RGB"))
        label = torch.tensor(np.load(self.labels[idx]).flatten())
        return img, label

# ViT Model Definition 
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
        return out

# Enhanced Training Function - Added resume training, early stopping, and adaptive optimizer
def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                      epochs=50, 
                      patience=5,             # Early stopping patience
                      resume_from=None,       # Resume training model path
                      lr=1e-4,
                      weight_decay=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create model save directory
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "vit_go_model_best.pth")
    
    # Set adaptive optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler - use cosine annealing for adaptive learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Resume training: if model path provided, load model and optimizer state
    start_epoch = 0
    best_val_acc = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Continuing training from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc']
        
        print(f"Resuming from epoch {start_epoch}, best validation accuracy: {best_val_acc:.4f}")
    
    # Early stopping setup
    early_stop_counter = 0
    
    # Training records
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\n✅ Starting Training (Device: {device})")
    t0 = time.time()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs}"):
            x, y = x.to(device), y.to(device)
            
            # Forward propagation
            pred = model(x)
            loss = loss_fn(pred.view(-1, 4), y.view(-1))
            
            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate statistics
            train_loss += loss.item()
            train_total += y.numel()
            train_correct += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_total += y.numel()
                val_correct += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
        
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save records
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Output current progress
        print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            torch.save(checkpoint, model_save_path)
            print(f"✓ Best model saved (Validation Accuracy: {val_acc:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"! Validation accuracy not improved ({early_stop_counter}/{patience})")
        
        # Check early stopping
        if early_stop_counter >= patience:
            print(f"⚠️ Early stopping triggered! No improvement for {patience} epochs.")
            break
    
    # Training completed
    t1 = time.time()
    print(f"⏱️ Training time: {t1 - t0:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, "vit_go_model_final_50.pth")
    torch.save({
        'epoch': start_epoch + epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history
    }, final_model_path)
    print(f"✓ Final model saved to {final_model_path}")
    
    # Plot training process curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'training_history.png'))
    
    # Evaluate with test set using best model
    print("\n🔍 Evaluating on test set")
    # Load best model
    best_checkpoint = torch.load(model_save_path, map_location=device)
    if 'model_state_dict' in best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(best_checkpoint)
    
    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    t2 = time.time()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            logits = model(x).argmax(dim=-1)
            all_preds.append(logits.view(-1).cpu())
            all_labels.append(y.view(-1).cpu())
    t3 = time.time()

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    # Output classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    
    # Save classification report
    with open(os.path.join(model_save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, where=cm_sum!=0) * 100
    
    labels = np.asarray([f"{perc:.1f}%\n({count})" for perc, count in zip(cm_perc.flatten(), cm.flatten())])
    labels = labels.reshape(cm.shape)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="YlGnBu",
                xticklabels=["Empty", "Black", "White", "Occluded"],
                yticklabels=["Empty", "Black", "White", "Occluded"])
    
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix.png'))
    
    # Calculate test time and accuracy
    test_time = t3 - t2
    test_acc = (y_pred == y_true).mean()
    print(f"⏱️ Test time: {test_time:.2f} seconds")
    print(f"🎯 Test accuracy: {test_acc:.4f}")
    
    # Return test results
    return {
        'test_accuracy': test_acc,
        'test_time': test_time,
        'history': history,
        'best_val_acc': best_val_acc
    }

# Main program
if __name__ == "__main__":
    # Set transformations
    tf = transforms.Compose([
        transforms.Resize((304, 304)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load datasets
    train_ds = WholeBoardDataset("images_train", "labels_train", tf)
    val_ds = WholeBoardDataset("images_val", "labels_val", tf)
    test_ds = WholeBoardDataset("images_test", "labels_test", tf)
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    # Initialize model
    model = ViTMultiPatchClassifier()
    
    # Resume training: check if best model exists
    resume_path = None
    if os.path.exists('models/vit_go_model_best.pth'):
        resume_path = 'models/vit_go_model_best.pth'
    
    # Train model
    results = train_and_evaluate(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        epochs=40,           # Maximum training epochs
        patience=5,          # Early stopping patience
        resume_from=resume_path,  # Resume training model path
        lr=5e-5,            # Learning rate
        weight_decay=1e-4    # Weight decay
    )
    
    print(f"\nTraining completed! Best validation accuracy: {results['best_val_acc']:.4f}, Test accuracy: {results['test_accuracy']:.4f}")