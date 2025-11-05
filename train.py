import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientNetB4Model
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameters ---
EPOCHS = 200
BATCH_SIZE = 16
BASE_LR = 1e-4
MAX_LR = 3e-3
WEIGHT_DECAY = 1e-4
TARGET_ACCURACY = 95.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Load Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Initialize Model
    model = EfficientNetB4Model(in_channels=in_channels, num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)
    
    # Initially freeze most layers
    total_layers = len(list(model.net.features))
    freeze_layers = int(total_layers * 0.8)
    for param in model.net.features[:freeze_layers].parameters():
        param.requires_grad = False
    
    # 3. Loss with class weights
    pos_weight = torch.tensor([1.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 4. Optimizer with different LRs for different parts
    optimizer = optim.AdamW([
        {'params': model.net.features[freeze_layers:].parameters(), 'lr': BASE_LR},
        {'params': model.net.classifier.parameters(), 'lr': BASE_LR * 10}
    ], weight_decay=WEIGHT_DECAY)
    
    # 5. Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[MAX_LR, MAX_LR * 10],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    # 6. Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Initialize tracking
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    # Training Loop
    for epoch in range(EPOCHS):
        # Progressive unfreezing
        if epoch in [40, 80, 120]:
            unfreeze_idx = freeze_layers - int((epoch // 40) * total_layers * 0.2)
            print(f"\nUnfreezing from layer {unfreeze_idx}...")
            
            for param in model.net.features[unfreeze_idx:].parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW([
                {'params': model.net.features[unfreeze_idx:].parameters(), 'lr': BASE_LR/2},
                {'params': model.net.classifier.parameters(), 'lr': BASE_LR * 5}
            ], weight_decay=WEIGHT_DECAY)
            
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[MAX_LR/2, MAX_LR * 5],
                epochs=EPOCHS-epoch,
                steps_per_epoch=len(train_loader),
                pct_start=0.1
            )
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Save best model and check early stopping
        if val_accuracy > best_val_acc or (val_accuracy == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_accuracy:.2f}% "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_accuracy:.2f}% "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_accuracy >= TARGET_ACCURACY:
            print(f"\nReached target accuracy of {TARGET_ACCURACY}%")
            break
        
        if patience_counter >= patience:
            print("\nEarly stopping triggered")
            break
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    
    # Load best model and visualize
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    train()