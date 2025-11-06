import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientNetB4Model
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 8
BASE_LR = 1e-3
MAX_LR = 5e-3
WEIGHT_DECAY = 1e-4
TARGET_ACCURACY = 80.0
DEVICE = torch.device('cpu')

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Initialize Model
    model = EfficientNetB4Model(in_channels=in_channels, num_classes=num_classes, pretrained=True)
    
    # Freeze fewer layers
    total_layers = len(list(model.net.features))
    freeze_layers = int(total_layers * 0.6)
    for param in model.net.features[:freeze_layers].parameters():
        param.requires_grad = False
    
    # 3. Loss function with class weights
    pos_weight = torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 4. Optimizer with higher learning rates
    optimizer = optim.AdamW([
        {'params': model.net.features[freeze_layers:].parameters(), 'lr': BASE_LR},
        {'params': model.net.classifier.parameters(), 'lr': BASE_LR * 5}
    ], weight_decay=WEIGHT_DECAY)
    
    # 5. Simpler scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[MAX_LR, MAX_LR * 5],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=50,
        anneal_strategy='linear'
    )
    
    # Initialize tracking
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    # Training Loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:  # More frequent updates
                print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
                
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
    model.load_state_dict(torch.load("best_model.pth"))
    try:
        visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == '__main__':
    train()