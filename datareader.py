import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import ChestMNIST
from PIL import Image

# Class configuration
CLASS_A_IDX = 1  # 'Cardiomegaly'
CLASS_B_IDX = 7  # 'Pneumothorax'
NEW_CLASS_NAMES = {0: 'Cardiomegaly', 1: 'Pneumothorax'}

class FilteredBinaryDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        dataset = ChestMNIST(split=split, transform=None, download=True)
        original_labels = dataset.labels

        # Find indices for single-label images of desired classes
        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        self.images = []
        self.labels = []

        # Balance dataset
        min_samples = min(len(indices_a), len(indices_b))
        indices_a = indices_a[:min_samples]
        indices_b = indices_b[:min_samples]

        for idx in indices_a:
            self.images.append(dataset[idx][0])
            self.labels.append(0)
        for idx in indices_b:
            self.images.append(dataset[idx][0])
            self.labels.append(1)

        print(f"Split: {split} | Cardiomegaly: {len(indices_a)} | Pneumothorax: {len(indices_b)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_data_loaders(batch_size):
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])

    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = FilteredBinaryDataset('train', train_transform)
    val_dataset = FilteredBinaryDataset('test', val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, 2, 3  # num_classes=2, in_channels=3