import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
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
        self.images = dataset.imgs

        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        min_samples = min(len(indices_a), len(indices_b))
        indices_a = indices_a[:min_samples]
        indices_b = indices_b[:min_samples]

        self.selected_indices = np.concatenate([indices_a, indices_b])
        self.labels = np.array([0] * len(indices_a) + [1] * len(indices_b))

        print(f"Split: {split} | Cardiomegaly: {len(indices_a)} | Pneumothorax: {len(indices_b)}")

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        image = self.images[self.selected_indices[idx]]
        label = self.labels[idx]

        # Convert to PIL and ensure RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def get_data_loaders(batch_size):
    # PERBAIKAN: Tambahkan normalisasi ImageNet
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Mengubah ke [0, 1]
        imagenet_norm,          # Normalisasi
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Mengubah ke [0, 1]
        imagenet_norm           # Normalisasi
    ])

    train_dataset = FilteredBinaryDataset('train', train_transform)
    val_dataset = FilteredBinaryDataset('test', val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, 2, 3