import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB4Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = efficientnet_b4(weights=weights)
        except:
            self.net = models.efficientnet_b4(pretrained=pretrained)
        
        # Get number of features
        num_ftrs = self.net.classifier[1].in_features
        
        # PERBAIKAN 1: Input ke classifier sudah (batch_size, num_ftrs)
        # Hapus AdaptiveAvgPool2d dan Flatten yang menyebabkan error
        self.net.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, x):
        # PERBAIKAN 2: Normalisasi sudah ditangani oleh dataloader
        return self.net(x)