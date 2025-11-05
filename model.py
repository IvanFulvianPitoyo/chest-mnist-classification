import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB4Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        try:
            from torchvision.models import EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.efficientnet_b4(weights=weights)
        except:
            self.net = models.efficientnet_b4(pretrained=pretrained)
        
        # Get number of features from efficientnet
        num_ftrs = self.net.classifier[1].in_features
        
        # Replace classifier with custom head
        self.net.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, x):
        # Handle grayscale input
        if self.in_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)