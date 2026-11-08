import torch
from torch import nn

class EMGCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features= nn.Sequential(
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self,x):
        x=self.features(x)
        x=x.squeeze(-1)
        return self.classifier(x)

