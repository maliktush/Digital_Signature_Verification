import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, x1, x2, y):
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = self.alpha * y * distance ** 2 + self.beta * (1 - y) * (torch.max(torch.zeros_like(distance), self.margin - distance) ** 2)
        return torch.mean(loss)

class SigNet(nn.Module):
    def __init__(self):
        super().__init__()
        #size = [155, 220, 1]
        self.conv1 = nn.Conv2d(1, 96, 11)  # size = [145,210]
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros')  # size = [72, 105]
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros')
        self.norm1 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)
        self.norm2 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)
        self.d1 = nn.Dropout2d(p=0.3)
        self.d2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(18 * 26 * 256, 1024)
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x1, x2):
        x1 = self.norm1(F.relu(F.max_pool2d(self.conv1(x1), 2)))
        x1 = self.norm2(F.max_pool2d(self.conv2(x1), 2))
        x1 = self.d1(x1)
        x1 = self.conv3(x1)
        x1 = F.max_pool2d(self.conv4(x1), 2)
        x1 = self.d1(x1)
        # x1 = nn.Flatten(1, -1)
        x1 = x1.view(x1.shape[0], -1)
        x1 = self.fc1(x1)
        x1 = self.d2(x1)
        x1 = self.fc2(x1)

        x2 = self.norm1(F.relu(F.max_pool2d(self.conv1(x2), 2)))
        x2 = self.norm2(F.max_pool2d(self.conv2(x2), 2))
        x2 = self.d1(x2)
        x2 = self.conv3(x2)
        x2 = F.max_pool2d(self.conv4(x2), 2)
        x2 = self.d1(x2)
        # x2 = nn.Flatten(1, -1)
        x2 = x2.view(x2.shape[0], -1)
        x2 = self.fc1(x2)
        x2 = self.d2(x2)
        x2 = self.fc2(x2)

        return x1, x2
