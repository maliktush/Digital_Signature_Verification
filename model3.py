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
        loss = self.alpha * (y) * distance ** 2 + self.beta * (1-y) * (torch.max(torch.zeros_like(distance), self.margin - distance) ** 2)
        return torch.mean(loss)


class SigNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = [155, 220, 1]
        self.conv1 = nn.Conv2d(1, 128, 8)  # size = [74,106]
        self.conv2 = nn.Conv2d(128, 256, 7, stride=2,padding=2)  # size = [36, 51]
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2)
        self.conv4 = nn.Conv2d(512, 256, 3)
        self.conv_bn1 = nn.BatchNorm2d(128)
        self.conv_bn2 = nn.BatchNorm2d(256)
        self.conv_bn3 = nn.BatchNorm2d(512)
        self.conv_bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(7 * 11 * 256, 2048)
        self.fc2 = nn.Linear(2048, 128)

    def forward(self, x1, x2):
        x1 = self.conv_bn1(F.relu(F.max_pool2d(self.conv1(x1), 2)))
        x1 = self.conv_bn2(F.tanh(self.conv2(x1)))
        x1 = self.conv_bn3(F.relu(self.conv3(x1)))
        x1 = self.conv_bn4(F.tanh(F.max_pool2d(self.conv4(x1), 2)))
        # x1 = nn.Flatten(1, -1)
        x1 = x1.view(x1.shape[0], -1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)

        x2 = self.conv_bn1(F.relu(F.max_pool2d(self.conv1(x2), 2)))
        x2 = self.conv_bn2(F.tanh(self.conv2(x2)))
        x2 = self.conv_bn3(F.relu(self.conv3(x2)))
        x2 = self.conv_bn4(F.tanh(F.max_pool2d(self.conv4(x2), 2)))
        # x1 = nn.Flatten(1, -1)
        x2 = x2.view(x2.shape[0], -1)
        x2 = self.fc1(x2)
        x2 = self.fc2(x2)

        return x1, x2
