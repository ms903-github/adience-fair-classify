import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.optim import lr_scheduler
import torchvision.models as models

class Classifier_resnet(nn.Module):
    def __init__(self):
        super(Classifier_resnet, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 9)
        self.batchnorm = nn.BatchNorm1d(500)
    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = F.relu(self.fc2(x))
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 9)
        self.batchnorm = nn.BatchNorm1d(500)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = F.relu(self.fc2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 1)
        self.batchnorm = nn.BatchNorm1d(500)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))

