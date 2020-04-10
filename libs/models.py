import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.optim import lr_scheduler
import torchvision.models as models

class Classifier_resnet(nn.Module):
    def __init__(self, n_class):
        super(Classifier_resnet, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, n_class)
        self.batchnorm = nn.BatchNorm1d(500)
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.fc2(x)
        return x
