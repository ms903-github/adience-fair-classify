import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import pandas as pd
from PIL import Image
import random

def load_pict(load_path, transform=None):
    class MyDataset(Dataset):
        def __init__(self, file_path, transform):
            self.df = pd.read_csv(file_path)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]["image_path"]
            img_path = change_path(img_path)
            label = self.df.iloc[idx]["label"]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
    return MyDataset(load_path, transform=transform)