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

# if data is given as (path label \n) in txt file
def load_pict(load_path, transform=None):
    class MyDataset(Dataset):
        def __init__(self, file_path, transform):
            pathlist = []
            labellist = []
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                path, label = line.split(" ")
                pathlist.append(path)
                labellist.append(int(label))
            self.pathlist = pathlist
            self.labellist = labellist
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_path = self.pathlist[idx]
            label = self.labellist[idx]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
    return MyDataset(load_path, transform=transform)


def load_pict2(load_path, transform=None, test=False):
    class MyDataset(Dataset):
        def __init__(self, file_path, transform, test):
            #path: ./datasets/train/0/*.jpg
            if not test:
                pathlist = glob.glob(os.path.join("load_path", "train/*/*.jpg"))
                labellist = []
                for path in pathlist:
                    labellist.append(int(path.split("/")[3]))
            else:
                pathlist = glob.glob(os.path.join("load_path", "test/*/*.jpg"))
                labellist = []
                for path in pathlist:
                    labellist.append(int(path.split("/")[3]))

            self.pathlist = pathlist
            self.labellist = labellist
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_path = self.pathlist[idx]
            label = self.labellist[idx]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
    return MyDataset(load_path, transform=transform, test=test)