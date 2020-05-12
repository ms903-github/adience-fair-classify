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
def load_pict(load_path, gen_mode="both", transform=None):
    class MyDataset(Dataset):
        def __init__(self, file_path, gen_mode=gen_mode, transform=None):
            self.gen_mode = gen_mode
            pathlist = []
            labellist = []
            genlist = []
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                path, age, gen = line.split()
                age = int(age)
                # male : 1 female : 0
                if self.gen_mode == "both":
                    if gen == "m":
                        genlist.append(1)
                    else:
                        genlist.append(0)
                elif self.gen_mode == "male":
                    if not gen == "m":
                        continue
                    genlist.append(1)
                elif self.gen_mode == "female":
                    if not gen == "f":
                        continue
                    genlist.append(0)
                pathlist.append(path)
                
                if 0<=age<4:
                    labellist.append(0)
                elif 4<=age<8:
                    labellist.append(1)
                elif 8<=age<15:
                    labellist.append(3)
                elif 15<=age<25:
                    labellist.append(4)
                elif 25<=age<38:
                    labellist.append(5)
                elif 38<=age<48:
                    labellist.append(6)
                elif 48<=age<53:
                    labellist.append(7)
                elif 53<=age:
                    labellist.append(8)
                else:
                    continue

            self.pathlist = pathlist
            self.agelist = labellist
            self.genlist = genlist
            self.transform = transform
            print(len(pathlist))
            print(len(labellist))

        def __len__(self):
            return len(self.pathlist)
        
        def __getitem__(self, idx):
            img_path = self.pathlist[idx]
            age = self.agelist[idx]
            gen = self.genlist[idx]
            img = Image.open(os.path.join("./datasets/faces", img_path))
            if self.transform:
                img = self.transform(img)
            return img, age, gen
    return MyDataset(load_path, transform=transform)


# 男女およびnum_sampleをそれぞれ指定するloader
def load_pict2(file_path, num_f_sample, num_m_sample, transform=None):
    with open(file_path, "r") as f:
        lines = f.readlines()
    f_lines = []
    m_lines = []
    for line in lines:
        path, age, gen = line.split()
        if gen == "f":
            f_lines.append(line)
        elif gen == "m":
            m_lines.append(line)
    f_lines = random.sample(f_lines, num_f_sample)
    m_lines = random.sample(m_lines, num_m_sample)

    f_pathlist = []
    m_pathlist = []
    f_labellist = []
    m_labellist = []

    for line in f_lines:
        path, age, _ = line.split()
        age = int(age)
        if 0<=age<4:
            f_labellist.append(0)
        elif 4<=age<8:
            f_labellist.append(1)
        elif 8<=age<15:
            f_labellist.append(3)
        elif 15<=age<25:
            f_labellist.append(4)
        elif 25<=age<38:
            f_labellist.append(5)
        elif 38<=age<48:
            f_labellist.append(6)
        elif 48<=age<53:
            f_labellist.append(7)
        elif 53<=age:
            f_labellist.append(8)
        else:
            continue
        f_pathlist.append(path)

    for line in m_lines:
        path, age, _ = line.split()
        age = int(age)
        if 0<=age<4:
            m_labellist.append(0)
        elif 4<=age<8:
            m_labellist.append(1)
        elif 8<=age<15:
            m_labellist.append(2)
        elif 15<=age<25:
            m_labellist.append(3)
        elif 25<=age<38:
            m_labellist.append(4)
        elif 38<=age<48:
            m_labellist.append(5)
        elif 48<=age<53:
            m_labellist.append(6)
        elif 53<=age:
            m_labellist.append(7)
        else:
            continue
        m_pathlist.append(path)

    pathlist = f_pathlist + m_pathlist
    agelist = f_labellist + m_labellist
    genlist = [0 for i in range(len(f_pathlist))] + [1 for i in range(len(m_pathlist))]
    transform = transform
    print(len(pathlist))
    print(len(agelist))

    class MyDataset(Dataset):
        def __init__(self, pathlist, labellist, genlist, transform=None):
            self.pathlist = pathlist
            self.labellist = labellist
            self.genlist = genlist
            self.transform = transform
        def __len__(self):
            return len(self.pathlist)
        
        def __getitem__(self, idx):
            img_path = self.pathlist[idx]
            label = self.labellist[idx]
            gen = self.genlist[idx]
            img = Image.open(os.path.join("./datasets/faces", img_path))
            if self.transform:
                img = self.transform(img)
            return img, label, gen
    return MyDataset(pathlist, agelist, genlist, transform=transform)

# 各クラスのサンプル数を指定するloader
def load_pict3(file_path, n_sample_list, transform=None):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    f_pathlist = [[] for i in range(8)]
    m_pathlist = [[] for i in range(8)]
    
    for line in lines:
        path, age, gen = line.split()
        age = int(age)
        if 0<=age<4:
            if gen == "f":
                f_pathlist[0].append(path)
            else:
                m_pathlist[0].append(path)
        elif 4<=age<=8:
            if gen == "f":
                f_pathlist[1].append(path)
            else:
                m_pathlist[1].append(path)
        elif 8<age<15:
            if gen == "f":
                f_pathlist[2].append(path)
            else:
                m_pathlist[2].append(path)
        elif 15<=age<25:
            if gen == "f":
                f_pathlist[3].append(path)
            else:
                m_pathlist[3].append(path)
        elif 25<=age<38:
            if gen == "f":
                f_pathlist[4].append(path)
            else:
                m_pathlist[4].append(path)
        elif 38<=age<48:
            if gen == "f":
                f_pathlist[5].append(path)
            else:
                m_pathlist[5].append(path)
        elif 48<=age<53:
            if gen == "f":
                f_pathlist[6].append(path)
            else:
                m_pathlist[6].append(path)
        elif 53<=age:
            if gen == "f":
                f_pathlist[7].append(path)
            else:
                m_pathlist[7].append(path)
        else:
            continue
    pathlist = []
    labellist = []
    genlist = []
    for c in range(8):
        print(n_sample_list[c][0])
        print(len(f_pathlist[c]))
        print(n_sample_list[c][1])
        print(len(m_pathlist[c]))
        pathlist += random.sample(f_pathlist[c], n_sample_list[c][0])
        pathlist += random.sample(m_pathlist[c], n_sample_list[c][1])
        labellist += [c for i in range(n_sample_list[c][0] + n_sample_list[c][1])]
        genlist += [0 for i in range(n_sample_list[c][0])]
        genlist += [1 for i in range(n_sample_list[c][1])]
    transform = transform
    print(len(pathlist))
    print(len(labellist))
    print(len(genlist))

    class MyDataset(Dataset):
        def __init__(self, pathlist, labellist, genlist, transform=None):
            self.pathlist = pathlist
            self.labellist = labellist
            self.genlist = genlist
            self.transform = transform
        def __len__(self):
            return len(self.pathlist)
        
        def __getitem__(self, idx):
            img_path = self.pathlist[idx]
            label = self.labellist[idx]
            gen = self.genlist[idx]
            img = Image.open(os.path.join("./datasets/faces", img_path))
            if self.transform:
                img = self.transform(img)
            return img, label, gen
    return MyDataset(pathlist, labellist, genlist, transform=transform)