import glob
import os
import random
import shutil
import argparse
import pandas as pd
import re

f0 = open("./datasets/fold_0_data.txt", "r")
f1 = open("./datasets/fold_1_data.txt", "r")
f2 = open("./datasets/fold_2_data.txt", "r")
f3 = open("./datasets/fold_3_data.txt", "r")
f4 = open("./datasets/fold_4_data.txt", "r")

lines_te = f4.readlines()[1:]
lines_tr = f0.readlines()[1:]
lines_tr += f1.readlines()[1:]
lines_tr += f2.readlines()[1:]
lines_tr += f3.readlines()[1:]

n_lines_tr = []
n_lines_te = []

for line in lines_tr:
    line.replace("\t", " ")
    fid= line.split()[0]
    imid = line.split()[1]
    pid = line.split()[2]
    age_l = line.split()[3]
    age_h = line.split()[4]
    if not "(" in age_l:
        continue
    age_l = age_l.strip("(").strip(",")
    age_h = age_h.strip(")")
    age = int(age_h) + int(age_l) // 2
    gen = line.split()[5]
    path = fid + "/" + "coarse_tilt_aligned_face." + pid + "." + imid
    n_lines_tr.append(path + " " + str(age) + " " + gen + "\n")

for line in lines_te:
    line.replace("\t", " ")
    fid= line.split()[0]
    imid = line.split()[1]
    pid = line.split()[2]
    age_l = line.split()[3]
    age_h = line.split()[4]
    if not "(" in age_l:
        continue
    age_l = age_l.strip("(").strip(",")
    age_h = age_h.strip(")")
    age = int(age_h) + int(age_l) // 2
    gen = line.split()[5]
    path = fid + "/" + "coarse_tilt_aligned_face." + pid + "." + imid
    n_lines_te.append(path + " " + str(age) + " " + gen + "\n")
    

f_tr = open("./datasets/tr_data4.txt", "w")
f_te = open("./datasets/te_data4.txt", "w")
f_tr.writelines(n_lines_tr)
f_te.writelines(n_lines_te)
