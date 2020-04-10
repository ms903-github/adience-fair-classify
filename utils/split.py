import os
import shutil
import glob

# 前提：データセットは./dataset/(classnum)/*.jpgの形で与えられていること
#     ./dataset/train/(classnum), ./dataset/test/(classnum)のディレクトリが用意されていること
# クラスフォルダが数字でなくアルファベットの場合，rename.pyで数字に変換する

base_dir = "./caltech101"
tr_dir = "./caltech101_sp/train"
te_dir = "./caltech101_sp/test"

path = base_dir
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
files_dir.sort()
print(files_dir)

for files in files_dir:
    pathlist = glob.glob(os.path.join(base_dir, files, "*.jpg"))
    dnum = len(pathlist)
    print(dnum)
    tr_pathlist = pathlist[:int(dnum*0.7)]
    te_pathlist = pathlist[int(dnum*0.7):]
    for path in tr_pathlist:
        shutil.copy(path, os.path.join(tr_dir, str(int(files))))
    for path in te_pathlist:
        shutil.copy(path, os.path.join(te_dir, str(int(files))))
