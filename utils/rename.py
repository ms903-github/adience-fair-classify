import os
import numpy as np
import json

path = "./"
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
files_dir.sort()
print(files_dir)

idx = np.arange(110).tolist()
idx = [str(i) for i in idx]
idx2label = dict(zip(idx, files_dir))
print(idx2label)
with open("./idx2label.json", "w") as f:
    json.dump(idx2label, f)

for i, files in enumerate(files_dir):
    os.rename(files, str(i))
