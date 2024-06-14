import os, glob, shutil

dirPath = "/media/sci/P44_Pro/AdrenalGland/分类"

dirs = []

for r, d, files in os.walk(dirPath):
    if len(files) > 30 and ("A1" in r or "A1.25" in r or "A1.5" in r):
        dirs.append(r)

for i in dirs:
    print(i)

mappingDict = {}

index = 1
for i in dirs:
    mappingDict[i] = f"ag_{index:>03}_0000.nii.gz"
    index += 1

import pandas as pd

df = pd.DataFrame.from_dict(mappingDict, orient="index")

df.to_excel("/media/sci/P44_Pro/AdrenalGland/Mapping.xlsx")