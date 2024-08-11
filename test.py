import pandas as pd
import os, shutil

excel = r"F:\Data\AdrenalNoduleClassification\Problems\ImagesMapping.xlsx"

df = pd.read_excel(excel)

Radiology_IDs = df['Radiology_ID'].values
New_Name = df['New_Name'].values

mapping = dict(zip(New_Name, Radiology_IDs))

files = os.listdir(r"F:\Data\AdrenalNoduleClassification\Problems\Predicts")
for i in files:
    os.rename(os.path.join(r"F:\Data\AdrenalNoduleClassification\Problems\Predicts", i),
              os.path.join(r"F:\Data\AdrenalNoduleClassification\Problems\Predicts", mapping[i]))