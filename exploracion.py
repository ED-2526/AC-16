import pandas as pd
import re
import numpy as np

def clean_date(txt):
    res = re.search("(\d+)(\D+)(\d+)",txt)
    res = re.search("(\d+)(\D+)(\d+)",txt)
    if res:
        #res.group(2) != "-" and (abs((int(res.group(1)) - int(res.group(3))) / int(res.group(1))) <  0.1):
        return True
    return False

columnas = ["ID", "FILE", "DATE","FORM", "TIMELINE"]
df = pd.read_csv("labels.csv", sep="\t", usecols=columnas)
df = df.loc[df["FORM"] == "painting"].copy()
print(df.loc[df["DATE"].apply(clean_date)])

