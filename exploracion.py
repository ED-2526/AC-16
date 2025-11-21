import pandas as pd
import re
import numpy as np

def clean_date(txt):
    res = re.search("(\d+)(\D+)(\d+)",txt) # primer mirem que no sigui un interval d'anys
    if res:
        n1 = int(res.group(1))
        n2 = int(res.group(3))
        if (abs(n1 - n2) / n1) < 0.1:
            return (n1 + n2)/2
        else:
            n1 = str(n1)
            n2 = str(n2)
            if len(n2) != 2:
                print(txt)
            return int(n1[:-2] + n2)
    res = re.search("(\d+)", txt)
    if res:
        return int(res.group())
    return np.nan

columnas = ["ID", "FILE", "DATE","FORM", "TIMELINE"]
df = pd.read_csv("labels.csv", sep="\t", usecols=columnas)
df = df.loc[df["FORM"] == "painting"].copy()
df["CLEAN_DATE"] = df["DATE"].apply(clean_date)
print(df.loc[(df["CLEAN_DATE"].isna())])

