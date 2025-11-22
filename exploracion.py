import pandas as pd
import re
import numpy as np

def get_fechas_autor(txt):
    res = re.search(r"(\d+)\D+(\d+)", txt)
    if res:
        n1 = res.group(1)
        n2 = res.group(2)
        if n1 < n2:
            return n1, n2
        print(n1, n2, txt)
    print(txt)
    

def clean_date(txt):
    res= re.search(r"(\d\d|\d)th",txt)
    if res:
        return int(res.group(1))*100
    res = re.search(r"(\d+)(\D+)(\d+)",txt) # primer mirem que no sigui un interval d'anys
    if res:
        n1 = int(res.group(1))
        n2 = int(res.group(3))
        if abs(n1 - n2)/n1 < 0.1:
            return (n1 + n2)/2
        else:
            n1 = str(n1)
            n2 = str(n2)
            if len(n2) != 2:
                return int(res.group(1)) + int(res.group(3))
            return int((int(n1)+int(n1[:-2] + n2))/2)
    res = re.search(r"(\d+)", txt)
    if res:
        return int(res.group())
    return np.nan

def dentro(anno, timeline):
    res = re.search(r"(\d+)-(\d+)", timeline)
    if res:
        n1 = int(res.group(1))
        n2 = int(res.group(2))
        if anno:
            if n1 <= anno <= n2:
                return 0
            else:
                return min(abs(anno - n1), abs(anno - n2))
    return np.nan
columnas = ["ID", "FILE", "AUTHOR", "BORN-DIED", "DATE","FORM", "TIMELINE"]
df = pd.read_csv("labels.csv", sep="\t", usecols=columnas)
df = df.loc[df["FORM"] == "painting"].copy()
df["CLEAN_DATE"] = df["DATE"].apply(clean_date)
print(df.loc[((df["CLEAN_DATE"].isna()) & (df["DATE"] != "-"))])
#print(df["CLEAN_DATE"].sort_values().head(20))
#print(df.isna().sum())
df["DENTRO"] = df.apply(lambda x: dentro(x["CLEAN_DATE"], x["TIMELINE"]), axis=1)
print(df["BORN-DIED"].apply(get_fechas_autor).isna().sum())
print(df.sort_values(by="DENTRO", ascending=False).head(20))

