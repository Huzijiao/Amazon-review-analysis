import csv
import pandas as pd
import numpy as np
import re
import sys
df=pd.read_csv(r"C:\Users\HP-DLC\Desktop\amazon问题分析\data_after_clean\pacifier_after_clean.csv",encoding="ISO-8859-1")
df=df.astype(str)
#丢弃市场这一栏不需要的元素
df.drop(['marketplace'],axis=1,inplace=True)
#查看估计哪些地方是0
#y=df[df["verified_purchase"].str.contains("n" or "N")]
#print(y)
#把为亚马逊说话的和买都没买就瞎评论的去掉
df=df[df["verified_purchase"]!="n"]
df=df[df["verified_purchase"]!="N"]
df=df[df["vine"]!="y"]
df=df[df["vine"]!="Y"]
df=df.to_csv(r"C:\Users\HP-DLC\Desktop\amazon问题分析\data_after_clean\pacifier_after_clean.csv",encoding="ISO-8859-1",index=0)
