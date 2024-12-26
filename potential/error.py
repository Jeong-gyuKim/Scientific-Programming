"""
Project Name    :Scientific Programming HW

File Name       :error.py
Date            :2024.11.07
Author          :Jeong-gyu, Kim
"""
import numpy as np
import pandas as pd
from functions import sampling, get_wos, invCDF, get_real
N_list = [int(10**(i/5+1)) for i in range(21)]
l = 25 #total radius to concern
n = 125 #bin number

x = np.linspace(0,l,n)
PDF = get_real(x)
index = PDF.index(max(PDF))

df = pd.DataFrame()

df["N"]=N_list
df["WOS"]=[sampling(get_wos,N,l,n)[index]/N for N in N_list]
df["WOP"]=[sampling(invCDF,N,l,n)[index]/N for N in N_list]
df["PDF"]=[PDF[index] for _ in N_list]
df.to_csv("error.csv", index=False)
