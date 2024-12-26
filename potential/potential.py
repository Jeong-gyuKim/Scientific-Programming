"""
Project Name    :Scientific Programming HW

File Name       :potential.py
Date            :2024.11.07
Author          :Jeong-gyu, Kim
"""

import numpy as np
import pandas as pd
from functions import get_real, sampling, get_wos, invCDF 

#main
N = 500000000 #MC step
l = 25 #total radius to concern
n = 125 #bin number

df = pd.DataFrame()

#bin
x = np.linspace(0,l,n)
df[f"{N}"] = x[:-1]

#PDF
PDF = np.array(get_real(x))*N
df["PDF"] = PDF

#WOS
WOS = sampling(get_wos,N,l,n)
df["WOS"] = WOS[:-1]

#WOP
WOP = sampling(invCDF,N,l,n)
df["WOP"] = WOP[:-1]

#save
df.to_csv("output.csv", index=False)
