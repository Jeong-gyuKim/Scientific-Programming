"""
Project Name    :Scientific Programming HW

File Name       :error.py
Date            :2024.11.07
Author          :Jeong-gyu, Kim
"""
import numpy as np
from potential import sampling, get_wos, invCDF, get_real
N_list = [int(10**(i/5+1)) for i in range(21)]
l = 25 #total radius to concern
n = 125 #bin number

x = np.linspace(0,l,n)
PDF = get_real(x)
index = PDF.index(max(PDF))

with open("error.csv","w") as out:
    out.writelines("N,WOS,WOP,PDF\n")
    for N in N_list:
        out.writelines(f"{N},{sampling(get_wos,N,l,n)[index]/N},{sampling(invCDF,N,l,n)[index]/N},{PDF[index]}\n")
