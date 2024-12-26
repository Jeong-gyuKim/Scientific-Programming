"""
Project Name    :Scientific Programming HW

File Name       :plot.py
Date            :2024.11.07
Author          :Jeong-gyu, Kim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

x_1, x_2 = 22,32 #graph in graph range by bin index

#def plot function
def error_plot(ax, x, y, sigma=2., alpha=.75, fmt="None", capsize=2, capthick=1,*args,**kwargs):
    ax.errorbar(x,y,sigma*np.sqrt(y), alpha=alpha,fmt=fmt,capsize=capsize,capthick=capthick,*args,**kwargs)

#load data 
df = pd.read_csv("output.csv")
x, PDF, WOS, WOP = df.to_numpy().T

#plot graph
f, ax = plt.subplots(1, facecolor="#F0F0F0", dpi=500)
ax.plot(x, PDF, label='PDF', color='r', linestyle='dashed')
error_plot(ax, x, WOS, label='WOS(C.L 95%)')
error_plot(ax, x, WOP, label='WOP(C.L 95%)', color='g')

#plot graph in graph
axins = zoomed_inset_axes(ax, zoom=5, loc='right', borderpad = 1.0)
error_plot(axins, x, WOS)
error_plot(axins, x, WOP, color='g')
axins.plot(x, PDF, label='PDF', color='r', linestyle='dashed')
axins.set(xlim=[min(x[x_1],x[x_2]), max(x[x_1],x[x_2])], ylim=[min(PDF[x_1], PDF[x_2]), max(PDF[x_1], PDF[x_2])])
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='gray')

#setting
ax.legend()
ax.set(xlabel="radius", 
       ylabel="counts[#]", 
       title = f"charge distribution\n#N = {df.columns[0]}")

plt.savefig("Figure_3.png")
