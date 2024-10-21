"""
Project Name    :Scientific Programming HW

File Name       :main.py
Date            :2024.09.25
Author          :Jeong-gyu, Kim
"""

from buffon import diffrence, LinearRegression
import numpy as np
import matplotlib.pyplot as plt

################################################################
#input parameter
l,d = 1,1
n_list = (10**np.arange(1,5,0.1)).astype(int)#(10**np.arange(1,6,0.1)).astype(int)

################################################################
#calculate error
err, derr, Ns, Pi = diffrence(n_list,l,d,N=30)
a,b = LinearRegression(np.log(n_list), np.log(err))

################################################################
#draw graph
#Figure_1
plt.figure()
plt.scatter(n_list, err, label="Buffon's needle")
plt.fill_between(n_list, err - derr, err + derr, alpha=.25)
plt.plot(n_list, np.exp(a*np.log(n_list)+b), label="linear regression\n(slope={})".format(round(a,3)))

plt.title("Result of the Scientific Programming HW")
plt.xlabel("# of throws")
plt.ylabel("Error")

plt.legend()
plt.loglog()

plt.savefig("Figure_1.png")

#Figure_2
plt.figure()
plt.scatter(np.log10(Ns), Pi, label="Buffon's needle", alpha=0.3)
plt.plot([1,6],[np.pi,np.pi],color='r',label="Pi")

plt.title("Result of the Scientific Programming HW")
plt.xlabel("log(# of throws)")
plt.ylabel("Estimated Pi")

plt.legend()
plt.savefig("Figure_2.png")