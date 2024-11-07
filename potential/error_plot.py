"""
Project Name    :Scientific Programming HW

File Name       :error_plot.py
Date            :2024.11.07
Author          :Jeong-gyu, Kim
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#load data 
df = pd.read_csv("error.csv")
N, WOS, WOP, PDF = df.to_numpy().T

def LinearRegression(x,y):#입력된 데이터에 y=a*x+b를 fitting한다.
    x, y = np.array(x), np.array(y)
    a = sum((x-np.mean(x))*(y-np.mean(y)))/sum((x-np.mean(x))**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b

def errorplot(X,Y, label):#선형회귀한 결과와 데이터를 loglog plot한다.
    a, b = LinearRegression(np.log(X), np.log(Y))
    plt.scatter(X, Y, label=label)
    plt.plot(X, np.exp(a*np.log(X)+b), label="linear regression\n(slope={})".format(round(a,3)))

#plot
plt.figure(facecolor="#F0F0F0", dpi=500)
errorplot(N,abs(WOS-PDF),"WOS")
errorplot(N,abs(WOP-PDF),"WOP")
plt.title("Result of the Scientific Programming HW")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.loglog()
plt.savefig("Figure_4.png")