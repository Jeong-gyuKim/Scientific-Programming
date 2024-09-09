from buffon import buffonWithPi, buffonWithoutPi, static, diffReal
import numpy as np
import matplotlib.pyplot as plt

n=10**7
l,d = 1,1
n_list = (10**np.arange(1,6,0.1)).astype(int)

diffReal(n_list,l,d, N=300)