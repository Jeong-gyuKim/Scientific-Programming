import numpy as np
from datetime import timedelta

N = 5e7
N_list = np.array(10**(np.linspace(2,np.log10(N),20)),dtype=int)

iter = sum(N_list)
sec = 0.00048*iter
td = timedelta(seconds=sec)

print(td)