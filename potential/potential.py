import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


#def class
class particle:
    #def init condition
    def __init__(self):
        self.epsilon = 1e-6
        self.passage = False
        self.x, self.y, self.z = 0,0,1
    
    #def add
    def __add__(self, tuple):
        self.x += tuple[0]
        self.y += tuple[1]
        self.z += tuple[2]
        if self.z < self.epsilon:
            self.passage = True
        
        return self

# load C
sprng = ctypes.CDLL("./sprng_interface.so")

# C funtion
sprng.init_sprng_stream.argtypes = []
sprng.init_sprng_stream.restype = None

sprng.get_sprng_random.argtypes = []
sprng.get_sprng_random.restype = ctypes.c_double

sprng.get_sprng_random_int.argtypes = []
sprng.get_sprng_random_int.restype = ctypes.c_int

def rand():
    return sprng.get_sprng_random()

def randint():
    return sprng.get_sprng_random_int()

# function
def sampling(f,N,l,n):
    arr = np.zeros(n)
    for _ in range(N):
        r = f()
        r = binning(r,n,l/n)
        arr[r] += 1
    return arr

def get_real(x):
    arr = [prob(x[i],x[i+1]) for i in range(len(x)-1)]
    return arr
################################################################
#ref. by physics CDF
"""
 By helmholtz theorem, The vector can be uniquly define 
when the curl, divergence, and boundary condition are given.

 In case of finding "charge density on an infinite conducting plane 
induced by an electric charge".
 boundary condition of electric potential(V) is defined by
V = 0 at z=0 and r->inf

 By the maxwell's equations and boundary condition,
Electric field and Magnetic field are unique.
Electric field is -gradiant of electric potential(V). 
So, Electric potential also unique.
Then we can guess another situation which makes same boundary condition 
makes same Electric potential in the given boundary!

 According this concept, we can use the image charge.
for this case, real charge q is located at (0,0,d).
to make V=0 at z=0 image charge -q will be located at (0,0,-d).

 By using image charge, get the electric potential will become much easier.
just add two electric potential produed by point charge.
$$V(x,y,z) = k \frac{q}{\sqrt{x^{2}+y^{2}+(z-d)^{2}}} 
            + k \frac{-q}{\sqrt{x^{2}+y^{2}+(z+d)^{2}}}$$

then, we have one more question how can we get 
charge density($\sigma$) by electric potential(V)?

by the gauss law, one of the maxwell's equation
set the cylinder has symmetry at origin and the top, bottom plane makes
parallel to z=0 plane. 
then integrated (electric field dot area) 
= integrated ((charge density / epsilon not) dot area)
which means that 
electric field = charge density / epsilon_{0} 
(direction of unit normal vector = z axis)
and also electric field = -gradiant of electric potential

By these relation, 
\sigma(x,y) = - \epsilon_{0} \frac{\partial V}{\partial z} (at z=0)
    = - \epsilon_{0} k q 
    (\frac{z+d}{(x^{2}+y^{2}+(z-d)^{2})^{3/2}} 
    -\frac{z-d}{(x^{2}+y^{2}+(z+d)^{2})^{3/2}}) (at z=0)
    = - 2 \epsilon_{0} k q d \frac{1}{(x^{2}+y^{2}+d^{2})^{3/2}}
let r = \sqrt{x^{2}+y^{2}}
\sigma(r) = - 2 \epsilon_{0} k q d \frac{1}{(r^{2}+d^{2})^{3/2}}

by area integrate infinite plate with charge density \sigma(r)
\int_a^b \int\limits_0^{2 \pi} \sigma(r) dot r dr dtheta
= - 2 \epsilon_{0} k q d * 2 \pi * (-\frac{1}{(r^{2}+d^{2})^{1/2}})|_a^b
= - 2 \epsilon_{0} k q d * 2 \pi * (\frac{1}{(a^{2}+d^{2})^{1/2}}-\frac{1}{(b^{2}+d^{2})^{1/2}})
= - q d * (\frac{1}{(a^{2}+d^{2})^{1/2}}-\frac{1}{(b^{2}+d^{2})^{1/2}})
since k = 1/(4 \pi \epsilon_{0})
total induced charge by a=0, b=inf to get -q

to make simple problem, WLOG d[m]=1[unit of length], -q[C]=1[unit of charge]
then \frac{1}{(a^{2}+1)^{1/2}} - \frac{1}{(b^{2}+1)^{1/2}}
"""

#probability between a to b
def prob(a,b):
    f = lambda r:(1+r**2)**-0.5
    return f(a)-f(b)

################################################################
#function define WOS
#sph->cart coord. transform function
def sphere2cartesian(r,theta, phi):
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return x, y, z

#sample diffusion process
def unit_sphere_sample(r=1):
    phi = np.arccos(2*rand()-1)
    theta = 2*np.pi*rand()
    return sphere2cartesian(r,theta, phi)

#do one particle sim.
def get_wos():
    a = particle()
    while not a.passage:
        a += unit_sphere_sample(a.z)
    r = np.sqrt(a.x**2+a.y**2)
    return r    

#get bin index
def binning(x, n=10, dx=1):
    x = x//dx
    if x >= n:
        x = n - 1
    return int(x)
################################################################
#function define WOP
def invCDF():
    return np.sqrt((1/(1-rand()))**2 - 1)
################################################################

#main
N = 5000 #MC step
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

################################################################
#plot
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
