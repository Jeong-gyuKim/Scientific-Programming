"""
Project Name    :Scientific Programming HW

File Name       :potential.py
Date            :2024.10.16
Author          :Jeong-gyu, Kim
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

#def class
class particle:
    #def init condition
    def __init__(self, epsilon=1e-6, x=0,y=0,z=1) -> None:
        self.epsilon = epsilon
        
        self.x, self.y, self.z = x,y,z
        if z < epsilon:
            self.passage = True
        else:
            self.passage = False
    
    #def add
    def __add__(self, tuple):
        self.x += tuple[0]
        self.y += tuple[1]
        self.z += tuple[2]
        if self.z < self.epsilon:
            self.passage = True
        
        return self
    
    # def set_pos(self, x,y,z):
    #     self.x, self.y, self.z = x,y,z
            
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
    phi = np.arccos(2*np.random.rand()-1)
    theta = 2*np.pi*np.random.rand()
    return sphere2cartesian(r,theta, phi)

#do one particle sim.
def get_r(epsilon=1e-6, x=0,y=0,z=1):
    a = particle(epsilon, x,y,z)
    while not a.passage:
        a += unit_sphere_sample(a.z)
    r = np.sqrt(a.x**2+a.y**2)
    return r    

#get bin index
def binning(x, len=10, dx=1):
    x = x//dx
    if x >= len:
        x = len - 1
    return int(x)
################################################################
#function define WOP
def invCDF(u):
    return np.sqrt((1/(1-u))**2 -1)

################################################################
#test code

#3D plot
def plot3d(X,Y,Z, xlabel='', ylabel='', zlabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X,Y,Z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    ax.set_xlim(-1.25,1.25)
    ax.set_ylim(-1.25,1.25)
    ax.set_zlim(-1,1)
    plt.show()

#for see the uniform sampling
def test_uniform_sphere_sample(f):
    X,Y,Z = [],[],[]
    for _ in range(2000):
        x,y,z = f()
        X.append(x)
        Y.append(y)
        Z.append(z)
    plot3d(X,Y,Z,
        xlabel='X Label',ylabel='Y Label', zlabel='Z Label')

#test_uniform_sphere_sample(unit_sphere_sample)
################################################################
#main
N = 50000
l = 25
n = 125

len = n
dx = l/n

f, ax = plt.subplots(1, facecolor="#F0F0F0", dpi=500)

#WOS
arr1 = np.zeros(len)
x = np.array([i*dx for i in range(len)])
for _ in range(N):
    r = get_r(1e-6)
    r = binning(r,len,dx)
    arr1[r] += 1
darr1 = np.sqrt(arr1)
#print(f"is empty bin exist?: {0 in arr}\n" ,arr)
ax.errorbar(x[:-1],arr1[:-1],darr1[:-1],# 0.5*dx*np.ones_like(x[:-1]), 
             alpha=.75, fmt="None", capsize=2, capthick=1,
             label='WOS')

#WOP
arr2 = np.zeros(len)
x = np.array([i*dx for i in range(len)])
for _ in range(N):
    r = invCDF(np.random.rand())
    r = binning(r,len,dx)
    arr2[r] += 1
darr2 = np.sqrt(arr2)
#print(f"is empty bin exist?: {0 in arr}\n" ,arr)
ax.errorbar(x[:-1],arr2[:-1],darr2[:-1],# 0.5*dx*np.ones_like(x[:-1]), 
             alpha=.75, fmt="None", capsize=2, capthick=1,
             label='WOP', color='g')

#PDF
range = np.linspace(0,l, 5000)
real = np.array([prob(i,i+dx) for i in range])
ax.plot(range,real*N, label='PDF', color='r', linestyle='dashed')

#Zoom
axins = zoomed_inset_axes(ax, zoom=5, loc='right', borderpad = 1.0)
axins.errorbar(x[:-1],arr1[:-1],darr1[:-1],# 0.5*dx*np.ones_like(x[:-1]), 
             alpha=.75, fmt="None", capsize=2, capthick=1,
             label='WOS')
axins.errorbar(x[:-1],arr2[:-1],darr2[:-1],# 0.5*dx*np.ones_like(x[:-1]), 
             alpha=.75, fmt="None", capsize=2, capthick=1,
             label='WOP', color='g')
axins.plot(range,real*N, label='PDF', color='r', linestyle='dashed')

for s in ['top', 'bottom', 'left', 'right']:
    axins.spines[s].set(color='grey', lw=1, linestyle='solid')
x_1, x_2 = 4.5, 6.0
axins.set(xlim=[min(x_1,x_2), max(x_1,x_2)], ylim=[min(prob(x_1,x_1+dx)*N, prob(x_2,x_2+dx)*N), max(prob(x_1,x_1+dx)*N, prob(x_2,x_2+dx)*N)])
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='gray')

#plot
ax.legend()
ax.set(xlabel="radius", 
       ylabel="counts[#]", 
       title = f"charge distribution\n#N = {N}")
plt.savefig("Figure_3.png")
