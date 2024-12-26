"""
Project Name    :Scientific Programming HW

File Name       :buffon.py
Date            :2024.09.25
Author          :Jeong-gyu, Kim
"""

import numpy as np
from scipy.stats.qmc import Sobol

def buffonWithPi(m,d):#get random value with reference value of pi
    sampler = Sobol(d=2, scramble=False)
    sample = sampler.random_base2(m=m).T
    x_random = sample[0] * (d / 2)
    theta_random = sample[1] * np.pi
    
    return x_random, theta_random

def buffon(m,l,d,f):#길이 l짜리 바늘을 간격 d의 평행선 사이에 n번 던져서 pi값을 계산한다.
    cnt = 1

    x_random, theta_random = f(m,d)
    touches = x_random < (l / 2) * np.sin(theta_random)
    cnt += np.sum(touches)
    
    if cnt:
        pi_estimate = (2 * (2**m) * l) / (cnt * d)
    else:
        raise ValueError
    return pi_estimate

def ErrStatic(li):#N회 buffon함수를 실행한 결과로 오차의 평균과 표준편차를 반환한다.
    li = abs(li-np.pi)
    return np.mean(li), np.std(li)

def diffrence(m_list, l, d, N=30, f=buffonWithPi):#n_list에 들어있는 n값마다 ErrStatic을 실행한 결과를 반환한다.
    Err = np.zeros_like(m_list).astype(float)
    dErr = np.zeros_like(m_list).astype(float)
    Ns = np.zeros(len(m_list)*N)
    Pi = np.zeros(len(m_list)*N)
    for i, m in enumerate(m_list):
        li = np.array([buffon(m,l,d,f) for _ in range(N)])
        Err[i], dErr[i]= ErrStatic(li)
        for j in range(N):
            Ns[i*N+j] = 2**m
            Pi[i*N+j] = li[j]
    return Err, dErr, Ns, Pi

def LinearRegression(x,y):#입력된 데이터에 y=a*x+b를 fitting한다.
    x, y = np.array(x), np.array(y)
    a = sum((x-np.mean(x))*(y-np.mean(y)))/sum((x-np.mean(x))**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b
