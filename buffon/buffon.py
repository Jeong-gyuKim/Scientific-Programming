"""
Project Name    :Scientific Programming HW

File Name       :buffon.py
Date            :2024.09.25
Author          :Jeong-gyu, Kim
"""

import numpy as np

def buffonWithPi(n,d):#get random value with reference value of pi
    x_random = np.random.rand(n) * (d / 2)
    theta_random = np.random.rand(n) * np.pi
    
    return x_random, theta_random

def buffonWithoutPi(n,d):#get random value without reference value of pi
    x_random = np.random.rand(n) * (d / 2)
    
    x, y = np.zeros(n), np.zeros(n)
    for i in range(n):
        while 1:
            X, Y = np.random.rand(), np.random.rand()
            if X**2 + Y**2 <= 1 :
                break
        x[i], y[i] = X, Y
    theta_random = np.arctan2(x,y)
    
    return x_random, theta_random

def buffon(n,l,d,f):#길이 l짜리 바늘을 간격 d의 평행선 사이에 n번 던져서 pi값을 계산한다.
    cnt = 1

    x_random, theta_random = f(n,d)
    touches = x_random < (l / 2) * np.sin(theta_random)
    cnt += np.sum(touches)
    
    if cnt:
        pi_estimate = (2 * n * l) / (cnt * d)
    else:
        raise ValueError
    return pi_estimate

def ErrStatic(li):#N회 buffon함수를 실행한 결과로 오차의 평균과 표준편차를 반환한다.
    li = abs(li-np.pi)
    return np.mean(li), np.std(li)

def diffrence(n_list, l, d, N=30, f=buffonWithoutPi):#n_list에 들어있는 n값마다 ErrStatic을 실행한 결과를 반환한다.
    Err = np.zeros_like(n_list).astype(float)
    dErr = np.zeros_like(n_list).astype(float)
    Ns = np.zeros(len(n_list)*N)
    Pi = np.zeros(len(n_list)*N)
    for i, n in enumerate(n_list):
        li = np.array([buffon(n,l,d,f) for _ in range(N)])
        Err[i], dErr[i]= ErrStatic(li)
        for j in range(N):
            Ns[i*N+j] = n
            Pi[i*N+j] = li[j]
    return Err, dErr, Ns, Pi

def LinearRegression(x,y):#입력된 데이터에 y=a*x+b를 fitting한다.
    x, y = np.array(x), np.array(y)
    a = sum((x-np.mean(x))*(y-np.mean(y)))/sum((x-np.mean(x))**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b
