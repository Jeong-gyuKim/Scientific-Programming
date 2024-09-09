import numpy as np
import matplotlib.pyplot as plt

def buffonWithPi(n,l,d):
    '''
    n: 시행 횟수
    l: 바늘 길이
    d: 평행선 간격
    '''

    x_random = np.random.rand(n) * (d / 2)
    theta_random = np.random.rand(n) * np.pi

    touches = x_random < (l / 2) * np.sin(theta_random)
    cnt = np.sum(touches)
    
    pi_estimate = (2 * n * l) / (cnt * d)
    
    return pi_estimate

def buffonWithoutPi(n,l,d):
    '''
    n: 시행 횟수
    l: 바늘 길이
    d: 평행선 간격
    '''
    
    return 0

def static(n,l,d,N=30,f=buffonWithPi):
    li = [f(n,l,d) for _ in range(N)]
    return np.mean(li), np.std(li)

def diffReal(n_list, l, d, N=30, f=buffonWithPi):
    M = np.zeros_like(n_list).astype(float)
    for i, n in enumerate(n_list):
        m, s = static(n,l,d,N,f)
        M[i] = m
    plt.scatter((n_list),(abs(M-np.pi)))
    plt.title("Buffon's needle")
    plt.xlabel("log10(n)")
    plt.ylabel("log10(Err)")
    plt.loglog()
    plt.show()
    return M
