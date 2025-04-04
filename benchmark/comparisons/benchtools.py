import math
import time
import numpy as np

def distributeZ2(D, p = 0.5):
    D0 = math.ceil(p*D)
    D1 = D - D0
    return [(0, D0), (1, D1)]

def distributeU1(D, p = 0.25):
    return distributeU1_poisson(D, p = p)

def distributeU1_exponential(D, p = 0.25):
    λ = (1-p)/(1+p)
    D0 = math.ceil(p*D)
    if (D-D0) % 2 == 1 :
        D0 = 2 if D0 == 1 else D0-1
    sectors = [(0, D0)]
    Drem = D - D0
    n = 1
    while Drem > 0 :
        pn = p * λ**n
        Dn = math.ceil(pn*D)
        sectors = sectors + [(n, Dn), (-n, Dn)]
        Drem -= 2*Dn
        n += 1
    sectors.sort(key = lambda d : d[0])
    return sectors

def distributeU1_poisson(D, p = 0.25):
    λ = math.log((1/p+1)/2)
    D0 = math.ceil(p*D)
    if (D-D0) % 2 == 1 :
        D0 = 2 if D0 == 1 else D0-1
    sectors = [(0, D0)]
    Drem = D - D0
    n = 1
    while Drem > 0 :
        pn = p * λ**n / math.factorial(n)
        Dn = math.ceil(pn*D)
        sectors = sectors + [(n, Dn), (-n, Dn)]
        Drem -= 2*Dn
        n += 1
    sectors.sort(key = lambda d : d[0])
    return sectors

def donothing(x):
    x

def timer(f, inner = 1, outer = 1):
    f()
    times = np.zeros(outer)
    for i in range(0, outer) :
        if inner == 1 :
            start = time.time_ns()
            donothing(f())
            stop = time.time_ns()
        else :
            start = time.time_ns()
            for j in range(0, inner) :
                donothing(f())
            stop = time.time_ns()
        times[i] = (stop-start)/1e9/inner
    return times
