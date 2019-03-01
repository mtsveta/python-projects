import math

def y(t):
    return math.exp(-100*t)

def f(t):
    return -100*y(t)

def fprime(t):
    return 10000*y(t)

def f_n(tn, yn):
    return -100*yn

def fprime_n(tn, yn):
    return 10000*yn