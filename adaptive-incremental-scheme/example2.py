import math

def y(t):
    return t*math.sin(math.pi*t)

def f(t):
    return -y(t)

def fprime(t):
    return y(t)

def f_n(yn):
    return -yn

def fprime_n(yn):
    return yn