import math

def y(t):
    return math.exp(-t)

def f(t):
    return -y(t)

def fprime(t):
    return y(t)

def d2fdt2(t):
    return -y(t)

def d3fdt3(t):
    return y(t)

def f_n(tn, yn):
    return -yn

def fprime_n(tn, yn):
    return yn