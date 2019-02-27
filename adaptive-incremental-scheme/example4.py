import math

def y(t):
    return (math.exp(t + 1) - 2) / (math.exp(t + 1) - 1)

def f(t):
    return (y(t) - 2) * (y(t) - 1)

def fy(t):
    return 2 * y(t) - 3

def fprime(t):
    return f(t) * fy(t)

def f_n(tn, yn):
    return (2 - yn) * (1 - yn)

def fy_n(tn, yn):
    return 2 * yn - 3

def fprime_n(tn, yn):
    return f_n(tn, yn) * fy_n(tn, yn)