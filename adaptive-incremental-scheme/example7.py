import math

def y(t):
    return t**4 + t + 1

def f(t):
    return 4 * t**3 + 1

def fy(t):
    return 0

def ft(t):
    return 12 * t**2

def fprime(t):
    return f(t) * fy(t) + ft(t)

def f_n(tn, yn):
    return 4 * tn**3 + 1

def fy_n(tn, yn):
    return 0

def ft_n(tn, yn):
    return 12 * tn**2

def fprime_n(tn, yn):
    return f_n(tn, yn) * fy_n(tn, yn) + ft_n(tn, yn)