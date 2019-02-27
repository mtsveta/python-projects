import math

def y(t):
    return math.exp(-2 * t) + 2 / 5 * math.sin(t) - 1 / 5 * math.cos(t)

def f(t):
    return - 2 * y(t) + math.sin(t)

def fy(t):
    return - 2

def ft(t):
    return math.cos(t)

def fprime(t):
    return f(t) * fy(t) + ft(t)

def f_n(tn, yn):
    return - 2 * yn + math.sin(tn)

def fy_n(tn, yn):
    return - 2

def ft_n(tn, yn):
    return math.cos(tn)

def fprime_n(tn, yn):
    return f_n(tn, yn) * fy_n(tn, yn) + ft_n(tn, yn)