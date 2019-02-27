import math

def y(t):
    return t + 1 / 4 * math.sin(2 * t) - t / 2 * math.cos(2 * t)

def f(t):
    return t * math.sin(2 * t) + 1

def fy(t):
    return 0

def ft(t):
    return math.sin(2 * t) + 2 * t * math.cos(2 * t)

def fprime(t):
    return f(t) * fy(t) + ft(t)

def f_n(tn, yn):
    return tn * math.sin(2 * tn) + 1

def fy_n(tn, yn):
    return 0

def ft_n(tn, yn):
    return math.sin(2 * tn) + 2 * tn * math.cos(2 * tn)

def fprime_n(tn, yn):
    return f_n(tn, yn) * fy_n(tn, yn) + ft_n(tn, yn)