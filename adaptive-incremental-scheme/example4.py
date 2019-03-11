import math

y = lambda t: (math.exp(t + 1) - 2) / (math.exp(t + 1) - 1)
f = lambda t: (y(t) - 2) * (y(t) - 1)
fy = lambda t: 2 * y(t) - 3
fyy = lambda t: 2
fyyy = lambda t: 0

dfdt = lambda t: f(t) * fy(t)
d2fdt2 = lambda t: (f(t))**2 * fyy(t) + f(t) * (fy(t))**2
d3fdt3 = lambda t: (f(t))**3 * fyyy(t) + 4*(f(t))**2 * fyy(t) * fy(t) + f(t) * (fy(t))**3

f_n = lambda tn, yn: (2 - yn) * (1 - yn)
fy_n = lambda tn, yn: 2 * yn - 3
dfdt_n = lambda tn, yn: f_n(tn, yn) * fy_n(tn, yn)
