import math

y = lambda t: math.exp(t) / (2 + math.exp(t))
f = lambda t: y(t) * (1 - y(t))
fy = lambda t:  1 - 2 * y(t)
fyy = lambda t: -2
fyyy = lambda t: 0

dfdt = lambda t: f(t) * fy(t)
d2fdt2 = lambda t: (f(t))**2 * fyy(t) + f(t) * (fy(t))**2
d3fdt3 = lambda t: (f(t))**3 * fyyy(t) + 4*(f(t))**2 * fyy(t) * fy(t) + f(t) * (fy(t))**3

f_n = lambda tn, yn: yn * (1 - yn)
fy_n = lambda tn, yn: 1 - 2 * yn
dfdt_n = lambda tn, yn: f_n(tn, yn) * fy_n(tn, yn)
