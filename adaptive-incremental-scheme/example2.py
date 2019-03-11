import math

y = lambda t: math.exp(-10*t)
f = lambda t: -10 * y(t)
fy = lambda t: -10
fyy = lambda t: 0
fyyy = lambda t: 0


dfdt = lambda t: f(t) * fy(t)
d2fdt2 = lambda t: (f(t))**2 * fyy(t) + f(t) * (fy(t))**2
d3fdt3 = lambda t: (f(t))**3 * fyyy(t) + 4*(f(t))**2 * fyy(t) * fy(t) + f(t) * (fy(t))**3

f_n = lambda tn, yn: -10*yn
dfdt_n = lambda tn, yn:  100*yn
