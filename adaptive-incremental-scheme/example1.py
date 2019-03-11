import math

y = lambda t: math.exp(-t)
f = lambda t: -y(t)
dfdt = lambda t: y(t)
d2fdt2 = lambda t: -y(t)
d3fdt3 = lambda t: y(t)

f_n = lambda tn, yn: -yn
dfdt_n = lambda tn, yn: yn

