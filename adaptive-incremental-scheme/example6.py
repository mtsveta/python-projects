import math

y = lambda t: t + 1 / 4 * math.sin(2 * t) - t / 2 * math.cos(2 * t)
f = lambda t: t * math.sin(2 * t) + 1
fy = lambda t: 0
fyy = fy
fyyy = fy

ft = lambda t: 2 * t * math.cos(2 * t) + math.sin(2 * t)
ftt = lambda t: - 4 * t * math.sin(2 * t) + 4 * math.cos(2 * t)
fttt = lambda t: - 8 * t * math.cos(2 * t) - 12 * math.sin(2 * t)

fty = lambda t: 0
ftty = fty
ftyy = fty

dfdt = lambda t: f(t) * fy(t) + ft(t)
d2fdt2 = lambda t: ftt(t) + 2 * fty(t) * f(t) + fyy(t) * (f(t))**2 + fy(t) * dfdt(t)
d3fdt3 = lambda t: fttt(t) + 3 * (ftty(t) * f(t) + ftyy(t) * (f(t))**2) + fyyy(t) * (f(t))**3 + 3 * dfdt(t) * (fyy(t) * f(t) + fty(t)) + fy(t) * d2fdt2(t)

f_n = lambda tn, yn: tn * math.sin(2 * tn) + 1
fy_n = lambda tn, yn: 0
ft_n = lambda tn, yn: math.sin(2 * tn) + 2 * tn * math.cos(2 * tn)
dfdt_n = lambda tn, yn: f_n(tn, yn) * fy_n(tn, yn) + ft_n(tn, yn)
