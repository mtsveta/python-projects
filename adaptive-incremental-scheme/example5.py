import math

y = lambda t: math.exp(-2 * t) + 2 / 5 * math.sin(t) - 1 / 5 * math.cos(t)
f = lambda t: - 2 * y(t) + math.sin(t)
fy = lambda t: - 2
fyy = lambda t: 0
fyyy = fyy

ft = lambda t: math.cos(t)
ftt = lambda t: -math.sin(t)
fttt = lambda t: -math.cos(t)

fty = lambda t: 0
ftty = fty
ftyy = fty

dfdt = lambda t: f(t) * fy(t) + ft(t)
d2fdt2 = lambda t: ftt(t) + 2 * fty(t) * f(t) + fyy(t) * f(t)**2 + fy(t) * dfdt(t)
d3fdt3 = lambda t: fttt(t) + 3 * (ftty(t) * f(t) + ftyy(t) * f(t)**2) + fyyy(t) * f(t)**3 + 3 * dfdt(t) * (fyy(t) * f(t) + fty(t)) + fy(t) * d2fdt2(t)

f_n = lambda tn, yn: - 2 * yn + math.sin(tn)
fy_n = lambda tn, yn: - 2
ft_n = lambda tn, yn: math.cos(tn)
dfdt_n = lambda tn, yn: f_n(tn, yn) * fy_n(tn, yn) + ft_n(tn, yn)