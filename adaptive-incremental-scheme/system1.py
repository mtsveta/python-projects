import numpy as np

# number of variables
n = 2

f = lambda t, y:        np.array([y[1], -y[0]])
y = lambda t:        np.array([np.sin(t), np.cos(t)])

J_y = lambda t, y:      np.array([[0, 1], [-1, 0]])
J_t = lambda t, y:      np.array([0, 0])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)
f_n = lambda t, y:      np.array([y[1], -y[0]])
dfdt_n = lambda t, y: np.dot(J_y(t, y), f(t, y)) + J_t(t, y)

#H = lambda t, y:    np.zeros((n, n, n))
#d2fdt2 = lambda t, y:    np.dot(f(t, y), f(t, y)) * H(t, y) + f((t, y)) * np.dot(J(t, y), J(t, y))
#d3fdt3 = lambda t: (f(t))**3 * fyyy(t) + 4*(f(t))**2 * fyy(t) * fy(t) + f(t) * (fy(t))**3

f0 = lambda t, y: y[1]
f1 = lambda t, y: - y[0]

F = [f0, f1]