import numpy as np

# number of variables
n = 2

f = lambda t, y:        np.array([y[1],
                                  -y[0] + t])
y = lambda t:           np.array([np.cos(t) + np.sin(t) + t,
                                  np.cos(t) - np.sin(t) + 1])
J_y = lambda t, y:      np.array([[0, 1], [-1, 0]])
J_t = lambda t, y:      np.array([0, 1])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)
f_n = f
dfdt_n = dfdt

f0 = lambda t, y: y[1]
f1 = lambda t, y: - y[0] + t

F = [f0, f1]