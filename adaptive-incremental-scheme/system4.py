import numpy as np

# number of variables
n = 2

f = lambda t, y:        np.array([- 10 * y[0] + 100 * y[1],
                                  - y[1]])
y = lambda t:           np.array([1 / 9 * np.exp(-10 * t) * (100 * np.exp(9 * t) - 91),
                                  np.exp(-t)])
J_y = lambda t, y:      np.array([[-10, 100], [0, -1]])
J_t = lambda t, y:      np.array([0, 0])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)
f_n = f
dfdt_n = dfdt

f0 = lambda t, y: 10 * y[0] - 100 * y[1]
f1 = lambda t, y: y[1]

F = [f0, f1]