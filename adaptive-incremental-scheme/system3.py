import numpy as np

# number of variables
n = 2

f = lambda t, y:        np.array([-y[0], -100 * y[1]])
y = lambda t:           np.array([np.exp(-t),
                                  np.exp(-100 * t)])
J_y = lambda t, y:      np.array([[-1, 0], [0, -100]])
J_t = lambda t, y:      np.array([0, 0])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)
f_n = f
dfdt_n = dfdt

f0 = lambda t, y: y[0]
f1 = lambda t, y: 100 * y[1]

F = [f0, f1]