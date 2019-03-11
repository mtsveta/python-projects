import numpy as np

# number of variables
n = 2

f = lambda t, y:        np.array([y[1], -y[0]])
y = lambda t, y:        np.array([np.sin(t), np.cos(t)])
J = lambda t, y:        np.array([[0, 1], [-1, 0]])
dfdt = lambda t, y:    np.dot(J(t, y), f_n(t, y))
f_n = lambda t, y:      np.array([y[1], -y[0]])
dfdt_n = lambda t, y: np.dot(J(t, y), f_n(t, y))
