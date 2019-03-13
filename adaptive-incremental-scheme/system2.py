import numpy as np
# Example from
# Two derivative Runge Kutta methods with minimum phase-lag and amplification error
# Z. Kalogiratou, Th. Monovasilis, and T. E. Simos
# 2018

# number of variables
n = 2

f = lambda t, y:        np.array([y[1], -100 * y[0] + 99 * np.sin(t)])
y = lambda t:           np.array([np.cos(10 * t) + np.sin(10 * t) + np.sin(t),
                                  -10 * np.sin(10 * t) + 10 * np.cos(10*t) + np.cos(t)])
J_y = lambda t, y:      np.array([[0, 1], [-100, 0]])
J_t = lambda t, y:      np.array([0, 99 * np.cos(t)])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)
f_n = f
dfdt_n = dfdt

f0 = lambda t, y: y[1]
f1 = lambda t, y: -100 * y[0] + 99 * np.sin(t)

F = [f0, f1]