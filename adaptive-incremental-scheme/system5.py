import numpy as np

# number of variables
n = 2

# solution
y = lambda t:           np.array([np.cos(t) + np.sin(t) + t,
                                  np.cos(t) - np.sin(t) + 1])

# rhs
f0 = lambda t, y: y[1]
f1 = lambda t, y: - y[0] + t
f = lambda t, y: np.array([ y[1],
                           -y[0] + t])
F = [f0, f1]

# jacobian of rhs
Jf0_y = lambda t, y: np.array([0, 1])
Jf1_y = lambda t, y: np.array([-1, 0])
J_y = lambda t, y:      np.array([[0, 1],
                                  [-1, 0]])
JF_y = [Jf0_y, Jf1_y]

# derivative wrt to t
J_t = lambda t, y:      np.array([0, 1])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)

df0dt = lambda t, y: - y[0] + t
df1dt = lambda t, y: - y[1] + 1

dFdt = [df0dt,
        df1dt]

f_n = f
dfdt_n = dfdt