import numpy as np

# number of variables
n = 2

# solution
y = lambda t:           np.array([1 / 9 * np.exp(-10 * t) * (100 * np.exp(9 * t) - 91),
                                  np.exp(-t)])
# rhs
f0 = lambda t, y: - 10 * y[0] + 100 * y[1]
f1 = lambda t, y: - y[1]

f = lambda t, y:  np.array([- 10 * y[0] + 100 * y[1],
                            - y[1]])
F = [f0, f1]

# jacobian of rhs
Jf0_y = lambda t, y: np.array([-10, 100])
Jf1_y = lambda t, y: np.array([0, -1])

J_y = lambda t, y:      np.array([[-10, 100],
                                  [0, -1]])
JF_y = [Jf0_y, Jf1_y]

# derivative wrt to t
J_t = lambda t, y:  np.array([0, 0])
dfdt = lambda t, y: np.dot(J_y(t, y), f(t, y)) + J_t(t, y)

df0dt = lambda t, y: 100 * (y[0] - 11 * y[1])
df1dt = lambda t, y: y[1]

dFdt = [df0dt,
        df1dt]

f_n = f
dfdt_n = dfdt

