import numpy as np

# number of variables
n = 3

# solution
y = lambda t:           np.array([1 / 9 * (100 * np.exp(-100 * t) - 91 * np.exp(-1000 * t)),
                                  np.exp(-100 * t)])
# rhs
f0 = lambda t, y: - 1000 * y[0] + 10000 * y[1]
f1 = lambda t, y: - 100 * y[1]

f = lambda t, y:  np.array([- 1000 * y[0] + 10000 * y[1],
                            - 100 *  y[1]])
F = [f0, f1]

# jacobian of rhs
Jf0_y = lambda t, y: np.array([-1000, 10000])
Jf1_y = lambda t, y: np.array([0, -100])

J_y = lambda t, y:      np.array([[-1000, 10000],
                                  [0, -100]])
JF_y = [Jf0_y, Jf1_y]

# derivative wrt to t
J_t = lambda t, y:  np.array([0, 0])
dfdt = lambda t, y: np.dot(J_y(t, y), f(t, y)) + J_t(t, y)

df0dt = lambda t, y: 1000000 * (y[0] - 11 * y[1])
df1dt = lambda t, y: 10000 * y[1]

dFdt = [df0dt,
        df1dt]

f_n = f
dfdt_n = dfdt

