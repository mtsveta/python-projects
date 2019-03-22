import numpy as np

# number of variables
n = 2

# solution
y = lambda t:           np.array([1,
                                  0,
                                  0])

# rhs
f0 = lambda t, y: - 0.04 * y[0] + 1e4 * y[1] * y[2]
f1 = lambda t, y:   0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1]**2
f2 = lambda t, y:                                     3 * 1e7 * y[1]**2

f = lambda t, y: np.array([ - 0.04 * y[0] + 1e4 * y[1] * y[2],
                              0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1]**2,
                              3 * 1e7 * y[1]**2])
F = [f0, f1, f2]

# jacobian of rhs
Jf0_y = lambda t, y: np.array([- 0.04,   1e4 * y[2]                 ,   1e4 * y[1]])
Jf1_y = lambda t, y: np.array([  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]])
Jf2_y = lambda t, y: np.array([  0.00,                6 * 1e7 * y[1],          0.0])

J_y = lambda t, y:      np.array([[- 0.04,   1e4 * y[2]                 ,   1e4 * y[1]],
                                  [  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]],
                                  [  0.00,                6 * 1e7 * y[1],          0.0]])
JF_y = [Jf0_y, Jf1_y, Jf2_y]

# derivative wrt to t
J_t = lambda t, y:      np.array([0, 0, 0])
dfdt = lambda t, y:     np.dot(J_y(t, y), f(t, y)) + J_t(t, y)

df0dt = lambda t, y: - 0.04 * (- 0.04 * y[0] + 1e4 * y[1] * y[2]) \
                     +  1e4 * y[2] * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1]**2) \
                     +  1e4 * y[1] * 3 * 1e7 * y[1]**2
df1dt = lambda t, y:   0.04 * (- 0.04 * y[0] + 1e4 * y[1] * y[2]) \
                     + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1]**2) \
                     - 1e4 * y[1] * 3 * 1e7 * y[1]**2
df2dt = lambda t, y: 6 * 1e7 * y[1] * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1]**2)

dFdt = [df0dt,
        df1dt,
        df2dt]

f_n = f
dfdt_n = dfdt