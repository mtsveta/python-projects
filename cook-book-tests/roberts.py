import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

t0, tT = 0, 4 * 1e3        # considered interval
y0 = np.array([1.0, 0.0, 0.0])   # initial condition

neq = 3

def f_y(y, t=0):
    """ Rhs equations for the problem"""
    return np.array([[- 0.04,                    1e4 * y[2],   1e4 * y[1]],
                     [  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]],
                     [  0.00,                6 * 1e7 * y[1],         0.0]])
def f(y, t=0):
    """ Rhs equations for the problem"""
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                  3 * 1e7 * y[1] ** 2])

num = 1e4
t = np.linspace(t0, tT, num)

y, info = odeint(f, y0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for i in range(neq):
    plt.plot(t, y[:, i], label=r'$y^{%d}$' % (i+1) )
plt.legend()
plt.show()

y, info = odeint(f, y0, t, full_output=True, Dfun=f_y)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for i in range(neq):
    plt.plot(t, y[:, i], label=r'$y^{%d}$' % (i+1) )
plt.legend()
plt.show()