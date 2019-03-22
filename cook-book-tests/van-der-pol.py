import matplotlib.pyplot as plt
import numpy as np
import scikits.odes as od
#from scikits.odes import od
from scikits.odes.odeint import odeint

import pkg_resources
print(pkg_resources.get_distribution("scikits.odes").version)
#od.test()

t0, tT = 1, 500        # considered interval
y0 = np.array([0.5, 0.5])   # initial condition

def van_der_pol(t, y, ydot):
    """ we create rhs equations for the problem"""
    ydot[0] = y[1]
    ydot[1] = 1000 * (1.0 - y[0]**2) * y[1] - y[0]

num = 200
t_n = np.linspace(t0, tT, num)

solution = odeint(van_der_pol, t_n, y0)
plt.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
plt.show()
print(solution.values.y)

solution = od.ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
plt.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
plt.show()