import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

def J_y(y, t, mu):
    return np.array([
        [0,                 1],
        [-1-2*mu*y[0]*y[1], mu*(1-y[0]**2)]  # EXERCISE: -1-2*mu*y[0]*y[1]
    ])
def f(y, t, mu):
    return [
        y[1],
        mu*(1-y[0]**2)*y[1] - y[0]
    ]

tout = np.linspace(0, 200, 1024)
y_init= [1, 0]
params = (17,)

y_odeint, info = odeint(f, y_init, tout, params, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))

plt.plot(tout, y_odeint, label='Van der Pol without Jacobian')
plt.show()


y_odeint, info = odeint(f, y_init, tout, params, full_output=True, Dfun=J_y)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
plt.plot(tout, y_odeint, label='Van der Pol with Jacobian')
plt.show()
