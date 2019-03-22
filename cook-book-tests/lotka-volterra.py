import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

t = np.linspace(0, 15,  1000)
y0 = np.array([10, 15])

a = 1.
b = 0.1
c = 1.5
d = 0.75

def f(y, t=0):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ a*y[0] -   b*y[0]*y[1] ,
                     -c*y[1] + d*b*y[0]*y[1] ])
def f_y(y, t=0):
    """ Return the Jacobian matrix evaluated in X. """
    return np.array([[a -b*y[1],   -b*y[0]     ],
                     [ b*d*y[1],   -c +b*d*y[0]]])

y, info = odeint(f, y0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

plt.plot(t, y, label='Lotka-Volterra without Jacobian')
plt.show()

y, info = odeint(f, y0, t, full_output=True, Dfun=f_y)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

plt.plot(t, y, label='Van der Pol with Jacobian')
plt.show()
