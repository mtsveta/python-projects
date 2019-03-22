import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math

t0, tT = 0, 4 * 1e3        # considered interval
y0 = np.array([1.0, 0.0, 0.0])   # initial condition

neq = 3
nsens = neq**2

def f_y(y, t=0):
    """ Jacobian of rhs equations for the problem"""
    return np.array([[- 0.04,                    1e4 * y[2],   1e4 * y[1]],
                     [  0.04, - 1e4 * y[2] - 6 * 1e7 * y[1], - 1e4 * y[1]],
                     [  0.00,                6 * 1e7 * y[1],         0.0]])
def f(y, t=0):
    """ Rhs equations for the problem"""
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                  3 * 1e7 * y[1] ** 2])
def F(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                       0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                         3 * 1e7 * y[1] ** 2,
                     - 0.04 * y[3] +    1e4 * y[2]                   * y[4] + 1e4 * y[1] * y[5],
                       0.04 * y[3] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * y[4] - 1e4 * y[1] * y[5],
                       0.00 * y[3] +                 6 * 1e7 * y[1]  * y[4] +        0.0 * y[5],
                     - 0.04 * y[6] +    1e4 * y[2]                   * y[7] + 1e4 * y[1] * y[8],
                       0.04 * y[6] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * y[7] - 1e4 * y[1] * y[8],
                       0.00 * y[6] +                 6 * 1e7 * y[1]  * y[7] +        0.0 * y[8],
                     - 0.04 * y[9] +    1e4 * y[2]                   * y[10] + 1e4 * y[1] * y[11],
                       0.04 * y[9] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * y[10] - 1e4 * y[1] * y[11],
                       0.00 * y[9] +                 6 * 1e7 * y[1]  * y[10] +        0.0 * y[11]])
def g(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    s = y[neq:]
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                       0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                         3 * 1e7 * y[1] ** 2,
                     - 0.04 * s[0] +    1e4 * y[2]                   * s[1] + 1e4 * y[1] * s[2],
                       0.04 * s[0] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[1] - 1e4 * y[1] * s[2],
                       0.00 * s[0] +                 6 * 1e7 * y[1]  * s[1] +        0.0 * s[2],
                     - 0.04 * s[3] +    1e4 * y[2]                   * s[4] + 1e4 * y[1] * s[5],
                       0.04 * s[3] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[4] - 1e4 * y[1] * s[5],
                       0.00 * s[3] +                 6 * 1e7 * y[1]  * s[4] +        0.0 * s[5],
                     - 0.04 * s[6] +    1e4 * y[2] *                   s[7] + 1e4 * y[1] * s[8],
                       0.04 * s[6] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[7] - 1e4 * y[1] * s[8],
                       0.00 * s[6] +                 6 * 1e7 * y[1]  * s[7] +        0.0 * s[8]])

def g_(y, t=0):
    """ Rhs equations for the sensitivity problem"""
    s = y[neq : 2 * neq]
    return np.array([- 0.04 * y[0] + 1e4 * y[1] * y[2],
                       0.04 * y[0] - 1e4 * y[1] * y[2] - 3 * 1e7 * y[1] ** 2,
                                                         3 * 1e7 * y[1] ** 2,
                     - 0.04 * s[0] +    1e4 * y[2]                   * s[1] + 1e4 * y[1] * s[2],
                       0.04 * s[0] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[1] - 1e4 * y[1] * s[2],
                       0.00 * s[0] +                 6 * 1e7 * y[1]  * s[1] +        0.0 * s[2]])


'''
def g(s, y, t=0):
    """ Rhs equations for the sensitivity problem"""
    return np.array([- 0.04 * s[0] +    1e4 * y[2]                   * s[1] + 1e4 * y[1] * s[2],
                       0.04 * s[0] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[1] - 1e4 * y[1] * s[2],
                       0.00 * s[0] +                 6 * 1e7 * y[1]  * s[1] +        0.0 * s[2],
                     - 0.04 * s[3] +    1e4 * y[2]                   * s[4] + 1e4 * y[1] * s[5],
                       0.04 * s[3] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[4] - 1e4 * y[1] * s[5],
                       0.00 * s[3] +                 6 * 1e7 * y[1]  * s[4] +        0.0 * s[5],
                     - 0.04 * s[6] +    1e4 * y[2] *                   s[7] + 1e4 * y[1] * s[8],
                       0.04 * s[6] + (- 1e4 * y[2] - 6 * 1e7 * y[1]) * s[7] - 1e4 * y[1] * s[8], 
                       0.00 * s[6] +                 6 * 1e7 * y[1]  * s[7] +        0.0 * s[8]])
'''


num = 1e4
t = np.linspace(t0, tT, num)

print("% ----------------------------------------------------------- %")
print("  Solving the system without Jacobian ")
print("% ----------------------------------------------------------- %")

y, info = odeint(f, y0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for k in range(neq):
    n = k + 1
    plt.plot(t, y[:, k], label=r'$y^{%d}$' % n)
plt.legend()
plt.title('Solutions to the Roberts system')
plt.show()

print("% ----------------------------------------------------------- %")
print("  Solving the system with all sensitivities ")
print("% ----------------------------------------------------------- %")

s0 = np.append(y0, np.ones(nsens))
s, info = odeint(g, s0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for k in range(neq):
    n = k + 1
    plt.plot(t, s[:, k], label=r'$y^{%d}$' % n)
plt.legend()
plt.title('Solutions to the Roberts system')
plt.show()
fig.savefig('approximations.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)


fig, ax = plt.subplots(3, 1, figsize=(6, 8))
for k in  range(nsens):
    n = k + 1
    (j, i) = divmod(n, neq)
    if i == 0: i = neq; j -= 1
    ax[j].plot(t, s[:, k + neq], label=r'$s^{%d, %d}$' % (i, j + 1))
    ax[j].legend()
ax[0].set_title('Sensitivities to the Roberts system')
fig.show()
fig.savefig('sensitivities.eps',
            dpi=1000, facecolor='w', edgecolor='w',
            orientation='portrait', format='eps',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

S = s[:, neq:]
print(S.shape)
S = S.reshape((int(num), neq, neq))
print(S.shape)
S = np.swapaxes(S, 1, 2)
print(S.shape)

# pertubation of the initial condition
delta  = 1e-1
y0_new = y0 + delta * np.ones(neq)
dy0 = y0_new - y0

y_new = y + np.dot(S, dy0)

for k in range(neq):
    num = k + 1
    plt.plot(t, y[:, k], label=r'$y^{%d}$' % num)
plt.legend()
plt.title('Solutions to the Roberts system')
plt.show()

print("% ----------------------------------------------------------- %")
print("  Solving system with just first n sensitivities ")
print("% ----------------------------------------------------------- %")

s0 = np.append(y0, np.ones(neq))
s, info = odeint(g_, s0, t, full_output=True)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for k in range(neq):
    num = k + 1
    plt.plot(t, s[:, k], label=r'$y^{%d}$' % num)
plt.legend()
plt.show()
plt.title('Solutions to the Roberts system')

for k in range(neq, 2 * neq):
    num = k - neq + 1
    (j, i) = divmod(num, neq)
    #i = num % neq
    #j = math.mod(num, neq)
    plt.plot(t, s[:, i], label=r'$s^{%d, %d}$' % (i, j + 1) )
plt.legend()
plt.show()
plt.title('Sensitivities to the Roberts system')


S = np.transpose(np.tile(s[neq : 2 * neq], (neq, 1)))
print(S)

# pertubation of the initial condition
delta  = 1e-1
y0_new = y0 + delta * np.ones(neq)


y_new = y + np.dot(S, y0_new - y0)

print("% ----------------------------------------------------------- %")
print("  Solving the system with Jacobian ")
print("% ----------------------------------------------------------- %")

y, info = odeint(f, y0, t, full_output=True, Dfun=f_y)
print("Number of function evaluations: %d, number of Jacobian evaluations: %d" % (info['nfe'][-1], info['nje'][-1]))
print(info['message'])

for i in range(neq):
    plt.plot(t, y[:, i], label=r'$y^{%d}$' % (i+1) )
plt.legend()
plt.show()

