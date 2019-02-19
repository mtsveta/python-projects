
import numpy as np
import math
import matplotlib.pyplot as plt

from example1 import y, f, fprime, f_n, fprime_n

# order or the scheme
p = 4

# given tolerances
eps_rel = 1e-6
eps_abs = 1e-4

# given time interval
t_final = 3.0
t_0 = 0.0

# given initial value
y_0 = 1.0

# function estimating an intermediate step h_star
def h_star_estimate(y_n, fprime_n):
    return  math.sqrt(2 * (eps_rel*math.fabs(y_n) + eps_abs) / fprime_n(y_n))

#
def h_estimate(h_star, y_n, y_star, f_n, fprime_n):
    C = math.fabs( 1 / (2 * h_star**3) * (f_n(y_n) - f_n(y_star))
                  -1 / (4 * h_star**2) * (fprime_n(y_n) - fprime_n(y_star)))
    return math.pow((eps_rel * math.fabs(y_n) + eps_abs)/C, 0.25)

def norm(val):
    return math.fabs(val)


t = t_0
y_n = y_0

yn_array = np.append(np.array([]), np.array([y_0]))
y_array = np.append(np.array([]), np.array([y_0]))
t_array = np.append(np.array([]), np.array([t_0]))

#y_array = np.append(np.empty((0,0), float), np.array([y_0]), axis=0)
#t_array = np.append(np.empty((0,0), float), np.array([t_0]), axis=0)

while t < t_final:

    print('% -------------------------------------------------------------------------------------------- %')
    print('t = %4.4e\n' % (t))
    print('% -------------------------------------------------------------------------------------------- %')

    # h_star step
    h_star = h_star_estimate(y_n, fprime_n)
    y_star = y_n + h_star / 2 * f_n(y_n) + h_star**2 / 8 * fprime_n(y_n)
    y_star_1st = y_n + h_star / 2 * f_n(y_n)
    print('h* = %4.4e\ny* = %4.4e\ny(t+h*) = %4.4e\n' % (h_star, y_star, y(t+h_star)))

    # analysis of the errors
    err = norm(y_star - y(t+h_star))
    LTE = norm(y_star - y_star_1st)
    eps = eps_abs * max(1, norm(y_n))
    print('err = %4.4e\nLTE = %4.4e\neps = %4.4e\n' %(err, LTE, eps))

    # h step
    h = h_estimate(h_star, y_n, y_star, f_n, fprime_n)
    rho = h / h_star
    y_n1 = y_n + h * (1 - rho**2 + rho**3 / 2) * f_n(y_n) \
               + h * (rho**2 - rho**3 / 2) * f_n(y_star) \
               + h / 2 * (1 - 4 * rho / 3 + rho**2 / 2)*fprime_n(y_n) \
               + h / 2 * (- 2 * rho / 3 + rho**2 / 2)*fprime_n(y_star)

    y_n1_3rd =  y_n + h * (1 - rho**2) * f_n(y_n) \
                    + h * rho**2 * f_n(y_star) \
                    + h / 2 * (1 - 4 * rho / 3)*fprime_n(y_n) \
                    + h / 2 * (- 2 * rho / 3)*fprime_n(y_star)
    print('h = %4.4e\ny_{n+1} = %4.4e\ny(t+h) = %4.4e\n' % (h, y_n1, y(t + h)))

    yn_array = np.append(yn_array, np.array([y_n]))
    y_array = np.append(y_array, np.array([y(t+h)]))
    t_array = np.append(t_array, np.array([t+h]))

    # analysis of the errors
    err = norm(y_n1 - y(t+h))
    LTE = norm(y_n1 - y_n1_3rd)
    eps = eps_abs * max(1, norm(y_n))
    print('err = %4.4e\nLTE = %4.4e\neps = %4.4e\n' %(err, LTE, eps))

    # predicted time-step
    const = 1.1
    h_new = const * math.pow(math.fabs(eps/LTE), 1/(p+1)) * h
    print('h_new = %4.4e\n' % (h_new))

    t += h
    y_n = y_n1

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t_array, y_array, 'o-', mew=1, ms=8,
            mec='w', label=f'y')

ax.plot(t_array, yn_array, '--', mew=1, ms=8,
            mec='w', label=f'y_n')
ax.legend()

#ax.set_xlim(t_0, t_final)
#ax.set_ylim(math.min(yn_array), math.max(yn_array))

plt.show()
