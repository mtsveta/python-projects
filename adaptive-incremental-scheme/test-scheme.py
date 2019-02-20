from example1 import *
from adaptive_schemes import *
from uniform_schemes import *
from plotting import *

import numpy as np


result_folder = 'example1'
result_path = create_results_folder(result_folder)

# order or the scheme
order = 4

# given tolerances
num_tol = 5
factor  = 1e0
eps_abs = np.logspace(-3, -8, num_tol)
eps_rel = factor * eps_abs

# given time interval
t_0   = 0.0
t_fin = 1.0

# given initial value
y_0 = y(t_0)

# error arrays

# Test an adaptive scheme
# -------------------------------------------------------------------------------------------------------------------- #
lte = np.zeros(num_tol)
err = np.zeros(num_tol)
n   = np.zeros(num_tol)
f_evals   = np.zeros(num_tol)

for i in range(0, num_tol):
    print('% -------------------------------------------------------------------------------------------- %')
    print(' integration for eps_abs = %4.4e' % (eps_abs[i]))
    print('% -------------------------------------------------------------------------------------------- %')
    lte[i], err[i], n[i], f_evals[i] = adaptive_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path)
plot_convergence(lte, err, n, f_evals, order, 'Adaptive scheme', result_path)


# Test the two-derivatives Runge-Kutta scheme (4th order)
# -------------------------------------------------------------------------------------------------------------------- #
length = t_fin - t_0
h = length * np.array([math.pow(2, -2), math.pow(2, -3), math.pow(2, -4), math.pow(2, -5),
                       math.pow(2, -6), math.pow(2, -7), math.pow(2, -8), math.pow(2, -9)])
lte = np.zeros(len(h))
err = np.zeros(len(h))
n   = np.zeros(len(h))
f_evals   = np.zeros(len(h))

for i in range(0, len(h)):
    lte[i], err[i], n[i], f_evals[i] = tdrk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)
plot_uniform_convergence(lte, err, n, h, f_evals, order, 'TDRK scheme', result_path)

# Test the 4th order Runge-Kutta scheme
# -------------------------------------------------------------------------------------------------------------------- #
for i in range(0, len(h)-1):
    err[i], n[i], f_evals[i] = rk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, result_path)
plot_uniform_convergence(lte, err, h, n, f_evals, order, '4th order RK scheme', result_path)
