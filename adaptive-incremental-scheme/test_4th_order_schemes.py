from adaptive_schemes import *
from uniform_schemes import *
from plotting_4th_order_schemes import *
import time

import example1
import example2
import example3
import example4
import example5
import example6
import example7

import numpy as np

def test_schemes(example, test_params):

    if example == 1:
        y = example1.y; f = example1.f; dfdt = example1.dfdt; d2fdt2 = example1.d2fdt2; d3fdt3 = example1.d3fdt3;
        f_n = example1.f_n; fprime_n = example1.fprime_n
    elif example == 2:
        y = example2.y; f = example2.f; dfdt = example2.dfdt; d2fdt2 = example2.d2fdt2; d3fdt3 = example2.d3fdt3;
        f_n = example2.f_n; fprime_n = example2.fprime_n
    elif example == 3:
        y = example3.y; f = example3.f; dfdt = example3.dfdt; d2fdt2 = example3.d2fdt2; d3fdt3 = example3.d3fdt3;
        f_n = example3.f_n; fprime_n = example3.fprime_n
    elif example == 4:
        y = example4.y; f = example4.f; dfdt = example4.dfdt; d2fdt2 = example4.d2fdt2; d3fdt3 = example4.d3fdt3;
        f_n = example4.f_n; fprime_n = example4.fprime_n
    elif example == 5:
        y = example5.y; f = example5.f; dfdt = example5.dfdt; d2fdt2 = example5.d2fdt2; d3fdt3 = example5.d3fdt3;
        f_n = example5.f_n; fprime_n = example5.fprime_n
    elif example == 6:
        y = example6.y; f = example6.f; dfdt = example6.dfdt; d2fdt2 = example6.d2fdt2; d3fdt3 = example6.d3fdt3;
        f_n = example6.f_n; fprime_n = example6.fprime_n
    elif example == 7:
        y = example7.y; f = example7.f; dfdt = example7.dfdt; d2fdt2 = example7.d2fdt2; d3fdt3 = example7.d3fdt3;
        f_n = example7.f_n; fprime_n = example7.fprime_n

    result_folder = 'example' + str(example)

    result_path = create_results_folder(result_folder)

    # order or the scheme
    order = 4

    # given tolerances
    factor  = 1e0 # 1 / max_{t_0, t_fin}(y)
    #factor = 1e0
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]) #
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6])
    #eps_abs = np.array([1e-10, 1e-12])
    #eps_abs = np.array([1e-12])
    num_tol = len(eps_abs)
    eps_rel = factor * eps_abs # 1e-16 * np.ones(num_tol)  #
    # given time interval
    t_0   = 0.0
    t_fin = 3.0

    # given initial value
    y_0 = y(t_0)

    # Test adaptive schemes with our h prediction
    # ---------------------------------------------------------------------------------------------------------------- #
    # 0 : our scheme with our h prediction
    # 1 : tdrk scheme  with our h prediction
    num_of_schemes = 5
    e_loc_adapt = np.zeros((num_of_schemes, num_tol))
    e_glob_adapt = np.zeros((num_of_schemes, num_tol))
    n_adapt   = np.zeros((num_of_schemes, num_tol))
    f_evals_adapt = np.zeros((num_of_schemes, num_tol))
    rej_num_adapt = np.zeros((num_of_schemes, num_tol))
    cpu_time = np.zeros((num_of_schemes, num_tol))


    print('% -------------------------------------------------------------------------------------------- %')
    print(' Taylor scheme')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        t_start = time.time()
        #e_loc_adapt[0, i], e_glob_adapt[0, i], n_adapt[0, i], f_evals_adapt[0, i], rej_num_adapt[0, i] \
        #    = adaptive_taylor_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f, dfdt, d2fdt2, d3fdt3,
        #                                f_n, fprime_n, result_path, test_params)
        cpu_time[0, i] = time.time() - t_start

    '''
    print('% -------------------------------------------------------------------------------------------- %')
    print(' Our scheme (our prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        t_start = time.time()
        e_loc_adapt[0, i], e_glob_adapt[0, i], n_adapt[0, i], f_evals_adapt[0, i], rej_num_adapt[0, i] \
            = adaptive_our_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path, test_params)
        cpu_time[0, i] = time.time() - t_start
    '''
    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK scheme (our prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        t_start = time.time()
        e_loc_adapt[1, i], e_glob_adapt[1, i], n_adapt[1, i], f_evals_adapt[1, i], rej_num_adapt[1, i] \
            = adaptive_tdrk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path,
                                      test_params)
        cpu_time[1, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK2 scheme (classic prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        t_start = time.time()
        e_loc_adapt[2, i], e_glob_adapt[2, i], n_adapt[2, i], f_evals_adapt[2, i], rej_num_adapt[2, i] \
            = adaptive_classic_tdrk_2nd_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path,
                                              test_params)
        cpu_time[2, i] = time.time() - t_start

    plot_convergence(e_loc_adapt[0:2, :], e_glob_adapt[0:2, :], n_adapt[0:2, :], f_evals_adapt[0:2, :], cpu_time[0:2, :],
                     ['taylor4', 'tdrk4'],
                     'adaptive-tdrk-vs-taylor-',
                     result_path)

    # 2 : tdrk scheme with classic h prediction
    # 3 : rk scheme with classic h prediction
    '''
    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK scheme (classic prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        t_start = time.time()
        e_loc_adapt[2, i], e_glob_adapt[2, i], n_adapt[2, i], f_evals_adapt[2, i], rej_num_adapt[2, i] \
            = adaptive_classic_tdrk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path,
                                     test_params)
        cpu_time[2, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' RK scheme (classic prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, 3):
        t_start = time.time()
        e_loc_adapt[3, i], e_glob_adapt[3, i], n_adapt[3, i], f_evals_adapt[3, i], rej_num_adapt[3, i] \
            = adaptive_classic_rk_4th_order(eps_rel[i]*1e2, eps_abs[i]*1e2, t_0, t_fin, y_0, y, f_n, fprime_n,
                                              result_path, test_params)
        cpu_time[3, i] = time.time() - t_start

    plot_convergence(e_loc_adapt, e_glob_adapt, n_adapt, f_evals_adapt, cpu_time, 'our-vs-classic-h-pred-',
                         result_path)

    # Test the two-derivatives Runge-Kutta scheme (4th order)
    # -------------------------------------------------------------------------------------------------------------------- #
    length = t_fin - t_0
    h = length * np.array([math.pow(2, -2), math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])
    h_length = len(h)
    num_of_schemes = 5

    # 0 : tdrk uniform
    # 1 : rk uniform
    # 2 : our scheme with different rho

    e_loc_unif = np.zeros((num_of_schemes, h_length))
    e_glob_unif = np.zeros((num_of_schemes, h_length))
    n_unif = np.zeros((num_of_schemes, h_length))
    f_evals_unif = np.zeros((num_of_schemes, h_length))
    cpu_time_unif = np.zeros((num_of_schemes, h_length))

    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK scheme uniform')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, h_length):
        t_start = time.time()
        e_glob_unif[0, i], n_unif[0, i], f_evals_unif[0, i] \
            = tdrk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)
        cpu_time_unif[0, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' RK scheme uniform')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, h_length):
        t_start = time.time()
        e_glob_unif[1, i], n_unif[1, i], f_evals_unif[1, i] \
            = rk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, result_path)
        cpu_time_unif[1, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' Our scheme uniform, rho = 2')
    print('% -------------------------------------------------------------------------------------------- %\n')
    rho = 2.0
    for i in range(0, h_length):
        t_start = time.time()
        e_glob_unif[2, i], n_unif[2, i], f_evals_unif[2, i] \
            = our_scheme_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, test_params, rho, result_path)
        cpu_time_unif[2, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' Our scheme uniform, rho = 3')
    print('% -------------------------------------------------------------------------------------------- %\n')

    rho = 3.0
    for i in range(0, h_length):
        t_start = time.time()
        e_glob_unif[3, i], n_unif[3, i], f_evals_unif[3, i] \
            = our_scheme_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, test_params, rho, result_path)
        cpu_time_unif[3, i] = time.time() - t_start

    print('% -------------------------------------------------------------------------------------------- %')
    print(' Our scheme uniform, rho = 4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    rho = 4.0
    for i in range(0, h_length):
        t_start = time.time()
        e_glob_unif[4, i], n_unif[4, i], f_evals_unif[4, i] \
            = our_scheme_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, test_params, rho, result_path)
        cpu_time_unif[4, i] = time.time() - t_start

    plot_uniform_results(e_loc_unif, e_glob_unif, n_unif, f_evals_unif, cpu_time_unif, '', result_path)

    plot_convergence_summary(e_loc_adapt, e_glob_adapt, n_adapt, f_evals_adapt, cpu_time,
                             e_loc_unif, e_glob_unif, n_unif, f_evals_unif, cpu_time_unif,
                             'adaptive-and-uniform-', result_path)
    '''
if __name__== "__main__":

    examples = [1, 2, 3, 4, 5, 6, 7]
    #examples = [1, 2, 3, 4]
    #examples = [5, 6]
    #examples = [6]
    test_params = dict(middle_step_order=4,
                       detailed_log=True,
                       polynomial_comparison=False) # just tested for example 1, the import of each example +
                                                    # implementation of derivatives is needed

    for example_num in range(0, len(examples)):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' Example %d' % examples[example_num])
        print('% -------------------------------------------------------------------------------------------- %\n')

        test_schemes(examples[example_num], test_params)
