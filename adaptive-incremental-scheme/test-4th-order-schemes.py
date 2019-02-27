from adaptive_schemes import *
from uniform_schemes import *
from plotting_4th_order_schemes import *

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
        y = example1.y; f_n = example1.f_n; fprime_n = example1.fprime_n
    elif example == 2:
        y = example2.y; f_n = example2.f_n; fprime_n = example2.fprime_n
    elif example == 3:
        y = example3.y; f_n = example3.f_n; fprime_n = example3.fprime_n
    elif example == 4:
        y = example4.y; f_n = example4.f_n; fprime_n = example4.fprime_n
    elif example == 5:
        y = example5.y; f_n = example5.f_n; fprime_n = example5.fprime_n
    elif example == 6:
        y = example6.y; f_n = example6.f_n; fprime_n = example6.fprime_n
    elif example == 7:
        y = example7.y; f_n = example7.f_n; fprime_n = example7.fprime_n
    result_folder = 'example' + str(example)

    result_path = create_results_folder(result_folder)

    # order or the scheme
    order = 4

    # given tolerances
    #factor  = 1e-4 # 1 / max_{t_0, t_fin}(y)
    factor = 1e0
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    #eps_abs = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]) #
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    eps_rel = factor * eps_abs
    num_tol = len(eps_abs)
    # given time interval
    t_0   = 0.0
    t_fin = 3.0

    # given initial value
    y_0 = y(t_0)

    # Test adaptive schemes with our h prediction
    # ---------------------------------------------------------------------------------------------------------------- #
    # 0 : our scheme with our h prediction
    # 1 : tdrk scheme  with our h prediction
    num_of_schemes = 4
    e_loc_adapt = np.zeros((num_of_schemes, num_tol))
    e_glo_adapt = np.zeros((num_of_schemes, num_tol))
    n_adapt   = np.zeros((num_of_schemes, num_tol))
    f_evals_adapt = np.zeros((num_of_schemes, num_tol))
    rej_num_adapt = np.zeros((num_of_schemes, num_tol))

    print('% -------------------------------------------------------------------------------------------- %')
    print(' Our scheme (our prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        e_loc_adapt[0, i], e_glo_adapt[0, i], n_adapt[0, i], f_evals_adapt[0, i], rej_num_adapt[0, i] \
            = adaptive_our_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path, test_params)

    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK scheme (our prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        e_loc_adapt[1, i], e_glo_adapt[1, i], n_adapt[1, i], f_evals_adapt[1, i], rej_num_adapt[1, i] \
            = adaptive_tdrk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path,
                                      test_params)
    # 2 : our scheme with classic h prediction
    # 3 : tdrk scheme  with classic h prediction
    # 4 : tdrk scheme  with classic h prediction

    print('% -------------------------------------------------------------------------------------------- %')
    print(' TDRK scheme (classic prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        e_loc_adapt[2, i], e_glo_adapt[2, i], n_adapt[2, i], f_evals_adapt[2, i], rej_num_adapt[2, i] \
            = adaptive_classic_tdrk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n, result_path,
                                     test_params)
    print('% -------------------------------------------------------------------------------------------- %')
    print(' RK scheme (classic prediction of h)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    for i in range(0, num_tol):
        e_loc_adapt[3, i], e_glo_adapt[3, i], n_adapt[3, i], f_evals_adapt[3, i], rej_num_adapt[3, i] \
            = adaptive_classic_rk_4th_order(eps_rel[i], eps_abs[i], t_0, t_fin, y_0, y, f_n, fprime_n,
                                              result_path,
                                              test_params)
    plot_convergence(e_loc_adapt, e_glo_adapt, n_adapt, f_evals_adapt, 'classic-h-pred-', result_path)

    '''
    plot_summary_adaptive_convergence(e_loc_adapt, e_glo_adapt, n_adapt, f_evals_adapt,
                                      'Global error', 'global error', 'global-error-')
    '''
    '''
    plot_summary_adaptive_convergence(lte_adapt, n_adapt, f_evals_adapt,
                                      lte_pred_rej, n_pred_rej, f_evals_pred_rej, result_path,
                                      'Local truncation error', 'loc. trunc. error', 'lte-')
    '''
    '''
    # Test the two-derivatives Runge-Kutta scheme (4th order)
    # -------------------------------------------------------------------------------------------------------------------- #
    length = t_fin - t_0
    h = length * np.array([math.pow(2, -4), math.pow(2, -5),
                           math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -9)])
    lte_tdrk4 = np.zeros(len(h))
    err_tdrk4 = np.zeros(len(h))
    n_tdrk4   = np.zeros(len(h))
    f_evals_tdrk4   = np.zeros(len(h))

    for i in range(0, len(h)):
        lte_tdrk4[i], err_tdrk4[i], n_tdrk4[i], f_evals_tdrk4[i] \
            = tdrk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)

    # Test the two-derivatives Runge-Kutta scheme (5th order)
    # -------------------------------------------------------------------------------------------------------------------- #
    lte_rk4 = np.zeros(len(h))
    err_rk4 = np.zeros(len(h))
    n_rk4   = np.zeros(len(h))
    f_evals_rk4   = np.zeros(len(h))

    for i in range(0, len(h)):
        err_rk4[i], n_rk4[i], f_evals_rk4[i] \
            = rk_4th_order(t_0, t_fin, h[i], y_0, y, f_n, result_path)

    lte_tdrk5 = np.zeros(len(h))
    err_tdrk5 = np.zeros(len(h))
    n_tdrk5   = np.zeros(len(h))
    f_evals_tdrk5   = np.zeros(len(h))

    for i in range(0, len(h)):
        lte_tdrk5[i], err_tdrk5[i], n_tdrk5[i], f_evals_tdrk5[i] \
            = tdrk_5th_order(t_0, t_fin, h[i], y_0, y, f_n, fprime_n, result_path)

    plot_summary_uniform_convergence(err_rk4, n_rk4, f_evals_rk4,
                                     err_tdrk4, n_tdrk4, f_evals_tdrk4,
                                     err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                     'Uniform schemes comparison', result_path)


    plot_summary_adaptive_uniform_convergence(err_adapt, n_adapt, f_evals_adapt,
                                      err_pred_rej, n_pred_rej, f_evals_pred_rej,
                                      err_rk4, n_rk4, f_evals_rk4,
                                      err_tdrk4, n_tdrk4, f_evals_tdrk4,
                                      err_tdrk5, n_tdrk5, f_evals_tdrk5,
                                      'Adaptive & uniform schemes comparison', result_path)
    '''
if __name__== "__main__":

    examples = [1]
    #examples = [1, 2, 3, 4, 5, 6, 7]
    test_params = dict(adaptive_stepping='our_prediction',  # 'classic_prediction'
                       middle_step_order=2)
    for example_num in range(0, len(examples)):
        test_schemes(examples[example_num], test_params)
