import math
import numpy as np

from plotting import *


def norm(val):
    return math.fabs(val)

def rk_4th_order(t_0, t_final, h, y_0, y, f_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    err_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        k_1 = h * f_n_val

        f_nk1_val = f_n(t_n + h / 2, y_n + k_1 / 2)
        k_2 = h * f_nk1_val

        f_nk2_val = f_n(t_n + h / 2, y_n + k_2 / 2)
        k_3 = h * f_nk2_val

        f_nk3_val = f_n(t_n + h, y_n + k_3)
        k_4 = h * f_nk3_val

        f_evals += 4

        y_n1 = y_n + k_1 / 6 + k_2 / 3 + k_3 / 3 + k_4 / 6

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))

        t_array = np.append(t_array, np.array([t_n + h]))
        err_array = np.append(err_array, np.array([err]))
        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    return err, n, f_evals

def tdrk_4th_order(t_0, t_final, h, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    err_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n <= t_final:

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # reconstruct approximation y_1/2 of the half step
        y_half = y_n + h / 2 * f_n_val + h ** 2 / 8 * fprime_n_val

        # evaluate f_* and (f')_*
        f_half_val = f_n(t_n + h / 2, y_half)
        fprime_half_val = fprime_n(t_n + h / 2, y_half)
        f_evals += 2

        y_n1 = y_n + h * f_n_val \
               + h**2 / 6 * fprime_n_val \
               + h**2 / 3 * fprime_half_val

        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n + h * f_n_val + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))

        t_array = np.append(t_array, np.array([t_n + h]))
        err_array = np.append(err_array, np.array([err]))
        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_uniform_results(t_array, yn_array, y_array, err_array, result_path)
    return err, n, f_evals


def tdrk_5th_order(t_0, t_final, h, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    err_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        # stage 1
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # stage 2
        a_21 = 2 / 5
        a_bar_21 = 2 / 25

        y_2 = y_n + a_21 * h * f_n_val + a_bar_21 * h ** 2 * fprime_n_val
        c_2 = 2 / 5

        f_2_val = f_n(t_n + c_2 * h, y_2)
        fprime_2_val = fprime_n(t_n + c_2 * h, y_2)
        f_evals += 2

        # stage 3
        a_31 = 139 / 64
        a_32 = -75 / 64
        a_bar_31 = 17 / 64
        a_bar_32 = 45 / 64

        y_3 = y_n + h * (a_31 * f_n_val + a_32 * f_2_val) \
                  + h**2 * (a_bar_31 * fprime_n_val + a_bar_32 * fprime_2_val)
        c_3 = 1

        f_3_val = f_n(t_n + c_3 * h, y_3)
        fprime_3_val = fprime_n(t_n + c_3 * h, y_3)
        f_evals += 2

        # reconstruct approximation y_{n+1} of the 5th order
        b_1 = 9/16
        b_2 = 125/432
        b_3 = 4/27
        b_bar_1 = 1/16
        b_bar_2 = 25/144
        b_bar_3 = 0

        y_n1 = y_n \
               + h * (b_1 * f_n_val + b_2 * f_2_val + b_3 * f_3_val) \
               + h**2 * (b_bar_1 * fprime_n_val + b_bar_2 * fprime_2_val + b_bar_3 * fprime_3_val)
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n \
                   + h * f_n_val \
                   + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_2nd)

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals


def our_scheme_4th_order(t_0, t_final, h, y_0, y, f_n, fprime_n, test_params, rho, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0
    rej_num = 0
    accum = 0.0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    e_glob_array = np.append(np.array([]), np.array([]))
    e_loc_array = np.append(np.array([]), np.array([]))

    h_star = h / rho
    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        if test_params['middle_step_order'] == 2:
            # reconstruct apprimation y_star with the 2nd order tdrk
            y_star = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
        elif test_params['middle_step_order'] == 4:
            # reconstruct apprimation y_star with the 4th order tdrk
            t_aux = h_star / 2
            y_aux = y_n + t_aux * f_n_val + t_aux ** 2 / 2 * fprime_n_val
            fprime_aux_val = fprime_n(t_aux, y_aux)
            y_star = y_n \
                     + h_star * f_n_val \
                     + h_star ** 2 / 6 * fprime_n_val \
                     + h_star ** 2 / 3 * fprime_aux_val
            f_evals += 1

        # evaluate f_* and (f')_*
        f_star_val = f_n(t_n + h_star, y_star)
        fprime_star_val = fprime_n(t_n + h_star, y_star)
        f_evals += 2

        # h step
        # --------------------------------------------------------------------------------------------------------------
        # if predicted h is beyond considered interval [t_0, t_final]
        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n
        # reconstruct apprimation y_{n+1} of the 4th order

        y_n1 = y_n \
               + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
               + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
               + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
               + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val

        # reconstruct apprimation y_{n+1} of the 3rd order
        y_n1_3rd = y_n \
                   + h * (1 - rho ** 2) * f_n_val \
                   + h * rho ** 2 * f_star_val \
                   + h**2 / 2 * (1 - 4 * rho / 3) * fprime_n_val \
                   + h**2 / 2 * (- 2 * rho / 3) * fprime_star_val
        '''
        if test_params['polynomial_comparison']:
            # comparison to the 4th order polynomials
            times = np.zeros(5)     # times of the comparison
            p4 = np.zeros(5)        # 4th order polynomial
            appr = np.zeros(5)      # our apprimatin

            k_1 = 1 / 2 * (2 - 2 * rho**2 + rho**3)
            k_2 = 1 / 2 * (2 * rho**2 - rho**3)
            k_3 = 1 / 6 * (6 - 8 * rho + 3 * rho**2)
            k_4 = 1 / 6 * (- 4 * rho + 3 * rho**2)

            b_ = (k_1 + k_2) * f(t_n)
            c_ = 1 / h**2 * (h * h_star * k_2 + 1 / 2 * h**2 * (k_3 + k_4)) * fprime(t_n)
            d_ = 1 / h**3 * (1 / 2 * h * h_star**2 * k_2 + 1 / 2 * h**2 * h_star * k_4) * d2fdt2(t_n)
            e_ = 1 / h**4 * (1 / 6 * h * h_star**3 * k_2 + 1 / 4 * h**2 * h_star**2 * k_4) * d3fdt3(t_n)

            b = f(t_n)
            c = fprime(t_n) / 2
            d = d2fdt2(t_n) / 6
            e = d3fdt3(t_n) / 24

            def p(t):
                return y_n + b * (t - t_n) + c * (t - t_n)**2 + d * (t - t_n)**3 + e * (t - t_n)**4

            def our_appr(t):
                h = t - t_n
                rho = h / h_star
                return y_n \
                       + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
                       + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
                       + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
                       + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val

            times[0] = t_n
            times[1] = t_n + h / 4
            times[2] = t_n + h / 2
            times[3] = t_n + 3 * h / 4
            times[4] = t_n + h

            p4[0] = p(times[0])
            p4[1] = p(times[1])
            p4[2] = p(times[2])
            p4[3] = p(times[3])
            p4[4] = p(times[4])

            appr[0] = our_appr(times[0])
            appr[1] = our_appr(times[1])
            appr[2] = our_appr(times[2])
            appr[3] = our_appr(times[3])
            appr[4] = our_appr(times[4])

            accum += appr[4] - p4[4]  # accumulated errors
            # y_n1 -= appr[4] - p4[4]   # correction of the approximation

            if test_params['detailed_log']:
                print('  b_ = %4.4e\tc_ = %4.4e\td_ = %4.4e\te_ = %4.4e' % (b_, c_, d_, e_))
                print('  b  = %4.4e\tc  = %4.4e\td  = %4.4e\te  = %4.4e\n' % (b, c, d, e))

                print('  e(t_n) = %4.4e\te(t_n + h/4) = %4.4e\te(t_n + h/2) = %4.4e\te(t_n + 3*h/4) = %4.4e\te(t_n + h) = %4.4e\n'
                      % ((appr[0] - p4[0]), (appr[1] - p4[1]), (appr[2] - p4[2]), (appr[3] - p4[3]), (appr[4] - p4[4])))

        '''
        # analysis of the errors
        e_glob = norm(y_n1 - y(t_n + h))
        e_loc = norm(y_n1 - y_n1_3rd)

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        e_glob_array = np.append(e_glob_array, np.array([e_glob]))
        e_loc_array = np.append(e_loc_array, np.array([e_loc]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1
    print('  h = %4.4e\te_glob = %4.4e\tn = %d\tf_evals = %d\n' % (h, e_glob, n, f_evals))
    return e_glob, n, f_evals
