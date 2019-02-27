'''
# reconstruct y_star with 3rd order tdrk method
y_aux_2 = y_n + h_star / 2 * f_n_val + h_star ** 2 / 8 * fprime_n_val
fprime_2_val = fprime_n(y_aux_2)
y_aux_3 = y_n + h_star * f_n_val + h_star ** 2 * (fprime_n_val / 6 + fprime_2_val / 3)
fprime_3_val = fprime_n(y_aux_3)
y_star = y_n + h_star * f_n_val + h_star ** 2 * (103 / 600 * fprime_n_val + 97 / 300 * fprime_2_val + 1 / 200 * fprime_3_val)
f_evals += 2
'''
'''
# reconstruct y_star with 3rd order tdrk method
y_aux = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
f_aux_val = f_n(y_aux)
y_star = y_n + 2 / 3 * h_star * f_n_val + 1 / 3 * h_star * f_aux_val + 1 / 6 * h_star ** 2 * fprime_n_val
f_evals += 1
'''


import math
import numpy as np

from plotting import *

def adaptive_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path):
    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # auxiliary functions estimating an intermediate step h_star
    def h_star_estimate(y_n, fprime_n):
        return math.sqrt(2 * (eps_rel * math.fabs(y_n) + eps_abs) / math.fabs(fprime_n))

    #
    def h_estimate(h_star, y_n, f_n, fprime_n, f_star, fprime_star):
        C = math.fabs(  1 / (2 * h_star ** 3) * (f_n - f_star)
                      + 1 / (4 * h_star ** 2) * (fprime_n + fprime_star))
        return math.pow((eps_rel * math.fabs(y_n) + eps_abs) / C, 0.25)

    def norm(val):
        return math.fabs(val)

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        #f_n_val = f_n(y_n)
        #fprime_n_val = fprime_n(y_n)
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val)
        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n
        h_star = h / 2

        # reconstruct approximation y_star with the 4th order tdrk
        '''
        t_aux = h_star / 2
        y_aux = y_n + t_aux * f_n_val + t_aux ** 2 / 2 * fprime_n_val
        fprime_aux_val = fprime_n(t_aux, y_aux)
        y_star = y_n \
               + h_star * f_n_val \
               + h_star ** 2 / 6 * fprime_n_val \
               + h_star ** 2 / 3 * fprime_aux_val
        f_evals += 1
        '''
        # reconstruct approximation y_star with the 2nd order tdrk
        y_star = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
        # reconstruct approximation for the 1st order
        y_1st = y_n + h_star * f_n_val
        print('y_star = %e' % y_star)

        # analysis of the errors of the h_star step
        err_star = norm(y_star - y(t_n + h_star))
        lte_star = norm(y_star - y_1st)

        # predicted time-step
        const = 1.1
        p = 2
        h_star_new = const * math.pow(math.fabs(eps_abs / lte_star), 1 / (p + 1)) * h_star

        # evaluate f_* and (f')_*
        # f_star_val = f_n(y_star)
        # fprime_star_val = fprime_n(y_star)
        f_star_val = f_n(t_n + h_star, y_star)
        fprime_star_val = fprime_n(t_n + h_star, y_star)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # reconstruct approximation y_{n+1} of the 4th order
        rho = h / h_star
        y_n1 = y_n \
               + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
               + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
               + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
               + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val
        print('y_n1 = %e' % y_n1)

        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_3rd = y_n \
                   + h * (1 - rho ** 2) * f_n_val \
                   + h * rho ** 2 * f_star_val \
                   + h**2 / 2 * (1 - 4 * rho / 3) * fprime_n_val \
                   + h**2 / 2 * (- 2 * rho / 3) * fprime_star_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_3rd)



        # predicted time-step
        const = 1.1
        p = 4
        h_new = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h

        print('h* = %4.4e             \t (h* pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e' % (h_star, h_star_new, err_star, lte_star))
        print('h  = %4.4e rho = %3.2f \t (h  pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, rho, h_new, err, lte))

        # predict h step
        h = h_estimate(h_star, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val)
        # h = 2 * h_star
        # if predicted h is beyond considered interval [t_0, t_final]
        #if t_n + h > t_final:
        #    # correct the step
        #    h = t_final - t_n

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))
        err_star_array = np.append(err_star_array, np.array([err_star]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals


def adaptive_tdrk_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0

    # auxiliary functions estimating an intermediate step h_star
    def h_star_estimate(y_n, fprime_n):
        return math.sqrt(2 * (eps_rel * math.fabs(y_n) + eps_abs) / math.fabs(fprime_n))

    #
    def h_estimate(h_star, y_n, f_n, fprime_n, f_star, fprime_star):
        C = math.fabs(  1 / (2 * h_star ** 3) * (f_n - f_star)
                      + 1 / (4 * h_star ** 2) * (fprime_n + fprime_star))
        return math.pow((eps_rel * math.fabs(y_n) + eps_abs) / C, 0.25)

    def norm(val):
        return math.fabs(val)

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))
    err_star_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        print('t = %4.4e' % (t_n))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        #f_n_val = f_n(y_n)
        #fprime_n_val = fprime_n(y_n)
        #f_evals += 2
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val)
        else:
            #h = h_new
            h = h_pred

        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n

        # reconstruct approximation y_star with the 2nd order tdrk
        y_star = y_n + h / 2 * f_n_val + h ** 2 / 8 * fprime_n_val

        # evaluate f_* and (f')_*
        #f_star_val = f_n(y_star)
        #fprime_star_val = fprime_n(y_star)
        #f_evals += 2
        f_star_val = f_n(t_n + h / 2, y_star)
        fprime_star_val = fprime_n(t_n + h / 2, y_star)
        f_evals += 2

        # reconstruct approximation y_{n+1} of the 4th order
        y_n1 = y_n \
               + h * f_n_val \
               + h**2 / 6 * fprime_n_val \
               + h**2 / 3 * fprime_star_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n \
                   + h * f_n_val \
                   + h**2 / 2 * fprime_n_val

        # analysis of the errors
        err = norm(y_n1 - y(t_n + h))
        lte = norm(y_n1 - y_n1_2nd)

        # predicted time-step
        const = 0.9
        p = 4
        h_pred = h_estimate(h/2, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val)
        if lte != 0.0:
            h_new = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h
        else:
            h_new = const * h
        print('h  = %4.4e \t (h_pred = %4.4e, h_new = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, h_pred, h_new, err, lte))

        t_array = np.append(t_array, np.array([t_n + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t_n + h)]))

        t_n += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals


def adaptive_classic_rk_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path, test_params):
    # initial data
    t_n = t_0
    y_n = y_0

    # number of time-steps
    n = 0
    f_evals = 0
    rej_num = 0

    # arrays with the test data
    yn_array = np.append(np.array([]), np.array([y_0]))
    y_array = np.append(np.array([]), np.array([y_0]))
    t_array = np.append(np.array([]), np.array([t_0]))
    h_array = np.append(np.array([]), np.array([]))
    err_array = np.append(np.array([]), np.array([]))

    # loop until we are inside interval [t_0, t_final]
    while t_n < t_final:

        eps_n = eps_rel * math.fabs(y_n) + eps_abs
        # evaluate f_n and (f')_n
        f_n_val = f_n(t_n, y_n)
        fprime_n_val = fprime_n(t_n, y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0 and rej_num == 0:
            h = h_star_estimate_1st(f_n_val, eps_n)
        else:
            h = h_pred

        if t_n + h > t_final:
            # correct the step
            h = t_final - t_n

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

        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_2nd = y_n + h * k_2

        # analysis of the errors
        e_glob = norm(y_n1 - y(t_n + h))
        e_loc = norm(y_n1 - y_n1_2nd)

        # predicted time-step
        const = 1.0
        p = 4
        if e_loc < eps_n:
            if e_loc != 0.0:
                h_pred = 1.1 * const * math.pow(math.fabs(eps_abs / e_loc), 1 / (p + 1)) * h
            else:
                h_pred = 1.1 * const * h

            t_array = np.append(t_array, np.array([t_n + h]))
            h_array = np.append(h_array, np.array([h]))

            err_array = np.append(err_array, np.array([e_glob]))

            yn_array = np.append(yn_array, np.array([y_n1]))
            y_array = np.append(y_array, np.array([y(t_n + h)]))

            t_n += h
            y_n = y_n1
            n += 1
            '''
            print('t = %4.4e, eps_n = %4.4e' % (t_n, eps_n))
            print('% -------------------------------------------------------------------------------------------- %')
            print('h = %4.4e \t h_pred = %4.4e\te_glo = %4.4e\tl_loc = %4.4e\n' % (h, h_pred, e_glob, e_loc))
            '''
        else:
            h_pred = 0.5 * math.pow(math.fabs(eps_abs / e_loc), 1 / (p + 1)) * h
            print('y_n1 is rejected: decreasen step from h = %4.4e to h_new = %4.4e' % (h, h_pred))
            rej_num += 1

    # plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    print('eps_abs = %4.4e\te_glob = %4.4e\te_loc = %4.4e\n' % (eps_abs, e_glob, e_loc))
    return e_loc, e_glob, n, f_evals, rej_num
