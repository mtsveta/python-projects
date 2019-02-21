import math
import numpy as np

from plotting import *

def adaptive_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path):
    # initial data
    t = t_0
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
    while t < t_final:

        print('t = %4.4e' % (t))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(y_n)
        fprime_n_val = fprime_n(y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        h_star = h_star_estimate(y_n, fprime_n_val)
        if t + h_star > t_final:
            # correct the step
            h_star = t_final - t

        # reconstruct approximation y_star with the 4th order tdrk
        #'''
        y_aux = y_n + h_star / 2 * f_n_val + h_star ** 2 / 8 * fprime_n_val
        fprime_aux_val = fprime_n(y_aux)
        y_star = y_n \
               + h_star * f_n_val \
               + h_star ** 2 / 6 * fprime_n_val \
               + h_star ** 2 / 3 * fprime_aux_val
        f_evals += 1
        #'''
        # reconstruct approximation y_star with the 2nd order tdrk
        #y_star = y_n + h_star * f_n_val + h_star ** 2 / 2 * fprime_n_val
        # reconstruct approximation for the 1st order
        y_1st = y_n + h_star * f_n_val

        # analysis of the errors of the h_star step
        err_star = norm(y_star - y(t + h_star))
        lte_star = norm(y_star - y_1st)

        # predicted time-step
        const = 1.1
        p = 2
        h_star_new = const * math.pow(math.fabs(eps_abs / lte_star), 1 / (p + 1)) * h_star

        # evaluate f_* and (f')_*
        f_star_val = f_n(y_star)
        fprime_star_val = fprime_n(y_star)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h step
        h = h_estimate(h_star, y_n, f_n_val, fprime_n_val, f_star_val, fprime_star_val)
        #h = 2 * h_star
        # if predicted h is beyond considered interval [t_0, t_final]
        if t + h > t_final:
            # correct the step
            h = t_final - t
        # reconstruct approximation y_{n+1} of the 4th order
        rho = h / h_star
        #y_n1 = y(t + h)
        y_n1 = y_n \
               + h * (1 - rho ** 2 + rho ** 3 / 2) * f_n_val \
               + h * (rho ** 2 - rho ** 3 / 2) * f_star_val \
               + h**2 / 2 * (1 - 4 * rho / 3 + rho ** 2 / 2) * fprime_n_val \
               + h**2 / 2 * (- 2 * rho / 3 + rho ** 2 / 2) * fprime_star_val
        # reconstruct approximation y_{n+1} of the 3rd order
        y_n1_3rd = y_n \
                   + h * (1 - rho ** 2) * f_n_val \
                   + h * rho ** 2 * f_star_val \
                   + h**2 / 2 * (1 - 4 * rho / 3) * fprime_n_val \
                   + h**2 / 2 * (- 2 * rho / 3) * fprime_star_val

        # analysis of the errors
        err = norm(y_n1 - y(t + h))
        lte = norm(y_n1 - y_n1_3rd)

        # predicted time-step
        const = 1.1
        p = 4
        h_new = const * math.pow(math.fabs(eps_abs / lte), 1 / (p + 1)) * h

        print('h* = %4.4e             \t (h* pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e' % (h_star, h_star_new, err_star, lte_star))
        print('h  = %4.4e rho = %3.2f \t (h  pred/rej = %4.4e)\terr = %4.4e\tlte = %4.4e\n' % (h, rho, h_new, err, lte))

        t_array = np.append(t_array, np.array([t + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))
        err_star_array = np.append(err_star_array, np.array([err_star]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t + h)]))

        t += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals


def adaptive_tdrk_4th_order(eps_rel, eps_abs, t_0, t_final, y_0, y, f_n, fprime_n, result_path):

    # initial data
    t = t_0
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
    while t < t_final:

        print('t = %4.4e' % (t))
        print('% -------------------------------------------------------------------------------------------- %')

        # evaluate f_n and (f')_n
        f_n_val = f_n(y_n)
        fprime_n_val = fprime_n(y_n)
        f_evals += 2

        # h* step
        # --------------------------------------------------------------------------------------------------------------
        # predict h_star step
        if n == 0:
            h = h_star_estimate(y_n, fprime_n_val)
        else:
            #h = h_new
            h = h_pred

        if t + h > t_final:
            # correct the step
            h = t_final - t

        # reconstruct approximation y_star with the 2nd order tdrk
        y_star = y_n + h / 2 * f_n_val + h ** 2 / 8 * fprime_n_val

        # evaluate f_* and (f')_*
        f_star_val = f_n(y_star)
        fprime_star_val = fprime_n(y_star)
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
        err = norm(y_n1 - y(t + h))
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

        t_array = np.append(t_array, np.array([t + h]))
        h_array = np.append(h_array, np.array([h]))

        err_array = np.append(err_array, np.array([err]))

        yn_array = np.append(yn_array, np.array([y_n1]))
        y_array = np.append(y_array, np.array([y(t + h)]))

        t += h
        y_n = y_n1
        n += 1

    #plot_results(t_array, yn_array, y_array, h_array, err_array, result_path)
    return lte, err, n, f_evals