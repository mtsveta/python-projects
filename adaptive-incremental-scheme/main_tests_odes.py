#from explicit_schemes import AdaptiveTDRK4Scheme, UniformTDRK4Scheme
from AdaptiveTDRK4Scheme import AdaptiveTDRK4Scheme
from UniformTDRK4Scheme import UniformTDRK4Scheme
from Test import Test
from odes import ODE

import numpy as np

def test_uniform_tdrk4_scheme():
    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODE(y, f, fprime, f_n, fprime_n, t_0, t_fin)

    # define the schemes to test
    tdrk4 = UniformTDRK4Scheme(ode)

    # define time stepping array
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -2), math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])

    test = Test(examples[example_num], test_params, tdrk4)

    test.test_uniform(dt_array)
    test.plot_uniform_results()


def test_adaptive_tdrk4_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODE(y, f, fprime, f_n, fprime_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk4 = AdaptiveTDRK4Scheme(ode)

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    eps_rel = factor * eps_abs

    test = Test(examples[example_num], test_params, adapt_tdrk4)
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_adaptive_results('adaptive-tdrk4-')

    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])


if __name__== "__main__":

    examples = [1, 2]
    test_params = dict(test_log=True,
                       scheme_log=True)  # just tested for example 1, the import of each example +
    # implementation of derivatives is needed

    for example_num in range(0, len(examples)):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' Example %d' % examples[example_num])
        print('% -------------------------------------------------------------------------------------------- %\n')

        if examples[example_num] == 1:
            from example1 import *
        elif examples[example_num] == 2:
            from example2 import *

        test_adaptive_tdrk4_scheme();
        #test_uniform_tdrk4_scheme();
