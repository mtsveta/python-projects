from explicit_schemes import AdaptiveTDRK2Scheme, AdaptiveClassicTDRK2Scheme, AdaptiveTDRK4Scheme, AdaptiveTaylor4Scheme, UniformTDRK4Scheme
from test import Test
from odes import ODE

import numpy as np

def test_uniform_tdrk4_scheme():
    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODE(y, f, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    tdrk4 = UniformTDRK4Scheme(ode)

    # define time stepping array
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -2), math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])

    test = Test(examples[example_num], test_params, tdrk4, 'system-example-')

    test.test_uniform(dt_array)
    test.plot_uniform_results()


def test_adaptive_tdrk4_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODE(y, f, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk4 = AdaptiveTDRK4Scheme(ode)

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk4, 'ode-example-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('adaptive-tdrk4-')

    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])


def test_adaptive_tdrk2_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODE(y, f, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK2Scheme(ode)

    # define tolerances
    factor = 1e-2
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk2, 'ode-example-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('adaptive-tdrk2-')

    '''
    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])
    '''

def test_uniform_tdrk4_scheme():
    # define ODE
    t_0 = 0.0
    t_fin = math.pi
    ode = ODE(y, f, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    tdrk4 = UniformTDRK4Scheme(ode)

    # define time stepping array
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])

    test = Test(examples[example_num], test_params, tdrk4, 'ode-')

    test.test_uniform(dt_array)
    test.plot_results('uniform-tdrk4-scheme-')

def test_comparison_adaptive_tdrk2_schemes():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODE(y, f, dfdt, d2fdt2, d3fdt3, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK2Scheme(ode, 'tdrk2 (our h)')
    adapt_classic_tdrk2 = AdaptiveClassicTDRK2Scheme(ode, 'tdrk2 (classic h)')

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' classic tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_aclassic_tdrk2 = Test(examples[example_num], test_params, adapt_classic_tdrk2, 'comparison-tdrk2-and-4-example-')
    test_aclassic_tdrk2.test_adaptive(eps_abs, eps_rel)
    test_aclassic_tdrk2.plot_results('adaptive-classi-tdrk2-')

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_adapt_tdrk2 = Test(examples[example_num], test_params, adapt_tdrk2, 'comparison-tdrk2-and-4-example-')
    test_adapt_tdrk2.test_adaptive(eps_abs, eps_rel)
    test_adapt_tdrk2.plot_results('adaptive-tdrk2-')


    test_adapt_tdrk2.compare_results([test_aclassic_tdrk2], ['tdrk2 (classic h)'], 'tdrk2s-')

def test_comparison_adaptive_taylor4_tdrk4():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODE(y, f, dfdt,  d2fdt2, d3fdt3, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_taylor = AdaptiveTaylor4Scheme(ode, 'taylor4')
    adapt_tdrk4 = AdaptiveTDRK4Scheme(ode, 'tdrk4')

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    #eps_abs = np.array([1e-8])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive taylor4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_taylor = Test(examples[example_num], test_params, adapt_taylor, 'example-')
    test_taylor.test_adaptive(eps_abs, eps_rel)
    #test_taylor.plot_results('adaptive-taylor4-')

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_tdrk4 = Test(examples[example_num], test_params, adapt_tdrk4, 'example-')
    test_tdrk4.test_adaptive(eps_abs, eps_rel)
    #test_tdrk4.plot_results('adaptive-tdrk4-')


    test_tdrk4.compare_results([test_taylor], ['tdrk4', 'taylor'], 'taylor4-tdrk4-')

if __name__== "__main__":

    examples = [1, 2, 3, 4, 5, 6, 7]
    test_params = dict(test_log=True,
                       scheme_log=False)

    for example_num in range(0, len(examples)):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' Example %d' % examples[example_num])
        print('% -------------------------------------------------------------------------------------------- %\n')

        if examples[example_num] == 1:
            from example1 import *
        elif examples[example_num] == 2:
            from example2 import *
        elif examples[example_num] == 3:
            from example3 import *
        elif examples[example_num] == 4:
            from example4 import *
        elif examples[example_num] == 5:
            from example5 import *
        elif examples[example_num] == 6:
            from example6 import *
        elif examples[example_num] == 7:
            from example7 import *

        #test_uniform_tdrk4_scheme()
        #test_adaptive_tdrk4_scheme()
        #test_adaptive_tdrk2_scheme()
        #test_uniform_tdrk4_scheme()
        #test_comparison_adaptive_tdrk2_schemes()
        test_comparison_adaptive_taylor4_tdrk4()
