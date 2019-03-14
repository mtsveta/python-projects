from explicit_schemes import \
    AdaptiveTDRK2Scheme, AdaptiveClassicTDRK2Scheme, AdaptiveTDRK4Scheme, AdaptiveTaylor4Scheme, UniformTDRK4Scheme
from explicit_schemes import AdaptiveTDRK4SingleRateScheme, AdaptiveTDRK2SingleRateScheme, \
    AdaptiveTDRK4MiltiRateScheme, AdaptiveTDRK2MultiRateScheme

from test import Test
from odes import ODEs

import math

import numpy as np

def test_uniform_tdrk4_scheme():
    # define ODE
    t_0 = 0.0
    t_fin = math.pi
    ode = ODEs(y, f, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    tdrk4 = UniformTDRK4Scheme(ode)

    # define time stepping array
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -2), math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])
    '''
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -2), math.pow(2, -4), math.pow(2, -6), math.pow(2, -7)])
    '''
    test = Test(examples[example_num], test_params, tdrk4, 'system-example-')

    test.test_uniform(dt_array)
    test.plot_uniform_results()

def test_adaptive_tdrk4_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk4 = AdaptiveTDRK4SingleRateScheme(ode, 'tdrk4')

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk4, 'system-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('adaptive-tdrk4-')

    # check if test is passed
    '''
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])
    '''

def test_comparison_srtdrk2_mrtdrk2():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin, F, JF_y, dFdt)

    # define the schemes to test
    tdrk2_sr = AdaptiveTDRK2SingleRateScheme(ode, 'sr-tdrk2')
    tdrk2_mr = AdaptiveTDRK2MultiRateScheme(ode, 'mr-tdrk2')

    # define tolerances
    factor = 1e0
    eps_abs = np.array([1e-2, 1e-4, 1e-6])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' multirate adaptive tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_sr = Test(examples[example_num], test_params, tdrk2_sr, 'system-')
    test_sr.test_adaptive(eps_abs, eps_rel)

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk2 (2 * neq function call per step)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test_mr = Test(examples[example_num], test_params, tdrk2_mr, 'system-')
    test_mr.test_adaptive(eps_abs, eps_rel)

    test_sr.compare_results([test_mr], ['sr-tdrk2', 'mr-tdrk2'], 'sr-vs-mr-tdrk2-')

def test_adaptive_multirate_tdrk2_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin, F, JF_y, dFdt)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK2MultiRateScheme(ode, 'mr-tdrk2')
    # define tolerances
    factor = 1e0
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    eps_abs = np.array([1e-2, 1e-4, 1e-6])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' multirate adaptive tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk2, 'system-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('mr-tdrk2-')

    '''
    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])
    '''

def test_adaptive_tdrk2_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin, F, JF_y, dFdt)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK2SingleRateScheme(ode, 'sr-tdrk2')
    # define tolerances
    factor = 1e-2
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-7])
    eps_abs = np.array([1e-2, 1e-4, 1e-6])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk2 (2 * neq function call per step)')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk2, 'system-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('adaptive-tdrk2-')

    '''
    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])
    '''
def test_adaptive_tdrk4_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin, F, JF_y, dFdt)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK4SingleRateScheme(ode, 'tdrk4')
    # define tolerances
    factor = 1e-2
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-7])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk4')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk2, 'system-')
    test.test_adaptive(eps_abs, eps_rel)
    test.plot_results('adaptive-tdrk2-')

    '''
    # check if test is passed
    for i in range(0, len(eps_abs)):
        success = (test.e_glob[i] < eps_abs[i] + eps_rel[i])
        assert success, "eps = %4.4e\t e_glob = %4.4e" % (eps_abs[i] + eps_rel[i], test.e_glob[i])
    '''
def test_adaptive_multirate_tdrk4_scheme():

    # define ODE
    t_0 = 0.0
    t_fin = 5.0
    ode = ODEs(y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    adapt_tdrk2 = AdaptiveTDRK4MiltiRateScheme(ode, 'mr-tdrk2')
    # define tolerances
    factor = 1e-2
    eps_abs = np.array([1e-2, 1e-4, 1e-6, 1e-7])
    #eps_abs = np.array([1e-2, 1e-4, 1e-6])
    eps_rel = factor * eps_abs

    print('% -------------------------------------------------------------------------------------------- %')
    print(' adaptive tdrk2')
    print('% -------------------------------------------------------------------------------------------- %\n')

    test = Test(examples[example_num], test_params, adapt_tdrk2, 'system-')
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
    ode = ODEs(y, f, dfdt, dfdt, dfdt, f_n, dfdt_n, t_0, t_fin)

    # define the schemes to test
    tdrk4 = UniformTDRK4Scheme(ode, 'uniform tdrk4')

    # define time stepping array
    dt_array = (t_fin - t_0) \
               * np.array([math.pow(2, -4), math.pow(2, -6), math.pow(2, -7),
                           math.pow(2, -8), math.pow(2, -10), math.pow(2, -12), math.pow(2, -14)])

    test = Test(examples[example_num], test_params, tdrk4, 'system-example-')

    test.test_uniform(dt_array)
    test.plot_results('uniform-tdrk4-scheme-')

def test_comparison_adaptive_tdrk2_schemes():

    # define ODE
    t_0 = 0.0
    t_fin = 3.0
    ode = ODEs(y, f, dfdt, dfdt, dfdt, f_n, dfdt_n, t_0, t_fin)

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
    ode = ODEs(y, f, dfdt,  dfdt, dfdt, f_n, dfdt_n, t_0, t_fin)

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

    # 1 is working
    # 4 is working
    examples = [2]
    # , 4, 5
    #examples = [3, 4, 5]
    test_params = dict(test_log=True,
                       scheme_log=False)

    for example_num in range(0, len(examples)):
        print('% -------------------------------------------------------------------------------------------- %')
        print(' Example %d' % examples[example_num])
        print('% -------------------------------------------------------------------------------------------- %\n')

        if examples[example_num] == 1:
            from system1 import *
        elif examples[example_num] == 2:
            from system2 import *
        elif examples[example_num] == 3:
            from system3 import *
        elif examples[example_num] == 4:
            from system4 import *
        elif examples[example_num] == 5:
            from system5 import *

        #test_adaptive_tdrk4_scheme()
        #test_adaptive_tdrk2_scheme()
        #test_adaptive_multirate_tdrk2_scheme()

        test_comparison_srtdrk2_mrtdrk2()

        print("")
        #test_uniform_tdrk4_scheme()

