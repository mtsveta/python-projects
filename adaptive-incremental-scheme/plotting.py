import matplotlib.pyplot as plt
import math
import numpy as np
import os, sys

def get_project_path():
    (project_path, src_folder) = os.path.split(os.path.abspath(os.path.dirname(__file__)))
    return project_path + '/' + src_folder

def create_results_folder(directory):
    # create the full directory of the result folder
    full_directory = get_project_path() + '/results/' + directory

    # if directory doesn't exist
    if not os.path.exists(full_directory):
        os.makedirs(full_directory)
    # writing to the log file
    # sys.stdout = open(full_directory + '/log-results.txt', "w+")
    return full_directory

def plot_results(time, approx, exact, h, err, result_path):

    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].plot(time, approx,
               color='orchid',
               linestyle='dashed',
               marker='',
               markerfacecolor='orchid',
               markersize=6,
               label=r'$y_n$')
    ax[0].plot(time, exact,
               color='coral',
               linestyle='',
               marker='o',
               markerfacecolor='coral',
               markersize=2,
               label=r'$y$')
    ax[0].legend()
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$y_n, y$')
    #plt.title('Comparison of y_n to exact y(t)')
    ax[0].set_title(r'Comparison of $y_n$ to exact $y(t)$')

    # time-steps wrt time
    ax[1].plot(time[1:], h,
               color='coral',
               linestyle='',
               marker='*',
               markerfacecolor='tan',
               markersize=2,
               label=f'$h$ by out approach')
    ax[1].legend()
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$h$')
    ax[1].set_title('Predicted $h$')

    # time-steps wrt time
    ax[2].semilogy(time[1:],  err,
                   color='red',
                   linestyle=':',
                   marker='',
                   markerfacecolor='tan',
                   markersize=6,
                   label=r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[2].legend()
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('$e$')
    ax[2].set_title('Error')

    plt.show()
    plt.savefig(result_path + ('/adaptive-scheme-approx-error-%d.eps' %(len(time))), dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

def plot_uniform_results(time, approx, exact, err, result_path):
    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(time, approx, color='green', linestyle='dashed', marker='', markerfacecolor='green', markersize=6,
               label=f'y_n')
    ax[0].plot(time, exact, color='blue', linestyle='', marker='o', markerfacecolor='blue', markersize=6, label=f'y')
    # ax[0].plot(time, approx, '--', mew=1, ms=4, xmec='w', label=f'y_n')
    # ax[0].plot(time, exact, 'o-', mew=1, ms=4, mec='w', label=f'y')
    ax[0].legend()
    plt.xlabel('time, t')
    plt.ylabel('y_n, y')
    # plt.title('Comparison of y_n to exact y(t)')
    ax[0].set_title('Comparison of y_n to exact y(t)')

    # time-steps wrt time
    ax[1].semilogy(time[1:], err, color='red', linestyle=':', marker='', markerfacecolor='tan', markersize=6,
                   label=r'$log_10(err)$')
    ax[1].legend()
    plt.xlabel('time, t')
    plt.ylabel('log_10(err)')
    ax[1].set_title('Error')

    plt.show()
    plt.savefig(result_path + '/tdrk-approx-error-%d.eps' %(len(time)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

def plot_convergence(err, n, f_evals, title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].loglog(n, err,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[0].loglog(n, np.power(n, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order convergence')
    ax[0].loglog(n, np.power(n, -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=r'$3$th order convergence')

    ax[0].legend()
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals, err,
                 color='blue',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].legend()

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/adaptive-convergence-%d.eps' % (len(err)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

def plot_uniform_convergence(err, n, f_evals, title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].loglog(n, err,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e$')
    ax[0].loglog(n, np.power(n, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order scheme')

    ax[0].legend()
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel('$e$')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals, err,
                 color='blue',
                 linestyle='',
                 marker='o',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'$e$')
    ax[1].legend()

    plt.xlabel('func. evaluations')
    plt.ylabel('$e$')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/tdrk-convergence-%d.eps' % (len(err)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

def plot_summary_adaptive_convergence(err_our, n_our, f_evals_our,
                                      err_tdrk, n_tdrk, f_evals_tdrk,
                                      title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].loglog(n_our, err_our,
                 color='orchid',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=f'global error / our method')
    ax[0].loglog(n_tdrk, err_tdrk,
                 color='coral',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='coral',
                 markersize=6,
                 label=f'global error / tdrk method')

    ax[0].loglog(n_tdrk, np.power(n_tdrk, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order convergence')
    ax[0].loglog(n_tdrk, np.power(n_tdrk, -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=f'3th order convergence')

    ax[0].legend()
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'e')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals_our, err_our,
                 color='green',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e$ / our method')
    ax[1].loglog(f_evals_tdrk, err_tdrk,
                 color='blue',
                 linestyle=':',
                 marker='x',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'$e$ / tdrk method')

    ax[1].legend()

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/adaptive-convergence-summary-our-%d-tdrk-%d.eps' % (len(err_our), len(err_tdrk)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


def plot_summary_uniform_convergence(err_rk, n_rk, f_evals_rk,
                                     err_tdrk, n_tdrk, f_evals_tdrk,
                                     title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].loglog(n_rk, err_rk,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=f'global error / rk method')
    ax[0].loglog(n_tdrk, err_tdrk,
                 color='blue',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='blue',
                 markersize=10,
                 label=f'global error / tdrk method')

    ax[0].loglog(n_tdrk, np.power(n_tdrk, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order convergence')

    ax[0].legend()
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'e')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals_rk, err_rk,
                 color='green',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e$ / rk method')
    ax[1].loglog(f_evals_tdrk, err_tdrk,
                 color='blue',
                 linestyle=':',
                 marker='x',
                 markerfacecolor='blue',
                 markersize=10,
                 label=r'$e$ / tdrk method')

    ax[1].legend()

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/adaptive-convergence-summary-our-%d-tdrk-%d.eps' % (len(err_rk), len(err_tdrk)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)



def plot_summary_adaptive_uniform_convergence(err_our, n_our, f_evals_our,
                                      err_pred_rej, n_pred_rej, f_evals_pred_rej,
                                      err_rk, n_rk, f_evals_rk,
                                      err_tdrk, n_tdrk, f_evals_tdrk,
                                      title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].loglog(n_our, err_our,
                 color='coral',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='coral',
                 markersize=6,
                 label=f'global error / our method')
    ax[0].loglog(n_pred_rej, err_pred_rej,
                 color='orchid',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orchid',
                 markersize=6,
                 label=f'global error / adaptive tdrk method')

    ax[0].loglog(n_rk, err_rk,
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=f'global error / rk method')
    ax[0].loglog(n_tdrk, err_tdrk,
                 color='blue',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='blue',
                 markersize=6,
                 label=f'global error / tdrk method')

    ax[0].loglog(n_tdrk, np.power(n_tdrk, -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=f'4th order convergence')
    ax[0].loglog(n_tdrk, np.power(n_tdrk, -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=f'3th order convergence')

    ax[0].legend()
    ax[0].set_xlabel(r'number of steps, $N$')
    ax[0].set_ylabel(r'e')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation

    ax[1].loglog(f_evals_our, err_our,
                 color='coral',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='coral',
                 markersize=6,
                 label=f'global error / our method')
    ax[1].loglog(f_evals_pred_rej, err_pred_rej,
                 color='orchid',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orchid',
                 markersize=12,
                 label=f'global error / adaptive tdrk method')

    ax[1].loglog(f_evals_rk, err_rk,
                 color='green',
                 linestyle=':',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'$e$ / rk method')
    ax[1].loglog(f_evals_tdrk, err_tdrk,
                 color='blue',
                 linestyle=':',
                 marker='x',
                 markerfacecolor='blue',
                 markersize=12,
                 label=r'$e$ / tdrk method')

    ax[1].legend()

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'$e = \|y_{n+1} - y(t_{final})\|$')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/adaptive-convergence-summary-our-%d-tdrk-%d.eps' % (len(err_our), len(err_tdrk)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

