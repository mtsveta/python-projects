import matplotlib.pyplot as plt
import math
import numpy as np
import os, sys


colors = ['aqua', 'darkblue',
                 'coral', 'crimson',
                 'green', 'darkgreen',
                 'orange', 'darkorange',
                 'pink', 'maroon',
                 'gold', 'brown']

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
    # sys.stdout = open(full_directory + '/log-results-2nd-order-middle-step.txt', "w+")
    return full_directory

def plot_approximation_result(time, approx, exact, h, e_glo, e_loc, method_name, method_tag, result_path):

    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
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
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$y_n, y$')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)
    ax[0].set_title(method_name)
    # time-steps wrt time
    ax[1].plot(time[1:], h,
               color='coral',
               linestyle='',
               marker='*',
               markerfacecolor='tan',
               markersize=2,
               label=f'$h$')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('predicted $h$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # time-steps wrt time
    ax[2].semilogy(time[1:], e_glo,
                   color='red',
                   linestyle=':',
                   marker='',
                   markerfacecolor='tan',
                   markersize=6,
                   label=r'global $e$')
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('global $e$')
    ax[2].grid(True, color='gray', linestyle=':', linewidth=0.5)

    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + ('/' + method_tag + 'approx-error-%d.eps' % len(time)),
                dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)

def plot_uniform_results(e_loc_unif, e_glo_unif, n_unif, f_evals_unif, cpu_time_unif, tag, result_path):
    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].loglog(n_unif[0, :], e_glo_unif[0, :],
                 color='blueviolet',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='blueviolet',
                 markersize=6,
                 label=r'tdrk (uniform)')
    ax[0].loglog(n_unif[1, :], e_glo_unif[1, :],
                 color='deepskyblue',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='deepskyblue',
                 markersize=6,
                 label=r'rk (uniform)')

    ax[0].loglog(n_unif[2, :], e_glo_unif[2, :],
                 color='darkorange',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='darkorange',
                 markersize=6,
                 label=r'our, rho = 2 (uniform)')

    ax[0].loglog(n_unif[3, :], e_glo_unif[3, :],
                 color='dimgrey',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='dimgrey',
                 markersize=6,
                 label=r'our, rho = 3 (uniform)')

    ax[0].loglog(n_unif[4, :], e_glo_unif[4, :],
                 color='hotpink',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='hotpink',
                 markersize=6,
                 label=r'our, rho = 4 (uniform)')

    # plot different convergences
    ax[0].loglog(n_unif[1, :], np.power(n_unif[1, :], -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=r'$3$th order')
    ax[0].loglog(n_unif[1, :], np.power(n_unif[1, :], -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(r'global $e$')
    ax[0].set_title('Uniform: errors wrt # of steps / func. evals.')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    ax[1].loglog(cpu_time_unif[0, :], e_glo_unif[0, :],
                 color='blueviolet',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='blueviolet',
                 markersize=6,
                 label=r'tdrk scheme (uniform)')
    ax[1].loglog(cpu_time_unif[1, :], e_glo_unif[1, :],
                 color='deepskyblue',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='deepskyblue',
                 markersize=6,
                 label=r'rk (uniform)')

    ax[1].loglog(cpu_time_unif[2, :], e_glo_unif[2, :],
                 color='darkorange',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='darkorange',
                 markersize=6,
                 label=r'our, rho = 2 (uniform)')

    ax[1].loglog(cpu_time_unif[3, :], e_glo_unif[3, :],
                 color='dimgrey',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='dimgrey',
                 markersize=6,
                 label=r'our, rho = 3 (uniform)')

    ax[1].loglog(cpu_time_unif[4, :], e_glo_unif[4, :],
                 color='hotpink',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='hotpink',
                 markersize=6,
                 label=r'our, rho = 4 (uniform)')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[1].set_xlabel(r'cpu time')
    ax[1].set_ylabel(r'global $e$')
    ax[1].set_title('Uniform: errors wrt # of steps / func. evals.')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    fig.subplots_adjust(right=0.6)
    fig.show()
    fig.savefig(result_path + '/' + tag + 'uniform-error-%d.eps' %(len(e_glo_unif)),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)

def plot_convergence(e_loc, e_glo, n, f_evals, cpu_time, labels, tag, result_path):

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    # plot global errors wrt number of steps
    (cols, rows) = e_glo.shape
    '''
    for i in range(cols):
        ax[0].loglog(n[i, :], e_glo[i, :],
                     color=colors[i],
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor=colors[i],
                     markersize=6,
                     label=labels[i])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(r'global $e$')
    ax[0].set_title('Adaptive: errors wrt # of steps / func. evals.')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)
    '''
    # plot global error wrt number of functions evaluation
    for i in range(cols):
        ax[0].loglog(f_evals[i, :], e_glo[i, :],
                     color=colors[i],
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor=colors[i],
                     markersize=6,
                     label=labels[i])
    # plot different convergences
    ax[0].loglog(f_evals[0, :], np.power(f_evals[0, :], -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=r'$3$th order')
    ax[0].loglog(f_evals[0, :], np.power(f_evals[0, :], -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order')

    '''
    ax[1].loglog(f_evals[0, :], e_glo[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'taylor')
    ax[1].loglog(f_evals[1, :], e_glo[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'tdrk (our h pred.)')
    ax[1].loglog(f_evals[2, :], e_glo[2, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'tdrk (classic h pred.)')
    ax[1].loglog(f_evals[3, 0:3], e_glo[3, 0:3],
                 color='orange',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orange',
                 markersize=6,
                 label=r'rk (classic h pred.)')
    '''
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[0].set_xlabel(r'func. evaluations')
    ax[0].set_ylabel(r'global $e$')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation

    for i in range(cols):
        ax[1].loglog(cpu_time[i, :], e_glo[i, :],
                     color=colors[i],
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor=colors[i],
                     markersize=6,
                     label=labels[i])
    '''
    ax[2].loglog(cpu_time[0, :], e_glo[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'taylor')
    ax[2].loglog(cpu_time[1, :], e_glo[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'tdrk (our h pred.)')
    ax[2].loglog(cpu_time[2, :], e_glo[2, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'tdrk (classic h pred.)')
    ax[2].loglog(cpu_time[3, 0:3], e_glo[3, 0:3],
                 color='orange',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orange',
                 markersize=6,
                 label=r'rk (classic h pred.)')
    '''
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'cpu time')
    ax[1].set_ylabel(r'global $e$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)
    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + tag + 'adaptive-conv-%d.eps' % (len(e_glo[0, :])),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)
def plot_convergence_summary(e_loc, e_glo, n, f_evals, cpu_time,
                             e_loc_unif, e_glo_unif, n_unif, f_evals_unif, cpu_time_unif, tag, result_path):

    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    # plot global errors wrt number of steps
    ax[0].loglog(n[0, :], e_glo[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'our scheme (our h pred.)')
    ax[0].loglog(n[1, :], e_glo[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'tdrk (our h pred.)')
    ax[0].loglog(n[2, :], e_glo[2, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'tdrk (classic h pred.)')
    ax[0].loglog(n[3, 0:3], e_glo[3, 0:3],
                 color='orange',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orange',
                 markersize=6,
                 label=r'rk (classic h pred.)')
    # plot uniform schemes
    ax[0].loglog(n_unif[0, :], e_glo_unif[0, :],
                 color='blueviolet',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='blueviolet',
                 markersize=6,
                 label=r'tdrk (uniform)')
    ax[0].loglog(n_unif[1, :], e_glo_unif[1, :],
                 color='deepskyblue',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='deepskyblue',
                 markersize=6,
                 label=r'rk (uniform)')
    ax[0].loglog(n_unif[2, :], e_glo_unif[2, :],
                 color='darkorange',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='darkorange',
                 markersize=6,
                 label=r'our scheme, rho = 2 (uniform)')
    ax[0].loglog(n_unif[3, :], e_glo_unif[3, :],
                 color='dimgrey',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='dimgrey',
                 markersize=6,
                 label=r'our scheme, rho = 3 (uniform)')
    ax[0].loglog(n_unif[4, :], e_glo_unif[4, :],
                 color='hotpink',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='hotpink',
                 markersize=6,
                 label=r'our scheme, rho = 4 (uniform)')

    # plot different convergences
    ax[0].loglog(n[1, :], np.power(n[1, :], -3),
                 color='gray',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='gray',
                 markersize=6,
                 label=r'$3$th order')
    ax[0].loglog(n[1, :], np.power(n[1, :], -4),
                 color='tan',
                 linestyle='dashed',
                 marker='',
                 markerfacecolor='tan',
                 markersize=6,
                 label=r'$4$th order')

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    ax[0].set_xlabel(r'number of steps')
    ax[0].set_ylabel(r'global $e$')
    ax[0].set_title('Adaptive: errors wrt # of steps / func. evals.')
    ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals[0, :], e_glo[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'our scheme (our h pred.)')
    ax[1].loglog(f_evals[1, :], e_glo[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'tdrk (our h pred.)')
    ax[1].loglog(f_evals[2, :], e_glo[2, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'tdrk (classic h pred.)')
    ax[1].loglog(f_evals[3, 0:3], e_glo[3, 0:3],
                 color='orange',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orange',
                 markersize=6,
                 label=r'rk (classic h pred.)')
    ax[1].loglog(f_evals_unif[0, :], e_glo_unif[0, :],
                 color='blueviolet',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='blueviolet',
                 markersize=6,
                 label=r'tdrk scheme (uniform)')
    ax[1].loglog(f_evals_unif[1, :], e_glo_unif[1, :],
                 color='deepskyblue',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='deepskyblue',
                 markersize=6,
                 label=r'rk (uniform)')
    ax[1].loglog(f_evals_unif[2, :], e_glo_unif[2, :],
                 color='darkorange',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='darkorange',
                 markersize=6,
                 label=r'our scheme, rho = 2 (uniform)')
    ax[1].loglog(f_evals_unif[3, :], e_glo_unif[3, :],
                 color='dimgrey',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='dimgrey',
                 markersize=6,
                 label=r'our scheme, rho = 3 (uniform)')
    ax[1].loglog(f_evals_unif[4, :], e_glo_unif[4, :],
                 color='hotpink',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='hotpink',
                 markersize=6,
                 label=r'our scheme, rho = 4 (uniform)')

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[1].set_xlabel(r'func. evaluations')
    ax[1].set_ylabel(r'global $e$')
    ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

    # plot global error wrt number of functions evaluation
    ax[2].loglog(cpu_time[0, :], e_glo[0, :],
                 color='green',
                 linestyle='dashed',
                 marker='o',
                 markerfacecolor='green',
                 markersize=6,
                 label=r'our scheme (our h pred.)')
    ax[2].loglog(cpu_time[1, :], e_glo[1, :],
                 color='blue',
                 linestyle='dashed',
                 marker='^',
                 markerfacecolor='blue',
                 markersize=6,
                 label=r'tdrk (our h pred.)')
    ax[2].loglog(cpu_time[2, :], e_glo[2, :],
                 color='darkred',
                 linestyle='dashed',
                 marker='d',
                 markerfacecolor='darkred',
                 markersize=6,
                 label=r'tdrk (classic h pred.)')
    ax[2].loglog(cpu_time[3, 0:3], e_glo[3, 0:3],
                 color='orange',
                 linestyle='dashed',
                 marker='x',
                 markerfacecolor='orange',
                 markersize=6,
                 label=r'rk (classic h pred.)')

    ax[2].loglog(cpu_time_unif[0, :], e_glo_unif[0, :],
                 color='blueviolet',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='blueviolet',
                 markersize=6,
                 label=r'tdrk (uniform)')
    ax[2].loglog(cpu_time_unif[1, :], e_glo_unif[1, :],
                 color='deepskyblue',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='deepskyblue',
                 markersize=6,
                 label=r'rk (uniform)')

    ax[2].loglog(cpu_time_unif[2, :], e_glo_unif[2, :],
                 color='darkorange',
                 linestyle='dashed',
                 marker='+',
                 markerfacecolor='darkorange',
                 markersize=6,
                 label=r'our scheme, rho = 2 (uniform)')
    ax[2].loglog(cpu_time_unif[3, :], e_glo_unif[3, :],
                 color='dimgrey',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='dimgrey',
                 markersize=6,
                 label=r'our scheme, rho = 3 (uniform)')
    ax[2].loglog(cpu_time_unif[4, :], e_glo_unif[4, :],
                 color='hotpink',
                 linestyle='dashed',
                 marker='*',
                 markerfacecolor='hotpink',
                 markersize=6,
                 label=r'our scheme, rho = 4 (uniform)')
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax[2].set_xlabel(r'cpu time')
    ax[2].set_ylabel(r'global $e$')
    ax[2].grid(True, color='gray', linestyle=':', linewidth=0.5)
    plt.subplots_adjust(right=0.6)
    plt.show()
    fig.savefig(result_path + '/' + tag + 'adaptive-conv-%d.eps' % (len(e_glo[0, :])),
                dpi=1000, facecolor='w', edgecolor='w',
                orientation='portrait', format='eps',
                transparent=True, bbox_inches='tight', pad_inches=0.1)
