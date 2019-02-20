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

def plot_results(time, approx, exact, h, h_pred, err, result_path):

    # plot comparison of approximate and exact solution wrt time
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].plot(time, approx, color='green', linestyle='dashed', marker='', markerfacecolor='green', markersize=6, label=f'y_n')
    ax[0].plot(time, exact, color='blue', linestyle='', marker='o', markerfacecolor='blue', markersize=2, label=f'y')
    #ax[0].plot(time, approx, '--', mew=1, ms=4, xmec='w', label=f'y_n')
    #ax[0].plot(time, exact, 'o-', mew=1, ms=4, mec='w', label=f'y')
    ax[0].legend()
    plt.xlabel('time, t')
    plt.ylabel('y_n, y')
    #plt.title('Comparison of y_n to exact y(t)')
    ax[0].set_title('Comparison of y_n to exact y(t)')

    # time-steps wrt time
    ax[1].plot(time[1:], h, color='magenta', linestyle='', marker='*', markerfacecolor='magenta', markersize=2, label=f'h / out approach')
    ax[1].plot(time[1:], h_pred, color='teal', linestyle='', marker='x', markerfacecolor='teal', markersize=2, label=f'h / pred./reject. approach')
    ax[1].legend()
    plt.xlabel('time, t')
    plt.ylabel('h')
    ax[1].set_title('Predicted h')

    # time-steps wrt time
    ax[2].semilogy(time[1:],  err, color='red', linestyle=':', marker='', markerfacecolor='magenta', markersize=6, label=f'log_10(err)')
    ax[2].legend()
    plt.xlabel('time, t')
    plt.ylabel('log_10(err)')
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
    ax[1].semilogy(time[1:], err, color='red', linestyle=':', marker='', markerfacecolor='magenta', markersize=6,
                   label=f'log_10(err)')
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

def plot_convergence(lte, err, n, f_evals, order, title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].loglog(n, err, color='green', linestyle='dashed', marker='o', markerfacecolor='green', markersize=6, label=f'global error')
    ax[0].loglog(n, np.power(n, -4), color='magenta', linestyle='dashed', marker='x', markerfacecolor='magenta',
                 markersize=6, label=f'4th order scheme')
    ax[0].loglog(n, np.power(n, -3), color='yellow', linestyle='dashed', marker='x', markerfacecolor='yellow',
                markersize=6, label=f'4th order scheme')
    ax[0].loglog(n, np.power(n, -2), color='green', linestyle='dashed', marker='x', markerfacecolor='green',
                markersize=6, label=f'4th order scheme')

    ax[0].legend()

    plt.xlabel('log_10(N^{-1})')
    plt.ylabel('log_10(LTE)')
    ax[0].set_title('Errors vs. number of steps')

    # plot global error wrt number of functions evaluation
    ax[1].loglog(f_evals, err, color='blue', linestyle=':', marker='o', markerfacecolor='blue', markersize=6, label=f'error')
    ax[1].legend()

    plt.xlabel('log_10(func. evaluations)')
    plt.ylabel('log_10(error)')
    ax[1].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/adaptive-convergence-%d.eps' % (len(lte)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

def plot_uniform_convergence(lte, err, n, h, f_evals, order, title, result_path):

    # plot local truncation error wrt number of steps
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].loglog(n, err, color='green', linestyle='dashed', marker='o', markerfacecolor='green', markersize=6,
                 label=f'global error')
    ax[0].loglog(n, np.power(n, -order), color='magenta', linestyle='dashed', marker='x', markerfacecolor='magenta',
                 markersize=6, label=f'4th order scheme')

    ax[0].legend()
    plt.xlabel('log_10(N^{-1})')
    plt.ylabel('log_10(LTE)')
    ax[0].set_title('Errors vs. number of steps')

    # plot local truncation error wrt step-size
    ax[1].loglog(h, err, color='green', linestyle='dashed', marker='o', markerfacecolor='green',
                     markersize=6, label=f'global error')
    ax[1].loglog(h, np.power(h, order), color='magenta', linestyle='dashed', marker='',
                     markerfacecolor='magenta', markersize=6, label=f'optimal line')
    ax[1].legend()
    plt.xlabel('log_10(N^{-1})')
    plt.ylabel('log_10(LTE)')
    ax[1].set_title('Errors vs. h')

    # plot global error wrt number of functions evaluation
    ax[2].loglog(f_evals, err, color='blue', linestyle='', marker='o', markerfacecolor='blue', markersize=6,
                 label=f'error')
    ax[2].legend()

    plt.xlabel('log_10(func. evaluations)')
    plt.ylabel('log_10(error)')
    ax[2].set_title('Errors vs. function evaluation')

    plt.title(title)
    plt.show()
    plt.savefig(result_path + '/tdrk-convergence-%d.eps' % (len(lte)),
                dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
