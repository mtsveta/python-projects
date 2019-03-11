import os
import math
import time

import numpy as np
import matplotlib.pyplot as plt

class Test():

    def __init__(self, example, params, scheme, example_name):
        self.example, self.scheme, self.scheme.params = example, scheme, params
        self.result_path = self.create_results_folder(example_name + str(self.example))
        self.scheme.set_log(params['scheme_log'])

        self.colors = ['aqua', 'darkblue',
                 'coral', 'crimson',
                 'green', 'darkgreen',
                 'orange', 'darkorange',
                 'pink', 'maroon',
                 'gold', 'brown']

    def get_project_path(self):
        (project_path, src_folder) = os.path.split(os.path.abspath(os.path.dirname(__file__)))
        return project_path + '/' + src_folder

    def create_results_folder(self, folder_name):
        # create the full directory of the result folder
        full_directory = self.get_project_path() + '/results/' + folder_name

        # if directory doesn't exist
        if not os.path.exists(full_directory):
            os.makedirs(full_directory)
        # writing to the log file
        # sys.stdout = open(full_directory + '/log-results-2nd-order-middle-step.txt', "w+")
        return full_directory

    def allocate_data_sets(self):

        self.e_glob = np.zeros(self.num)
        self.n = np.zeros(self.num)
        self.f_evals = np.zeros(self.num)
        self.cpu_time = np.zeros(self.num)

    def test_adaptive(self, eps_abs, eps_rel):

        self.num = len(eps_abs)
        self.allocate_data_sets()

        for i in range(0, self.num):
            self.scheme.set_tolerance(eps_rel[i], eps_abs[i])
            t_start = time.time()
            self.e_glob[i], self.n[i], self.f_evals[i] = self.scheme.solve()
            self.cpu_time[i] = time.time() - t_start
            if self.scheme.params['test_log']:
                print('eps_abs = %4.4e\te_glob = %4.4e\t\tn = %6d\tf_evals = %6d\n' % (
                      eps_abs[i], self.e_glob[i], self.n[i], self.f_evals[i]))
            self.plot_approximation_result()
            self.scheme.refresh()

    def test_uniform(self, dt_array):

        self.num = len(dt_array)
        self.allocate_data_sets()

        for i in range(0, self.num):
            self.scheme.set_time_step(dt_array[i])
            t_start = time.time()
            self.e_glob[i], self.n[i], self.f_evals[i] = self.scheme.solve()
            self.cpu_time[i] = time.time() - t_start
            if self.scheme.params['test_log']:
                print('h = %4.4e\te_glob = %4.4e\t\tn = %6d\tf_evals = %6d\n'
                      % (dt_array[i], self.e_glob[i], self.n[i], self.f_evals[i]))
            self.scheme.refresh()
            #self.scheme.plot_approximation_result()

    def plot_results(self, tag):
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        # plot global errors wrt number of steps
        ax[0].loglog(self.f_evals, self.e_glob,
                     color='green',
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor='green',
                     markersize=6,
                     label=r'our scheme (our h pred.)')
        # plot different convergences
        ax[0].loglog(self.f_evals, np.power(self.f_evals, -self.scheme.p),
                     color='gray',
                     linestyle='dashed',
                     marker='',
                     markerfacecolor='gray',
                     markersize=6,
                     label=('order %d' %self.scheme.p))
        ax[0].loglog(self.f_evals, np.power(self.f_evals, -self.scheme.p + 1),
                     color='tan',
                     linestyle='dashed',
                     marker='',
                     markerfacecolor='tan',
                     markersize=6,
                     label=('order %d' % (self.scheme.p - 1)))

        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[0].set_xlabel(r'func. evals.')
        ax[0].set_ylabel(r'global $e$')
        ax[0].set_title('errors wrt # of steps / func. evals.')
        ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

        # plot global error wrt number of functions evaluation
        ax[1].loglog(self.cpu_time, self.e_glob,
                     color='green',
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor='green',
                     markersize=6,
                     label=r'our scheme (our h pred.)')
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[1].set_xlabel(r'cpu time')
        ax[1].set_ylabel(r'global $e$')
        ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

        plt.subplots_adjust(right=0.6)
        plt.show()
        fig.savefig(self.result_path + '/' + tag + 'error-convergence.eps',
                    dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', format='eps',
                    transparent=True, bbox_inches='tight', pad_inches=0.1)

    def plot_approximation_result(self):
        # plot comparison of approximate and exact solution wrt time
        fig, ax = plt.subplots(3, 1, figsize=(6, 8))

        for i in range(self.scheme.neq):
            if self.scheme.neq == 1:
                y_n = self.scheme.y_n
            else:
                y_n = self.scheme.y_n[:, i]
            ax[0].plot(self.scheme.t_n, y_n,
                       color=self.colors[2*i],
                       linestyle='dashed',
                       marker='',
                       markerfacecolor=self.colors[2*i],
                       markersize=6,
                       label=r'$y_n^{' + str(i) + '}$')

        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[0].set_xlabel(r'$t$')
        ax[0].set_ylabel(r'$y_n, y$')
        ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)
        ax[0].set_title(self.scheme.tag)
        # time-steps wrt time
        ax[1].semilogy(self.scheme.t_n, self.scheme.e_n,
                   color='coral',
                   linestyle='dashed',
                   marker='*',
                   markerfacecolor='coral',
                   markersize=4,
                   label=f'$e$')
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('predicted $h$')
        ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

        # time-steps wrt time
        ax[2].semilogy(self.scheme.t_n, self.scheme.dt_n,
                       color='#7aa0c4',
                       linestyle=':',
                       marker='^',
                       markerfacecolor='#7aa0c4',
                       markersize=4,
                       label=f'$h$')
        ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('predicted $h$')
        ax[2].grid(True, color='gray', linestyle=':', linewidth=0.5)

        plt.subplots_adjust(right=0.6)
        plt.show()
        fig.savefig(self.result_path + '/' + self.scheme.tag + '-solutions-convergence-%d.eps' % len(self.scheme.t_n),
                    dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', format='eps',
                    transparent=True, bbox_inches='tight', pad_inches=0.1)

    def compare_results(self, tests, labels, tag):

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        # plot global errors wrt number of steps
        ax[0].loglog(self.f_evals, self.e_glob,
                     color=self.colors[1],
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor=self.colors[1],
                     markersize=6,
                     label=self.scheme.tag)
        for i in range(len(tests)):
            ax[0].loglog(tests[i].f_evals, self.e_glob,
                         color=self.colors[2*i],
                         linestyle='dashed',
                         marker='o',
                         markerfacecolor=self.colors[2*i],
                         markersize=6,
                         label=labels[i+1])

        # plot different convergences
        ax[0].loglog(self.f_evals, np.power(self.f_evals, -self.scheme.p),
                     color='gray',
                     linestyle='dashed',
                     marker='',
                     markerfacecolor='gray',
                     markersize=6,
                     label=('order %d' % self.scheme.p))
        ax[0].loglog(self.f_evals, np.power(self.f_evals, -self.scheme.p + 1),
                     color='tan',
                     linestyle='dashed',
                     marker='',
                     markerfacecolor='tan',
                     markersize=6,
                     label=('order %d' % (self.scheme.p - 1)))

        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[0].set_xlabel(r'func. evals.')
        ax[0].set_ylabel(r'global $e$')
        ax[0].set_title('errors wrt # of steps / func. evals.')
        ax[0].grid(True, color='gray', linestyle=':', linewidth=0.5)

        # plot global error wrt number of functions evaluation
        ax[1].loglog(self.cpu_time, self.e_glob,
                     color=self.colors[1],
                     linestyle='dashed',
                     marker='o',
                     markerfacecolor=self.colors[1],
                     markersize=6,
                     label=self.scheme.tag)

        for i in range(len(tests)):
            ax[1].loglog(tests[i].cpu_time, self.e_glob,
                         color=self.colors[2*i],
                         linestyle='dashed',
                         marker='o',
                         markerfacecolor=self.colors[2*i],
                         markersize=6,
                         label=labels[i+1])

        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax[1].set_xlabel(r'cpu time')
        ax[1].set_ylabel(r'global $e$')
        ax[1].grid(True, color='gray', linestyle=':', linewidth=0.5)

        plt.subplots_adjust(right=0.6)
        plt.show()
        fig.savefig(self.result_path + '/' + tag + 'error-convergence.eps',
                    dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', format='eps',
                    transparent=True, bbox_inches='tight', pad_inches=0.1)

