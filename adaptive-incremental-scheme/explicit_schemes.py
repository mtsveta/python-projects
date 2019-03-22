from abc import ABC, abstractmethod

import numpy as np
import math

class TimeIntegrationScheme(ABC):

    def __init__(self, ode, scheme_tag):
        super().__init__()
        self.y, self.f, self.f_n, self.y0, self.t_fin, self.tag = ode.y, ode.f, ode.f_n, ode.y0, ode.t_fin, scheme_tag
        self.neq = ode.neq
        self.f_evals = 0

        # Debugging parameter
        self.scheme_log = True

    def set_log(self, param):
        self.scheme_log = param

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def approximate(self):
        pass

    @abstractmethod
    def advance_in_time(self):
        pass

    def norm(self, val):
        if isinstance(val, (float,int)):
            return math.fabs(val)
        else:
            return np.linalg.norm(val)

class UniformScheme(TimeIntegrationScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)

    def set_time_step(self, dt):
        self.dt = dt
        self.n = int(self.t_fin / dt)

        self.t_n = np.zeros(self.n + 1)
        self.dt_n = np.zeros((self.n + 1, self.neq))
        self.e_n = np.zeros(self.n + 1)

        if self.neq == 1:
            self.y_n = np.zeros(self.n + 1)
            self.d_n = np.zeros(self.n + 1)
        else:
            self.y_n = np.zeros((self.n + 1, self.neq))
            self.d_n = np.zeros((self.n + 1, self.neq))

        self.f_evals = 0

    @abstractmethod
    def refresh(self, dt):
        pass

    def advance_in_time(self):
        k = self.k

        self.dt_n[k + 1] = self.dt
        self.t_n[k + 1] = self.t_n[k] + self.dt
        self.y_n[k + 1] = self.y_n1
        self.e_n[self.k + 1] = self.norm(self.y_n1 - self.y(self.t_n[self.k + 1]))
        self.d_n[k + 1] = np.abs(self.y_n[k + 1] - self.y(self.t_n[k + 1]))

    def solve(self):

        self.y_n[0] = self.y0
        self.t_n[0] = float(0)

        for k in range(self.n):
            self.k = k
            self.approximate()
            self.advance_in_time()
        return self.e_n[k+1], self.n, self.f_evals

class AdaptiveScheme(TimeIntegrationScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.t_n = np.array([])
        self.dt_n = np.array([])
        self.e_n = np.array([])
        self.eps_n = np.array([])
        self.dt = 0.0

        if self.neq == 1:
            self.y_n = np.zeros(1)
            self.d_n = np.zeros(1)
        else:
            self.y_n = np.zeros((1, self.neq))
            self.d_n = np.zeros((1, self.neq))

        # vector of time steps predicted at each step
        self.dts = np.array(self.neq)

        # potential approximation
        self.y_n1 = 0.0

        self.rejects = 0

    def refresh(self):

        self.t_n = np.array([])
        self.dt_n = np.array([])
        self.e_n = np.array([])
        self.eps_n = np.array([])

        if self.neq == 1:
            self.y_n = np.zeros(1)
            self.d_n = np.zeros(1)
        else:
            self.y_n = np.zeros((1, self.neq))
            self.d_n = np.zeros((1, self.neq))

        self.dt = 0.0

        self.rejects = 0

        self.f_evals = 0

    def set_tolerance(self, eps_rel, eps_abs):

        self.eps_rel = eps_rel
        self.eps_abs = eps_abs

    def advance_in_time(self):
        k, dt, y_n1, y = self.k, self.dt, self.y_n1, self.y

        self.dt_n = np.append(self.dt_n, np.array([dt]))
        self.t_n = np.append(self.t_n, np.array([self.t_n[k] + dt]))
        self.y_n = np.r_['0, 2', self.y_n, y_n1]  # append y_n1 to th  bottom of y_n

        self.e_n = np.append(self.e_n, np.array([self.norm(y_n1 - y(self.t_n[k + 1]))]))
        self.d_n = np.r_['0, 2', self.d_n, np.array([np.abs(y_n1 - y(self.t_n[k + 1]))])]

        if self.scheme_log:
            print('n = %6d\teps_n = %4.4e\tt = %4.4e\tdt = %4.4e\te_glob = %4.4e'
                  % (k + 1, self.eps_n[k], self.t_n[k + 1], dt, self.e_n[k + 1]))

    def solve(self):

        self.t_n = np.append(self.t_n, np.array([float(0.0)]))
        self.dt_n = np.append(self.dt_n, np.array([float(self.dt)]))
        self.e_n = np.append(self.e_n, np.array([float(0.0)]))
        self.y_n[0] = self.y0

        k = 0

        while self.t_n[k] < self.t_fin:
            self.k = k
            self.estimate_dt()
            self.approximate()
            self.advance_in_time()

            k += 1
        return self.e_n[k], k, self.f_evals, self.y_n[k]

    @abstractmethod
    def approximate(self):
        pass

    @abstractmethod
    def estimate_dt(self):
        pass

class UniformTDRK4Scheme(UniformScheme):
    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(ode.t_0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(ode.t_0, ode.y0)
        self.f_evals += 2

        # order
        self.p = 4

    def refresh(self):

        #super().refresh()
        self.f_n_val = self.f_n(0.0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2

        # reconstruct approximation y_1/2 of the dt / 2
        y_aux = y_n + dt / 2 * self.f_n_val + dt ** 2 / 8 * self.dfdt_n_val

        # evaluate f_1/2 and (f')_1/2
        self.f_aux_val = self.f_n(t_n + dt / 2, y_aux)
        self.dfdt_aux_val = self.dfdt_n(t_n + dt / 2, y_aux)
        self.f_evals += 2

        self.y_n1 = y_n + dt * self.f_n_val + dt**2 / 6 * self.dfdt_n_val + dt**2 / 3 * self.dfdt_aux_val

class AdaptiveTaylor2Scheme(AdaptiveScheme): #

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt

        # order
        self.p = 4

    def estimate_dt(self):
        k, t_n = self.k, self.t_n[self.k]

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * math.fabs(self.y_n[k]) + self.eps_abs)]))

        self.dfdt_n_val = self.dfdt(t_n)
        self.f_evals += 1

        C = math.fabs(self.dfdt_n_val / 2)
        if C != 0.0:
            self.dt = math.pow(self.eps_n[k] / C, 1 / 2)
        else:
            delta = 1e-2
            C_delta = math.fabs(self.dfdt(t_n + delta) / 2)
            self.f_evals += 1
            self.dt = math.pow(self.eps_n[k] / C_delta, 1 / 2)

            # Check if the estimated step within the [t_0, t_fin] interval
        if t_n + self.dt > self.t_fin:
            self.dt = self.t_fin - t_n

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n, (f')_n, (f'')_n, and (f''')_n
        f_n_val = self.f(t_n)
        self.f_evals += 1

        self.y_n1 = y_n \
                    + dt * f_n_val \
                    + dt**2 / 2 * self.dfdt_n_val \

class AdaptiveTaylor4Scheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.d2fdt2 = ode.d2fdt2
        self.d3fdt3 = ode.d3fdt3

        # order
        self.p = 4

    def estimate_dt(self):
        k, t_n = self.k, self.t_n[self.k]

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * math.fabs(self.y_n[k]) + self.eps_abs)]))

        self.d3fdt3_n_val = self.d3fdt3(t_n)
        self.f_evals += 1

        C = math.fabs(self.d3fdt3_n_val / 24)
        if C != 0.0:
            self.dt = math.pow(self.eps_n[k] / C, 0.25)
        else:
            delta = 1e-2
            C_delta = math.fabs(self.d3fdt3(t_n + delta) / 24)
            self.f_evals += 1
            self.dt = math.pow(self.eps_n[k] / C_delta, 0.25)

            # Check if the estimated step within the [t_0, t_fin] interval
        if t_n + self.dt > self.t_fin:
            self.dt = self.t_fin - t_n

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n, (f')_n, (f'')_n, and (f''')_n
        f_n_val = self.f(t_n)
        dfdt_n_val = self.dfdt(t_n)
        d2fdt2_n_val = self.d2fdt2(t_n)
        self.f_evals += 3

        self.y_n1 = y_n \
                    + dt * f_n_val \
                    + dt**2 / 2 * dfdt_n_val \
                    + dt**3 / 6 * d2fdt2_n_val \
                    + dt**4 / 24 * self.d3fdt3_n_val

class AdaptiveTDRK4Scheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(0, ode.y0)
        self.dfdt_n_val_ = self.dfdt_n(0, ode.y0)
        self.f_evals += 2 * self.neq

        # order
        self.p = 4

        if self.neq == 1:
            self.f_prev_val = 0
            self.dfdt_prev_val = 0
            self.dfdt_aux_val = 0
            self.f_aux_val = 0
        else:
            self.f_prev_val = np.zeros(self.neq)
            self.dfdt_prev_val = np.zeros(self.neq)
            self.dfdt_aux_val = np.zeros(self.neq)
            self.f_aux_val = np.zeros(self.neq)

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0, self.y0)
        self.f_evals += 2 * self.neq

        if self.neq == 1:
            self.f_prev_val = 0
            self.dfdt_prev_val = 0
        else:
            self.f_prev_val = np.zeros(self.neq)
            self.dfdt_prev_val = np.zeros(self.neq)

    def estimate_dt(self):
        k, dt = self.k, self.dt

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * math.fabs(self.y_n[k]) + self.eps_abs)]))

        if self.k == 0:
            # if self.dfdt_n_val.all() != 0.0:
            if self.dfdt_n_val != 0.0:
                # print (self.dfdt_n_val)
                C = math.fabs(self.dfdt_n_val) / 2
                dt = math.sqrt(self.eps_n[k] / C)
            else:
                delta = 1e-2
                C_delta = math.fabs(self.dfdt_n(self.t_n[k] + delta, self.y_n[k])) / 2
                dt = math.sqrt(self.eps_n[k] / C_delta)
            dt *= 2
        else:
            #'''
            # using half step and f_aux, dfdt_aux
            C = math.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                          + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            dt = math.pow(self.eps_n[k - 1] / C, 0.25)
            #'''
            '''
            C = math.fabs(1 / (2 * dt ** 3) * (self.f_n_val - self.f_prev_val)
                          + 1 / (4 * dt ** 2) * (self.dfdt_n_val + self.dfdt_prev_val))
            dt = math.pow(self.eps_n[k - 1] / C, 0.25)
            '''
        # Check if the estimated step within the [t_0, t_fin] interval
        if self.t_n[k] + dt > self.t_fin:
            dt = self.t_fin - self.t_n[k]

        self.dt = dt

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_prev_val = self.f_n_val
            self.dfdt_prev_val = self.dfdt_n_val

            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2 * self.neq

        # reconstruct approximation y_1/2 of the dt / 2
        y_aux = y_n + dt / 2 * self.f_n_val + dt ** 2 / 8 * self.dfdt_n_val

        # evaluate f_1/2 and (f')_1/2
        self.f_aux_val = self.f_n(t_n + dt / 2, y_aux)
        self.dfdt_aux_val = self.dfdt_n(t_n + dt / 2, y_aux)
        self.f_evals += 2 * self.neq

        self.y_n1 = y_n + dt * self.f_n_val + dt**2 / 6 * self.dfdt_n_val + dt**2 / 3 * self.dfdt_aux_val

class AdaptiveTDRK4SingleRateScheme(AdaptiveTDRK4Scheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)

        self.dts = np.array(self.neq)
        self.dts_ = np.array(self.neq)

        self.f_prev_val = np.zeros(self.neq)
        self.dfdt_prev_val = np.zeros(self.neq)

        self.m = 0
        self.T = 0

    def advance_in_time(self):
        k, dt, y_n1, y = self.k, self.dt, self.y_n1, self.y

        self.dt_n = np.append(self.dt_n, np.array([dt]))
        self.t_n = np.append(self.t_n, np.array([self.t_n[k] + dt]))
        self.y_n = np.r_['0, 2', self.y_n, y_n1]  # append y_n1 to th  bottom of y_n

        self.e_n = np.append(self.e_n, np.array([self.norm(y_n1 - y(self.t_n[k + 1]))]))
        self.d_n = np.r_['0, 2', self.d_n, np.array([np.abs(y_n1 - y(self.t_n[k + 1]))])]

        if self.scheme_log:
            print('n = %d\teps_n = %4.4e\tt = %4.4e\tdt = %4.4e\te_glob = %4.4e\tf_evals = %6d\tdT = %4.4e\tm = %d'
                  % (k, self.eps_n[k], self.t_n[k + 1], dt, self.e_n[k + 1], self.f_evals, self.dT, self.m))
            #print('d_n'); print(self.d_n)


    def estimate_dt(self):
        k, dt = self.k, self.dt

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

        if self.k == 0:
            # if self.dfdt_n_val.all() != 0.0:
            self.dfdt_n_val_[self.dfdt_n_val == 0.0] = 1e-16
            if self.dfdt_n_val.any() != 0.0:
                #print (self.dfdt_n_val)
                C = np.abs(self.dfdt_n_val_) / 2
                self.dts = np.sqrt(self.eps_n[k] / C)
            else:
                delta = 1e-2
                C_delta = np.abs(self.dfdt_n(self.t_n[k] + delta, self.y_n[k])) / 2
                self.dts = np.sqrt(self.eps_n[k] / C_delta)
            # self.dts *= 2 # might be screwing the convergence
        else:
            C = np.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                          + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            self.dts = np.power(self.eps_n[k - 1] / C, 0.25)
            #'''
            C_ = np.fabs(1 / (2 * dt**3) * (self.f_n_val - self.f_prev_val)
                         + 1 / (4 * dt**2) * (self.dfdt_n_val + self.dfdt_prev_val))
            self.dts_ = np.power(self.eps_n[k - 1] / C_, 0.25)
            #'''
            '''
            print('prediction: dt = ')
            print(self.dt)
            print('dt_ = ')
            print(self.dt)
            print('')
            '''
        rem_dt = self.t_fin - self.t_n[k]

        def geom_mean(array):
            return len(array) * np.exp(np.sum(np.log(array))) / np.sum(array)
        dt_geom = geom_mean(self.dts)

        self.dts[self.dts >= rem_dt] = dt_geom
        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)
        dt = dt_min

        self.m = int(dt_max / dt_min)

        # Check if the estimated step within the [t_0, t_fin] interval
        if self.t_n[k] + dt > self.t_fin:
            dt = self.t_fin - self.t_n[k]
        self.dt = dt

        if self.m % 2 and self.m > 1:
            self.m = self.m - 1
        self.dT = self.m * self.dt

class AdaptiveTDRK4MultiRateScheme(AdaptiveTDRK4Scheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)

        self.dts = np.array(self.neq)
        self.dts_ = np.array(self.neq)

        self.J_y = ode.J_y
        self.F = ode.F
        self.JF_y = ode.JF_y
        self.dFdt = ode.dFdt

        self.yaux = np.zeros(self.neq)

        self.dt = 0
        self.dT = 0

        self.f_n_val_old = self.f_n_val
        self.dfdt_val_old = self.dfdt_n_val

    def refresh(self):
        super().refresh()

        self.yaux = np.zeros(self.neq)

        self.f_n_val_old = self.f_n_val
        self.dfdt_val_old = self.dfdt_n_val

    def advance_in_time(self):
        m, k, dt, y_n1, y = self.m, self.k, self.dt, self.y_n1, self.y

        self.dt_n = np.append(self.dt_n, np.array([dt]))
        self.t_n = np.append(self.t_n, np.array([self.t_n[k] + m * dt]))
        self.y_n = np.r_['0, 2', self.y_n, y_n1]  # append y_n1 to th  bottom of y_n

        self.e_n = np.append(self.e_n, np.array([self.norm(y_n1 - y(self.t_n[k + 1]))]))
        self.d_n = np.r_['0, 2', self.d_n, np.array([np.abs(y_n1 - y(self.t_n[k + 1]))])]

        if self.scheme_log:
            print('n = %d\teps_n = %4.4e\tt = %4.4e\tdt = %4.4e\te_glob = %4.4e\tf_evals= %6d\tdT = %4.4e\tm = %d'
                  % (k, self.eps_n[k], self.t_n[k + 1], dt, self.e_n[k + 1], self.f_evals, self.dT, self.m))
            # print('d_n'); print(self.d_n)

    def estimate_dt(self):
        k, dt, dT, t, y_n, t_n  = self.k, self.dt, self.dt, self.dT, self.y_n[self.k], self.t_n[self.k]
        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

        if self.k == 0:
            # if self.dfdt_n_val.all() != 0.0:
            self.dfdt_n_val[self.dfdt_n_val == 0.0] = 1e-16
            if self.dfdt_n_val.any() != 0.0:
                # print (self.dfdt_n_val)
                C = np.abs(self.dfdt_n_val) / 2
                self.dts = np.sqrt(self.eps_n[k] / C)
            else:
                delta = 1e-2
                C_delta = np.abs(self.dfdt_n(self.t_n[k] + delta, self.y_n[k])) / 2
                self.dts = np.sqrt(self.eps_n[k] / C_delta)
        else:
            C_ = np.fabs(1 / (2 * dt ** 3) * (self.f_n_val - self.f_prev_val)
                        + 1 / (4 * dt ** 2) * (self.dfdt_n_val + self.dfdt_prev_val))
            self.dts_ = np.power(self.eps_n[k - 1] / C_, 0.25)

            C = np.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                        + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            self.dts = np.power(self.eps_n[k - 1] / C, 0.25)

            C__ = np.fabs(1 / (2 * (dT / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                        + 1 / (4 * (dT / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            self.dts__ = np.power(self.eps_n[k - 1] / C__, 0.25)

            C_1 = np.abs(self.dfdt_n_val) / 2
            self.dts_1 = np.sqrt(self.eps_n[k] / C_1)

            # Check if the estimated step within the [t_0, t_fin] interval
            # max possible time step
        rem_dt = self.t_fin - self.t_n[k]

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)

        def geom_mean(array):
            # return len(array) * np.prod(array) / np.sum(array) # product might cause an overflow
            # Prod_i a_i = exp(Sum_i log(a_i))
            return len(array) * np.exp(np.sum(np.log(array))) / np.sum(array)

        # def geom_mean_(array):
        #    return np.exp(np.sum(np.log(array)))

        dt_geom = geom_mean(self.dts)

        # if some of the steps is too big replace it with geom mean of
        self.dts_tmp = self.dts
        self.dts_tmp[self.dts >= rem_dt] = dt_geom

        self.dt = dt_min

        if self.dt > rem_dt:
            self.dt = self.t_fin - self.t_n[k]

        # get indices of the sorted element
        self.indx = np.argsort(self.dts)
        # to get first
        self.indx_sorted = np.sort(self.indx)
        # get sorted array
        self.dt_sorted = np.array(self.dts_tmp)[self.indx]
        # self.steps     = np.zeros()

        self.m = int(self.dt_sorted[1] / self.dt_sorted[0])
        #self.dT = self.m * self.dt
        if self.m % 2 and self.m > 1:
            self.m = self.m - 1
        self.dT = self.m * self.dt

        self.fast_indx, = np.where(self.indx == 0)
        self.slow_indx, = np.where(self.indx == 1)
        '''
        print('dts = ');    print(self.dts);
        print('m = ');      print(self.m);
        print('');
        '''
    def approximate(self):

        #if self.k == 0: # force the first step being the single-rate tdrk
        #    self.m = 1
        #    self.approximate_all()
        #else:
        if self.m == 1:
            self.approximate_all()
        elif self.m > 1:
            self.approximate_fast_slow()

    def approximate_all(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2 * self.neq

        # reconstruct approximation y_1/2 of the dt / 2
        y_aux = y_n + dt / 2 * self.f_n_val + dt ** 2 / 8 * self.dfdt_n_val

        # evaluate f_1/2 and (f')_1/2
        self.f_aux_val = self.f_n(t_n + dt / 2, y_aux)
        self.dfdt_aux_val = self.dfdt_n(t_n + dt / 2, y_aux)
        self.f_evals += 2 * self.neq

        self.y_n1 = y_n + dt * self.f_n_val + dt**2 / 6 * self.dfdt_n_val + dt**2 / 3 * self.dfdt_aux_val

    def approximate_fast_slow(self):
        dt, dT, y_n, t_n, fast, slow \
            = self.dt, self.dT, self.y_n[self.k], self.t_n[self.k], self.fast_indx, self.slow_indx

        # storage for intermediate y_n1
        yn1 = np.zeros(self.neq)
        yaux = np.zeros(self.neq)

        # evaluate f_n and (f')_n
        if self.k > 0:
            # save previous values
            self.f_prev_val = self.f_n_val
            self.dfdt_prev_val = self.dfdt_n_val

            # calculate actual values
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2 * self.neq

        yn_f = y_n[fast]
        yn_s = y_n[slow]

        yaux_f = yaux[fast]
        yaux_s = yaux[slow]

        f_f = self.f_n_val[fast]
        f_s = self.f_n_val[slow]

        dfdt_f = self.dfdt_n_val[fast]
        dfdt_s = self.dfdt_n_val[slow]

        y_n_old = y_n
        self.f_n_val_old = self.f_n_val
        self.dfdt_n_val_old = self.dfdt_n_val

        # approximate m fast components
        for i in range(self.m):
            yaux[fast] = yn_f + dt / 2 * f_f + dt ** 2 / 8 * dfdt_f
            yaux[slow] = yn_s + (i + 1 / 2) * dt * f_s + ((i + 1 / 2) * dt) ** 2 / 2 * dfdt_s

            # update fast functions and their derivatives
            dfdt_aux_f = np.array([self.dFdt[i](t_n + (i + 1 / 2) * dt, yaux) for i in fast])
            self.f_evals += len(fast)

            yn1[fast] = yn_f + dt * f_f + dt ** 2 / 6 * dfdt_f + dt ** 2 / 3 * dfdt_aux_f
            yn1[slow] = yn_s + (i + 1) * dt * f_s + ((i + 1) * dt) ** 2 / 2 * dfdt_s

            # update  f_f and sfst_f for the next fast component subcycle
            if i + 1 < self.m:
                f_f    = np.array([self.F[i](t_n + (i + 1) * dt, yn1) for i in fast])
                dfdt_f = np.array([self.dFdt[i](t_n + (i + 1) * dt, yn1) for i in fast])
                self.f_evals += 2 * len(fast)

            # save y_n1, f_n1, dfdt_n1 as an auxiliary yaux_f approximation
            if i + 1 == self.m:
                yaux_f = yaux[fast]
                self.f_aux_val[fast] =  np.array([self.F[i](t_n + (i + 1 / 2) * dt, yaux) for i in fast])
                self.f_aux_val[slow] = np.array([self.F[i](t_n + (i + 1 / 2) * dt, yaux) for i in slow])
                self.dfdt_aux_val[fast] = dfdt_aux_f
                self.dfdt_aux_val[slow] = np.array([self.dFdt[i](t_n + (i + 1 / 2) * dt, yaux) for i in slow])
                self.f_evals += 2 * len(slow) + len(fast)
            #------------------------------------------------------------------------#
            # single-rate way of calculation to compare with multi-rate
            y_aux_old = y_n_old + dt / 2 * self.f_n_val_old + dt**2 / 8 * self.dfdt_n_val_old

            # evaluate f_1/2 and (f')_1/2
            self.dfdt_aux_val_old = self.dfdt_n(t_n + (i + 1 / 2) * dt, y_aux_old)

            self.y_n1_old = y_n_old + dt * self.f_n_val_old + dt**2 / 6 * self.dfdt_n_val_old + dt**2 / 3 * self.dfdt_aux_val_old
            '''
            print('y_aux = ');
            print(yaux)
            print('y_aux_old = ');
            print(y_aux_old)
            print('y_n1 = ');
            print(yn1)
            print('y_n1_old = ');
            print(self.y_n1_old)
            print('')
            '''
            # update fast yn tp be yn1
            yn_f = yn1[fast]
            y_n_old = self.y_n1_old
            self.f_n_val_old = self.f_n(t_n + (i + 1) * dt, self.y_n1_old)
            self.dfdt_n_val_old = self.dfdt_n(t_n + (i + 1) * dt, self.y_n1_old)

            #------------------------------------------------------------------------#

        # approximate slow components by the tdrk4 scheme with the step dT
        yaux_s = yn_s + dT / 2 * f_s + dT**2 / 8 * dfdt_s
        #yaux = y_n + dT / 2 * self.f_n_val + dT ** 2 / 8 * self.dfdt_n_val

        # refresh yaux for the slow component
        yaux[slow] = yaux_s
        yaux[fast] = yaux_f

        # update
        dfdt_aux_val_s = np.array([self.dFdt[i](t_n + dT / 2, yaux) for i in slow])
        #self.f_aux_val[slow]    = np.array([self.F[i](t_n + dT / 2, yaux) for i in slow])
        self.f_evals += len(slow)

        yn1[slow] = yn_s + dT * f_s \
                         + dT**2 / 6 * dfdt_s \
                         + dT**2 / 3 * dfdt_aux_val_s

        # save the approximation
        self.y_n1 = yn1

        '''
        # ------------------------------------------------------------------------#
        # single-rate way of calculation to compare with multi-rate
        y_aux_old = y_n + dT / 2 * self.f_n_val + dT**2 / 8 * self.dfdt_n_val
        # evaluate f_1/2 and (f')_1/2
        self.f_aux_val_old = self.f_n(t_n + dT / 2, y_aux_old)
        self.dfdt_aux_val_old = self.dfdt_n(t_n + dT / 2, y_aux_old)
        self.f_aux_val = self.f_aux_val_old
        self.dfdt_aux_val = self.dfdt_aux_val_old
        #self.f_evals += self.neq

        self.y_n1_old = y_n + dT * self.f_n_val \
                            + dT**2 / 6 * self.dfdt_n_val \
                            + dT**2 / 3 * self.dfdt_aux_val_old
        '''
        '''
        print('y_aux = ');
        print(yaux)
        print('y_aux_old = ');
        print(y_aux_old)

        print('y_n1 = ');
        print(self.y_n1)
        print('y_n1_old = ');
        print(self.y_n1_old)
        print('')
        '''
        # ------------------------------------------------------------------------#

class AdaptiveTDRK2Scheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(0.0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2

        # order
        self.p = 2

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0.0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2

    def estimate_dt(self):
        k, t_n = self.k, self.t_n[self.k]

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

        C = math.fabs(self.dfdt_n_val) / 2
        self.dt = math.sqrt(self.eps_n[k] / C)

        # Check if the estimated step within the [t_0, t_fin] interval
        if t_n + self.dt > self.t_fin: self.dt = self.t_fin - t_n

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2

        # reconstruct approximation y_1/2 of the dt / 2
        self.y_n1 = y_n + dt * self.f_n_val + dt ** 2 / 2 * self.dfdt_n_val

class AdaptiveTDRK2SingleRateScheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        # order
        self.p = 2

        self.dts = np.array(self.neq)

    def estimate_dt(self):
        k, t_n, y_n = self.k, self.t_n[self.k], self.y_n[self.k]

        # evaluate f_n and (f')_n
        self.f_n_val = self.f_n(t_n, y_n)
        self.dfdt_n_val = self.dfdt_n(t_n, y_n)
        self.f_evals += 2 * self.neq

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(y_n) + self.eps_abs)]))

        # if self.dfdt_n_val.all() != 0.0:
        self.dfdt_n_val[self.dfdt_n_val == 0.0] = 1e-16
        if self.dfdt_n_val.any() != 0.0:
            # print (self.dfdt_n_val)
            C = np.abs(self.dfdt_n_val) / 2
            self.dts = np.sqrt(self.eps_n[k] / C)
        else:
            delta = 1e-2
            C_delta = np.abs(self.dfdt_n(self.t_n[k] + delta, y_n)) / 2
            self.dts = np.sqrt(self.eps_n[k] / C_delta)

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)
        dt_geom = 2 * dt_max * dt_min / (dt_max + dt_min)
        #dt = dt_min
        dt = dt_geom
        #dt = dt_min if (dt_max > (self.t_fin - t_n)) else dt_max

        # Check if the estimated step within the [t_0, t_fin] interval
        if t_n + self.dt > self.t_fin: self.dt = self.t_fin - t_n

        self.dt = dt

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # reconstruct approximation y_1/2 of the dt / 2
        self.y_n1 = y_n + dt * self.f_n_val + dt ** 2 / 2 * self.dfdt_n_val

class AdaptiveTDRK2MultiRateScheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        # order
        self.p = 2

        self.dts = np.array(self.neq)

        self.J_y = ode.J_y

        self.F = ode.F
        self.JF_y = ode.JF_y
        self.dFdt = ode.dFdt

    def refresh(self):

        super().refresh()

    def approximate(self):
        if self.m == 1:
            self.approximate_all()
        elif self.m > 1:
            self.approximate_fast_slow()

    def advance_in_time(self):
        m, k, dt, y_n1, y = self.m, self.k, self.dt, self.y_n1, self.y

        self.dt_n = np.append(self.dt_n, np.array([dt]))
        self.t_n = np.append(self.t_n, np.array([self.t_n[k] + m * dt]))
        self.y_n = np.r_['0, 2', self.y_n, y_n1] # append y_n1 to th  bottom of y_n

        self.e_n = np.append(self.e_n, np.array([self.norm(y_n1 - y(self.t_n[k + 1]))]))
        self.d_n = np.r_['0, 2', self.d_n, np.array([np.abs(y_n1 - y(self.t_n[k + 1]))])]

        if self.scheme_log:
            print('eps_n = %4.4e\tt = %4.4e\tdt = %4.4e\te_glob = %4.4e' % (
            self.eps_n[k], self.t_n[k + 1], m * dt, self.e_n[k + 1]))

    def estimate_dt(self):
        k, dt = self.k, self.dt
        t, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

        self.f_n_val = self.f_n(t_n, y_n)
        self.dfdt_n_val = self.dfdt_n(t_n, y_n)
        self.f_evals += 2 * self.neq

        # if self.dfdt_n_val.all() != 0.0:
        self.dfdt_n_val[self.dfdt_n_val == 0.0] = 1e-16

        if self.dfdt_n_val.any() != 0.0:
            C = np.abs(self.dfdt_n_val) / 2
            self.dts = np.sqrt(self.eps_n[k] / C)
        else:
            delta = 1e-2
            C_delta = np.abs(self.dfdt_n(self.t_n[k] + delta, self.y_n[k])) / 2
            self.dts = np.sqrt(self.eps_n[k] / C_delta)

        # Check if the estimated step within the [t_0, t_fin] interval
        # max possible time step
        rem_dt = self.t_fin - self.t_n[k]

        if dt > rem_dt:
            dt = self.t_fin - self.t_n[k]

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)

        def geom_mean(array):
            # return len(array) * np.prod(array) / np.sum(array) # product might cause an overflow
            # Prod_i a_i = exp(Sum_i log(a_i))
            return len(array) * np.exp(np.sum(np.log(array))) / np.sum(array)

        dt_geom = geom_mean(self.dts)

        # if some of the steps is too big replace it with geom mean of
        self.dts_tmp = self.dts
        self.dts_tmp[self.dts >= rem_dt] = dt_geom

        self.dt = dt_min

        # get indices of the sorted element
        self.indx = np.argsort(self.dts)
        # to get first
        self.indx_sorted = np.sort(self.indx)
        # get sorted array
        self.dt_sorted = np.array(self.dts_tmp)[self.indx]
        # self.steps     = np.zeros()

        self.m = int(self.dt_sorted[1] / self.dt_sorted[0])
        self.dT = self.m * self.dt

        self.fast_indx, = np.where(self.indx == 0)
        self.slow_indx, = np.where(self.indx == 1)
        #print('dts = ');    print(self.dts);
        #self.slow_indx = np.zeros(self.neq).astype(int)
        #self.fast_indx = np.zeros(self.neq).astype(int)

        # fast component will be the one with the smallest step
        # whose index after in the argsorting returned 0
        #s elf.fast_indx[self.indx == 0] = int(1)
        # slow component will be the rest
        # self.slow_indx[self.fast_indx != 1] = int(1)
        # for i in range(len(self.dts_tmp)):
        #print('')

    def approximate_all(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # reconstruct approximation y_1/2 of the dt / 2
        self.y_n1 = y_n + dt * self.f_n_val + dt ** 2 / 2 * self.dfdt_n_val

    def rational_interpolation(self, h, y_i, f_i, f, Df_i):
        return y_i + 2 * h * np.power(f_i, 2) / (2 * f_i - h * np.dot(Df_i, f))

    def approximate_fast_slow(self):
        dt, dT, y_n, t_n = self.dt, self.dT, self.y_n[self.k], self.t_n[self.k]
        fast, slow, f_n_val, dfdt_n_val = self.fast_indx, self.slow_indx, self.f_n_val, self.dfdt_n_val

        # storage for intermidiate y_n1
        yn1 = np.zeros(self.neq)

        yn_f = y_n[fast]
        yn_s = y_n[slow]

        f_f = f_n_val[fast]
        f_s = f_n_val[slow]

        dfdt_f = dfdt_n_val[fast]
        dfdt_s = dfdt_n_val[slow]

        # approximate m fast components
        for i in range(self.m):
            # approximate fast components by the tdrk2 scheme
            yn1[fast] = yn_f + dt * f_f + dt ** 2 / 2 * dfdt_f
            yn1[slow] = yn_s + (i + 1) * dt * f_s + ((i + 1) * dt) ** 2 / 2 * dfdt_s

            # update fast functions and their derivatives
            f_f = np.array([self.F[i](t_n + (i + 1) * dt, yn1) for i in fast])
            dfdt_f = np.array([self.dFdt[i](t_n + (i + 1) * dt, yn1) for i in fast])
            self.f_evals += 2 * len(fast)

            # update fast yn tp be yn1
            yn_f = yn1[fast]

        # approximate slow components by the tdrk2 scheme with the step dT
        yn1[slow] = yn_s + dT * f_s + dT ** 2 / 2 * dfdt_s
        self.y_n1 = yn1

class AdaptiveClassicTDRK2Scheme(AdaptiveTDRK2Scheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)

        # order of the scheme
        self.p = 2

        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(ode.t_0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(ode.t_0, ode.y0)
        self.f_evals += 2

        # track the number of rejects
        self.rejects = 0
        self.e_loc = 0

        # flag to check if the current step is rejected
        self.is_rejected = False

        # potential approximation
        self.y_n1 = 0.0

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0.0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2

    def estimate_dt(self):

        if not self.is_rejected:
            self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * math.fabs(self.y_n[self.k]) + self.eps_abs)]))

        if self.is_rejected:
            # calculate the factor based on the current eps_n
            C = 0.9
            factor = C * math.pow(self.eps_n[self.k] / self.e_loc, 1 / (self.p + 1))
        else:
            if self.k == 0:
                C = math.fabs(self.dfdt_n_val) / 2
                self.dt = math.sqrt(self.eps_n[self.k] / C)
                factor = 1.0
            else:
                # calculate the factor based on the previous eps_n
                C = 1.1
                factor = C * math.pow(self.eps_n[self.k - 1] / self.e_loc, 1 / (self.p + 1))
        self.dt *= factor

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2

        # reconstruct approximation y_1/2 of the dt / 2
        self.y_n1 = y_n + dt * self.f_n_val + dt ** 2 / 2 * self.dfdt_n_val

    def approx_is_accepted(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        y_n1 = y_n + dt * self.f_n_val
        self.e_loc = self.norm(self.y_n1 - y_n1)

        return self.e_loc < self.eps_n[self.k]

    def solve(self):

        self.y_n = np.append(self.y_n, np.array([float(self.y0)]))
        self.t_n = np.append(self.t_n, np.array([float(0.0)]))
        self.dt_n = np.append(self.dt_n, np.array([float(self.dt)]))
        self.e_n = np.append(self.e_n, np.array([float(0.0)]))

        k = 0
        while self.t_n[k] < self.t_fin:
            self.k = k
            self.estimate_dt()
            self.approximate()

            # check if the tolerance of the obtained approximation is acceptable
            if self.approx_is_accepted():
                self.is_rejected = False
                self.advance_in_time()
                k += 1
            else:
                self.is_rejected = True
                self.rejects += 1


        #if self.scheme_log:
        print('number of rejects = %d' % self.rejects)
        return self.e_n[self.k+1], self.k, self.f_evals
