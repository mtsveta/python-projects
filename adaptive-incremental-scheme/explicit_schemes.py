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

        self.dt_n = np.append(self.dt_n, np.array([self.dt]))
        self.t_n = np.append(self.t_n, np.array([self.t_n[self.k] + self.dt]))
        self.e_n = np.append(self.e_n, np.array([self.norm(self.y_n1 - self.y(self.t_n[self.k + 1]))]))
        self.d_n = np.append(self.e_n, np.array([np.abs(self.y_n1 - self.y(self.t_n[self.k + 1]))]))
        if self.scheme_log: print('eps_n = %4.4e\tt = %4.4e\tdt = %4.4e\te_glob = %4.4e' % (
                             self.eps_n[self.k], self.t_n[self.k], self.dt, self.e_n[self.k + 1]))

        self.y_n = np.r_['0, 2', self.y_n, self.y_n1] # append y_n1 to th  bottom of y_n

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
        return self.e_n[self.k+1], k, self.f_evals

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
        self.f_evals += 2

        self.dfdt_aux_val = 0
        self.f_aux_val = 0

        # order
        self.p = 4

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0, self.y0)
        self.f_evals += 2

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
            C = math.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                          + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            dt = math.pow(self.eps_n[k - 1] / C, 0.25)

        # Check if the estimated step within the [t_0, t_fin] interval
        if self.t_n[k] + dt > self.t_fin:
            dt = self.t_fin - self.t_n[k]

        self.dt = dt

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

class AdaptiveTDRK4SingleRateScheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(0, ode.y0)
        self.f_evals += 2 * self.neq

        self.dfdt_aux_val = 0
        self.f_aux_val = 0

        # order
        self.p = 4

        self.dts = np.array(self.neq)

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0, self.y0)
        self.f_evals += 2 * self.neq

    def estimate_dt(self):
        k, dt = self.k, self.dt

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

        if self.k == 0:
            # if self.dfdt_n_val.all() != 0.0:
            self.dfdt_n_val[self.dfdt_n_val == 0.0] = 1e-16
            if self.dfdt_n_val.any() != 0.0:
                #print (self.dfdt_n_val)
                C = np.abs(self.dfdt_n_val) / 2
                self.dts = np.sqrt(self.eps_n[k] / C)
            else:
                delta = 1e-2
                C_delta = np.abs(self.dfdt_n(self.t_n[k] + delta, self.y_n[k])) / 2
                self.dts = np.sqrt(self.eps_n[k] / C_delta)
            self.dts *= 2
        else:
            C = np.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                          + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            self.dts = np.power(self.eps_n[k - 1] / C, 0.25)

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)
        dt_geom = 2 * dt_max * dt_min / (dt_max + dt_min)
        #dt = dt_min
        dt = dt_geom

        # Check if the estimated step within the [t_0, t_fin] interval
        if self.t_n[k] + dt > self.t_fin:
            dt = self.t_fin - self.t_n[k]
        self.dt = dt

    def approximate(self):
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

class AdaptiveTDRK4MiltiRateScheme(AdaptiveScheme):

    def __init__(self, ode, scheme_tag):
        super().__init__(ode, scheme_tag)
        self.dfdt = ode.dfdt
        self.dfdt_n = ode.dfdt_n

        self.f_n_val = self.f_n(0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(0, ode.y0)
        self.f_evals += 2

        self.dfdt_aux_val = 0
        self.f_aux_val = 0

        # order
        self.p = 4

        self.dts = np.array(self.neq)

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0, self.y0)
        self.f_evals += 2

    def solve(self):

        self.t_n = np.append(self.t_n, np.array([float(0.0)]))
        self.dt_n = np.append(self.dt_n, np.array([float(self.dt)]))
        self.e_n = np.append(self.e_n, np.array([float(0.0)]))
        self.y_n[0] = self.y0

        k = 0

        while self.t_n[k] < self.t_fin:
            self.k = k
            self.estimate_dt()
            self.approximate_fast()
            self.approximate_slow()
            self.advance_in_time()

            k += 1
        return self.e_n[self.k+1], k, self.f_evals

    def estimate_dt(self):
        k, dt = self.k, self.dt

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
            self.dts *= 2
        else:
            C = np.fabs(1 / (2 * (dt / 2) ** 3) * (self.f_n_val - self.f_aux_val)
                        + 1 / (4 * (dt / 2) ** 2) * (self.dfdt_n_val + self.dfdt_aux_val))
            self.dts = np.power(self.eps_n[k - 1] / C, 0.25)

        # Check if the estimated step within the [t_0, t_fin] interval
        if self.t_n[k] + dt > self.t_fin:
            dt = self.t_fin - self.t_n[k]

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)

        def geom_mean(array):
            return len(array) * np.prod(array) / np.sum(array)

        def geom_mean_(array):
            return np.exp(np.sum(np.log(array)))

        dt_geom = 2 * dt_max * dt_min / (dt_max + dt_min)
        dt_geom1 = geom_mean(self.dts)
        dt_geom2 = geom_mean_(self.dts)

        # if some of the steps is too big replace it with geom mean of
        self.dts_tmp = self.dts
        self.dts_tmp[self.dfdt_n_val >= self.t_n] = dt_geom

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

        '''
        if self.m:
            self.approximate()
        elif self.m >= 1:
            self.approximate_fast()
            self.approximate_slow()
        '''
        
        self.slow_indx = np.zeros(self.neq)
        self.fast_indx = np.zeros(self.neq)

        # fast component will be the one with the smallest step
        self.fast_indx[self.indx == 0] = 1
        # fast component will be the one with the smallest step
        self.slow_indx = [self.fast_indx != 1]
        # for i in range(len(self.dts_tmp)):
        print('')

    def approximate_fast(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        for i in range(self.m):
            # evaluate f_n and (f')_n
            if self.k > 0:
                self.f_n_val = self.f_n(t_n, y_n)
                self.dfdt_n_val = self.dfdt_n(t_n, y_n)
                self.f_evals += 2
                self.Jy_n = self.J_y(t_n, y_n)


            # reconstruct approximation y_1/2 of the dt / 2
            y_aux_old = y_n + dt / 2 * self.f_n_val + dt ** 2 / 8 * self.dfdt_n_val
            y_aux_f = y_n[self.fast_indx] + dt / 2 * self.f_n_val[self.fast_indx] + dt ** 2 / 8 * self.dfdt_n_val[
                self.fast_indx]

            #
            def rational_interpolation(h, y_i, f_i, f, Df_i):
                return y_i + 2 * h * f_i ** 2 / (2 * f_i - h * np.dot(Df_i, f))

            def Jy_i(Jy, i):
                e_i = np.zeros(self.neq)
                e_i[i] = 1
                return np.dot(Jy, e_i)

            y_aux_s = rational_interpolation(dt / 2,
                                             y_n[self.slow_indx],
                                             self.f_n_val[self.slow_indx],
                                             self.f_n_val,
                                             Jy_i(self.Jy_n, self.slow_indx))
            y_aux = np.zeros(self.neq)
            y_aux[self.slow_indx] = y_aux_s
            y_aux[self.fast_indx] = y_aux_f

            # evaluate f_1/2 and (f')_1/2
            self.f_aux_val_old = self.f_n(t_n + dt / 2, y_aux_old)
            self.dfdt_aux_val_old = self.dfdt_n(t_n + dt / 2, y_aux_old)

            self.f_aux_val = self.f_n(t_n + dt / 2, y_aux)
            self.dfdt_aux_val = self.dfdt_n(t_n + dt / 2, y_aux)
            self.f_evals += 2
            self.Jy_aux = self.J_y(t_n + dt / 2, y_aux)

            y_n1_f = y_n + dt * self.f_n_val + dt ** 2 / 6 * self.dfdt_n_val + dt ** 2 / 3 * self.dfdt_aux_val
            y_n1_s = rational_interpolation(dt,
                                            y_n[self.slow_indx],
                                            self.f_n_val[self.slow_indx],
                                            self.f_n_val,
                                            Jy_i(self.Jy_n, self.slow_indx))

            self.y_n1 = np.zeros(self.neq)
            self.y_n1[self.slow_indx] = y_n1_s
            self.y_n1[self.fast_indx] = y_n1_f

            self.y_n1_old = y_n + dt * self.f_n_val + dt ** 2 / 6 * self.dfdt_n_val_old + dt ** 2 / 3 * self.dfdt_aux_val_old
            y_n = self.y_n1

            print('')

        def approximate_slow(self):
            dt, dT, y_n, t_n = self.dt, self.dT, self.y_n[self.k], self.t_n[self.k]

            # evaluate f_n and (f')_n
            if self.k > 0:
                self.f_n_val = self.f_n(t_n, y_n)
                self.dfdt_n_val = self.dfdt_n(t_n, y_n)
                self.f_evals += 2

                self.Jy_n = self.J_y(t_n, y_n)

            # reconstruct approximation y_1/2 of the dt / 2
            y_aux_old = y_n + dT / 2 * self.f_n_val + dT ** 2 / 8 * self.dfdt_n_val
            y_aux_s = y_n[self.slow_indx] \
                      + self.dT / 2 * self.f_n_val[self.slow_indx] \
                      + self.dT ** 2 / 8 * self.dfdt_n_val[self.slow_indx]
            #
            def rational_interpolation(h, y_i, f_i, f, Df_i):
                return y_i + 2 * h * f_i ** 2 / (2 * f_i - h * np.dot(Df_i, f))

            def Jy_i(Jy, i):
                e_i = np.zeros(self.neq)
                e_i[i] = 1
                return np.dot(Jy, e_i)

            y_aux_f = self.y_n1[self.fast_indx]

            y_aux = np.zeros(self.neq)
            y_aux[self.slow_indx] = y_aux_s
            y_aux[self.fast_indx] = y_aux_f

            # evaluate f_1/2 and (f')_1/2
            self.f_aux_val_old = self.f_n(t_n + dt / 2, y_aux_old)
            self.dfdt_aux_val_old = self.dfdt_n(t_n + dt / 2, y_aux_old)

            self.f_aux_val = self.f_n(t_n + dt / 2, y_aux)
            self.dfdt_aux_val = self.dfdt_n(t_n + dt / 2, y_aux)
            self.f_evals += 2
            self.Jy_aux = self.J_y(t_n + dt / 2, y_aux)

            y_n1_f = y_n + dt * self.f_n_val + dt ** 2 / 6 * self.dfdt_n_val + dt ** 2 / 3 * self.dfdt_aux_val
            y_n1_s = rational_interpolation(dt,
                                            y_n[self.slow_indx],
                                            self.f_n_val[self.slow_indx],
                                            self.f_n_val,
                                            Jy_i(self.Jy_n, self.slow_indx))

            self.y_n1 = np.zeros(self.neq)
            self.y_n1[self.slow_indx] = y_n1_s
            self.y_n1[self.fast_indx] = y_n1_f

            self.y_n1_old = y_n + dt * self.f_n_val + dt ** 2 / 6 * self.dfdt_n_val_old + dt ** 2 / 3 * self.dfdt_aux_val_old
            print('')

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

        self.f_n_val = self.f_n(0.0, ode.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2 * self.neq

        # order
        self.p = 2

        self.dts = np.array(self.neq)

    def refresh(self):

        super().refresh()

        self.f_n_val = self.f_n(0.0, self.y0)
        self.dfdt_n_val = self.dfdt_n(0.0, self.y0)
        self.f_evals += 2 * self.neq

    def estimate_dt(self):
        k, t_n = self.k, self.t_n[self.k]

        self.eps_n = np.append(self.eps_n, np.array([float(self.eps_rel * self.norm(self.y_n[k]) + self.eps_abs)]))

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

        dt_max = np.max(self.dts)
        dt_min = np.min(self.dts)
        dt_geom = 2 * dt_max * dt_min / (dt_max + dt_min)
        dt = dt_geom

        # Check if the estimated step within the [t_0, t_fin] interval
        if t_n + self.dt > self.t_fin: self.dt = self.t_fin - t_n

        self.dt = dt

    def approximate(self):
        dt, y_n, t_n = self.dt, self.y_n[self.k], self.t_n[self.k]

        # evaluate f_n and (f')_n
        if self.k > 0:
            self.f_n_val = self.f_n(t_n, y_n)
            self.dfdt_n_val = self.dfdt_n(t_n, y_n)
            self.f_evals += 2 * self.neq

        # reconstruct approximation y_1/2 of the dt / 2
        self.y_n1 = y_n + dt * self.f_n_val + dt ** 2 / 2 * self.dfdt_n_val

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
