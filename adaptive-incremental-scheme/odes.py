class ODE():
    def __init__(self, y, f, dfdt, d2fdt2, d3fdt3, f_n, dfdt_n, t_0, t_fin):
        # check if input data are callable functions
        if not callable(y):
            raise TypeError('y is %s,not a fucntion' % type(y))
        if not callable(f):
            raise TypeError('f is %s,not a fucntion' % type(f))
        if not callable(dfdt):
            raise TypeError('f\' is %s,not a fucntion' % type(dfdt))
        self.y, self.f, self.dfdt, self.d2fdt2, self.d3fdt3, self.t_0, self.t_fin, self.f_n, self.dfdt_n = \
            y, f, dfdt, d2fdt2, d3fdt3, t_0, t_fin, f_n, dfdt_n
        self.y0 = self.y(t_0)
        self.neq = 1
    def __str__(self):
        return ''

    def __call__(self, y, t):
        return self.f(y, t)

class ODEs():
    def __init__(self, y, f, dfdt, J_y, J_t, f_n, dfdt_n, t_0, t_fin, F, JF_y, dFdt):
        # check if input data are callable functions
        if not callable(y):
            raise TypeError('y is %s,not a fucntion' % type(y))
        if not callable(f):
            raise TypeError('f is %s,not a fucntion' % type(f))
        if not callable(dfdt):
            raise TypeError('f\' is %s,not a fucntion' % type(dfdt))
        self.y, self.f, self.dfdt, self.J_y, self.J_t, self.t_0, self.t_fin, self.f_n, self.dfdt_n = \
            y, f, dfdt, J_y, J_t, t_0, t_fin, f_n, dfdt_n
        self.y0 = self.y(t_0)
        self.neq = len(self.y0)

        self.F = F
        self.JF_y = JF_y
        self.dFdt = dFdt

    def __str__(self):
        return ''

    def __call__(self, y, t):
        return self.f(y, t)