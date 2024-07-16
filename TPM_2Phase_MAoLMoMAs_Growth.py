
from helper import write_field2output, nonlinvarsolver, norm
import dolfin
import ufl

class InitialCondition(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, nS_0S, dim, **kwargs):
        self.subdomains = subdomains
        self.nS_0S = nS_0S
        self.dim = dim
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.dim == 3:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = self.nS_0S  # nS
        else:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # p
            values[3] = self.nS_0S  # nS

    def value_shape(self):
        if self.dim == 3:
            return (5, )
        else:
            return (4, )

class Production(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, growthArea, hatrhoS, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.growthArea = growthArea
        self.hatrhoS = hatrhoS

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # hatrhoS
        for area in self.growthArea:
            if self.subdomains[cell.index] == area:
                values[0] = self.hatrhoS

    def value_shape(self):
        return ()

class TPM_2Phase_MAoLMoMAs_Growth:
    """
    The two phase model implements a two phase material in the continuum-mechanical framework of the Theory of Porous
    Media. The material is split into a fluid and solid part, wherein the fluid part multiple components can be
    resolved. The user can either set free defined functions, constants or load xdmf input files to set initial
    conditions. In order to have time dependent production terms or to couple the production terms to other software
    the production terms will be actualised in every time step.
    """
    def __init__(self):
        self.id = "TPM_2Phase_MAoLMoMAs"
        self.output_file = None
        self.dim = 2
        self.growth_time_series = None

        self.finite_element = None
        self.function_space = None
        self.internal_function_spaces = None
        self.V1 = None
        self.V2 = None
        self.V3 = None

        self.type_u = None
        self.type_p = None
        self.type_nS = None
        self.order_u = None
        self.order_p = None
        self.order_nS = None
        self.mesh = None
        self.domain = None
        self.growthArea = None

        self.dx = None
        self.n_bound = None
        self.d_bound = None
        self.initial_condition = None
        self.production = None
        self.hatrhoS = None
        self.alpha_g = None

        self.rhoSR = None
        self.rhoFR = None
        self.lambdaS = None
        self.muS = None
        self.gammaFR = None
        self.nS_0S = None
        self.kF_0S = None

        self.solver_param = None

        self.dt = None
        self.T_end = None
        self.output_interval = None
        self.growth_time = None

    def set_initial_condition(self):
        self.initial_condition = InitialCondition(self.domain, self.nS_0S, self.dim)
        self.production = Production(self.domain, self.growthArea, self.hatrhoS)

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def set_param(self, input):
        """
        sets parameter needed for model class
        """
        self.dim = input.geom.dim
        self.growth_time_series = input.param.gen.growth_time_series
        self.eval_points = input.param.gen.eval_points
        self.output_file = input.param.gen.output_file
        self.type_u = input.param.fem.type_u
        self.type_p = input.param.fem.type_p
        self.type_nS = input.param.fem.type_nS
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nS = input.param.fem.order_nS
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        self.growthArea = input.geom.growthArea
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.hatrhoS = dolfin.Constant(input.param.mat.hatrhoS)
        self.alpha_g = dolfin.Constant(input.param.mat.alpha_g)
        self.rhoSR = dolfin.Constant(input.param.mat.rhoSR)
        self.rhoFR = dolfin.Constant(input.param.mat.rhoFR)
        self.lambdaS = dolfin.Constant(input.param.mat.lambdaS)
        self.muS = dolfin.Constant(input.param.mat.muS)
        self.gammaFR = dolfin.Constant(input.param.mat.gammaFR)
        self.nS_0S = dolfin.Constant(input.param.mat.nS_0S)
        self.kF_0S = dolfin.Constant(input.param.mat.kF_0S)
        self.solver_param = input.param.fem.solver_param
        self.dt = input.param.time.dt
        self.T_end = input.param.time.T_end
        self.output_interval = input.param.time.output_interval
        self.growth_time = input.param.time.growth_time

    def set_function_spaces(self):
        """
        sets function space for primary variables u, p, nS and for internal variables
        """
        element_u = dolfin.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)
        element_p = dolfin.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_nS = dolfin.FiniteElement(self.type_nS, self.mesh.ufl_cell(), self.order_nS)
        self.finite_element = dolfin.MixedElement(element_u, element_p, element_nS)
        self.function_space = dolfin.FunctionSpace(self.mesh, self.finite_element)
        self.V1 = dolfin.FunctionSpace(self.mesh, "P", 1)
        self.V2 = dolfin.VectorFunctionSpace(self.mesh, "P", 1)
        self.V3 = dolfin.TensorFunctionSpace(self.mesh, "P", 1)

    def solve(self):
        def output(w, time):
            # Calculate solutions
            u_, p_, nS_ = w.split()
            hatrhoS_ = dolfin.project(hatrhoS, self.V1)
            u_av = dolfin.project(norm(u_), self.V1)
            nFw_FS_ = dolfin.project(nFw_FS, self.V2)
            F_S_ = dolfin.project(F_S, self.V3)
            F_SE_ = dolfin.project(F_Se, self.V3)
            F_SG_ = dolfin.project(F_Sg, self.V3)
            J_S_ = dolfin.project(J_S, self.V1)
            J_SE_ = dolfin.project(J_Se, self.V1)
            J_SG_ = dolfin.project(J_Sg, self.V1)
            T_ = dolfin.project(T, self.V3)
            TE_ = dolfin.project((T + p * dolfin.Identity(len(u))), self.V3)
            T0 = dolfin.project(T[0, 0] + p, self.V1)
            u0 = dolfin.project(u_[0], self.V1)

            write_field2output(self.output_file, u_, "u", time)
            write_field2output(self.output_file, u_av, "u_av", time)
            write_field2output(self.output_file, p_, "p", time)
            write_field2output(self.output_file, nS_, "nS", time)
            write_field2output(self.output_file, hatrhoS_, "hatrhoS", time)
            write_field2output(self.output_file, nFw_FS_, "nFW_FS", time)
            write_field2output(self.output_file, F_S_, "F_S", time)
            write_field2output(self.output_file, F_SE_, "F_SE", time)
            write_field2output(self.output_file, F_SG_, "F_SG", time)
            write_field2output(self.output_file, J_S_, "J_S", time)
            write_field2output(self.output_file, J_SE_, "J_SE", time)
            write_field2output(self.output_file, J_SG_, "J_SG", time)
            write_field2output(self.output_file, T_, "total stress", time)
            write_field2output(self.output_file, TE_, "extra stress", time)
            write_field2output(self.output_file, T0, "total stress 0", time)
            write_field2output(self.output_file, u0, "u_r", time)

        prm = dolfin.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

        # Store history values for time integration
        w_n = dolfin.Function(self.function_space)
        hatrhoS = dolfin.Function(self.V1) 
        intGrowth = dolfin.Function(self.V1)
        intGrowth_n = dolfin.Function(self.V1)
        time = dolfin.Constant(0)
        u_n, p_n, nS_n = dolfin.split(w_n)

        # Get Ansatz and test functions
        w = dolfin.Function(self.function_space)
        _w = dolfin.TestFunction(self.function_space)
        u, p, nS = dolfin.split(w)
        _u, _p, _nS = dolfin.split(_w)

        # Integration over domain
        dx = dolfin.Measure("dx", metadata={'quadrature_degree': 2})

        # Calculate volume fractions
        hatnS = hatrhoS / self.rhoSR

        # Calculate kinematics
        integral = self.alpha_g * hatnS * (1-self.rhoSR/self.rhoFR) * self.dt
        intGrowth = intGrowth_n + dolfin.conditional(ufl.eq(hatrhoS, 0.0), 0.0, ufl.conditional(ufl.gt(time, 0), integral, 0.0))
        J_Sg = dolfin.exp(intGrowth)
        I = ufl.Identity(len(u))
        F_Sg = J_Sg ** (1 / len(u)) * I
        F_S = I + ufl.grad(u)
        C_S = F_S.T * F_S
        J_S = ufl.det(F_S)
        F_Se = F_S * ufl.inv(F_Sg)
        J_Se = ufl.det(F_Se)
        B_Se = F_Se * F_Se.T

        # Calculate velocity and time dependent variables
        v = (u - u_n) / self.dt
        div_v = ufl.inner(ufl.grad(v), ufl.inv(F_S.T))  # ufl.inner(D_S, ufl.Identity(len(u)))
        dnSdt = (nS - nS_n) / self.dt

        # Calculate Stresses
        TS_E = (self.muS * (B_Se - I) + self.lambdaS * ufl.ln(J_Se) * I) / J_Se
        T = TS_E - p * dolfin.Identity(len(u))
        P = J_S * T * ufl.inv(F_S.T)

        # Calculate seepage-velocity (w_FS)
        k_D = self.kF_0S / self.gammaFR
        nFw_FS = - J_S * k_D * ufl.dot(ufl.grad(p), ufl.inv(C_S))

        # Define weak forms
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo = res_LMo1

        res_MMo1 = J_S * div_v * _p * dx
        res_MMo2u = ufl.dot(ufl.grad(p), ufl.inv(F_S))
        res_MMo2 = + J_S * k_D * ufl.inner(res_MMo2u, ufl.inv(F_S.T) * ufl.grad(_p)) * dx
        res_MMo3 = - J_S * hatnS * (1 - self.rhoSR / self.rhoFR) * _p * dx
        res_MMo = res_MMo1 + res_MMo2 + res_MMo3

        res_MMs1 = J_S * (dnSdt - hatnS) * _nS * dx
        res_MMs2 = nS * J_S * div_v * _nS * dx
        res_MMs = res_MMs1 + res_MMs2

        res_tot = res_LMo + res_MMo + res_MMs

        # Define problem solution
        solver = nonlinvarsolver(res_tot, w, self.d_bound, self.solver_param)

        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)
        hatrhoS.interpolate(self.production)

        # Initialize old step and calc step
        u, p, nS = w.split()

        # Initialize solution time
        timeseries_intGrowth = dolfin.TimeSeries(self.growth_time_series)
        t = 0
        out_count = 0.0

        output(w, t)
        timeseries_intGrowth.store(dolfin.project(intGrowth, self.V1).vector(), t)
        # Time loop
        while t < self.T_end:
            # Increment solution time
            t += self.dt
            time.assign(t)
            out_count += self.dt

            if t > self.growth_time:
                hatrhoS.interpolate(Production(self.domain, self.growthArea, dolfin.Constant(0)))

            # Calculate current solution
            n_iter, converged = solver.solve()

            # Output solution
            if out_count >= self.output_interval:
                out_count = 0.0
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
                output(w, t)

            timeseries_intGrowth.store(dolfin.project(intGrowth, self.V1).vector(), t)
            intGrowth_n.assign(dolfin.project(intGrowth, self.V1))
            w_n.assign(w)

            if converged and n_iter == 0:
                break
