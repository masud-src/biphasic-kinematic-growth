
from helper import Problem, write_field2output, norm, nonlinvarsolver
import dolfin
import ufl

class InitialCondition(dolfin.UserExpression):
    def __init__(self, subdomains, dim, **kwargs):
        self.subdomains = subdomains
        self.dim = dim
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.dim==3:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
        else:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y

    def value_shape(self):
        if self.dim==3:
            return (3, )
        else:
            return (2, )

class Production(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, growthArea, J_G, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.growthArea = growthArea
        self.J_G = J_G

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # hatrhoS
        for area in self.growthArea:
            if self.subdomains[cell.index] == area:
                values[0] = self.J_G
    def value_shape(self):
        return ()

class Rodriguez1994:
    """
    t.b.d.
    """
    def __init__(self):
        self.dim = None
        self.growth_time_series = None
        self.output_file = None
        self.finite_element = None
        self.function_space = None
        self.internal_function_spaces = None
        self.V1 = None
        self.V2 = None
        self.V3 = None
        self.type_u = None
        self.order_u = None
        self.mesh = None
        self.domain = None
        self.growthArea = None
        self.dx = None
        self.n_bound = None
        self.d_bound = None
        self.initial_condition = None
        self.internal_condition = None
        self.rhoSR = None
        self.lambdaS = None
        self.muS = None
        self.solver_param = None
        self.dt = None
        self.T_end = None
        self.output_interval = None
        self.growth_time = None

    def set_initial_condition(self):
        self.initial_condition = InitialCondition(self.domain, self.dim)
        self.internal_condition = Production(self.domain, self.growthArea, self.hatrhoS)

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def set_param(self, input: Problem):
        """
        sets parameter needed for model class
        """
        self.dim = input.geom.dim
        self.growth_time_series = input.param.gen.growth_time_series
        self.eval_points = input.param.gen.eval_points
        self.output_file = input.param.gen.output_file
        self.type_u = input.param.fem.type_u
        self.order_u = input.param.fem.order_u
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        self.growthArea = input.geom.growthArea
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.hatrhoS = input.param.mat.hatrhoS
        self.rhoSR = input.param.mat.rhoSR
        self.lambdaS = input.param.mat.lambdaS
        self.muS = input.param.mat.muS
        self.solver_param = input.param.fem.solver_param
        self.dt = input.param.time.dt
        self.T_end = input.param.time.T_end
        self.output_interval = input.param.time.output_interval
        self.growth_time = input.param.time.growth_time

    def set_function_spaces(self):
        """
        sets function space for primary variables u, p, nS and for internal variables
        """
        element_u = dolfin.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)  # defines vector approximation for displacement
        self.finite_element = element_u
        self.function_space = dolfin.FunctionSpace(self.mesh, self.finite_element)
        self.V1 = dolfin.FunctionSpace(self.mesh, "P", 1)
        self.V2 = dolfin.VectorFunctionSpace(self.mesh, "P", 1)
        self.V3 = dolfin.TensorFunctionSpace(self.mesh, "P", 1)

    def solve(self):
        """
        """
        def output(time):
            u_av = dolfin.project(norm(u), self.V1)
            F_S_ = dolfin.project(F_S, self.V3)
            F_SE_ = dolfin.project(F_Se, self.V3)
            F_SG_ = dolfin.project(F_Sg, self.V3)
            J_SE_ = dolfin.project(J_Se, self.V1)
            J_SG_ = dolfin.project(J_Sg, self.V1)
            T_ = dolfin.project(T, self.V3)
            T0 = dolfin.project(T[0, 0], self.V1)
            u0 = dolfin.project(u[0], self.V1)

            write_field2output(self.output_file, u, "u", time)
            write_field2output(self.output_file, u_av, "u_av", time)
            write_field2output(self.output_file, F_S_, "F_S", time)
            write_field2output(self.output_file, F_SE_, "F_Se", time)
            write_field2output(self.output_file, F_SG_, "F_Sg", time)
            write_field2output(self.output_file, J_SE_, "J_Se", time)
            write_field2output(self.output_file, J_SG_, "J_Sg", time)
            write_field2output(self.output_file, T_, "stress", time)
            write_field2output(self.output_file, T0, "stress 0", time)
            write_field2output(self.output_file, u0, "displ 0", time)

        prm = dolfin.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        timeseries_intGrowth = dolfin.TimeSeries(self.growth_time_series)

        # Store history values for time integration
        u_n = dolfin.Function(self.function_space)
        intGrowth = dolfin.Function(self.V1)
        hatrhoS = dolfin.Function(self.V1)
        time = dolfin.Constant(0.0)

        # Get Ansatz and test functions
        u = dolfin.Function(self.function_space)
        _u = dolfin.TestFunction(self.function_space)

        # Integration over domain
        dx = dolfin.Measure("dx", metadata={'quadrature_degree': 2})

        # Calculate Deforamtion Measures
        J_Sg = dolfin.exp(intGrowth)
        I = ufl.Identity(len(u))
        F_Sg = J_Sg ** (1 / len(u)) * I
        F_S = I + ufl.grad(u)
        J_S = ufl.det(F_S)
        F_Se = F_S * ufl.inv(F_Sg)
        J_Se = ufl.det(F_Se)
        B_Se = F_Se * F_Se.T

        # Calculate Stresses
        TS_E = (self.muS * (B_Se - I) + self.lambdaS * ufl.ln(J_Se) * I) / J_Se
        T = TS_E

        # Backmapping
        P = J_S * ufl.dot(T, ufl.inv(F_Se))

        # Define weak forms
        res_BLM = ufl.inner(P, ufl.grad(_u)) * dx

        res_tot = res_BLM

        # Define problem solution
        solver = nonlinvarsolver(res_tot, u, self.d_bound, self.solver_param)

        # Set initial conditions
        u_n.interpolate(self.initial_condition)
        hatrhoS.interpolate(self.internal_condition)

        # Initialize solution time
        t = 0
        out_count = 0.0
        timeseries_intGrowth.retrieve(intGrowth.vector(), t)

        output(t)

        # Time loop
        while t < self.T_end:
            # Increment solution time
            t += self.dt
            time.assign(t)
            out_count += self.dt
            timeseries_intGrowth.retrieve(intGrowth.vector(), t)
            # Print current time
            dolfin.info("Time: {}".format(t))

            if t > self.growth_time:
                hatrhoS.interpolate(Production(self.domain, self.growthArea, dolfin.Constant(0)))

            # Calculate current solution
            n_iter, converged = solver.solve()

            # Output solution
            if out_count >= self.output_interval:
                out_count = 0.0
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
                output(t)

            u_n.assign(u)

            if converged and n_iter == 0:
                break
