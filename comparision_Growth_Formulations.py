########################################################################
# Imports
########################################################################
from itertools import product
import os
import dolfin as df
from TPM_2Phase_MAoLMoMAs_Growth import TPM_2Phase_MAoLMoMAs_Growth
from Rodriguez1994 import Rodriguez1994
from helper import set_output_file, mkdir_if_not_exist, Problem, create_Quarter_Circle, create_Rectangle, get_geometry
########################################################################
# Routines for specific model
########################################################################
def run_tpm(prob):
    tpm = TPM_2Phase_MAoLMoMAs_Growth()
    prob.param.gen.output_file = set_output_file(sol_folder + os.sep + str(prob.param.gen.title))
    print("run ", prob.param.gen.title)
    tpm.set_param(prob)
    tpm.set_function_spaces()
    bcs = []
    if prob.param.gen.geom_title == "Terzaghi":
        bcs.append(df.DirichletBC(tpm.function_space.sub(0).sub(0), 0.0, prob.geom.facet_function, 4))
        bcs.append(df.DirichletBC(tpm.function_space.sub(0).sub(0), 0.0, prob.geom.facet_function, 6))
        bcs.append(df.DirichletBC(tpm.function_space.sub(0).sub(1), 0.0, prob.geom.facet_function, 3))
        if prob.geom.bool_pressure:
            bcs.append(df.DirichletBC(tpm.function_space.sub(1), 0.0, prob.geom.facet_function, 5))

    else:
        bcs.append(df.DirichletBC(tpm.function_space.sub(0).sub(0), 0.0, prob.geom.facet_function, 4))
        bcs.append(df.DirichletBC(tpm.function_space.sub(0).sub(1), 0.0, prob.geom.facet_function, 3))
        if prob.geom.bool_pressure:
            bcs.append(df.DirichletBC(tpm.function_space.sub(1), 0.0, prob.geom.facet_function, 5))
    tpm.set_boundaries(bcs, None)
    tpm.set_initial_condition()
    tpm.solve()


def run_rodriguez(prob):
    rodriguez94 = Rodriguez1994()
    prob.param.gen.output_file = set_output_file(sol_folder + os.sep + str(prob.param.gen.title) + "_Rod")
    print("run ", prob.param.gen.title + "/Rodriguez")
    rodriguez94.set_param(prob)
    rodriguez94.set_function_spaces()
    bcs = []
    bcs.append(df.DirichletBC(rodriguez94.function_space.sub(1), 0.0, prob.geom.facet_function, 3))
    bcs.append(df.DirichletBC(rodriguez94.function_space.sub(0), 0.0, prob.geom.facet_function, 4))
    rodriguez94.set_boundaries(bcs, None)
    rodriguez94.set_initial_condition()
    rodriguez94.solve()


########################################################################
# Output directories
########################################################################
output_folder = mkdir_if_not_exist("output")
sol_folder = mkdir_if_not_exist(output_folder + os.sep + "sol")
########################################################################
# Specific Problems
########################################################################
# Problem 1
# General
p_2DSC = Problem()
p_2DSC.param.gen.geom_title = "SplitCircle"
# Geometry
raw_file = output_folder + os.sep + "2D_QuarterCircle"
geom_path = create_Quarter_Circle(0.01, 10, 1, 5, 80, raw_file, True)  # m
p_2DSC.param.gen.eval_points = []
p_2DSC.geom = get_geometry(geom_path)
p_2DSC.geom.growthArea = [2]
########################################################################
# Problem 2
# General
p_2DFC = Problem()
p_2DFC.param.gen.geom_title = "FullCircle"
# Geometry
raw_file = output_folder + os.sep + "2D_QuarterCircle"
geom_path = create_Quarter_Circle(0.01, 10, 1/5, 1, 80, raw_file, True)  # m
p_2DFC.param.gen.eval_points = []
p_2DFC.geom = get_geometry(geom_path)
p_2DFC.geom.growthArea = [1, 2]
########################################################################
# Problem 3
# General
p_2DTE = Problem()
p_2DTE.param.gen.geom_title = "Terzaghi"
# Geometry
raw_file = output_folder + os.sep + "2D_Terzaghi"
geom_path = create_Rectangle(0.1, 0.1, 1, 5, 1, raw_file)  # m
p_2DTE.param.gen.eval_points = []
p_2DTE.geom = get_geometry(geom_path)
p_2DTE.geom.growthArea = [2]
########################################################################
# Parameters
########################################################################
problems = [p_2DSC]  # [p_2DFC, p_2DSC]
alpha_g_val = [0.0, 0.25, 1.0]
nS_0S_val = [0.3]
kF_0S_val = [1.0e-1]                        # m/s
gammaFR_val = [1.0]                         # m/s
e_modulus_val = [1.0]                       # N/m^2
poisson_val = [0.01, 0.1, 0.4]              # -
rhoSR_val = [1]                             # kg/m^3
rhoFR_val = [2]                             # kg/m^3
hatrhoS_val = [1.0e-2]                      # kg/(m^3*s)
T_end_val = [800]                           # s
output_interval_val = [1]                   # s
dt_val = [1]                                # s
growth_time_val = [50]                      # s
bounds = ["withoutOutflow", "withOutflow"]
########################################################################
# Calculation loop
########################################################################
print("Start calculations")
df.set_log_level(80)
for p in problems:
    for alpha_g, nS_0S, kF_0S, gammaFR, e_modulus, poisson, rhoSR, rhoFR, hatrhoS, T_end, output_interval, dt, \
        growth_time \
            in product(alpha_g_val, nS_0S_val, kF_0S_val, gammaFR_val, e_modulus_val, poisson_val, rhoSR_val, rhoFR_val, 
                       hatrhoS_val, T_end_val, output_interval_val, dt_val, growth_time_val):
        # Material Parameters
        p.param.mat.nS_0S = nS_0S
        p.param.mat.kF_0S = kF_0S  
        p.param.mat.gammaFR = gammaFR  
        p.param.mat.lambdaS = e_modulus * poisson / ((1 + poisson) * (1 - 2 * poisson))  # N/m^2
        p.param.mat.muS = e_modulus * 1 / (2 * (1 + poisson))  # N/m^2
        p.param.mat.rhoSR = rhoSR
        p.param.mat.rhoFR = rhoFR
        p.param.mat.hatrhoS = hatrhoS
        p.param.mat.alpha_g = alpha_g
        p.param.gen.growth_time_series = output_folder + os.sep + "intGrowth_Series"
        # Time Parameters
        p.param.time.T_end = T_end
        p.param.time.output_interval = output_interval
        p.param.time.dt = dt
        p.param.time.growth_time = growth_time
        # FEM Paramereters
        p.param.fem.solver_param.solver_type = "mumps"
        p.param.fem.solver_param.maxIter = 10
        p.param.fem.solver_param.rel = 1E-9
        p.param.fem.solver_param.abs = 1E-8
        p.param.fem.type_u = "CG"
        p.param.fem.order_u = 2
        p.param.fem.type_p = "CG"
        p.param.fem.order_p = 1
        p.param.fem.type_nS = "CG"
        p.param.fem.order_nS = 1

        param_string = ""
        for x in [alpha_g, nS_0S, kF_0S, gammaFR, e_modulus, poisson, rhoSR, rhoFR, hatrhoS, T_end,
                  output_interval, dt, growth_time]:
            param_string += "_" + str(x)

        for bound in bounds:
            p.param.gen.bounds = bound
            p.param.gen.title = p.param.gen.geom_title + param_string + "_" + bound
            p.geom.bool_pressure = True
            if bound == "withoutOutflow":
                p.geom.bool_pressure = False
            run_tpm(p)

        #for problem in problems:
        #    run_rodriguez(p)
