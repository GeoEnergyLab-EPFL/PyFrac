# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo and Andreas Möri on Tue 11:26:51 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import numpy as np

# -------------- Physical parameter -------------- #

# -- Solid -- #

sigma_o = 1e10
# confining stress [Pa]

E = 1e11
# Young's modulus [Pa]

nu = 0.25
# Poisson's ratio [-]

KIc = .5e6
# Fracture toughness of the solid [Pa m^(1/2)]

rho_s = 2700
# density of the solid [kg m^(-3)]

cl = 1e-10
# Carters leak of coefficient [m s^^-1/2)]

# -- Fluid -- #

mu = 1e-3
# dynamic fluid viscosity [Pa s]

rho_f = 1e3
# fluid density [kg m^(-3)]


# -- Injection -- #

t_change = [0.0, 60, 100, 160]
# times when the injection rate changes [s]
# Note: For a continuous injection use "t_change = []". For other injection histories the first entry always needs
# to be 0.0!

Qo = [0.01, 0.0, 0.015, 0.0]
# Injection rates in the time windows defined in t_change [m^3/s]
# Note: For a continuous injection use "Qo = your_injection_rate". For injection histories the number of entries in Qo
# must match with the number of entries in t_change. First entry of Qo is injection for t between the first and second
# entry of t_change.

# -- Geometry -- #

r_init = 0.05
# initial radius of the fracture [m]


# -------------- Essential simulation parameter -------------- #

# -- Space discretization -- #

domain_limits = [-0.1, 0.1, -0.1, 0.1]
# Limits of the simulated domain [m]. Defined as [min(x), max(x), min(y), max(y)] for the fracture in a x|y plane.

number_of_elements = [61, 61]
# Number of elements [-]. First is the number of elements in direction x, second in direction y.
# Note: We recommend to use a discretization with a minimum of 41 elements (precision) and a maximum of 101 elements
# (computational cost) per direction.

# -- Time discretization (optional) -- #
# Note: Time discretisation is optional as an automatic time-stepping is implemented. You can however specify some
# features that are important for your simulation.

fixed_times = [30, 75, 150]
# The fracture will automatically be saved at the times fixed within fixed_times [s]. If you leave it empty "fixed_times
# = []" the default scheme is applied.

max_timestep = np.inf
# The maximum a time step can take [s].
# Using np.inf means that we do not fix a maximum for the time-step


# -- Miscellaneous -- #

sim_name = 'Sample_Simulation'
# Name you want to give your simulation. The folder where the data is stored will appear as such.

save_folder = "./Data/Sample_Simulations"
# The folder where the results of your simulation get saved within.

final_time = 2e4
# The time when your simulation should stop [s]

gravity = False
# Boolean to decide if gravity is used. True for yes, False for no.

run_the_simualtion = True
# Boolean to decide if the simulation will be run.

post_process_the_results = True
# Boolean to decide you want to post-process the results.
# If run_the_simualtion = True and post_process_the_results = False, then the last simulation inside "save_folder"
# will be post processed.


# <editor-fold desc="# -------------- Simulation run (do not modify this part) -------------- #">

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console
from elasticity import load_isotropic_elasticity_matrix_toepliz
from fracture_initialization import get_radial_survey_cells
import warnings
warnings.filterwarnings("ignore")

if run_the_simualtion:
    Ep = E/(1 - nu * nu)
    mup = 12 * mu
    Kp = np.sqrt(32/np.pi)*KIc
    if type(Qo) == list:
        inj = np.asarray([t_change,
                          Qo])
    else:
        inj = Qo

    # setting up the verbosity level of the log at console
    setup_logging_to_console(verbosity_level='info')

    # creating mesh
    Mesh = CartesianMesh(domain_limits[:2], domain_limits[2:], number_of_elements[0], number_of_elements[1])

    # Injection
    Injection = InjectionProperties(inj, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=mu, density=rho_f)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = final_time
    simulProp.set_outputFolder(save_folder)  # the disk address where the files are saved
    simulProp.set_simulation_name(sim_name)
    simulProp.useBlockToeplizCompression = True
    if len(fixed_times) != 0:
        simulProp.set_solTimeSeries(np.asarray(fixed_times))
    simulProp.timeStepLimit = max_timestep

    # material properties
    if gravity:
        def sigmaO_func(x, y):
            return sigma_o - y * rho_s * 9.8
        Solid = MaterialProperties(Mesh,
                                   Ep,
                                   KIc,
                                   Carters_coef=cl,
                                   confining_stress_func=sigmaO_func)
        simulProp.gravity = True
        simulProp.set_mesh_extension_factor(1.2)
        if rho_s > rho_f:
            simulProp.set_mesh_extension_direction(['top'])
        else:
            simulProp.set_mesh_extension_direction(['bottom'])
    else:
        Solid = MaterialProperties(Mesh,
                                   Ep,
                                   KIc,
                                   Carters_coef=cl,
                                   confining_stress=sigma_o)

    # initializing fracture
    surv_cells, surv_cells_dist, inner_cells = get_radial_survey_cells(Mesh, r_init)
    Fr_geometry = Geometry(shape='level set',
                           survey_cells=surv_cells,
                           tip_distances=surv_cells_dist,
                           inner_cells=inner_cells)

    C = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep)
    init_param = InitializationParameters(Fr_geometry,
                                          regime='static',
                                          net_pressure=1e-5,
                                          width=1e-8,
                                          elasticity_matrix=C)

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp)

    # create a Controller
    controller = Controller(Fr,
                            Solid,
                            Fluid,
                            Injection,
                            simulProp)

    # run the simulation
    controller.run()
# </editor-fold>

# -------------- Post Processing -------------- #

if post_process_the_results:

    # -- loading simulation results -- #
    from visualization import *
    Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name,  step_size=1)  # load all fractures
    time_srs = get_fracture_variable(Fr_list, variable='time')  # list of times
    Solid, Fluid, Injection, simulProp = properties


    # -- FIGURE 1) plot fracture footprint VS time -- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])
    plot_prop = PlotProperties()
    Fig_R = plot_fracture_list(sol_at_selected_times,
                               variable='footprint',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list(sol_at_selected_times,
                               fig=Fig_R,
                               variable='mesh',
                               mat_properties=properties[0],
                               plot_prop=plot_prop)


    # -- FIGURE 2) plot fracture radius VS time -- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the linestyle to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    Fig_R = plot_fracture_list(sol_at_selected_times,
                               variable='d_mean',
                               plot_prop=plot_prop)


    # -- FIGURES 3) and 4) plot fracture opening VS time at a given point (my_X, my_Y)-- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    my_X = 0.0
    my_Y = 0.0
    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])

    Fig_w = plot_fracture_list_at_point(sol_at_selected_times,
                                        variable='w',
                                        point=[my_X, my_Y],
                                        plot_prop=plot_prop)


    # -- FIGURES 5) and 6) plot net fluid pressure VS time at a given point (my_X, my_Y)-- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    my_X = 0.0
    my_Y = 0.0

    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])

    Fig_pf = plot_fracture_list_at_point(sol_at_selected_times,
                                        variable='pn',
                                        point=[my_X, my_Y],
                                        plot_prop=plot_prop)


    # --  FIGURE 7) plot fracture opening along a segment cutting the fracture at given times -- #
    # --  the segment is horizontal and passing at (my_X, my_Y) -- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    my_X = 0.
    my_Y = 0.

    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])

    ext_pnts = np.empty((2, 2), dtype=np.float64)
    Fig_WS = plot_fracture_list_slice(sol_at_selected_times,
                                      variable='w',
                                      projection='2D',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts,
                                      orientation='horizontal',
                                      point1=[my_X, my_Y]
                                      )


    # --  FIGURE 8) plot fracture fluid pressure along a segment cutting the fracture at given times -- #
    # --  the segment is vertical and passing at (my_X, my_Y) -- #
    # plot the solution every "tsJump" time steps
    tsJump = 10
    sol_at_selected_times = []
    for i in np.arange(0, len(Fr_list), tsJump):
        sol_at_selected_times.append(Fr_list[i])
    my_X = 0.
    my_Y = 0.
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    Fig_PN = plot_fracture_list_slice(sol_at_selected_times,
                                      variable='pf',
                                      projection='2D',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts,
                                      orientation='vertical',
                                      point1=[my_X, my_Y]
                                      )
    plt.show(block=True)


# -------------- END OF FILE -------------- #



