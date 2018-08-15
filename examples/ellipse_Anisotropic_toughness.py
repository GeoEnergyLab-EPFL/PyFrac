# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(8, 4, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

# the function below will make the fracture propagate in the form of an ellipse (see Zia and Lecampion 2018)
def K1c_func(alpha):
    K1c_1 = 1.e6                    # fracture toughness along x-axis
    K1c_2 = 1.5e6                   # fracture toughness along y-axis

    beta = np.arctan((K1c_1 / K1c_2)**2 * np.tan(alpha))
    return 4 * (2/np.pi)**0.5 * K1c_2 * (np.sin(beta)**2 + (K1c_1 / K1c_2)**4 * np.cos(beta)**2)**0.25

Solid = MaterialProperties(Mesh,
                           Eprime,
                           anisotropic_K1c=True,
                           K1c_func=K1c_func)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-5)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 500               # the time at which the simulation stops
simulProp.set_volumeControl(True)     # to set up the solver in volume control mode (inviscid fluid)
simulProp.set_tipAsymptote('K')       # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.outputTimePeriod = 1e-10      # save after every time step
simulProp.tolFractFront = 4e-3          # increase tolerance for the anisotropic case
simulProp.remeshFactor = 1.5            # the factor by which the mesh will be compressed.
simulProp.set_outputFolder(".\\Data\\ellipse") # the disk address where the files are saved
simulProp.set_simulation_name('anisotropic_toughness_benchmark')
simulProp.verbosity = 2
# simulProp.plotFigure = True
simulProp.plotAnalytical = True
simulProp.analyticalSol = 'E_K'
simulProp.explicitProjection = True
simulProp.symmetric = True

# initializing fracture
minor_axis = 2.
gamma = (K1c_func(np.pi/2) / K1c_func(0))**2    # gamma = (Kc1/Kc3)**2
init_param = ("E_K", "length", minor_axis, gamma)

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


####################
# plotting results #
####################

# loading simulation results
time_srs = 2 ** np.linspace(np.log2(1), np.log2(500), 8)
Fr_list, properties = load_fractures(address=".\\Data\\ellipse",
                                    simulation='anisotropic_toughness_benchmark',
                                    time_srs=time_srs)
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')

# plotting footprint
Fig_FP = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='2D')
Fig_FP = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='2D',
                            fig=Fig_FP)
Fig_FP = plot_analytical_solution('E_K',
                                  'footprint',
                                  Solid,
                                  Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_FP,
                                  projection='2D',
                                  time_srs=time_srs)

# loading fractures
time_srs = 2 ** np.linspace(np.log2(3), np.log2(500), 5)
Fr_list, properties = load_fractures(address=".\\Data\\ellipse",
                                            simulation='anisotropic_toughness_benchmark',
                                            time_srs=time_srs)
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')

#plotting width
Fig_w_slice = plot_fracture_list_slice(Fr_list,
                                       variable='width',
                                       point1=[-Fr_list[-1].mesh.Lx, 0],
                                       point2=[Fr_list[-1].mesh.Lx, 0])
Fig_w_slice = plot_analytical_solution_slice('E_K',
                                             variable='width',
                                             mat_prop=Solid,
                                             inj_prop=Injection,
                                             fluid_prop=Fluid,
                                             fig=Fig_w_slice,
                                             point1=[-Fr_list[-1].mesh.Lx, 0],
                                             point2=[Fr_list[-1].mesh.Lx, 0],
                                             time_srs=time_srs,
                                             plt_2D_image=False)

# loading all fractures
Fr_list, properties = load_fractures(address=".\\Data\\ellipse",
                                     simulation='anisotropic_toughness_benchmark')
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')
# making a plot properties object with desired linestyle and scaling
plot_prop = PlotProperties(line_style='.',
                           graph_scaling='loglog')
# plotting minor axis length
labels = get_labels('d_min', 'wm', '2D')
labels.figLabel = 'Minor axis length'
Fig_len_a = plot_fracture_list(Fr_list,
                             variable='d_min',
                             plot_prop=plot_prop,
                             labels=labels)
Fig_len_a = plot_analytical_solution('E_K',
                                   'd_min',
                                   Solid,
                                   Injection,
                                   fluid_prop=Fluid,
                                   fig=Fig_len_a,
                                   time_srs=time_srs,
                                   labels=labels)
# plotting major axis length
labels = get_labels('d_max', 'whole_mesh', '2D')
labels.figLabel = 'Major axis length'
Fig_len_b = plot_fracture_list(Fr_list,
                             variable='d_max',
                             plot_prop=plot_prop,
                             labels=labels)
Fig_len_b = plot_analytical_solution('E_K',
                                   'd_max',
                                   Solid,
                                   Injection,
                                   fluid_prop=Fluid,
                                   fig=Fig_len_b,
                                   time_srs=time_srs,
                                   labels=labels)

plt.show()
