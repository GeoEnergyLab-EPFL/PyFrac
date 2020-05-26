# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from visualization import *
from scipy import interpolate

# creating mesh
Mesh = CartesianMesh(.3, .3, 41, 41)

# solid properties
nu = 0.15                               # Poisson's ratio
youngs_mod = 3e10                       # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2)     # plain strain modulus
K_Ic = 2e6                              # fracture toughness
sigma0 = 15e6                           # confining stress

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic, 
                            # Carters_coef = 1e-6,
                           confining_stress=sigma0,
                           minimum_width=1e-5)

Q0 = 0.001/60
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=0.22, rheology='HBF', n=0.617, k=0.22, T0=2.3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 6e7                           # the time at which the simulation stops
simulProp.set_outputFolder("./Data/HB")                # the disk address where the files are saved
simulProp.set_simulation_name('HBF_6.17e-1_2.2e-1_2.3_71_And_reg')
simulProp.saveYieldRatio = True
simulProp.plotVar = ['ev', 'y', 'w']
simulProp.saveEffVisc = True
simulProp.relaxation_factor = 0.3
simulProp.tmStpPrefactor = 0.5
# simulProp.elastohydrSolver = 'implicit_Picard'
# simulProp.set_tipAsymptote('HBF_num_quad')
# simulProp.frontAdvancing = 'implicit'
simulProp.maxSolverItrs = 400
simulProp.Anderson_parameter = 10
simulProp.plotFigure = False
# simulProp.saveToDisk = False
# simulProp.blockFigure = True

# initialization parameters
Fr_geometry = Geometry('radial', radius=0.2)
from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry, regime='static', net_pressure=1e6, elasticity_matrix=C)

# # creating fracture object
# Fr = Fracture(Mesh,
#               init_param,
#               Solid,
#               Fluid,
#               Injection,
#               simulProp)
Fr = load_fractures(address="./Data/HB", sim_name='HBF_6.17e-1_2.2e-1_2.3_71_And_reg')[0][-1]
# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

# run the simulation
# controller.run()

####################
# plotting results #
####################

from visualization import *


# loading simulation results
Fr_list, properties = load_fractures(address="./Data/HB",
                                     sim_name="HBF_6.17e-1_2.2e-1_2.3_71_And_reg__2020-05-26__15_09_24")      # load all fractures
time_srs = get_fracture_variable(Fr_list, variable='time')                      # list of times

animate_simulation_results(Fr_list, variable=['y', 'ev'])
# plot fracture radius
plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
Fig_R = plot_fracture_list(Fr_list,
                           variable='d_mean',
                           plot_prop=plot_prop)

# plot analytical radius
plot_prop_k = PlotProperties(line_color='b')
Fig_R = plot_analytical_solution(regime='K',
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  time_srs=time_srs,
                                  fig=Fig_R,
                                  plot_prop=plot_prop)

# plot width at center
Fig_w = plot_fracture_list_at_point(Fr_list,
                                    variable='w',
                                    plot_prop=plot_prop)
# plot analytical width at center
Fig_w = plot_analytical_solution_at_point('K',
                                          'w',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_w)


plt.show(block=True)

