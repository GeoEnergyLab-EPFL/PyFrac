# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Aug 31 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(104, 63, 105, 85, symmetric=True)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

# the function below will make the fracture propagate a specific shape at large time (see Zia et al. IJF 2018)
# somehow "eye" like at large time
def K1c_func(alpha):
    K1c_1 = 2.0e6                    # fracture toughness along x-axis
    K1c_2 = 3.0e6                    # fracture toughness along y-axis
# the evolution between the 0 and 90 deg angle is a smooth Heaviside starting at "sharp" angle  3 pi/20
    j = 3 * np.pi / 20
    f = 1 / (1 + np.e ** (-2 * 5 * (alpha - j)))
    return K1c_1 + (K1c_2 - K1c_1) * f

Solid = MaterialProperties(Mesh,
                           Eprime,
                           anisotropic_K1c=True,
                           K1c_func=K1c_func)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)    # toughness dominated solution

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 4000            # the time at which the simulation stops
simulProp.set_volumeControl(True)     # to set up the solver in volume control mode (inviscid fluid)
simulProp.tolFractFront = 4e-3        # increase tolerance for the anisotropic case
simulProp.set_outputFolder("./Data/toughness_jump") # the disk address where the files are saved
simulProp.set_simulation_name('anisotropic_toughness_jump')
simulProp.symmetric = True            # set the fracture to symmetric
simulProp.projMethod = 'ILSA_orig'
simulProp.set_tipAsymptote('U')

# initializing fracture
gamma = (K1c_func(np.pi/2) / K1c_func(0))**2    # gamma = (Kc1/Kc3)**2
Fr_geometry = Geometry('elliptical', minor_axis=15., gamma=gamma)
init_param = InitializationParameters(Fr_geometry, regime='E_K')

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

if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    time_srs = 2 ** np.linspace(np.log2(40), np.log2(5000), 8)
    Fr_list, properties = load_fractures(address="./Data/toughness_jump",
                                    sim_name ='anisotropic_toughness_jump',
                                    time_srs=time_srs)
    # plotting footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D')
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP)

    #plotting toughness dominated radial and elliptical solutions
    plot_prop = PlotProperties(line_color_anal='orange')
    Fig_FP = plot_analytical_solution('K',
                                      'footprint',
                                      Solid,
                                      Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_FP,
                                      plot_prop=plot_prop,
                                      projection='2D',
                                      time_srs=[Fr_list[-1].time])
    plot_prop.lineColorAnal = 'b'
    Fig_FP = plot_analytical_solution('E_K',
                                      'footprint',
                                      Solid,
                                      Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_FP,
                                      plot_prop=plot_prop,
                                      projection='2D',
                                      time_srs=[Fr_list[-1].time])

    # loading all fractures
    Fr_list, properties = load_fractures(address="./Data/toughness_jump",
                                         sim_name='anisotropic_toughness_jump')
    plot_prop = PlotProperties(line_style='.-',
                               graph_scaling='semilogx')
    Fig_len_a = plot_fracture_list(Fr_list,
                                    variable='ar',
                                    plot_prop=plot_prop)

    plt.show(block=True)