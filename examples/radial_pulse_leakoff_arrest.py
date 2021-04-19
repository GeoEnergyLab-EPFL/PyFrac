# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
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
Mesh = CartesianMesh(2, 2, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 0                             # Zero toughness case
Cprime = 5e-9


# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cprime)

# injection parameters
Q0 = np.asarray([[0.0, 5],
                 [0.01, 0]])  # injection rate

Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 0.001
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e6                            # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 5, 3      # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/Pulse")             # the disk address where the files are saved

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=0.05)

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

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/Pulse")       # load all fractures
    time_srs = get_fracture_variable(Fr_list,                             # list of times
                                     variable='time')

    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the linestyle to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    label = LabelProperties('d_mean')
    label.legend = 'radius'
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop) # numerical radius

    # plot analytical M-vertex solution for radius
    plt_prop = PlotProperties(line_color_anal='b')
    label = LabelProperties('d_mean')
    label.legend = 'M solution'
    Fig_R = plot_analytical_solution(regime='M',
                                     variable='d_mean',
                                     labels=label,
                                     mat_prop=properties[0],
                                     inj_prop=properties[2],
                                     fluid_prop=properties[1],
                                     time_srs=time_srs,
                                     plot_prop=plt_prop,
                                     fig=Fig_R)

    # plot analytical M-pulse-vertex solution for radius
    plt_prop = PlotProperties(line_color_anal='m')
    label = LabelProperties('d_mean')
    label.legend = 'M-pulse solution'
    Fig_R = plot_analytical_solution(regime='Mp',
                                     variable='d_mean',
                                     labels=label,
                                     mat_prop=properties[0],
                                     inj_prop=properties[2],
                                     fluid_prop=properties[1],
                                     time_srs=time_srs,
                                     plot_prop=plt_prop,
                                     fig=Fig_R)

    # plot the toughness arrest radius
    plt_prop = PlotProperties(line_color_anal='g')
    label = LabelProperties('d_mean')
    label.legend = 'Leak-off arrest'
    Fig_R = plot_analytical_solution(regime='La',
                                     variable='d_mean',
                                     labels=label,
                                     mat_prop=properties[0],
                                     inj_prop=properties[2],
                                     fluid_prop=properties[1],
                                     time_srs=time_srs,
                                     plot_prop=plt_prop,
                                     fig=Fig_R)

    plt.show(block=True)