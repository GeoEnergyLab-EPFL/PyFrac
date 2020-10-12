# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
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
Mesh = CartesianMesh(0.15, [-0.175, 0.05], 47, 71)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e9                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if y > 0.025:
        return 11.2e6
    elif y < -0.025:
        return 5.0e6
    else:
        return 7.0e6

Solid = MaterialProperties(Mesh,
                           Eprime,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-8)

# injection parameters
Q0 = np.asarray([[0, 31, 151], [0.0009e-6, 0.0065e-6, 0.0023e-6]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=30)

# simulation properties
simulProp = SimulationProperties()
simulProp.bckColor = 'confining stress'           # the parameter according to which the background is color coded
simulProp.frontAdvancing = 'explicit'
simulProp.set_outputFolder('./Data/Wu_et_al')
simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))
simulProp.plotVar = ['footprint']

# initializing fracture
Fr_geometry = Geometry('radial', radius=0.019)
init_param = InitializationParameters(Fr_geometry, regime='M')

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

# loading the experiment data file
import csv
with open('./wu_et_al_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.asarray(list(reader), dtype=np.float64)


####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from visualization import *

    # plotting fracture footprint
    Fr_list, properties = load_fractures(address='./Data/Wu_et_al',
                                         time_srs=simulProp.get_solTimeSeries())
    Fig = plot_fracture_list(Fr_list,
                             variable='mesh',
                             backGround_param='sigma0',
                             mat_properties=Solid)
    plot_prop = PlotProperties(line_color='darkmagenta')
    Fig = plot_fracture_list(Fr_list,
                             variable='footprint',
                             fig=Fig,
                             plot_prop=plot_prop)

    # plotting footprint from experiment
    ax = Fig.get_axes()[0]
    ax.plot(data[:, 0]*1e-3, -1e-3*data[:, 1], 'k')
    ax.plot(data[:, 2]*1e-3, -1e-3*data[:, 3], 'k')
    ax.plot(data[:, 4]*1e-3, -1e-3*data[:, 5], 'k')
    ax.plot(data[:, 6]*1e-3, -1e-3*data[:, 7], 'k')
    ax.plot(data[:, 8]*1e-3, -1e-3*data[:, 9], 'k')

    blue_patch = mpatches.mlines.Line2D([], [], color='k', label='experiment')
    black_patch = mpatches.mlines.Line2D([], [], color='darkmagenta', label='numerical computation')
    plt.legend(handles=[blue_patch, black_patch])
    ax.set_ylim(-170e-3, 50e-3,)


    #plotting in 3D
    plot_prop_mesh = PlotProperties(disp_precision=2)
    Fig_Fr = plot_fracture_list(Fr_list,            #plotting mesh
                                variable='mesh',
                                projection='3D',
                                backGround_param='sigma0',
                                mat_properties=Solid,
                                plot_prop=plot_prop_mesh)

    Fig_Fr = plot_fracture_list(Fr_list,            #plotting footprint
                                variable='footprint',
                                projection='3D',
                                fig=Fig_Fr)

    plot_prop = PlotProperties(alpha=0.6, text_size=2.)           #plotting width
    Fig_Fr = plot_fracture_list(Fr_list,
                                variable='w',
                                projection='3D',
                                fig=Fig_Fr,
                                plot_prop=plot_prop)

    plt.show(block=True)