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
Mesh = CartesianMesh(5, 5, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 5e5 / (32 / np.pi)**0.5       # K' = 5e5
Cl = 0.5e-6                         # Carter's leak off coefficient

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 0.001 / 12  # mu' =0.001
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e7                           # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 5, 5   # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/MtoK_leakoff")   # the disk address where the files are saved
simulProp.frontAdvancing = 'explicit'               # setting up explicit front advancing
simulProp.plotVar = ['regime', 'w']

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=0.5)

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
    Fr_list, properties = load_fractures("./Data/MtoK_leakoff")
    time_srs = get_fracture_variable(Fr_list,
                                     'time')
    # plotting efficiency
    plot_prop = PlotProperties(graph_scaling='loglog',
                               line_style='.')
    label = LabelProperties('efficiency')
    label.legend = 'fracturing efficiency'
    Fig_eff = plot_fracture_list(Fr_list,
                               variable='efficiency',
                               plot_prop=plot_prop,
                               labels=label)
    t = 2**np.linspace(np.log2(0.5),np.log2(1e7),30)
    # solution taken from matlab code provided by Dontsov EV (2016)
    eff_analytical = np.asarray([0.9923, 0.9904, 0.9880, 0.9850, 0.9812, 0.9765, 0.9708, 0.9636, 0.9547, 0.9438, 0.9305,
                                 0.9142, 0.8944, 0.8706, 0.8423, 0.8089, 0.7700, 0.7256, 0.6757, 0.6209, 0.5622, 0.5011,
                                 0.4393, 0.3789, 0.3215, 0.2688, 0.2218, 0.1809, 0.1461, 0.1171])
    ax_eff = Fig_eff.get_axes()[0]
    ax_eff.semilogx(t, eff_analytical, 'r-', label='semi-analytical fracturing efficiency')
    ax_eff.legend()


    label = LabelProperties('d_mean')
    label.legend = 'radius'
    Fig_r = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop,
                               labels=label)
    # solution taken from matlab code provided by Dontsov EV (2016)
    r_analytical = np.asarray([0.0035, 0.0046, 0.0059, 0.0076, 0.0099, 0.0128, 0.0165, 0.0212, 0.0274, 0.0352, 0.0453,
                               0.0581, 0.0744, 0.0951, 0.1212, 0.1539, 0.1948, 0.2454, 0.3075, 0.3831, 0.4742, 0.5829,
                               0.7114, 0.8620, 1.0370, 1.2395, 1.4726, 1.7406, 2.0483, 2.4016])*1e3
    ax_r = Fig_r.get_axes()[0]
    ax_r.loglog(t, r_analytical, 'r-', label='semi-anlytical radius')
    ax_r.legend()

    plt.show(block=True)