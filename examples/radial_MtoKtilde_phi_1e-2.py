# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os

import numpy as np

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
Mesh = CartesianMesh(0.5, 0.5, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 1e6                           # Fracture toughness
Cl = 5.875e-6                         # Carter's leak off coefficient

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 1e-3
Fluid = FluidProperties(viscosity=viscosity)

# value of the trajectory parameter phi
phi = (2 * Cl) ** 4 * Eprime ** 11 * (12 * viscosity) ** 3 * Q0 / ((32 / np.pi) ** (1/2) * K1c) ** 14

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e5                           # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 5, 5   # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/MtoK_leakoff")   # the disk address where the files are saved
simname = 'phi_001'
simulProp.set_simulation_name(simname)              # set the nime of the simulation
simulProp.frontAdvancing = 'explicit'               # setting up explicit front advancing
simulProp.plotVar = ['regime', 'w']

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=1e-3)

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
    Fr_list, properties = load_fractures("./Data/MtoK_leakoff",
                                         sim_name=simname)
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
    t = np.asarray([0.001, 0.00160372, 0.00257191, 0.00412463, 0.00661474, 0.0106082,
    0.0170125, 0.0272833, 0.0437548, 0.0701704, 0.112534, 0.180472,
    0.289427, 0.464159, 0.74438, 1.19378, 1.91448, 3.07029, 4.92388,
    7.89652, 12.6638, 20.3092, 32.5702, 52.2335, 83.7678, 134.34,
    215.443, 345.511, 554.102, 888.624, 1425.1, 2285.46, 3665.24,
    5878.02, 9426.68, 15117.8, 24244.6, 38881.6, 62355.1, 100000.])
    # solution taken from matlab code provided by Dontsov EV (2016)
    eff_analytical = np.asarray([0.995112, 0.994179, 0.993034, 0.991678, 0.99007, 0.98789, 0.985505,
    0.982858, 0.979418, 0.975125, 0.97028, 0.964953, 0.958285, 0.950267,
    0.940702, 0.928993, 0.916286, 0.90124, 0.883589, 0.862549, 0.838824,
    0.812694, 0.783958, 0.74958, 0.711572, 0.671169, 0.629418, 0.583406,
    0.536243, 0.488384, 0.440254, 0.393285, 0.348094, 0.305188, 0.267377,
    0.230932, 0.197317, 0.168732, 0.144201, 0.122716])
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
    r_analytical = np.asarray([0.1726, 0.212748, 0.262185, 0.323053, 0.397975, 0.490095, 0.603465,
    0.742955, 0.914298, 1.12467, 1.38308, 1.70045, 2.08924, 2.56516,
    3.14698, 3.85638, 4.72309, 5.77686, 7.05496, 8.59759, 10.4592,
    12.7025, 15.3974, 18.5791, 22.3335, 26.7772, 32.0535, 38.0903,
    45.0873, 53.1663, 62.3592, 72.8777, 84.817, 98.2476, 113.145,
    130.122, 149.291, 170.609, 194.392, 221.091])
    ax_r = Fig_r.get_axes()[0]
    ax_r.loglog(t, r_analytical, 'r-', label='semi-anlytical radius')
    ax_r.legend()

    plt.show(block=True)