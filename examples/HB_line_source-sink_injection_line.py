# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from controller import Controller
from fracture import Fracture
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(.75, .75, 41, 41)

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
                           Carters_coef=1e-6,
                           confining_stress=sigma0,
                           minimum_width=1e-5)


def sink_location(x, y):
    """ This function is used to evaluate if a point is a sink."""
    return abs(y) >= .6 and abs(y) <= 1. and abs(x) < 0.01

def sink_vel(x, y):
    """ This function gives the sink velocity of point."""
    return 6e-4

def source_location(x, y):
    """ This function is used to evaluate if a point is a source, i.e.
        the fluid is injected at the given point.
    """
    # the condition
    return abs(x) < 0.01 and abs(y) < 0.5


Q0 = 0.001/60
Injection = InjectionProperties(Q0,                             # see documentation of the class for details
                                Mesh,
                                source_loc_func=source_location,
                                sink_loc_func=sink_location,
                                sink_vel_func=sink_vel,
                                model_inj_line=True,
                                il_compressibility=1e-9,
                                il_volume=1e-3,
                                perforation_friction=0,
                                initial_pressure=np.nan)        # the initial pressure in injection line is set below

# fluid properties
Fluid = FluidProperties(viscosity=0.617,
                        rheology='HBF',                         # set fluid rheology to Herschel-Bulkley
                        compressibility=1e-11,
                        n=0.617, k=0.22, T0=2.3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 86                                # the time at which the simulation stops
simulProp.set_outputFolder("./Data/HB")                 # the disk address where the files are saved
simulProp.set_simulation_name('HB_injection_line_sink') # setting simulation name
simulProp.saveG = True                                  # enable saving the coefficient G
simulProp.plotVar = ['ir', 'w']                         # plot width of fracture
simulProp.saveEffVisc = True                            # enable saving of the effective viscosity
simulProp.relaxation_factor = 0.3                       # relax Anderson iteration
simulProp.maxSolverItrs = 200                           # set maximum number of Anderson iterations to 200
simulProp.Anderson_parameter = 10                       # save last 10 iterations in Anderson iteration
simulProp.collectPerfData = True                        # enable collect performance data
simulProp.fixedTmStp = np.asarray([[0, 0.5], [0.01, None]])     # set auto time step size after propagation start
simulProp.tolFractFront = 0.003                         # relaxing tolerance for front iteration
simulProp.set_tipAsymptote('HBF')                       # setting tip asymptote to Herschel-Bulkley fluid

# starting simulation with a static radial fracture with radius 20cm and net pressure of 1MPa
Fr_geometry = Geometry('radial', radius=0.2)
from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      net_pressure=1e6,
                                      elasticity_matrix=C)

# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)
Fr.pInjLine = Fr.pFluid[Mesh.CenterElts]

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
    Fr_list, properties = load_fractures(address="./Data/HB",
                                         sim_name='HB_injection_line_sink')

    # see evolution of the injection rate
    animate_simulation_results(Fr_list, variable=['ir'])

    # plotting injection line pressure and fracture radius versus time
    plt_prop = PlotProperties(line_style='.-')
    Fig_p = plot_fracture_list(Fr_list, variable='injection line pressure', plot_prop=plt_prop)
    Fig_r = plot_fracture_list(Fr_list, variable='d_mean', plot_prop=plt_prop)

    plt.show(block=True)