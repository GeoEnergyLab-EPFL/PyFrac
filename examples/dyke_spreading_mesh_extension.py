# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
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
from elasticity import load_isotropic_elasticity_matrix_toepliz
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(425, [-425, 455], 14, 15)

# solid properties
nu = 0.25                           # Poisson's ratio
youngs_mod = 1.125e9                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus


def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""
    density_high = 2700
    density_low = 2300
    layer = 2900
    Ly = 3400
    if y > layer:
        return (Ly - y) * density_low * 9.8

    return (Ly - y) * density_high * 9.8 - (Ly - layer) * (density_high - density_low) * 9.8

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           toughness=6.5e6,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-5)

# injection parameters
Q0 = np.asarray([[0.0,  500],
                [2000,    0]])  # injection rate
Injection = InjectionProperties(Q0,
                                Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=30, density=2400)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 560000                                    # the time at which the simulation stops
simulProp.set_outputFolder("./Data/neutral_buoyancy")           # the disk address where the files are saved
simulProp.gravity = True                                        # set up the gravity flag
simulProp.tolFractFront = 3e-3                                  # increase the tolerance for fracture
                                                                # front iteration
simulProp.plotTSJump = 4                                        # plot every fourth time step
simulProp.saveTSJump = 2                                        # save every second time step
simulProp.maxSolverItrs = 200                                   # increase the Anderson iteration limit for the
                                                                # elastohydrodynamic solver
simulProp.tmStpPrefactor = np.asarray([[0, 80000], [0.5, 0.1]]) # set up the time step prefactor
simulProp.timeStepLimit = 5000                                  # time step limit
simulProp.plotVar = ['w', 'v']                                  # plot fracture width and fracture front velocity
simulProp.set_mesh_extension_direction(['top', 'horizontal'])   # allow the fracture to extend in positive y and x
simulProp.set_mesh_extension_factor(1.2)                        # set the extension factor to 1.4
simulProp.useBlockToeplizCompression = True                     # use the Toepliz elasticity matrix to save memory

# initializing a static fracture
C = load_isotropic_elasticity_matrix_toepliz(Mesh, Solid.Eprime)
Fr_geometry = Geometry('radial', radius=300)
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      net_pressure=0.5e6,
                                      elasticity_matrix=C)

Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)

# create a controller
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
    time_srs = np.asarray([50, 350,  700, 1100, 2500, 12000, 50000, 560000])
    Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                         time_srs=time_srs)
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # plot footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D',
                                mat_properties=properties[0],
                                backGround_param='confining stress')
    plt_prop = PlotProperties(plot_FP_time=False)
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP,
                                plot_prop=plt_prop)

    # plot width in 3D
    plot_prop_magma=PlotProperties(color_map='jet', alpha=0.2)
    Fig_Fr = plot_fracture_list(Fr_list[2:],
                                variable='width',
                                projection='3D',
                                plot_prop=plot_prop_magma
                                )
    Fig_Fr = plot_fracture_list(Fr_list[1:],
                                variable='footprint',
                                projection='3D',
                                fig=Fig_Fr)

    plt.show(block=True)