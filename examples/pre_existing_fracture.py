# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
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
Mesh = CartesianMesh(6, 6, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1e7                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           minimum_width=1e-9)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 1.1e-3
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 50                      # the time at which the simulation stops
simulProp.set_outputFolder("./Data/star")     # the address of the output folder
simulProp.plotTSJump = 4

# initializing fracture
from fracture_initialization import get_radial_survey_cells
initRad = np.pi
surv_cells, _, inner_cells = get_radial_survey_cells(Mesh, initRad)
surv_cells_dist = np.cos(Mesh.CenterCoor[surv_cells, 0]) + 2.5 - abs(Mesh.CenterCoor[surv_cells, 1])
Fr_geometry = Geometry(shape='level set',
                       survey_cells=surv_cells,
                       tip_distances=surv_cells_dist,
                       inner_cells=inner_cells)

from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      net_pressure=1e3,
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

####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/star")
    time_srs = get_fracture_variable(Fr_list,
                                     'time')

    # plotting maximum distance of the front from the injection point
    plot_prop = PlotProperties(line_style='.')
    Fig_d = plot_fracture_list(Fr_list,
                               variable='d_max',
                               plot_prop=plot_prop)
    Fig_FP = plot_analytical_solution(regime='K',
                                     variable='d_max',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fig=Fig_d,
                                     time_srs=time_srs)

    # loading five fractures from the simulation separated with equal time period
    Fr_list, properties = load_fractures(address="./Data/star",
                                         time_srs=np.linspace(0, 50, 5))
    # getting exact time of the loaded fractures
    time_srs = get_fracture_variable(Fr_list,
                                     'time')

    # plotting footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                    variable='mesh')
    Fig_FP = plot_fracture_list(Fr_list,
                                    variable='footprint',
                                    fig=Fig_FP)
    Fig_FP = plot_analytical_solution(regime='K',
                                     variable='footprint',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fig=Fig_FP,
                                     time_srs=time_srs)

    # plotting fracture in 3D
    Fig_3D = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='3D')
    Fig_3D = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='3D',
                                fig=Fig_3D)
    plot_prop = PlotProperties(alpha=0.3)
    Fig_3D = plot_fracture_list(Fr_list,
                                variable='surface',
                                projection='3D',
                                fig=Fig_3D,
                                plot_prop=plot_prop)


    plt.show(block=True)