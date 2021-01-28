# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
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
Mesh = CartesianMesh(0.3, 0.3, 41, 41, symmetric=True)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=0)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e2               # the time at which the simulation stops
simulProp.set_tipAsymptote('K')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.set_volumeControl(True)       # use the inviscid fluid solver(toughness dominated), imposing volume balance
simulProp.set_outputFolder("./Data/K_radial_symmetric") # the disk address where the files are saved
simulProp.symmetric = True              # assume fracture geometry to be symmetric (only available for volume control)
simulProp.plotTSJump = 10               # plotting every 10 time steps

# initializing fracture
Fr_geometry = Geometry('radial', radius=0.15)
init_param = InitializationParameters(Fr_geometry, regime='K')

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
    Fr_list, properties = load_fractures(address="./Data/K_radial_symmetric")       # load all fractures
    time_srs = get_fracture_variable(Fr_list,                                       # list of times
                                     variable='time')

    plot_prop = PlotProperties()

    # plot fracture radius
    plot_prop.lineStyle = '.'
    plot_prop.graphScaling = 'loglog'
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop)
    # plot analytical radius
    Fig_R = plot_analytical_solution('K',
                                     'd_mean',
                                     Solid,
                                     Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)

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

    time_srs = np.geomspace(1, 1e5, 8)
    Fr_list, properties = load_fractures(address="./Data/K_radial_symmetric",
                                         time_srs=time_srs)
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # plot footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D')
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP)
    # plot analytical footprint
    Fig_FP = plot_analytical_solution('K',
                                      'footprint',
                                      Solid,
                                      Injection,
                                      fluid_prop=Fluid,
                                      time_srs=time_srs,
                                      projection='2D',
                                      fig=Fig_FP)


    # plot slice
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    Fig_WS = plot_fracture_list_slice(Fr_list,
                                      variable='w',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts)
    #plot slice analytical
    Fig_WS = plot_analytical_solution_slice('K',
                                            'w',
                                            Solid,
                                            Injection,
                                            fluid_prop=Fluid,
                                            fig=Fig_WS,
                                            time_srs=time_srs,
                                            point1=ext_pnts[0],
                                            point2=ext_pnts[1])


    plt.show(block=True)