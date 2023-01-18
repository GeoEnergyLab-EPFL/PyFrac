# -*- coding: utf-8 -*-
"""
Created by Carlo Peruzzo.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters

run = True
plot = True
if run:
    # creating mesh
    Mesh = CartesianMesh(0.0025, 0.0025, 91, 91)

    # solid properties
    nu = 0.5                            # Poisson's ratio
    Eprime = 425000   # plain strain modulus
    youngs_mod = Eprime * (1 - nu**2)   # Young's modulus
    gammas= 3.6                         # J/m^2 surface energy
    K1c = np.sqrt(2*gammas*Eprime)         # fracture toughness (+/- 1)
    Cl = 0.                             # Carter's leak off coefficient

    # material properties
    Solid = MaterialProperties(Mesh,
                               Eprime,
                               K1c,
                               Carters_coef=Cl)

    # injection parameters
    Q0 = 23/1000/60/1000  # injection rate
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    viscosity = 1.13 # Pa.s
    Fluid = FluidProperties(viscosity=viscosity)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = 90                           # the time at which the simulation stops
    simulProp.saveTSJump, simulProp.plotTSJump = 1, 1   # save and plot after every 5 time steps
    simulProp.set_outputFolder("./Data/Exp_15_single_fracture")   # the disk address where the files are saved
    simulProp.frontAdvancing = 'implicit'               # setting up explicit front advancing
    simulProp.projMethod = 'LS_continousfront'
    simulProp.plotTSJump = 20
    simulProp.set_solTimeSeries(np.asarray([40.6,41.0]))

    # initializing fracture
    Fr_geometry = Geometry('radial',radius=0.0005)
    init_param = InitializationParameters(Fr_geometry)

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
if plot:
    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures("./Data/Exp_15_single_fracture")
    Solid, Fluid, Injection, simulProp = properties
    time_srs = get_fracture_variable(Fr_list, variable='time')                      # list of times

    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the line style to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop)
    # plot analytical radius
    Fig_R = plot_analytical_solution(regime='M',
                                     variable='d_mean',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)
    # plot analytical radius
    Fig_R = plot_analytical_solution(regime='K',
                                     variable='d_mean',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)
    plt.show(block=True)