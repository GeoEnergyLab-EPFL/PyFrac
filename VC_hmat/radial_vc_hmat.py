# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
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
import math
from utility import setup_logging_to_console
from Hdot import Hdot_3DR0opening
from Hdot import gmres_counter

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
run = False
if run:
    # creating mesh
    #Mesh = CartesianMesh(0.018, 0.018, 81, 81)
    Mesh = CartesianMesh(    34, 34, 81, 81)

    # solid properties
    # nu = 0.48                               # Poisson's ratio
    # youngs_mod = 125000                     # Young's modulus
    # Eprime = youngs_mod / (1 - nu ** 2)     # plain strain modulus
    # K_Ic = np.sqrt(2*4.4*Eprime)            # fracture toughness

    nu = 0.4                            # Poisson's ratio
    youngs_mod = 3.3e10                 # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
    K_Ic = 0.5e6                          # fracture toughness


    properties = [youngs_mod, nu]

    # def K1c_func(x, y):
    #     """ The function providing the toughness"""
    #     if (np.floor(abs(y)) % 5) > 2 and abs(y) > 2.1:
    #         return 0.9e6
    #     else:
    #         return 0.6e6

    # material properties
    # def sigmaO_func(x, y):
    #     return 0
        # # comment the following section if you would like to consider field of stress
        # # caracterized by the presence of less heterogeneities.
        # lx = 0.20
        # ly = 0.20
        # if math.trunc(abs(x) / lx) >0:
        #     if math.trunc(abs(x) / lx) %2 == 0:
        #         x = abs(x) - (math.trunc(abs(x) / lx)) * lx
        #     else :
        #         x = abs(x) - (math.trunc(abs(x) / lx) + 1) * lx
        #
        # if math.trunc(abs(y) / ly) > 0:
        #     if math.trunc(abs(y) / ly) %2 == 0:
        #         y = abs(y) - (math.trunc(abs(y) / ly)) * ly
        #     else :
        #         y = abs(y) - (math.trunc(abs(y) / ly)+1) * ly
        # # comment up to here
        #
        #
        # """ The function providing the confining stress"""
        # R=0.05
        # x1=0.
        # y1=0.2
        #
        # if (abs(x)-x1)**2+(abs(y)-y1)**2 < R**2:
        #    return 60.0e6
        # if (abs(x)-y1)**2+(abs(y)-x1)**2 < R**2:
        #    return 60.0e6
        # else:
        #    return 5.0e6


    # Solid = MaterialProperties(Mesh,
    #                            Eprime,
    #                            K1c_func=K1c_func,
    #                            confining_stress=0.,
    #                            minimum_width=0.)
    Solid = MaterialProperties(Mesh,
                               Eprime,
                               toughness= K_Ic ,
                               confining_stress=0.,
                               minimum_width=0.)


    # set the Hmatrix for elasticity
    C = Hdot_3DR0opening()
    max_leaf_size = 100
    eta = 10
    eps_aca = 0.001
    data = [max_leaf_size, eta, eps_aca, properties, Mesh.VertexCoor, Mesh.Connectivity, Mesh.hx, Mesh.hy]
    C.set(data)


    # injection parameters
    #Q0 =  20/1000/60/1000  # injection rate
    Q0 = 0.001
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=1.1e-3)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.set_volumeControl(True)
    simulProp.volumeControlHMAT = True

    simulProp.bckColor = 'K1c'
    simulProp.finalTime =1e5                          # the time at which the simulation stops
    simulProp.tmStpPrefactor = 0.8                       # decrease the pre-factor due to explicit front tracking
    simulProp.set_outputFolder("./Data/radial_VC_hmat") # the disk address where the files are saved
    simulProp.set_tipAsymptote('K')  # the tip asymptote is evaluated with the toughness dominated assumption
    #simulProp.plotVar = ['pf','w']
    simulProp.plotVar = ['footprint','pf']
    simulProp.frontAdvancing = 'implicit'
    simulProp.projMethod = 'LS_continousfront' # <--- mandatory use
    simulProp.saveToDisk = True

    #simulProp.set_mesh_extension_factor(1.1)
    #simulProp.set_mesh_extension_direction(['all'])

    # initialization parameters
    #Fr_geometry = Geometry('radial', radius=0.002)
    Fr_geometry = Geometry('radial', radius=5.)

    if not simulProp.volumeControlHMAT:
        from elasticity import load_isotropic_elasticity_matrix
        C = load_isotropic_elasticity_matrix(Mesh, Eprime)

    init_param = InitializationParameters(Fr_geometry, regime='K')
    # init_param = InitializationParameters(Fr_geometry, time=1.021,
    #                                       regime='static',
    #                                       net_pressure=18000,
    #                                      elasticity_matrix=C,
    #                                       )


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
                            simulProp,
                            C = C)

    # run the simulation
    controller.run()

####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/radial_VC_hmat",step_size=1)                  # load all fractures
    time_srs = get_fracture_variable(Fr_list, variable='time')                                                 # list of times
    Solid, Fluid, Injection, simulProp = properties


    # plot fracture radius
    plot_prop = PlotProperties()
    Fig_R = plot_fracture_list(Fr_list,
                               variable='footprint',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list(Fr_list,
                               fig=Fig_R,
                               variable='mesh',
                               mat_properties=properties[0],
                               backGround_param='sigma0',
                               plot_prop=plot_prop)


    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the linestyle to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop)

    # plot analytical radius
    Fig_R = plot_analytical_solution(regime='K',
                                     variable='d_mean',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)

    # Fr_list, properties = load_fractures(address="./Data/LAB25",step_size=1)                  # load all fractures
    # time_srs = get_fracture_variable(Fr_list, variable='time')                                                 # list of times
    # Solid, Fluid, Injection, simulProp = properties
    # plot_prop.lineColor = (1.,0.5,0.)
    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='d_mean',
    #                            plot_prop=plot_prop,
    #                            fig=Fig_R)
    plt.show(block=True)