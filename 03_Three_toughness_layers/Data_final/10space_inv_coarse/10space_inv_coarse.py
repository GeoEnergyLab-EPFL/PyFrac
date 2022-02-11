# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# imports
import numpy as np
import os
# local imports
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console
from utilities.postprocess_fracture import load_fractures
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

########## OPTIONS #########
run = True
run_dir =  "./"
restart= False
############################

if run:
    # creating mesh
    Mesh = CartesianMesh(0.4, 0.4, 135, 135)

    # solid properties
    nu = 0.4  # Poisson's ratio
    youngs_mod = 3.3e10  # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus

    def smoothing(K1, K2, r, delta, x):
        # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
        #### LINEAR - DIRAC DELTA ####
        x = np.abs(x)
        if  x < r-delta :
            return K1
        elif x >= r-delta and x<r :
            K12 = K1 + (K2-K1)*0.5
            a = (K12 - K1) / (delta)
            b = K1 - a * (r - delta)
            return a * x + b
        elif x >= r:
            return K2
        else:
            print("ERROR")


    def K1c_func(x,y, alpha):
        """ The function providing the toughness"""
        K_Ic = 1.e6  # fracture toughness
        r = 1.48
        delta = 0.001
        #return smoothing(K_Ic, 4.5*K_Ic, r, delta, x)
        return K_Ic

    def sigmaO_func(x, y):
        return 0

    Solid = MaterialProperties(Mesh,
                              Eprime,
                              K1c_func=K1c_func,
                              confining_stress_func = sigmaO_func,
                              minimum_width=0.)

    # injection parameters
    Q0 = 1.
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=1.)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = 1000005.12
    simulProp.tmStpPrefactor = 0.8
    simulProp.saveToDisk = True
    simulProp.tolFractFront = 0.0001
    simulProp.maxFrontItrs = 100
    simulProp.bckColor = 'K1c'
    simulProp.set_outputFolder(run_dir)
    simulProp.EHL_iter_lin_solve = True
    simulProp.plotVar = ['custom', 'regime','footprint']
    simulProp.frontAdvancing = 'implicit'
    simulProp.projMethod = 'LS_continousfront'
    simulProp.customPlotsOnTheFly = True
    simulProp.LHyst__ = []
    simulProp.tHyst__ = []

    # setting up mesh extension options
    # simulProp.meshExtensionAllDir = True
    # simulProp.set_mesh_extension_factor(1.5)
    # # simulProp.set_mesh_extension_direction(['vertical'])
    # simulProp.meshReductionPossible = False

    simulProp.simID = 'mtok' # do not use _

    simulProp.EHL_iter_lin_solve = True
    simulProp.gmres_Restart = 1000
    simulProp.gmres_maxiter = 1000


    # initialization parameters
    Fr_geometry = Geometry('radial')
    init_param = InitializationParameters(Fr_geometry, regime='M', time=0.0001)

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp)

    ################################################################################
    # the following lines are needed if you want to restart an existing simulation #
    ################################################################################
    if restart:
        from src.utilities.visualization import *
        Fr_list, properties = load_fractures(address=run_dir, step_size=100)
        Solid, Fluid, Injection, simulProp = properties
        Fr = Fr_list[-1]

        Solid = MaterialProperties(Fr.mesh,
                                  Eprime,
                                  K1c_func=K1c_func,
                                  confining_stress_func = sigmaO_func,
                                  confining_stress=0.,
                                  minimum_width=0.)

        Injection = InjectionProperties(Q0, Fr.mesh)
        simulProp.meshReductionPossible = True
        simulProp.meshExtensionAllDir = False
        simulProp.finalTime = 10.**30
    ############################################################################

    # create a Controller
    controller = Controller(Fr,
                            Solid,
                            Fluid,
                            Injection,
                            simulProp
                            )

    # run the simulation
    controller.run()

