# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
import shutil
import os
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        print("File or dir {} has not been removed.".format(path))
        #raise ValueError("file {} is not a file or dir.".format(path))

def run_radial_vertex(my_front_reconstruction,my_front_advancement,my_vertex,my_param):
    # setting up the verbosity level of the log at console
    # setup_logging_to_console(verbosity_level='error')

    outputfolder = "./Temp_Data/" + my_vertex + "_radial_" + my_front_advancement + "_" + my_front_reconstruction
    remove(outputfolder)

    # creating mesh
    Mesh = CartesianMesh(my_param['Lx'], my_param['Ly'], my_param['Nx'], my_param['Ny'])

    # solid properties
    nu = my_param['nu']                 # Poisson's ratio
    youngs_mod = my_param['youngs_mod'] # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
    K_Ic = my_param['K_Ic']             # fracture toughness
    Cl = my_param['Cl']                 # Carter's leak off coefficient

    # material properties
    Solid = MaterialProperties(Mesh,Eprime,K_Ic,Carters_coef=Cl)

    # injection parameters
    Q0 = my_param['Q0']  # injection rate
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=my_param['viscosity'])

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = my_param['finalTime']         # the time at which the simulation stops
    simulProp.set_tipAsymptote(my_vertex)               # tip asymptote is evaluated with the viscosity dominated assumption
    simulProp.frontAdvancing = my_front_advancement     # to set explicit front tracking
    simulProp.plotFigure= False
    simulProp.saveTSJump, simulProp.plotTSJump = 5, 5   # save and plot after every five time steps
    simulProp.set_outputFolder(outputfolder)
    simulProp.projMethod = my_front_reconstruction
    simulProp.log2file = False

    # initialization parameters
    Fr_geometry = Geometry('radial', radius=my_param['initialR'])
    init_param = InitializationParameters(Fr_geometry, regime=my_vertex)

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
    exitcode = controller.run()
    return exitcode, outputfolder