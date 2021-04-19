# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
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
import math
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(0.65, 0.65, 69, 69)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic1 = 5.6e6                       # fracture toughness


def My_KIc_func(x, y):
    """ The function providing the fracture toughness"""
    return K_Ic1


# material properties
def sigmaO_func(x, y):

    # comment the following section if you would like to consider field of stress
    # caracterized by the presence of less heterogeneities.
    lx = 0.20
    ly = 0.20
    if math.trunc(abs(x) / lx) >0:
        if math.trunc(abs(x) / lx) %2 == 0:
            x = abs(x) - (math.trunc(abs(x) / lx)) * lx
        else :
            x = abs(x) - (math.trunc(abs(x) / lx) + 1) * lx

    if math.trunc(abs(y) / ly) > 0:
        if math.trunc(abs(y) / ly) %2 == 0:
            y = abs(y) - (math.trunc(abs(y) / ly)) * ly
        else :
            y = abs(y) - (math.trunc(abs(y) / ly)+1) * ly
    # comment up to here


    """ The function providing the confining stress"""
    R=0.05
    x1=0.
    y1=0.2

    if (abs(x)-x1)**2+(abs(y)-y1)**2 < R**2:
       return 60.0e6
    if (abs(x)-y1)**2+(abs(y)-x1)**2 < R**2:
       return 60.0e6
    else:
       return 5.0e6


Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c_func=My_KIc_func,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-8)


# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationProperties()
simulProp.bckColor = 'sigma0'
simulProp.finalTime = 0.28                          # the time at which the simulation stops
simulProp.outputTimePeriod = 1e-4                    # to save after every time step
simulProp.tmStpPrefactor = 0.5                       # decrease the pre-factor due to explicit front tracking
simulProp.set_outputFolder("./Data/stress_heterogeneities") # the disk address where the files are saved
simulProp.saveFluidFluxAsVector = True
simulProp.plotVar = ['ffvf']
simulProp.projMethod = 'LS_continousfront' # <--- mandatory use
simulProp.saveToDisk = True
simulProp.set_mesh_extension_factor(1.1)
simulProp.set_mesh_extension_direction(['all'])
simulProp.useBlockToeplizCompression = True

# initialization parameters
Fr_geometry = Geometry('radial', radius=0.12)
init_param = InitializationParameters(Fr_geometry, regime='M')


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

if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/stress_heterogeneities",step_size=10)                  # load all fractures
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

    plt.show(block=True)