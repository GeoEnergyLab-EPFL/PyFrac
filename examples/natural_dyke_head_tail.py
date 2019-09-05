# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from elasticity import load_isotropic_elasticity_matrix

# creating mesh
Mesh = CartesianMesh(500, 1500, 61, 141)

# solid properties
nu = 0.25                           # Poisson's ratio
youngs_mod = 20e9                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus


def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""
    density_high = 2700
    density_low = 2400
    #layer = 1000
    Ly = 2800
    #if y > layer:
        #return (Ly - y) * density_low * 9.8
    # only dependant on the depth
    #return (Ly - y) * density_high * 9.8 - (Ly - layer) * (density_high - density_low) * 9.8
    return (Ly - y) * density_high * 9.8

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           toughness=2.5e6,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-5)

# injection parameters
Q0 = np.asarray([[0.0,  50],
                [50,    0]])  # injection rate
Injection = InjectionProperties(Q0,
                                Mesh,
                                source_coordinates=[0, -1000])

# fluid properties
Fluid = FluidProperties(viscosity=50, density=2650)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 56000000000
#simulProp.frontAdvancing = 'implicit'  # the time at which the simulation stops
simulProp.set_outputFolder("./Data/neutral_buoyancy") # the disk address where the files are saved
simulProp.gravity = True                    # set up the gravity flag
simulProp.tolFractFront = 3e-3              # increase the tolerance for fracture front iteration
simulProp.plotTSJump = 5                    # plot every fourth time step
simulProp.saveTSJump = 2                    # save every second time step
simulProp.maxSolverItrs = 500               # increase the Picard iteration limit for the elastohydrodynamic solver
#simulProp.tmStpPrefactor = np.asarray([[0, 80000], [0.3, 0.1]]) # set up the time step prefactor
#simulProp.timeStepLimit = 500             # time step limit
simulProp.plotVar = ['w']              # plot fracture width and fracture front velocity
#simulProp.blockFigure = True
#simulProp.saveToDisk = False

# initializing a static fracture
C = load_isotropic_elasticity_matrix(Mesh, Solid.Eprime)
Fr_geometry = Geometry('radial', radius=80)
#init_param = InitializationParameters(Fr_geometry, regime='M')
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      net_pressure=6e6,
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

from visualization import *

# loading simulation results
time_srs = np.asarray([50, 10e5, 10e6, 302066618.7602621,902164089.99799, 50e8])
Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                     time_srs=time_srs)
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')

# plot footprint
Fig_FP = None
Fig_FP = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='2D',
                            mat_properties=Solid,
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

#plt.show(block=True)
#  set block=True and comment last 2 lines if you want to keep the window open
plt.show(block=False)
plt.pause(5)
plt.close()