# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(0.15, 0.2, 41, 75)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e9                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0.5e6                        # set toughness to a very low value

def sigmaO_func(x, y):
    """ The function providing the confining stress variation with space"""

    if abs(y) > 0.05 and abs(y) < 0.07:
        sigma0 = 75.e6
    elif abs(y) < 0.05:
        sigma0 = 5.0e6
    else:
        sigma0 = 0.1e6

    if abs(x) > 0.1:
        sigma0 = 85.e6

    return sigma0

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           confining_stress_func=sigmaO_func)

# injection parameters
Q0 = 0.001
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 0.05              # the time at which the simulation stops
simulProp.bckColor = 'sigma0'           # the parameter according to which the background is color coded
simulProp.set_outputFolder('./Data/pinch_point')
simulProp.plotVar = ['footprint']

# initializing fracture
Fr_geometry = Geometry('radial', radius=0.048)
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

# loading results
Fr_list, properties = load_fractures(address='./Data/pinch_point')

# plotting fracture width slice for last six time steps
plt_prop = PlotProperties(line_style='.-')
Fig = plot_fracture_list_slice(Fr_list[-6:],
                                variable='w',
                                point1=[-0.09, -Fr_list[-1].mesh.Ly],
                                plot_cell_center=True,
                                orientation='vertical',
                                plot_prop=plt_prop)

plt.show(block=True)
