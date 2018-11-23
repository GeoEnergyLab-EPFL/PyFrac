# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(15, 4, 61, 35)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1e5                          # fracture toughness of the material

def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if abs(y) > 3:
        return 7.5e6
    else:
        return 1e6

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           SigmaO_func=sigmaO_func)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 27.           # the time at which the simulation stops
simulProp.outputEveryTS = 1         # the fracture file will be saved after every time step
simulProp.bckColor = 'sigma0'       # setting the parameter according to which the mesh is color coded
simulProp.set_outputFolder(".\\Data\\confined_propagation")
simulProp.plotFigure = True         # the fracture footprint will be plotted during the simulation

# initializing fracture
initRad = 2.8
init_param = ('M', "length", initRad)

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

# loading simulation results
Fr_list, properties = load_fractures(address=".\\Data\\confined_propagation",
                                     time_srs=np.linspace(1, 27, 10))

# animate results
animate_simulation_results(Fr_list,
                           variable='footprint',
                           projection='2D',
                           backGround_param='sigma0',
                           mat_properties=Solid)


# loading simulation results
Fr_list, properties = load_fractures(address=".\\Data\\confined_propagation",
                                     time_srs=np.linspace(1, 27, 4))

#plotting in 3D
Fig_Fr = plot_fracture_list(Fr_list,            #plotting mesh
                            variable='mesh',
                            projection='3D',
                            backGround_param='sigma0',
                            mat_properties=Solid)

Fig_Fr = plot_fracture_list(Fr_list,            #plotting footprint
                            variable='footprint',
                            projection='3D',
                            fig=Fig_Fr)

plot_prop = PlotProperties(alpha=0.3)           #plotting width
Fig_Fr = plot_fracture_list(Fr_list,
                            variable='width',
                            projection='3D',
                            fig=Fig_Fr,
                            plot_prop=plot_prop)

plt.show()
