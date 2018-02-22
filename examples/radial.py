# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcess import *

# creating mesh
Mesh = CartesianMesh(30, 30, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 5e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties()

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 1e9               # the time at which the simulation stops
simulProp.set_volumeControl(True)       # to set up the solver in volume control mode (inviscid fluid)
simulProp.set_tipAsymptote('K')         # the tip asymptote is evaluated with the toughness dominated assumption
# simulProp.set_outFileAddress(".\\Data\\radial") # the disk address where the files are saved



# initializing fracture
initRad = 10.
init_param = ("K", "length", initRad)

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

# plot results
plot_simulation_results(simulProp.get_outFileAddress(),         # the address where the results are stored
                        plot_at_times=simulProp.get_solTimeSeries(),# the time series at which the solution is plotted
                        analytical_sol='K')                     # analytical solution for reference
