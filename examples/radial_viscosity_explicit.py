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
Mesh = CartesianMesh(0.3, 0.3, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0.5e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 1e5               # the time at which the simulation stops
simulProp.set_tipAsymptote('M')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.frontAdvancing = 'explicit'   # to set explicit front tracking
simulProp.outputTimePeriod = 1e-4        # to save after every time step
simulProp.tmStpPrefactor = 0.5          # decrease the pre-factor due to explicit front tracking
# simulProp.set_outFileAddress(".\\Data\\radial") # the disk address where the files are saved

# initializing fracture
initRad = 0.1
init_param = ("M", "length", initRad)

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

plot_footprint(simulProp.get_outFileAddress(),         # the address where the results are stored
                        plot_at_times=10**np.linspace(-2,5,15),# the time series at which the solution is plotted
                        analytical_sol='M')            # analytical solution for reference
plt.show()
plot_radius(simulProp.get_outFileAddress(),
                        r_type='mean',
                        analytical_sol='M')
plt.show()
plot_at_injection_point(simulProp.get_outFileAddress(),
                        plt_pressure=False,
                        analytical_sol='M')
plt.show()
plot_profile(simulProp.get_outFileAddress(),
                        plot_at_times=10**np.linspace(-2,5,10),
                        analytical_sol='M')
plt.show()

