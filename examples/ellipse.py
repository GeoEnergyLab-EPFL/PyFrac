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
Mesh = CartesianMesh(8., 4., 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

# the function below will make the fracture propagate in the form of an ellipse (see Zia and Lecampion 2018)
def Kprime_func(alpha):
    K1c_1 = 1.e6                    # fracture toughness along x-axis
    K1c_2 = 1.5e6                   # fracture toughness along y-axis

    beta = np.arctan((K1c_1 / K1c_2)**2 * np.tan(alpha))
    return 4 * (2/np.pi)**0.5 * K1c_2 * (np.sin(beta)**2 + (K1c_1 / K1c_2)**4 * np.cos(beta)**2)**0.25

Solid = MaterialProperties(Mesh,
                           Eprime,
                           anisotropic_flag=True,
                           Kprime_func= Kprime_func)


# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties()

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 100               # the time at which the simulation stops
simulProp.plotFigure = False            # to disable plotting of figures while the simulation runs
simulProp.saveToDisk = True             # to enable saving the results (to hard disk)
simulProp.set_outFileAddress(".\\Data\\ellipse") # the disk address where the files are saved
simulProp.set_volumeControl(True)       # to set up the solver in volume control mode (inviscid fluid)
simulProp.set_tipAsymptote('K')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.outputTimePeriod = 1e-10      # save after every time step
simulProp.tolFractFront = 2.5e-3        # increase tolerance for the anisotropic case
simulProp.verbosity = 2


# initializing fracture
minor_axis = 2.0
init_param = ("E", "length", minor_axis)

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

# plotting footprint
plot_footprint(simulProp.get_outFileAddress(),
                        time_period=15,
                        analytical_sol='E')

# plotting length along minor axis
plot_radius(simulProp.get_outFileAddress(),
                        r_type='min',
                        analytical_sol='E')

# plotting length along major axis
plot_radius(simulProp.get_outFileAddress(),
                        r_type='max',
                        analytical_sol='E')

# plotting width and pressure at the injection point
plot_at_injection_point(simulProp.get_outFileAddress(),
                        plt_pressure=True,
                        analytical_sol='E')
plt.show()
