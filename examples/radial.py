# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# adding src folder to the path
import sys
if "win32" in sys.platform or "win64" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '..' + slash + 'src' in sys.path:
    sys.path.append('..' + slash + 'src')
if not '.' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

# imports
from src.Controller import *
from src.Fracture import *
from src.PostProcess import plot_simulation_results

# creating mesh
Mesh = CartesianMesh(30, 30, 41, 41)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K_Ic = 5e6
sigma0 = np.full((Mesh.NumberOfElts,), 1e6, dtype=np.float64)
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           SigmaO=sigma0)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
muPrime = 1.1e-3
Fluid = FluidProperties(muPrime, Mesh)

# simulation properties
simulProp = SimulationParameters(".\\Data\\radial")

# initializing fracture
initRad = 10.
init_data = ("K", "l", initRad)

# creating fracture object
Fr = Fracture(Mesh,
              init_data,
              Solid,
              Fluid,
              Injection,
              simulProp)

# create a Controller
controller = Controller(Fr, Solid, Fluid, Injection, simulProp)

# run the simulation
controller.run()

# plot results
plot_simulation_results('.\\Data\\radial', sol_t_srs=simulProp.solTimeSeries, analytical_sol='K')

