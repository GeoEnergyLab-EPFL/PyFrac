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
import numpy as np
from src.CartesianMesh import CartesianMesh
from src.Fracture import Fracture
from src.Properties import *
from src.Controller import Controller
from src.PostProcess import animate_simulation_results


# creating mesh
Mesh = CartesianMesh(200, 200, 41, 41)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K_Ic = 1e6
sigma0 = np.full((Mesh.NumberOfElts,), 1e6, dtype=np.float64)
d_grain = 1e-5
def Kprime_function(alpha):
    return (32 / math.pi) ** 0.5 * (1e6 + 0.4e6 * np.sin(alpha))
Solid = MaterialProperties(Mesh, Eprime, K_Ic, SigmaO=sigma0, Kprim_func=Kprime_function, anisotropic_flag=True)

# injection parameters
Q0 = 0.001  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=False)

# simulation properties
req_sol_time = np.linspace(2300, 45000, 11)
simulProp = SimulationParameters(tip_asymptote="U",
                                 output_time_period=0.002,
                                 plot_figure=False,
                                 save_to_disk=True,
                                 out_file_folder=".\\Data\\anisotropic", # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=True,
                                 tmStp_prefactor=0.4,
                                 final_time=44000,
                                 analytical_sol="K",
                                 req_sol_at=req_sol_time)


# initializing fracture
initRad = 50 # initial radius of fracture

# creating fracture object
Fr = Fracture(Mesh,
              initRad,
              'radius',
              'K',
              Solid,
              Fluid,
              Injection,
              simulProp)


# create a Controller
controller = Controller(Fr, Solid, Fluid, Injection, simulProp)

# run the simulation
controller.run()

# plot results
animate_simulation_results(simulProp.outFileAddress, sol_time_series=simulProp.solTimeSeries)



