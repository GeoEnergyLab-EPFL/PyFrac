# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# adding src folder to the path
import sys
if "win" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '..' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

# imports
import numpy as np
from src.CartesianMesh import *
from src.Fracture import *
from src.Properties import *
from src.Controller import *
from src.PostProcess import animate_simulation_results
from src.PostProcessRadial import plot_radial_data

# creating mesh
Mesh = CartesianMesh(.15,0.04,55,25)

# solid properties
nu = 0.4
Eprime = 3.3e9 / (1 - nu ** 2)
K_Ic = np.full((Mesh.NumberOfElts,), 0.5e6, dtype=np.float64)
sigma0 = np.full((Mesh.NumberOfElts,), 1e6, dtype=np.float64)

# high stressed layers
stressed_layer = np.where(abs(Mesh.CenterCoor[:,1]) > 0.025)[0]
sigma0[stressed_layer] = 5e6
K_Ic[stressed_layer] = 1.1e6
Solid = MaterialProperties(Mesh, Eprime, K_Ic, SigmaO=sigma0)

# injection parameters
Q0 = 0.0023*1.e-6  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(30, Mesh, turbulence=False)

# simulation properties
req_sol_time = np.linspace(10.,530.,11)
simulProp = SimulationParameters(tip_asymptote="U",
                                 plot_figure=False,
                                 save_to_disk=True,
                                 out_file_folder=".\\Data\\Confined2", # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=False,
                                 tmStp_prefactor=0.6,
                                 req_sol_at=req_sol_time)



#initialization data tuple
initRad = 0.015
init_data = (initRad, 'radius', 'M')
C = None

# creating fracture object
Fr = Fracture(Mesh,
              # 'general',
              'analytical',
              Solid,
              Fluid,
              Injection,
              simulProp,
              # general_init_data=init_data)
              analyt_init_data=init_data)

# create a Controller
controller = Controller(Fr, Solid, Fluid, Injection, simulProp, C=C)

# run the simulation
controller.run()

# plot fracture evolution
animate_simulation_results(simulProp.outFileAddress, sol_time_series=simulProp.solTimeSeries)
