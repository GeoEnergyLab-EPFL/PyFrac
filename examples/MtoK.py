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

# import numpy as np
# from src.CartesianMesh import *
# from src.Fracture import *
# from src.Properties import *
from src.Controller import *
from src.Utility import ReadFracture
from src.PostProcess import animate_simulation_results
from src.PostProcessRadial import plot_radial_data
from src.FractureInitilization import *



# creating mesh
Mesh = CartesianMesh(2., 2., 40, 40)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K1c_1 = 1.5e6
sigma0 = np.full((Mesh.NumberOfElts,), 0, dtype=np.float64)
Solid = MaterialProperties(Mesh, Eprime, K1c_1, SigmaO=sigma0, anisotropic_flag=False)
# Solid = MaterialProperties(Mesh, Eprime, K1c_2, SigmaO=sigma0, anisotropic_flag=False)

# injection parameters
Q0 = 0.01  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=False)

# simulation properties
req_sol_time = np.linspace(250.,5400.,15)
simulProp = SimulationParameters(tip_asymptote="U",
                                 output_time_period=0.01,
                                 plot_figure=True,
                                 save_to_disk=False,
                                 out_file_folder=".\\Data\\remesh_test", # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=True,
                                 tmStp_prefactor=0.5,
                                 final_time=10000000,
                                 volume_control=False,
                                 analytical_sol="M",
                                 tol_toughness=0.002,
                                 max_toughnessItr=20)
                                 # req_sol_at=req_sol_time)


#initialization data tuple
initRad = 0.5
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

# Fr = ReadFracture('.\\Data\\remesh_test\\file_196')
# simulProp.lastSavedFile = 196

# create a Controller
controller = Controller(Fr, Solid, Fluid, Injection, simulProp, C=C)

# run the simulation
controller.run()

# plot results
animate_simulation_results(simulProp.outFileAddress, sol_time_series=simulProp.solTimeSeries)
# fig_wdth, fig_radius, fig_pressure = plot_radial_data(simulProp.outFileAddress,
#                                                         regime="M",
#                                                         loglog=True,
#                                                         plot_w_prfl=True,
#                                                         plot_p_prfl=True)


