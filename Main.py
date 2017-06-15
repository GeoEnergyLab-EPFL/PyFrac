# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 23 17:49:21 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# adding src folder to the path
import sys
if "win" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '.' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

# imports
import numpy as np
from src.CartesianMesh import *
from src.Fracture import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Elasticity import *
from src.Properties import *
from src.FractureFrontLoop import *

# creating mesh
Mesh = CartesianMesh(3, 3, 41, 41)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K_Ic = 1e6

sigma0 = np.full((Mesh.NumberOfElts,), 1e6, dtype=np.float64)
# high stressed layers
stressed_layer = np.where(abs(Mesh.CenterCoor[:,1]) > 1.5-Mesh.hx/2)[0]
sigma0[stressed_layer] = 7e6

d_grain = 1e-5
Solid = MaterialProperties(Eprime, K_Ic, 0., sigma0, d_grain, Mesh)

# injection parameters
Q0 = 0.001  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=False)

# simulation properties
simulProp = SimulationParameters(tip_asymptote="U",
                                 output_time_period=0.02,
                                 plot_figure=False,
                                 save_to_disk=True,
                                 out_file_folder=".\\Data\\Test", # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=True,
                                 tmStp_prefactor=0.4)


# initializing fracture
initRad = 0.8 # initial radius of fracture

# creating fracture object
Fr = Fracture(Mesh,
              initRad,
              'radius',
              'M',
              Solid,
              Fluid,
              Injection,
              simulProp)


# load elasticity matrix
C = load_elasticity_matrix(Mesh, Solid.Eprime)

simulProp.maxTimeSteps=20
# starting time stepping loop
i = 0
Fr_k = Fr

while (Fr.time < simulProp.FinalTime) and (i < simulProp.maxTimeSteps):

    i = i + 1

    TimeStep = simulProp.tmStpPrefactor * min(Fr.mesh.hx, Fr.mesh.hy) / np.max(Fr.v)
    status, Fr_k = advance_time_step(Fr_k, C, Solid, Fluid, simulProp, Injection, TimeStep)

    Fr = copy.deepcopy(Fr_k)


