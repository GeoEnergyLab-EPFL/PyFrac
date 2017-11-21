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
Mesh = CartesianMesh(8, 4, 80, 80)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K1c_1 = 1.e6
K1c_2 = 1.732e6
sigma0 = np.full((Mesh.NumberOfElts,), 0, dtype=np.float64)
def Kprime_function(alpha):
    K1c_1 = 1.e6
    K1c_2 = 1.732e6

    # return 5.81e6 + (K1c_2-5.81e6) * np.sin(beta)

    # return 4 * (2/np.pi)**0.5 * K1c_1 + 2*alpha/np.pi * (K1c_2-K1c_1)

    beta = np.arctan((K1c_1 / K1c_2)**2 * np.tan(alpha))
    return 4 * (2/np.pi)**0.5 * K1c_2 * (np.sin(beta)**2 + (K1c_1 / K1c_2)**4 * np.cos(beta)**2)**(0.25)

    # const = np.cos(np.pi/4)+np.sin(np.pi/4)
    # a = (K1c_2-K1c_1)/(const-1.)
    # c = K1c_1-(K1c_2-K1c_1)/(const-1.)
    # K1c = (np.cos(alpha)+np.sin(alpha))*a + c

    # delta = 0.3
    # k = 4
    # K1c = K1c_2 * (1 + delta * np.cos(k * alpha))

    # k = 20
    # jump_at = np.pi/4. + np.pi/20.
    # K1c = K1c_1 + (K1c_2-K1c_1) * 1/(1+np.e**(-2*k*(alpha-jump_at)))

    return 4 * (2/np.pi)**0.5 * K1c
Solid = MaterialProperties(Mesh, Eprime, K1c_2, SigmaO=sigma0, anisotropic_flag=True, Kprime_func= Kprime_function, Toughness_perp=K1c_1)
# Solid = MaterialProperties(Mesh, Eprime, K1c_2, SigmaO=sigma0, anisotropic_flag=False)

# injection parameters
Q0 = 0.001  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=False)

# simulation properties
req_sol_time = np.linspace(250.,5400.,15)
simulProp = SimulationParameters(tip_asymptote="U",
                                 toleranceFractureFront=0.002,
                                 output_time_period=0.0000,
                                 plot_figure=False,
                                 save_to_disk=True,
                                 out_file_folder=".\\Data\\Ellipse3MtoK", # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=True,
                                 tmStp_prefactor=0.35,
                                 final_time=200000.,
                                 volume_control=False,
                                 analytical_sol="M",
                                 tol_toughness=0.002,
                                 max_toughnessItr=20)
                                 # req_sol_at=req_sol_time)


# # initializing fracture
# b = 0.8
# t, a, w, p = anisotropic_toughness_elliptical_solution(K1c_2*np.ones((Mesh.NumberOfElts,),), K1c_1, Eprime, Q0, Mesh, b)
# from src.FractureInitilization import get_circular_survey_cells
# # surv_cells, channel_cells = get_circular_survey_cells(Mesh, b)
# surv_cells, channel_cells = get_eliptical_survey_cells(Mesh, a, b)
# surv_cells_dist = np.zeros((surv_cells.size, ), dtype=np.float64)
# for i in range(0, len(surv_cells)):
#     surv_cells_dist[i] = Distance_ellipse(a, b, Mesh.CenterCoor[surv_cells[i], 0],
#                                                    Mesh.CenterCoor[surv_cells[i], 1])
# C = load_elasticity_matrix(Mesh, Eprime)
# v = 0.8
# init_data = (surv_cells, channel_cells, surv_cells_dist, w, p, C, None, v)

# R = 10.
# from src.FractureInitilization import get_circular_survey_cells
# surv_cells, channel_cells = get_circular_survey_cells(Mesh, R)
# surv_cells_dist = np.zeros((surv_cells.size, ), dtype=np.float64)
# for i in range(0, len(surv_cells)):
#     surv_cells_dist[i] = R - (Mesh.CenterCoor[surv_cells[i], 0]**2 + Mesh.CenterCoor[surv_cells[i], 1]**2)**0.5
# C = load_elasticity_matrix(Mesh, Eprime)
# v = 0.5
# init_data = (surv_cells, channel_cells, surv_cells_dist, None, None, C, Q0*t, v)

# #initialization data tuple
initRad = 1
init_data = (initRad, 'radius', 'M')
C = None

# creating fracture object
# Fr = Fracture(Mesh,
#               # 'general',
#               'analytical',
#               Solid,
#               Fluid,
#               Injection,
#               simulProp,
#               # general_init_data=init_data)
#               analyt_init_data=init_data)
# Fr_coarse = Fr.remesh(1.5, C, Solid, Fluid, Injection, simulProp)
Fr = ReadFracture('./Data/Ellipse3MtoK/file_971')
simulProp.lastSavedFile = 971


# fig1 = Fr.plot_fracture("complete", "footPrint", identify=np.hstack((Fr.EltRibbon, Fr.EltTip)), mat_Properties=Solid)
# fig1 = Fr.plot_fracture("complete", "footPrint")
# plt.show()
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


