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
from src.anisotropy import *

# creating mesh
Mesh = CartesianMesh(6., 3., 41, 41)

# solid properties

Cij = np.zeros((6, 6), dtype=float)

# isotropic
# epsilon = 0.001
# m = 10
# lam = 5
# Cij[0, 0] = (1 + 7 * epsilon) * (2 * m + lam)
# Cij[5, 5] = (1 + 3 * epsilon) * m
# Cij[0, 1] = Cij[0, 0] - 2 * Cij[5, 5]
# Cij[0, 2] = (1 + 5 * epsilon) * lam
# Cij[2, 2] = (1 + 9 * epsilon) * (2 * m + lam)
# Cij[3, 3] = (1 + epsilon) * m
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# slate
# Cij[0, 0] = 38
# Cij[5, 5] = 0.5*(38-0.35)
# Cij[0, 1] = 0.35
# Cij[0, 2] = 0.98
# Cij[2, 2] = 27
# Cij[3, 3] = 15
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# calov
# Cij[0, 0] = 20.50
# Cij[5, 5] = 0.5*(20.50-8.16)
# Cij[0, 1] = 8.16
# Cij[0, 2] = 4.87
# Cij[2, 2] = 13.11
# Cij[3, 3] = 5.22
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# Opa
# Cij[0, 0] = 57.65
# Cij[5, 5] = 0.5*(57.65-54.61)
# Cij[0, 1] = 54.61
# Cij[0, 2] = 38.7
# Cij[2, 2] = 28.8
# Cij[3, 3] = 0.9
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# SiltMud
# Cij[0, 0] = 50.7
# Cij[0, 1] = 21.24
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 20.82
# Cij[2, 2] = 36.15
# Cij[3, 3] = 10.99
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# KimClay
# Cij[0, 0] = 48.34
# Cij[0, 1] = 14.41
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 16.31
# Cij[2, 2] = 27.21
# Cij[3, 3] = 7.8
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# shale
# Cij[0, 0] = 46.674
# Cij[5, 5] = 16.652
# Cij[0, 1] = 13.369
# Cij[0, 2] = 12.008
# Cij[2, 2] = 23.803
# Cij[3, 3] = 8.7
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# mica
# Cij[0, 0] = 89.74
# Cij[0, 1] = 22.22
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 23.51
# Cij[2, 2] = 65.87
# Cij[3, 3] = 24
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# # mica
# Cij[0, 0] = 22.08
# Cij[0, 1] = 8.36
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 1.25
# Cij[2, 2] = 10.91
# Cij[3, 3] = 3.71
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# modelgam05
Cij[0, 0] = 47.5
Cij[0, 1] = 16.55
Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
Cij[0, 2] = 20.34
Cij[2, 2] = 28.86
Cij[3, 3] = 7.32
Cij[1, 1] = Cij[0, 0]
Cij[1, 0] = Cij[0, 1]
Cij[2, 0] = Cij[0, 2]
Cij[2, 1] = Cij[0, 2]
Cij[1, 2] = Cij[0, 2]
Cij[4, 4] = Cij[3, 3]
Cij=Cij*1e9

# # modelgam06
# Cij[0, 0] = 53.9
# Cij[0, 1] = 16.55
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 20.34
# Cij[2, 2] = 28.86
# Cij[3, 3] = 7.32
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# # Clay
# Cij[0, 0] = 44.9
# Cij[0, 1] = 21.7
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 18.1
# Cij[2, 2] = 24.2
# Cij[3, 3] = 3.7
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

# Calcamud
# Cij[0, 0] = 90.4
# Cij[0, 1] = 51.57
# Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
# Cij[0, 2] = 39.85
# Cij[2, 2] = 35.13
# Cij[3, 3] = 6.49
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9


# Cij[0, 0] = 25
# Cij[5, 5] = 7
# Cij[0, 1] = 11
# Cij[0, 2] = 10
# Cij[2, 2] = 19
# Cij[3, 3] = 5
# Cij[1, 1] = Cij[0, 0]
# Cij[1, 0] = Cij[0, 1]
# Cij[2, 0] = Cij[0, 2]
# Cij[2, 1] = Cij[0, 2]
# Cij[1, 2] = Cij[0, 2]
# Cij[4, 4] = Cij[3, 3]
# Cij=Cij*1e9

Eprime = TI_plain_strain_modulus(np.pi/2, Cij)# plain strain modulus

# the function below will make the fracture propagate in the form of an ellipse (see Zia and Lecampion 2018)
def K1c_func(alpha):

    K1c_3 = 2e6                    # fracture toughness along x-axis
    gamma = 1.5

    Cij = np.zeros((6, 6), dtype=float)
    Cij[0, 0] = 47.5
    Cij[0, 1] = 16.55
    Cij[5, 5] = 0.5 * (Cij[0, 0] - Cij[0, 1])
    Cij[0, 2] = 20.34
    Cij[2, 2] = 28.86
    Cij[3, 3] = 7.32
    Cij[1, 1] = Cij[0, 0]
    Cij[1, 0] = Cij[0, 1]
    Cij[2, 0] = Cij[0, 2]
    Cij[2, 1] = Cij[0, 2]
    Cij[1, 2] = Cij[0, 2]
    Cij[4, 4] = Cij[3, 3]
    Cij = Cij * 1e9

    Eprime_ratio = TI_plain_strain_modulus(alpha,Cij) / TI_plain_strain_modulus(np.pi/2, Cij)
    beta = np.arctan(np.tan(alpha) / gamma)

    return K1c_3 * Eprime_ratio * ((np.sin(beta))**2 + (np.cos(beta)/gamma)**2)**0.25


Solid = MaterialProperties(Mesh,
                           Eprime=Eprime,
                           anisotropic_K1c=True,
                           Toughness=K1c_func(np.pi/2),
                           K1c_func=K1c_func,
                           TI_elasticity=True,
                           Cij=Cij)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-5)




# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 200              # the time at which the simulation stops
# simulProp.set_volumeControl(True)       # to set up the solver in volume control mode (inviscid fluid)
# simulProp.set_tipAsymptote("K")     # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.outputTimePeriod = 1e-3      # save after every time step
simulProp.tolFractFront = 2e-3          # increase tolerance for the anisotropic case
simulProp.maxToughnessItr = 5           # set maximum iterations to 5 for faster simulation
simulProp.remeshFactor = 1.5            # the factor by which the mesh will be compressed.
simulProp.verbosity=2
simulProp.set_outputFolder(".\\data\\TIelasticityuniformKModelgam06")
simulProp.TI_KernelExecPath = 'C:\\Users\\Haseeb\\Documents\\GitHubPyFrac\\src_TI_Kernel\\'
# simulProp.plotFigure = True
# simulProp.set_volumeControl(True)

# simulProp.plotFigure=True
# simulProp.plotAnalytical=True
# simulProp.analyticalSol='E'
simulProp.frontAdvancing="implicit"
simulProp.tmStpPrefactor=0.4

# simulProp.set_outFileAddress(".\\Data\\ellipse") # the disk address where the files are saved
initRad = 1.5
surv_cells, inner_cells = get_circular_survey_cells(Mesh, initRad)
surv_cells, inner_cells = get_eliptical_survey_cells(Mesh, 3, 2)
surv_dist = np.zeros((surv_cells.size, ), dtype=np.float64)
# # get minimum distance from center of the survey cells
for i in range(0, surv_cells.size):
     surv_dist[i] = Distance_ellipse(3, 2, Mesh.CenterCoor[surv_cells[i], 0],
                                                    Mesh.CenterCoor[surv_cells[i], 1])

# initializing fracture
t, b, p, winit, v, actvElts = HF_analytical_sol('E_E',
                                            Mesh,
                                            Eprime,
                                            Q0,
                                            gamma=1.5,
                                            Cij=Cij,
                                            Kprime=(32 / np.pi) ** 0.5 * K1c_func(np.pi / 2),
                                            length=2)

C = load_TI_elasticity_matrix(Mesh, Solid, simulProp)

init_param = ('G',              # type of initialization
              surv_cells,       # the given survey cells
              inner_cells,      # the cell enclosed by the fracture
              surv_dist,        # the distance of the survey cells from the front
              winit,            # the given width
              p,                # the pressure (uniform in this case)
              C,                # the elasticity matrix
              None,             # the volume of the fracture
              0)                # the velocity of the propagating front (stationary in this case)


#  creating fracture object
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
                        simulProp,
                        C=C)

# run the simulation
# controller.run()

# loading simulation results
Fr_list, properties = load_fractures(address=".\\data\\TIelasticityuniformKModelgam06")      # load all fractures
time_srs = get_fracture_variable_whole_mesh(Fr_list,                            # list of times
                                            variable='time')
Fig_FP = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='2D')
Fig_FP = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='2D',
                            fig=Fig_FP)

plt.show()
