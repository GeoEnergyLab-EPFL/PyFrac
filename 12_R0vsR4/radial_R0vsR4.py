# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab
import logging
import time
import matplotlib.pyplot as plt

# local imports
from mesh_obj.mesh import CartesianMesh
from fracture_obj.fracture_initialization import Geometry, get_survey_points, generate_footprint
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from utilities.utility import setup_logging_to_console
from linear_solvers.preconditioners.prec_back_subst_EHL import EHL_iLU_Prec
from linear_solvers.linear_iterative_solver import iteration_counter
from level_set.continuous_front_reconstruction import plot_two_fronts
# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
log = logging.getLogger('PyFrac.solve_width_pressure')

def w_radial_solution(x,y,Young,nu,p,R):
    rr = x**2 + y**2
    return 8. * (1 - nu * nu) * p * np.sqrt(R**2 - rr) / (np.pi * Young)


def get_solution(C, p, EltCrack):

    # prepare preconditioner
    Aprec = EHL_iLU_Prec(C._get9stencilC(EltCrack))
    counter = iteration_counter(log=log)  # to obtain the number of iteration and residual
    C._set_domain_IDX(EltCrack)
    C._set_codomain_IDX(EltCrack)

    sol_GMRES = bicgstab(C, p, M=Aprec, atol=10.e-14, tol=1.e-9, maxiter=1000, callback=counter)

    if sol_GMRES[1] > 0:
        print("EHL system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
    elif sol_GMRES[1] == 0:
        print(" --> GMRES EHL converged after " + str(counter.niter) + " iter. ")
    return sol_GMRES[0]

# -----------------------
# creating mesh
Mesh = CartesianMesh([-4,4], [-4,4], 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus

# initializing fracture
R = 3.
Fr_geometry = Geometry('radial', radius=R)
surv_cells, surv_dist, inner_cells = get_survey_points(Fr_geometry,Mesh)
EltChannel, EltTip, EltCrack, \
EltRibbon, ZeroVertex, CellStatus, \
l, alpha, FillF, sgndDist, \
Ffront, number_of_fronts, fronts_dictionary = generate_footprint(Mesh, surv_cells, inner_cells, surv_dist, 'LS_continousfront')

plot_two_fronts(Mesh, newfront=Ffront, oldfront=None , fig=None, grid=True, cells = EltCrack)


p = 1000000000*np.ones(len(EltCrack))
sol_R4 = np.zeros(Mesh.NumberOfElts)
sol_R0 = np.zeros(Mesh.NumberOfElts)
abs_err_R0 = []
abs_err_R4 = []
rel_err_R0 = []
rel_err_R4 = []

print("loading R4")
load_time = -time.time()
C_R4 = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision=np.float64, useHMATdot=False, nu=None, HMATparam = None, Kernel='R4')
load_time = load_time+time.time()
print(f"loaded R4 {load_time}")

print("solving R4")
sol_time = -time.time()
sol_R4[EltCrack] = get_solution(C_R4, p, EltCrack)
sol_time = sol_time+time.time()
print(f"solved R4 {sol_time}")

C_R0 = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision=np.float64, useHMATdot=False, nu=None, HMATparam = None, Kernel='R0')
sol_R0[EltCrack] = get_solution(C_R0, p, EltCrack)

true_solution = np.zeros(Mesh.NumberOfElts)

r = []
radim = []
radim_all = []
for el in EltCrack:
    x = Mesh.CenterCoor[el,0]
    y = Mesh.CenterCoor[el,1]
    r.append(np.sqrt(x*x + y*y))
    radim_all.append(r[-1] / R)
    if (R - np.sqrt((x**2 + y**2))) > 0.:
        true_solution[el] = w_radial_solution(x, y, youngs_mod, nu, p[0], R)
        radim.append(r[-1] / R)
        rel_err_R4.append(100. * np.abs(true_solution[el] - sol_R4[el]) / np.abs(true_solution[el]))
        rel_err_R0.append(100. * np.abs(true_solution[el] - sol_R0[el]) / np.abs(true_solution[el]))
        abs_err_R4.append(np.abs(true_solution[el] - sol_R4[el]))
        abs_err_R0.append(np.abs(true_solution[el] - sol_R0[el]))

# generate true solution
r_sol = np.arange(0,1,0.01)
w_sol = []
for rsol in r_sol:
    w_sol.append(w_radial_solution(0., rsol * R, youngs_mod, nu, p[0], R))

print("Statistics:\n")
print(f" Num. of elts in the crack: {Mesh.NumberOfElts}")
print("  - Absolute error")
print(f"    R0: {np.min(abs_err_R0)}")
print(f"    R4: {np.min(abs_err_R4)}")

print("  - Relative error [%]")
print(f"    R0: {np.max(rel_err_R0)}")
print(f"    R4: {np.max(rel_err_R4)}")

fig2 = plt.figure()
plt.suptitle('Fracture opening')
#plt.scatter(radim, true_solution[EltCrack], c='b', marker="+")
plt.scatter(r_sol, w_sol, c='b', marker="+")
plt.scatter(radim_all, sol_R0[EltCrack], c='r', marker="+")
plt.scatter(radim_all, sol_R4[EltCrack], c='g', marker="+")

fig3 = plt.figure()
plt.suptitle('Relative error')
plt.scatter(radim, rel_err_R0, c='r', marker="+")
plt.scatter(radim, rel_err_R4, c='g', marker="+")

fig3 = plt.figure()
plt.suptitle('Absolute error')
plt.scatter(radim, abs_err_R0, c='r', marker="+")
plt.scatter(radim, abs_err_R4, c='g', marker="+")

plt.show()