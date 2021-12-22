# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy.sparse.linalg import bicgstab
import logging
import time
import matplotlib.pyplot as plt
import json

# local imports
from utilities.postprocess_fracture import append_to_json_file
from utilities.utility import plot_as_matrix
from mesh_obj.mesh import CartesianMesh
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from utilities.utility import setup_logging_to_console
from linear_solvers.preconditioners.prec_back_subst_EHL import EHL_iLU_Prec
from linear_solvers.linear_iterative_solver import iteration_counter
from level_set.continuous_front_reconstruction import plot_two_fronts

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
log = logging.getLogger('PyFrac.solve_width_pressure')


def plot_3d_scatter(zdata, xdata, ydata, zlabel = 'z', xlabel = 'x', ylabel = 'y'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel);
    return fig

# ----------------------------------------------
def get_solution(C, p, EltCrack, x0=None):
    """

    :param C: elasticity matrix
    :param p: pressure
    :param EltCrack: IDs of the elements in the crack
    :return: solution x of the system C.x = p
    """

    # prepare preconditioner
    Aprec = EHL_iLU_Prec(C._get9stencilC(EltCrack))
    counter = iteration_counter(log=log)  # to obtain the number of iteration and residual
    C._set_domain_IDX(EltCrack)
    C._set_codomain_IDX(EltCrack)

    sol_ = bicgstab(C, p, x0=x0, M=Aprec, atol=10.e-14, tol=1.e-9, maxiter=1000, callback=counter)

    if sol_[1] > 0:
        print("     --> iterative solver did NOT converge after " + str(sol_[1]) + " iterations!")
    elif sol_[1] == 0:
        print("     --> iterative solver converged after " + str(counter.niter) + " iter. ")
    return sol_[0]


# ----------------------------------------------
def get_mesh(sim_info, refinement_ID):

    ar = sim_info["aspect ratio"]
    lx = sim_info["domain x"][1] - sim_info["domain x"][0]
    ly = lx * ar
    d_x = sim_info["domain x"]
    d_y = [-ly / 2., ly / 2.]

    # define nx
    if refinement_ID < sim_info["n. of refinements x"]:
        nx = refinement_ID * sim_info["nx min"]
    else:
        nx = sim_info["nx min"]

    # define ny
    if refinement_ID < sim_info["n. of refinements y"]:
        ny = refinement_ID * sim_info["nx min"] * ar
    else:
        ny = sim_info["nx min"] * ar

    # define ny should be an odd number
    if ny % 2 == 0 : ny = ny + 1
    if nx % 2 == 0 : nx = nx + 1

    return CartesianMesh(d_x, d_y, nx, ny)


# ----------------------------------------------
run = False
file_name = "results_rectangular.json"

if run:
    sim_info = {}

    # deciding the aspect ratio
    # - it is Ly/Lx
    sim_info["aspect ratio"] = 2

    # number of mesh refinements
    #   - along x and y
    sim_info["n. of refinements x"] = 10
    sim_info["n. of refinements y"] = 10
    sim_info["max. n. of refinements"] = np.maximum(sim_info["n. of refinements x"], sim_info["n. of refinements y"])

    # the coarsest mesh (the y direction will be a function of x)
    sim_info["nx min"] = 7

    # set the domain size (the y direction will be a function of x)
    sim_info["domain x"] = [-2,2]

    # solid properties
    sim_info["nu"] = 0.4                                                    # Poisson's ratio
    sim_info["youngs mod"] = 3.3e10                                         # Young's modulus
    sim_info["Eprime"] = sim_info["youngs mod"] / (1 - sim_info["nu"]**2)   # plain strain modulus

    # uniform load
    sim_info["p"] =1.10**12

    maxref = sim_info["max. n. of refinements"]

    results = {"nx" : [],
               "ny" : [],
               "Lx" : [],
               "Ly" : [],
               "max w R0" : [],
               "max w R4" : [],
               "frac volume R0": [],
               "frac volume R4": [],
               "n. of Elts" : []}

    for refinement_ID in np.arange(1, maxref, 1):


        print(f"Step {refinement_ID} of {maxref} :")
        st = 0

        # creating mesh & plotting
        Mesh = get_mesh(sim_info, refinement_ID)
        EltCrack = np.arange(Mesh.NumberOfElts)
        #plot_two_fronts(Mesh, newfront=None, oldfront=None , fig=None, grid=True, cells = EltCrack, my_marker = " ")
        results["n. of Elts"].append(int(len(EltCrack)))
        results["nx"].append(int(Mesh.nx))
        results["ny"].append(int(Mesh.ny))
        results["Lx"].append(int(Mesh.Lx))
        results["Ly"].append(int(Mesh.Ly))
        print(f"     --> DoF {len(EltCrack)}")

        # setting the load
        p = np.full(len(EltCrack), sim_info["p"])

        st = st + 1
        print(f" {st}) loading R_4 matrix")
        dummy = - time.time()
        C_R4 = load_isotropic_elasticity_matrix_toepliz(Mesh, sim_info["Eprime"], Kernel='R4')
        dummy = dummy + time.time()
        print(f"     --> done in {dummy}")

        st = st + 1
        print(f" {st}) solving R_4 matrix")
        dummy = - time.time()
        sol_R4 = get_solution(C_R4, p, EltCrack)
        dummy = dummy + time.time()
        print(f"     --> done in {dummy}")

        st = st + 1
        print(f" {st}) loading R_0 matrix")
        dummy = - time.time()
        C_R0 = load_isotropic_elasticity_matrix_toepliz(Mesh, sim_info["Eprime"], Kernel='R0')
        dummy = dummy + time.time()
        print(f"     --> done in {dummy}")

        st = st + 1
        print(f" {st}) solving R_0 matrix")
        dummy = - time.time()
        sol_R0 = get_solution(C_R0, p, EltCrack, x0=sol_R4)
        dummy = dummy + time.time()
        print(f"     --> done in {dummy}")

        # some plots
        #rel_err_num = 100 * np.abs(sol_R0 - sol_R4) / sol_R4
        #plot_as_matrix(rel_err_num, mesh=Mesh) # 2D plot
        #plot_3d_scatter(sol_R4, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1]) # 3D plot

        st = st + 1
        print(f" {st}) saving stats.")
        results["max w R0"].append(sol_R0.max())
        results["max w R4"].append(sol_R4.max())
        results["frac volume R0"].append(np.sum(sol_R0) * Mesh.hx * Mesh.hy)
        results["frac volume R4"].append(np.sum(sol_R4) * Mesh.hx * Mesh.hy)
        print(" ------------------- \n")

    print("Saving to file")
    content = results
    action = 'dump_this_dictionary'
    append_to_json_file(file_name, content, action, delete_existing_filename=True)

if not run:
    with open(file_name, "r+") as json_file:
        results = json.load(json_file)  # get the data

print("Plotting results")

# w max
fig1 = plt.figure()
plt.suptitle('w_max VS DOF')
plt.plot(results["n. of Elts"], results["max w R0"], c='r', marker="+")
plt.plot(results["n. of Elts"], results["max w R4"], c='b', marker="+")
plt.xlabel('# of DOF')
plt.ylabel('w max')
plt.legend(('R0', 'R4'),loc='upper center', shadow=True)
plt.xscale('log')
plt.yscale('log')

# volume
fig1 = plt.figure()
plt.suptitle('fractue volume VS DOF')
plt.plot(results["n. of Elts"], results["frac volume R0"], c='r', marker="+")
plt.plot(results["n. of Elts"], results["frac volume R4"], c='b', marker="+")
plt.xlabel('# of DOF')
plt.ylabel('frac volume')
plt.legend(('R0', 'R4'),loc='upper center', shadow=True)
plt.xscale('log')
plt.yscale('log')

print(" <<<< DONE >>>>")




#
# rel_err_R4_R0 = []
# abs_err_R4_R0 = []
# abs_err_cell_names = []
# rel_err_cell_names = []
# r = []
# radim_rel_err = []
# radim_abs_err = []
#
# for el in np.arange(Mesh.NumberOfElts):
#     x = Mesh.CenterCoor[el,0]
#     y = Mesh.CenterCoor[el,1]
#     rel_err_R4_R0.append(100. * np.abs(sol_R0[el] - sol_R4[el]) / np.abs(sol_R4[el]))
#     abs_err_R4_R0.append(np.abs(sol_R0[el] - sol_R4[el]))
#     radim_rel_err.append(x)
#     radim_abs_err.append(x)
#     rel_err_cell_names.append(el)
#     abs_err_cell_names.append(el)
#
#
# print("Statistics:\n")
# print(f" Num. of elts in the crack: {Mesh.NumberOfElts}")
# print("  - Absolute error")
# print(f"    R0 vs R4: {np.min(rel_err_R4_R0)}")
#
# print("  - Relative error [%]")
# print(f"    R0 vs R4: {np.max(abs_err_R4_R0)}")
#
# fig2 = plt.figure()
# plt.suptitle('Fracture opening')
# plt.scatter(radim_abs_err, sol_R0[EltCrack], c='r', marker="+")
# plt.scatter(radim_abs_err, sol_R4[EltCrack], c='g', marker="+")
#
# fig3 = plt.figure()
# plt.suptitle('Relative error')
# plt.scatter(radim_rel_err, rel_err_R4_R0, c='r', marker="+")
#
# fig3 = plt.figure()
# plt.suptitle('Absolute error')
# plt.scatter(radim_abs_err, abs_err_R4_R0, c='r', marker="+")
#
# from utilities.utility import plot_as_matrix
# A=np.full(Mesh.NumberOfElts,np.NaN)
# A[rel_err_cell_names] = abs_err_R4_R0
# plot_as_matrix(A,mesh=Mesh)
# plt.suptitle('abs. err. R4 R0')
#
# plt.show()
