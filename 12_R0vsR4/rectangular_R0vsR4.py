# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# external imports
import time
import json

# local imports
from common_rect_and_radial_tests import *
from level_set.continuous_front_reconstruction import plot_two_fronts
from utilities.postprocess_fracture import append_to_json_file
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz


# ----------------------------------------------
run = True
file_name = "results_rectangular.json"

if run:
    # deciding the aspect ratio (if as > 1 -> Ly > Lx)
    # - it is Ly/Lx
    # number of mesh refinements
    #   - along x and y

    sim_info = {"aspect ratio": 20, "n. of refinements x": 5, "n. of refinements y": 5}

    sim_info["max. n. of refinements"] = np.maximum(sim_info["n. of refinements x"], sim_info["n. of refinements y"])

    # the coarsest mesh (the y direction will be a function of x)
    sim_info["nx min"] = 7

    # set the domain size (the y direction will be a function of x)
    sim_info["domain x"] = [-100, 100]

    # solid properties
    sim_info["nu"] = 0.4                                                    # Poisson's ratio
    sim_info["youngs mod"] = 3.3e4                                          # Young's modulus
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
               "n. of Elts" : [],
               "nu": sim_info["nu"],  # Poisson's ratio
               "youngs mod": sim_info["youngs mod"],
               "H" : sim_info["domain x"][1] - sim_info["domain x"][0],
               "p": sim_info["p"],
               }

    for refinement_ID in np.arange(1, maxref, 1):


        print(f"Step {refinement_ID} of {maxref} :")
        st = 0

        # creating mesh & plotting
        Mesh = get_mesh(sim_info, refinement_ID)
        EltCrack = np.arange(Mesh.NumberOfElts)

        if refinement_ID == 1 or  refinement_ID == maxref -1:
           plot_two_fronts(Mesh, newfront=None, oldfront=None , fig=None, grid=True, cells = EltCrack, my_marker = " ")

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
        sol_R0 = get_solution(C_R0, p, EltCrack)
        dummy = dummy + time.time()
        print(f"     --> done in {dummy}")

        # some plots
        # rel_err_num = 100 * np.abs(sol_R0 - sol_R4) / sol_R4
        # plot_as_matrix(rel_err_num, mesh=Mesh) # 2D plot
        # plot_3d_scatter(sol_R4, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1]) # 3D plot

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
    append_to_json_file(file_name, [content], action, delete_existing_filename=True)

if not run:
    with open(file_name, "r+") as json_file:
        results = json.load(json_file)  # get the data

print("Plotting results")

# w max
fig1 = plt.figure()
plt.suptitle('Rectangular crack test')
plt.plot(results["n. of Elts"], results["max w R0"], c='r', marker="+")
plt.plot(results["n. of Elts"], results["max w R4"], c='b', marker="+")

y_ana = wmax_plane_strain_solution(results["youngs mod"], results["nu"],results["p"], results["H"])
y_ana = np.full(2, y_ana)
plt.plot([results["n. of Elts"][0],results["n. of Elts"][-1]], y_ana, c='black', marker=" ")
plt.xlabel('# of DOF in the crack')
plt.ylabel('w max')
plt.legend(('R0 - tip corr not needed', 'R4 - tip corr not needed'),loc='lower left', shadow=True)
plt.xscale('log')
plt.yscale('log')

# volume
fig1 = plt.figure()
plt.suptitle('Rectangular crack test')
plt.plot(results["n. of Elts"], results["frac volume R0"], c='r', marker="+")
plt.plot(results["n. of Elts"], results["frac volume R4"], c='b', marker="+")
plt.xlabel('# of DOF in the crack')
plt.ylabel('frac volume')
plt.legend(('R0 - tip corr not needed', 'R4 - tip corr not needed'), loc='lower left', shadow=True)
plt.xscale('log')
plt.yscale('log')

# rel err w max
fig1 = plt.figure()
plt.suptitle('Rectangular crack test')

y_ana = wmax_plane_strain_solution(results["youngs mod"], results["nu"],results["p"], results["H"])
# plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R0"]) - y_ana)/y_ana, c='r', marker="+")
# plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R4"]) - y_ana)/y_ana, c='b', marker="+")
plt.plot(results["nx"], 100 * np.abs(np.asarray(results["max w R0"]) - y_ana)/y_ana, c='r', marker="+")
plt.plot(results["nx"], 100 * np.abs(np.asarray(results["max w R4"]) - y_ana)/y_ana, c='b', marker="+")
plt.tick_params(labeltop=True, labelright=True)
plt.grid(True, which="both", ls="-")

plt.xlabel('# of DOF in the crack')
plt.ylabel('rel. err. w max [%]')
plt.legend(('R0 - tip corr not needed', 'R4 - tip corr not needed'), loc='lower left', shadow=True)
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
