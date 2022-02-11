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
from functools import reduce

# local imports
import matplotlib.pyplot as plt

from common_rect_and_radial_tests import *
from level_set.continuous_front_reconstruction import plot_two_fronts
from utilities.postprocess_fracture import append_to_json_file
from utilities.utility import plot_as_matrix
from utilities.visualization import EPFLcolor
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from src.tip.tip_inversion import StressIntensityFactor, StressIntensityFactorFromVolume

# ----------------------------------------------
# ----------------------------------------------
# RUN
# ----------------------------------------------
# ----------------------------------------------


run = True


if run:

    ## --- Setting the different parameters --- ##
    Nx_set = [7, 15, 25, 50, 75, 100, 150, 200] # Number of elements in the cross section
    aspect_ratio_set = [2, 5, 10, 25, 50, 75, 100, 200] # Aspect ratios
    leaf_size_set = [5, 10, 100, 500] # leaf size
    eta_set = [3, 5, 10] # distance threshold
    epsilon_set = [1e-3, 1e-4, 1e-5] # limit on final accuracy

    # solid properties
    sim_info = {"nu": 0.4 , "youngs mod": 3.3e4} # Poissons'ratio and young's modulus
    sim_info["Eprime"] = sim_info["youngs mod"] / (1 - sim_info["nu"] ** 2)  # plain strain modulus

    # set the domain size
    sim_info["domain x"] = [-100, 100]

    # uniform opening
    sim_info["wset"] = 1

    for Nx_i in Nx_set:
        for ar_i in aspect_ratio_set:
            new_ar = True
            ## --- Define the Infos on the simulations --- ##
            # aspect ratio and number of elements
            sim_info["aspect ratio"] = ar_i
            sim_info["Nx"] = Nx_i
            sim_info["Ny"] = Nx_i * ar_i

            ## --- creating the mesh --- ##
            Mesh = get_mesh_Hmat(sim_info)
            EltTip = np.asarray(Mesh.get_Boundarylist())
            EltRibbon = Mesh.get_Frontlist()
            EltCrack = np.arange(Mesh.NumberOfElts)
            FillF = np.ones(len(EltTip))
            centralElts = np.where(np.abs(Mesh.CenterCoor[:, 1]) < 0.9 * Mesh.hy)[0]
            H = (sim_info["domain x"][1] - sim_info["domain x"][0]) + Mesh.hx
            sim_info["H"] = H

            # setting the constant opening DD
            w = np.full(len(EltCrack), sim_info["wset"])

            ## --- prepare the results array --- ##
            results = {"aspect ratio": [],
                       "leaf size": [],
                       "eta": [],
                       "epsilon": [],
                       "nx": [],
                       "ny": [],
                       "Lx": [],
                       "Ly": [],
                       "max p": [],
                       "n. of Elts": [],
                       "nu": sim_info["nu"],  # Poisson's ratio
                       "youngs mod": sim_info["youngs mod"],
                       # "H": [],
                       "w": sim_info["wset"],
                       # "x_center_section": [],
                       # "p": [],
                       # "allp": [],
                       "t_Hmat": [],
                       "t_Dot": [],
                       "max rel_err": [],
                       "rel_err": [],
                       "compression ratio": []
                       }

            for ls_i in leaf_size_set:
                for eta_i in eta_set:
                    for epsilon_i in epsilon_set:
                        ## --- Append first results/sim parameters --- ##
                        # results["x_center_section"].append(Mesh.CenterCoor[centralElts, 0].tolist())
                        # results["H"].append(H)
                        results["n. of Elts"].append(int(len(EltCrack)))
                        results["nx"].append(int(Mesh.nx))
                        results["ny"].append(int(Mesh.ny))
                        results["Lx"].append(int(Mesh.Lx))
                        results["Ly"].append(int(Mesh.Ly))
                        results["aspect ratio"].append(int(Mesh.Ly / Mesh.Lx))
                        results["leaf size"].append(int(ls_i))
                        results["eta"].append(int(eta_i))
                        results["epsilon"].append(int(epsilon_i))

                        ## --- Print all the parameters --- ##
                        print(f"Nx: {int(Mesh.nx)}")
                        print(f"Aspect ratio: {int(Mesh.Ly/Mesh.Lx)}")
                        print(f"leaf size: {ls_i}")
                        print(f"eta: {eta_i}")
                        print(f"epsilon: {epsilon_i}")
                        print(f"DoF: {len(EltCrack)}")

                        ## --- prepare the HMat parameters --- ##
                        HMATparam = [ls_i, eta_i, epsilon_i]

                        ## --- load HMat --- ##
                        print(f" Loading the HMat")
                        tHmatLoad = - time.time()
                        C = load_isotropic_elasticity_matrix_toepliz(Mesh, sim_info["Eprime"], C_precision=np.float64,
                                                                     useHMATdot=True, nu=sim_info["nu"],
                                                                     HMATparam=HMATparam)
                        tHmatLoad = tHmatLoad + time.time()
                        print(f"     --> HMat loaded in {tHmatLoad} [s]")
                        results["t_Hmat"].append(tHmatLoad)

                        ## --- Do the multiplication --- ##
                        print(f" Solving for pressure")
                        tHmatDot = - time.time()
                        sol_p = C.HMAT._matvec(w)
                        tHmatDot = tHmatDot + time.time()
                        print(f"     --> HMat dot product solved in {tHmatDot} [s]")
                        results["t_Dot"].append(tHmatDot)
                        results["compression ratio"].append(C.HMAT.HMATtract.getCompressionRatio()*100)

                        ## --- get the analytical solution --- ##
                        if new_ar:
                            print(f" getting the analytical solution")
                            dummy = - time.time()
                            p_ana = C.get_normal_traction_at(Mesh.CenterCoor, Mesh.CenterCoor, w)
                            dummy = dummy + time.time()
                            print(f"     --> Analytical solution in {dummy} [s]")
                            new_ar = False

                        # some plots
                        rel_err_num = 100 * np.abs(sol_p - p_ana) / p_ana
                        # plot_as_matrix(rel_err_num, mesh=Mesh) # 2D plot
                        # plot_3d_scatter(sol_p, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1]) # 3D plot
                        # plot_3d_scatter(p_ana, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1])  # 3D plot
                        print(f" saving stats")
                        dummy = - time.time()
                        results["max p"].append(sol_p.max())
                        results["max rel_err"].append(rel_err_num.max())

                        ## --- extract all p --- ##
                        all_p = np.zeros(Mesh.NumberOfElts)
                        all_p[EltCrack] = sol_p

                        ## --- store nonzero p and elements index --- ##

                        # results["p"].append(sol_p[centralElts].tolist())
                        # results["allp"].append(sol_p.tolist())
                        dummy = dummy + time.time()
                        print(f"     --> Stored in {dummy} [s]")
                        print(" ------------------- \n")

            ## --- State that one ar is done --- ##
            print(F" <<<< FINISHED Nx = {Nx_i}, Ar = {ar_i}  >>>>")
            print("Saving to file")
            content = results
            action = 'dump_this_dictionary'
            append_to_json_file("HMatConvergence_Nx_" + str(Nx_i) + "_Ar_" + str(ar_i) , [content], action, delete_existing_filename=True)

        ## --- State that one Nx is done --- ##
        print(F" <<<< FINISHED Nx = {Nx_i} completely  >>>>")

    print("Saving to file")
    content = results
    action = 'dump_this_dictionary'
    append_to_json_file("HMatConvergence_Results", [content], action, delete_existing_filename=True)

# ----------------------------------------------
# ----------------------------------------------
# POSTPROCESS
# ----------------------------------------------
# ----------------------------------------------
post = False
if post:
    ## --- Define colors marker etc. --- ##
    cmap = EPFLcolor()
    colors = cmap([0, .2, .3, .4, .5, .6, .8, 1.0])
    markers = ["+", "o", "s", "d", "v"]

    ## --- load the data --- ##
    file_name = "HMatConvergence_Results.json"
    with open(file_name, "r+") as json_file:
        results_loaded = json.load(json_file)[0]  # get the data
    print("Plotting results")

    ## --- prepare all the unique parts --- ##
    # Nx --> color
    unique_Nx = np.unique(results_loaded["nx"])
    Nx_indexes = []
    for Nx_i in unique_Nx:
        Nx_indexes.append(np.where(results_loaded["nx"] == Nx_i)[0])

    # leaf size
    unique_ls = np.unique(results_loaded["leaf size"])
    ls_indexes = []
    for ls_i in unique_ls:
        ls_indexes.append(np.where(results_loaded["leaf size"] == ls_i)[0])

    # eta
    unique_eta = np.unique(results_loaded["eta"])
    eta_indexes = []
    for eta_i in unique_eta:
        eta_indexes.append(np.where(results_loaded["eta"] == eta_i)[0])

    # epsilon
    unique_epsilon = np.unique(results_loaded["epsilon"])
    epsilon_indexes = []
    for epsilon_i in unique_epsilon:
        epsilon_indexes.append(np.where(results_loaded["epsilon"] == epsilon_i)[0])

    #######################
    # max relative error  #
    #######################

    ## --- prepare the plot --- ##
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test: leaf size')
    legend = []
    for ind in range(len(unique_Nx)):
        for ind2 in range(len(unique_ls)):
            legend.append('Nx = ' + str(unique_Nx[ind]) + ', ls = ' + str(unique_ls[ind2]))
            indexes = reduce(np.intersect1d, (Nx_indexes[ind], ls_indexes[ind2], eta_indexes[0], epsilon_indexes[0]))
            plt.plot(np.asarray(results_loaded["aspect ratio"])[indexes],
                     np.asarray(results_loaded["max rel_err"])[indexes], c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Rel err on pressure for uniform opening DD [%]')
    plt.legend(tuple(legend), loc='upper right', shadow=True,
               title='eta = ' + str(unique_eta[0]) + ', eps = ' + str(unique_epsilon[0]))
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig1 = plt.figure()

    #####################
    # compression ratio #
    #####################

    ## --- prepare the plot --- ##
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test: leaf size')
    legend = []
    for ind in range(len(unique_Nx)):
        for ind2 in range(len(unique_ls)):
            legend.append('Nx = ' + str(unique_Nx[ind]) + ', ls = ' + str(unique_ls[ind2]))
            indexes = reduce(np.intersect1d, (Nx_indexes[ind], ls_indexes[ind2], eta_indexes[0], epsilon_indexes[0]))
            plt.plot(np.asarray(results_loaded["aspect ratio"])[indexes],
                     np.asarray(results_loaded["compression ratio"])[indexes],  c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Compression ratio [%]')
    plt.legend(tuple(legend), loc='upper right', shadow=True,
               title='eta = ' + str(unique_eta[0]) + ', eps = ' + str(unique_epsilon[0]))
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig2 = plt.figure()

    #####################
    # Hmat loading time #
    #####################

    ## --- prepare the plot --- ##
    fig3 = plt.figure()
    ax = fig3.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test: leaf size')
    legend = []
    for ind in range(len(unique_Nx)):
        for ind2 in range(len(unique_ls)):
            legend.append('Nx = ' + str(unique_Nx[ind]) + ', ls = ' + str(unique_ls[ind2]))
            indexes = reduce(np.intersect1d, (Nx_indexes[ind], ls_indexes[ind2], eta_indexes[0], epsilon_indexes[0]))
            plt.plot(np.asarray(results_loaded["aspect ratio"][indexes]), np.asarray(results_loaded["t_Hmat"])[indexes],
                     c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Computation time for HMat [s]')
    plt.legend(tuple(legend), loc='upper right', shadow=True,
               title='eta = ' + str(unique_eta[0]) + ', eps = ' + str(unique_epsilon[0]))
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig3 = plt.figure()

    ####################
    # Dot product time #
    ####################

    ## --- prepare the plot --- ##
    fig4 = plt.figure()
    ax = fig4.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test: leaf size')
    legend = []
    for ind in range(len(unique_Nx)):
        for ind2 in range(len(unique_ls)):
            legend.append('Nx = ' + str(unique_Nx[ind]) + ', ls = ' + str(unique_ls[ind2]))
            indexes = reduce(np.intersect1d, (Nx_indexes[ind], ls_indexes[ind2], eta_indexes[0], epsilon_indexes[0]))
            plt.plot(np.asarray(results_loaded["aspect ratio"])[indexes], np.asarray(results_loaded["t_Dot"])[indexes],
                     c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Computation time for Dot product [s]')
    plt.legend(legend, loc='upper right', shadow=True,
               title='eta = ' + str(unique_eta[0]) + ', eps = ' + str(unique_epsilon[0]))
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig4 = plt.figure()
    plt.show()

print(" <<<< FINISHED with All >>>>")

