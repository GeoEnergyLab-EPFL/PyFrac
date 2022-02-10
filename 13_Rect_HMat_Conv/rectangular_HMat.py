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
    Nx_set = [7, 15, 25, 50, 75, 100, 200] # Number of elements in the cross section
    aspect_ratio_set = [2.5, 5, 10, 25, 50, 75, 100, 200] # Aspect ratios
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
               "H": [],
               "w": sim_info["wset"],
               "x_center_section": [],
               "p": [],
               "allp": [],
               "t_Hmat": [],
               "t_Dot": [],
               "max rel_err": [],
               "rel_err": [],
               "compression ratio": []
               }

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

            for ls_i in leaf_size_set:
                for eta_i in eta_set:
                    for epsilon_i in epsilon_set:
                        ## --- Append first results/sim parameters --- ##
                        results["x_center_section"].append(Mesh.CenterCoor[centralElts, 0].tolist())
                        results["H"].append(H)
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

                        results["p"].append(sol_p[centralElts].tolist())
                        results["allp"].append(sol_p.tolist())
                        dummy = dummy + time.time()
                        print(f"     --> Stored in {dummy} [s]")
                        print(" ------------------- \n")

            ## --- State that one ar is done --- ##
            print(F" <<<< FINISHED Nx = {Nx_i}, Ar = {ar_i}  >>>>")

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
    colors = cmap([0, .2, .4, .5, .6, .8, 1.0])
    markers = ["+", "o", "s", "d", "v"]

    ## --- load the data --- ##
    file_name = "HMatConvergence_Results"
    with open(file_name, "r+") as json_file:
        results_loaded = json.load(json_file)[0]  # get the data
    print("Plotting results")

    ## --- prepare all the unique parts --- ##
    # Nx --> color
    unique_Nx = np.unique(results_loaded["Nx"])
    Nx_indexes = []
    for Nx_i in Nx_indexes:
        Nx_indexes.append(np.where(results_loaded["Nx"] == Nx_i)[0])

    # leaf size
    unique_ls = np.unique(results_loaded["leaf size"])
    ls_indexes = []
    for ls_i in ls_indexes:
        ls_indexes.append(np.where(results_loaded["leaf size"] == ls_i)[0])

    # eta
    unique_eta = np.unique(results_loaded["eta"])
    eta_indexes = []
    for eta_i in eta_indexes:
        eta_indexes.append(np.where(results_loaded["eta"] == eta_i)[0])

    # epsilon
    unique_epsilon = np.unique(results_loaded["epsilon"])
    epsilon_indexes = []
    for epsilon_i in epsilon_indexes:
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
            plt.plot(results_loaded["aspect ratio"][indexes], results_loaded["max rel_err"][indexes], c=colors[ind],
                     marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Rel err on pressure for uniform opening DD [%]')
    plt.legend(legend, loc='upper right', shadow=True, title='eta = ' + unique_eta[0] + ', eps = ' + unique_epsilon[0])
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
            plt.plot(results_loaded["aspect ratio"][indexes], results_loaded["compression ratio"][indexes],
                     c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Compression ratio [%]')
    plt.legend(legend, loc='upper right', shadow=True, title='eta = ' + unique_eta[0] + ', eps = ' + unique_epsilon[0])
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig2 = plt.figure()

    #####################
    # Hmat loading time #
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
            plt.plot(results_loaded["aspect ratio"][indexes], results_loaded["t_Hmat"][indexes],
                     c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Computation time for HMat [s]')
    plt.legend(legend, loc='upper right', shadow=True, title='eta = ' + unique_eta[0] + ', eps = ' + unique_epsilon[0])
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig2 = plt.figure()

    #########################
    # Dot product time time #
    #########################

    ## --- prepare the plot --- ##
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test: leaf size')
    legend = []
    for ind in range(len(unique_Nx)):
        for ind2 in range(len(unique_ls)):
            legend.append('Nx = ' + str(unique_Nx[ind]) + ', ls = ' + str(unique_ls[ind2]))
            indexes = reduce(np.intersect1d, (Nx_indexes[ind], ls_indexes[ind2], eta_indexes[0], epsilon_indexes[0]))
            plt.plot(results_loaded["aspect ratio"][indexes], results_loaded["t_Dot"][indexes],
                     c=colors[ind], marker=markers[ind2])

    plt.xlabel('Fracture aspect ratio')
    plt.ylabel('Computation time for Dot product [s]')
    plt.legend(legend, loc='upper right', shadow=True, title='eta = ' + unique_eta[0] + ', eps = ' + unique_epsilon[0])
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')
    fig2 = plt.figure()

    # plt.suptitle('Rectangular crack test')
    #
    # res = [results1, results2, results3]
    # for res_i in range(3):
    #     y_ana = []
    #     for H_i in range(len(res[res_i]["H"])):
    #         y_ana.append(wmax_plane_strain_solution(res[res_i]["youngs mod"], res[res_i]["nu"],res[res_i]["p"], res[res_i]["H"][H_i]))
    #     y_ana = np.asarray(y_ana)
    #     n1 = 0.6+0.1*res_i
    #     n2 = 0.6 - 0.1 * res_i
    #     plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R0"]) - y_ana))/(100 *y_ana), c=(n1, n2, 0.), marker="+")
    #     plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R0 with tipcorr"]) - y_ana)) / (100 * y_ana), c=(n1, n2, 0.),marker="o")
    #     plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R4"]) - y_ana))/(100 *y_ana), c=(0., n2, n1), marker="+")
    #     plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R4 with tipcorr"]) - y_ana)) / (100 * y_ana), c=(0., n2, n1), marker="o")
    #
    # plt.tick_params(labeltop=True, labelright=True)
    # plt.grid(True, which="both", ls="-")
    #
    # plt.xlabel('# of DOF in the transversal direction')
    # plt.ylabel('rel. err. w max [%]')
    # plt.legend(('R0 - aspect ratio 10',
    #             'R0 with tip corr - aspect ratio 10',
    #             'R4 - aspect ratio 10',
    #             'R4 with tip corr - aspect ratio 10',
    #             'R0 - aspect ratio 20',
    #             'R0 with tip corr - aspect ratio 20',
    #             'R4 - aspect ratio 20',
    #             'R4 with tip corr - aspect ratio 20',
    #             'R0 - aspect ratio 30',
    #             'R0 with tip corr - aspect ratio 30',
    #             'R4 - aspect ratio 30',
    #             'R4 with tip corr - aspect ratio 30'
    #             ), loc='lower left', shadow=True, title='tip corr as Ryder & Napier 1985' )
    # plt.xscale('log')
    # plt.yscale('log')
    #
    # ####################
    # # SIF              #
    # ####################
    #
    # fig1 = plt.figure()
    # plt.suptitle('Rectangular crack test')
    #
    # plt.plot(results3["nx"], results3["KI R0"], c='r', marker="+")
    # plt.plot(results3["nx"], results3["KI R0 with tipcorr"], c='r', marker=".")
    # plt.plot(results3["nx"], results3["KI R4"], c='b', marker="+")
    # plt.plot(results3["nx"], results3["KI R4 with tipcorr"], c='b', marker=".")
    #
    # # SIF from vol
    # plt.plot(results3["nx"], results3["KI R0_fromVol"], c='r', ls='--', marker="+")
    # plt.plot(results3["nx"], results3["KI R0_fromVol with tipcorr"], c='r', ls='--', marker=".")
    # plt.plot(results3["nx"], results3["KI R4_fromVol"], c='b',ls='--', marker="+")
    # plt.plot(results3["nx"], results3["KI R4_fromVol with tipcorr"], c='b', ls='--', marker=".")
    #
    # plt.tick_params(labeltop=True, labelright=True)
    # plt.grid(True, which="both", ls="-")
    #
    # plt.xlabel('# of DOF in x direction crack')
    # plt.ylabel('rel. err. KI [%]')
    # plt.legend(('R0 - NO tip corr',  'R0 - tip corr as Ryder & Napier 1985',
    #             'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985',
    #             'R0 - NO tip corr - vol est', 'R0 - tip corr - vol est',
    #             'R4 - NO tip corr - vol est', 'R4 - tip corr - vol est'
    #             ), loc='lower right', shadow=True, title='Aspect ratio = 30')
    # plt.xscale('log')
    # plt.yscale('log')
    #
    # ####################
    # # w(x) adim        #
    # ####################
    # res = [results1, results2, results3]
    # ar = [10,20,30]
    # for res_i in range(3):
    #     fig1 = plt.figure()
    #     plt.suptitle('Rectangular crack test')
    #     xadim = np.asarray(res[res_i]["x_center_section"][-1])/((res[res_i]["H"][-1])/2.)
    #     xsol = np.sqrt(1 - xadim * xadim)
    #     w_R0 = np.asarray(res[res_i]["w_R0"][-1]) / np.asarray(res[res_i]["w_R0"][-1]).max()
    #     w_R0_tipcorr = np.asarray(res[res_i]["w_R0_tipcorr"][-1]) / np.asarray(res[res_i]["w_R0_tipcorr"][-1]).max()
    #     w_R4 = np.asarray(res[res_i]["w_R4"][-1]) / np.asarray(res[res_i]["w_R4"][-1]).max()
    #     w_R4_tipcorr = np.asarray(res[res_i]["w_R4_tipcorr"][-1]) / np.asarray(res[res_i]["w_R4_tipcorr"][-1]).max()
    #
    #     re_w_R0 = 100 * np.abs(w_R0-xsol) / (xsol)
    #     re_w_R0_tc = 100 * np.abs(w_R0_tipcorr - xsol) / (xsol)
    #     re_w_R4 = 100 * np.abs(w_R4-xsol) / (xsol)
    #     re_w_R4_tc = 100 * np.abs(w_R4_tipcorr - xsol) / (xsol)
    #
    #     plt.plot(xadim, re_w_R0, c='r', marker="+")
    #     plt.plot(xadim, re_w_R0_tc, c='r', marker=".")
    #     plt.plot(xadim, re_w_R4, c='b', marker="+")
    #     plt.plot(xadim, re_w_R4_tc, c='b', marker=".")
    #     plt.tick_params(labeltop=True, labelright=True)
    #     plt.grid(True, which="both", ls="-")
    #
    #     plt.ylabel('rel error w [%]')
    #     plt.xlabel('x/(H/2)')
    #     plt.legend(('R0 - NO tip corr',  'R0 - tip corr as Ryder & Napier 1985',
    #                 'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985'), loc='upper center', shadow=True, title='Aspect ratio = '+str(ar[res_i]))
    #     plt.xlim([0,1])
    #
    # ####################
    # # w(x)             #
    # ####################
    # res = [results1, results2, results3]
    # ar = [10, 20, 30]
    # for res_i in range(3):
    #     fig1 = plt.figure()
    #     plt.suptitle('Rectangular crack test')
    #     xadim = np.asarray(res[res_i]["x_center_section"][-1]) / ((res[res_i]["H"][-1]) / 2.)
    #     xsol = np.sqrt(1 - xadim * xadim) * wmax_plane_strain_solution(res[res_i]["youngs mod"], res[res_i]["nu"],res[res_i]["p"], res[res_i]["H"][H_i])
    #     w_R0 = np.asarray(res[res_i]["w_R0"][-1])
    #     w_R0_tipcorr = np.asarray(res[res_i]["w_R0_tipcorr"][-1])
    #     w_R4 = np.asarray(res[res_i]["w_R4"][-1])
    #     w_R4_tipcorr = np.asarray(res[res_i]["w_R4_tipcorr"][-1])
    #
    #     re_w_R0 = 100 * np.abs(w_R0 - xsol) / (xsol)
    #     re_w_R0_tc = 100 * np.abs(w_R0_tipcorr - xsol) / (xsol)
    #     re_w_R4 = 100 * np.abs(w_R4 - xsol) / (xsol)
    #     re_w_R4_tc = 100 * np.abs(w_R4_tipcorr - xsol) / (xsol)
    #
    #     plt.plot(xadim, re_w_R0, c='r', marker="+")
    #     plt.plot(xadim, re_w_R0_tc, c='r', marker=".")
    #     plt.plot(xadim, re_w_R4, c='b', marker="+")
    #     plt.plot(xadim, re_w_R4_tc, c='b', marker=".")
    #     plt.tick_params(labeltop=True, labelright=True)
    #     plt.grid(True, which="both", ls="-")
    #
    #     plt.ylabel('rel error w [%]')
    #     plt.xlabel('x/(H/2)')
    #     plt.legend(('R0 - NO tip corr', 'R0 - tip corr as Ryder & Napier 1985',
    #                 'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985'), loc='upper center', shadow=True,
    #                title='Aspect ratio = ' + str(ar[res_i]))
    #     plt.xlim([0, 1])
    #
    #
    # ####################
    # # SIF by volume    #
    # ####################
    # Ep = results3["youngs mod"] / (1 - results3["nu"]**2)
    # hx = 2. * results3["Lx"][-1] / (results3["nx"][-1] - 1)
    # KI_ana = KI_2DPS_solution(results3["p"], results3["H"][-1])
    # w1 = results3["w_R0_tipcorr"][-1][-1]
    # w2 = results3["w_R0_tipcorr"][-1][-2]
    #
    # # A: from tip to h/2
    # KI_A = 3. * Ep * np.sqrt(np.pi) * w1 / (8. * np.sqrt(hx))
    # KI_A_relerr = 100 * np.abs(KI_A - KI_ana)/KI_ana
    # print(f'rel error KI A: {KI_A_relerr} %')
    #
    # # B: from tip to h
    # KI_B = 3. * Ep * np.sqrt(np.pi*0.5) * w1 / (8. * np.sqrt(hx))
    # KI_B_relerr = 100 * np.abs(KI_B - KI_ana)/KI_ana
    # print(f'rel error KI B: {KI_B_relerr} %')
    #
    # # C: from tip to 3 h/2
    # KI_C =  Ep * np.sqrt(np.pi/3.) * (2*w1 + w2) / (8. * np.sqrt(hx))
    # KI_C_relerr = 100 * np.abs(KI_C - KI_ana)/KI_ana
    # print(f'rel error KI C: {KI_C_relerr} %')
    #
    # # D: from tip to 2 h
    # KI_D = 3. * Ep * np.sqrt(np.pi) * (w1 + w2) / (32. * np.sqrt(hx))
    # KI_D_relerr = 100 * np.abs(KI_D - KI_ana)/KI_ana
    # print(f'rel error KI D: {KI_D_relerr} %')
    #
    # # E: from h to 3 h/2
    # KI_E =  3. * (4. + 3. * np.sqrt(6.)) * Ep * np.sqrt(np.pi/2.) * (w2) / (152. * np.sqrt(hx))
    # KI_E_relerr = 100 * np.abs(KI_E - KI_ana)/KI_ana
    # print(f'rel error KI E: {KI_E_relerr} %')
    #
    # # F: from h to 2 h
    # KI_F = 3. * (4. + np.sqrt(2.)) * Ep * np.sqrt(np.pi) * (w2) / (112. * np.sqrt(hx))
    # KI_F_relerr = 100 * np.abs(KI_F - KI_ana)/KI_ana
    # print(f'rel error KI F: {KI_F_relerr} %')
    # plt.show()
print(" <<<< FINISHED with All >>>>")

