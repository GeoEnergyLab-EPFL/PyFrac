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
import matplotlib.pyplot as plt

from common_rect_and_radial_tests import *
from level_set.continuous_front_reconstruction import plot_two_fronts
from utilities.postprocess_fracture import append_to_json_file
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from src.tip.tip_inversion import StressIntensityFactor

# ----------------------------------------------
# ----------------------------------------------
# RUN
# ----------------------------------------------
# ----------------------------------------------


run = False


if run:

    file_name = "results_rectangular_as10.json"
    aspec_ratio_set = [10, 20, 30]
    file_name_set = ["results_rectangular_as10.json", "results_rectangular_as20.json", "results_rectangular_as30.json"]

    aspec_ratio_set = [30]
    file_name_set = ["results_rectangular_as30.json"]


    for ar_i in range(len(aspec_ratio_set)):
        file_name = file_name_set[ar_i]
        ar = aspec_ratio_set[ar_i]
        print("\n ------")
        print(f"ASPECT RATIO: {ar}")
        print("\n ------")
        # deciding the aspect ratio (if as > 1 -> Ly > Lx)
        # - it is Ly/Lx
        # number of mesh refinements
        #   - along x and y

        sim_info = {"aspect ratio": ar, "n. of refinements x": 6, "n. of refinements y": 6}

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
                   "max w R0 with tipcorr": [],
                   "max w R4 with tipcorr": [],
                   "frac volume R0": [],
                   "frac volume R4": [],
                   "frac volume R0 with tipcorr": [],
                   "frac volume R4 with tipcorr": [],
                   "KI R0": [],
                   "KI R4": [],
                   "KI R0 with tipcorr": [],
                   "KI R4 with tipcorr": [],
                   "n. of Elts" : [],
                   "nu": sim_info["nu"],  # Poisson's ratio
                   "youngs mod": sim_info["youngs mod"],
                   "H" : [],
                   "p": sim_info["p"],
                   "x_center_section": [],
                   "w_R0": [],
                   "w_R4": [],
                   "w_R0_tipcorr": [],
                   "w_R4_tipcorr": []
                   }


        # loop over the different meshes
        for refinement_ID in np.arange(1, maxref, 1):


            print(f"Step {refinement_ID} of {maxref} :")
            st = 0

            # creating mesh & plotting
            Mesh = get_mesh(sim_info, refinement_ID)
            EltTip = np.asarray(Mesh.get_Boundarylist())
            EltRibbon = Mesh.get_Frontlist()
            EltCrack = np.arange(Mesh.NumberOfElts)
            FillF = np.ones(len(EltTip))
            centralElts = np.where(np.abs(Mesh.CenterCoor[:,1]) < 0.9 * Mesh.hy)[0]
            H = (sim_info["domain x"][1] - sim_info["domain x"][0]) + Mesh.hx

            # if refinement_ID == 1 or  refinement_ID == maxref -1:
            #   plot_two_fronts(Mesh, newfront=None, oldfront=None , fig=None, grid=True, cells = EltCrack, my_marker = " ")

            results["x_center_section"].append(Mesh.CenterCoor[centralElts,0].tolist())
            sim_info["H"] = H
            results["H"].append(H)
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
            print(f" {st}) solving R_4 matrix (with tip corr)")
            dummy = - time.time()
            TipCorr = [FillF, EltTip]
            sol_R4_tipcorr = get_solution(C_R4, p, EltCrack, TipCorr=TipCorr)
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

            st = st + 1
            print(f" {st}) solving R_0 matrix (with tip corr)")
            dummy = - time.time()
            TipCorr = [FillF, EltTip]
            sol_R0_tipcorr = get_solution(C_R0, p, EltCrack, TipCorr=TipCorr)
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
            results["max w R0 with tipcorr"].append(sol_R0_tipcorr.max())
            results["max w R4 with tipcorr"].append(sol_R4_tipcorr.max())
            results["frac volume R0"].append(np.sum(sol_R0) * Mesh.hx * Mesh.hy)
            results["frac volume R4"].append(np.sum(sol_R4) * Mesh.hx * Mesh.hy)
            results["frac volume R0 with tipcorr"].append(np.sum(sol_R0_tipcorr) * Mesh.hx * Mesh.hy)
            results["frac volume R4 with tipcorr"].append(np.sum(sol_R4_tipcorr) * Mesh.hx * Mesh.hy)


            # > SIF estimation <

            KI_ana = KI_2DPS_solution(sim_info["p"], H)
            # compute sgndDist
            sgndDist = np.zeros(Mesh.NumberOfElts)

            for cell in range(Mesh.NumberOfElts):
                coorC = Mesh.CenterCoor[cell]
                xC = coorC[0]; yC = coorC[1]
                temp = np.asarray([coorC[1], coorC[1], coorC[0], coorC[0]])
                # remember Mesh.domainLimits = [yminF,ymaxF,xminF,xmaxF]
                sgndDist[cell] = - np.min(np.abs(Mesh.domainLimits - temp))

            # from utilities.utility import plot_as_matrix
            # plot_as_matrix(sgndDist, Mesh)

            #
            all_w = np.zeros(Mesh.NumberOfElts)
            all_w[EltCrack] = sol_R0
            KIPrime_R0 = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                               sgndDist,
                                               EltTip,
                                               EltRibbon,
                                               np.full(len(EltTip), True),
                                               Mesh,
                                               Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))

            # find all the elem at the tip with coordinate y==0
            centralTips = np.where(np.abs(Mesh.CenterCoor[EltTip,1]) < 0.9 *Mesh.hy)[0][0]
            relerr_KIPrime_R0 = 100. * (np.abs(KIPrime_R0[centralTips] - KI_ana) / KI_ana)
            results["KI R0"].append(relerr_KIPrime_R0)

            #
            all_w[EltCrack] = sol_R4

            KIPrime_R4 = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                               sgndDist,
                                               EltTip,
                                               EltRibbon,
                                               np.full(len(EltTip), True),
                                               Mesh,
                                               Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))
            relerr_KIPrime_R4 = 100 * (np.abs(KIPrime_R4[centralTips] - KI_ana) / KI_ana)
            results["KI R4"].append(relerr_KIPrime_R4)

            #
            all_w[EltCrack] = sol_R0_tipcorr
            KIPrime_R0_tipcorr = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                                                              sgndDist,
                                                                              EltTip,
                                                                              EltRibbon,
                                                                              np.full(len(EltTip), True),
                                                                              Mesh,
                                                                              Eprime=np.full(Mesh.NumberOfElts,
                                                                                             sim_info["Eprime"]))
            relerr_KIPrime_R0_tipcorr = 100 * (np.abs(KIPrime_R0_tipcorr[centralTips] - KI_ana) / KI_ana)
            results["KI R0 with tipcorr"].append(relerr_KIPrime_R0_tipcorr)
            #
            all_w[EltCrack] = sol_R4_tipcorr
            KIPrime_R4_tipcorr = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                                                              sgndDist,
                                                                              EltTip,
                                                                              EltRibbon,
                                                                              np.full(len(EltTip), True),
                                                                              Mesh,
                                                                              Eprime=np.full(Mesh.NumberOfElts,
                                                                                             sim_info["Eprime"]))
            relerr_KIPrime_R4_tipcorr = 100 * (np.abs(KIPrime_R4_tipcorr[centralTips] - KI_ana) / KI_ana)
            results["KI R4 with tipcorr"].append(relerr_KIPrime_R4_tipcorr)

            # > store nonzero w and elements index <

            results["w_R0"].append(sol_R0[centralElts].tolist())
            results["w_R4"].append(sol_R4[centralElts].tolist())
            results["w_R0_tipcorr"].append(sol_R0_tipcorr[centralElts].tolist())
            results["w_R4_tipcorr"].append(sol_R4_tipcorr[centralElts].tolist())

            print(" ------------------- \n")

            print("Saving to file")
            content = results
            action = 'dump_this_dictionary'
            append_to_json_file(file_name, [content], action, delete_existing_filename=True)

# ----------------------------------------------
# ----------------------------------------------
# POSTPROCESS
# ----------------------------------------------
# ----------------------------------------------
post = True
if post:
    file_name_1 = "results_rectangular_as10.json"
    file_name_2 = "results_rectangular_as20.json"
    file_name_3 = "results_rectangular_as30.json"
    with open(file_name_1, "r+") as json_file:
        results1 = json.load(json_file)[0]  # get the data
    with open(file_name_2, "r+") as json_file:
        results2 = json.load(json_file)[0]  # get the data
    with open(file_name_3, "r+") as json_file:
        results3 = json.load(json_file)[0]  # get the data
    print("Plotting results")

    ####################
    # rel error w max  #
    ####################
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    plt.suptitle('Rectangular crack test')
    y_ana = []
    for H_i in range(len(results1["H"])):
        y_ana.append(wmax_plane_strain_solution(results1["youngs mod"], results1["nu"],results1["p"], results1["H"][H_i]))
    y_ana = np.asarray(y_ana)
    re_maxwR0    = (100 * np.abs(np.asarray(results1["max w R0"])              - y_ana)) / (100 * y_ana)
    re_maxwR0_tc = (100 * np.abs(np.asarray(results1["max w R0 with tipcorr"]) - y_ana)) / (100 * y_ana)
    re_maxwR4    = (100 * np.abs(np.asarray(results1["max w R4"])              - y_ana))/ (100 * y_ana)
    re_maxwR4_tc = (100 * np.abs(np.asarray(results1["max w R4 with tipcorr"]) - y_ana))/ (100 * y_ana)

    plt.plot(results1["n. of Elts"], 100 * re_maxwR0, c='r', marker="+")
    plt.plot(results1["n. of Elts"], 100 * re_maxwR0_tc, c='r', marker="o")
    plt.plot(results1["n. of Elts"], 100 * re_maxwR4, c='b', marker="+")
    plt.plot(results1["n. of Elts"], 100 * re_maxwR4_tc, c='b', marker="o")

    # y_ana = np.full(2, y_ana)
    # plt.plot([results1["n. of Elts"][0],results1["n. of Elts"][-1]], y_ana, c='black', marker=" ")
    plt.xlabel('# of DOF in the crack')
    plt.ylabel('Rel err max(w) against plane strain sol. [%]')
    plt.legend(('R0', 'R0 - tip corr as Ryder & Napier 1985', 'R4', 'R4 - tip corr as Ryder & Napier 1985'),loc='upper right', shadow=True, title='Aspect ratio = 10')
    plt.xscale('log')
    plt.yscale('log')
    # Add a grid
    ax.grid(which='both')

    ####################
    # volume           #
    ####################
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(1, 1, 1)
    #
    # # Major ticks every 20, minor ticks every 5
    # major_ticks_y = np.arange(0, 101, 20)
    # minor_ticks_y = np.arange(0, 101, 5)
    # major_ticks_x = np.arange(0, 1.01, 0.2)
    # minor_ticks_x = np.arange(0, 1.01, 0.05)
    #
    # ax.set_xticks(major_ticks_x)
    # ax.set_xticks(minor_ticks_x, minor=True)
    # ax.set_yticks(major_ticks_y)
    # ax.set_yticks(minor_ticks_y, minor=True)
    #
    # # And a corresponding grid
    # ax.grid(which='both')
    #
    # plt.suptitle('Rectangular crack test')
    # plt.plot(results1["n. of Elts"], results1["frac volume R0"], c='r', marker="+")
    # plt.plot(results1["n. of Elts"], results1["frac volume R0 with tipcorr"], c='r', marker="o")
    # plt.plot(results1["n. of Elts"], results1["frac volume R4"], c='b', marker="+")
    # plt.plot(results1["n. of Elts"], results1["frac volume R4 with tipcorr"], c='b', marker="o")
    # plt.xlabel('# of DOF in the crack')
    # plt.ylabel('frac volume')
    # plt.legend(('R0', 'R0 - tip corr as Ryder & Napier 1985', 'R4', 'R4 - tip corr as Ryder & Napier 1985'),loc='upper right', shadow=True, title='Aspect ratio = 10')
    #
    # plt.xscale('log')
    # plt.yscale('log')

    #################################
    # rel err w max VS aspect ratio #
    #################################
    #
    fig1 = plt.figure()
    plt.suptitle('Rectangular crack test')

    res = [results1, results2, results3]
    for res_i in range(3):
        y_ana = []
        for H_i in range(len(res[res_i]["H"])):
            y_ana.append(wmax_plane_strain_solution(res[res_i]["youngs mod"], res[res_i]["nu"],res[res_i]["p"], res[res_i]["H"][H_i]))
        y_ana = np.asarray(y_ana)
        n1 = 0.6+0.1*res_i
        n2 = 0.6 - 0.1 * res_i
        plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R0"]) - y_ana))/(100 *y_ana), c=(n1, n2, 0.), marker="+")
        plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R0 with tipcorr"]) - y_ana)) / (100 * y_ana), c=(n1, n2, 0.),marker="o")
        plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R4"]) - y_ana))/(100 *y_ana), c=(0., n2, n1), marker="+")
        plt.plot(res[res_i]["nx"], 100 * (100 * np.abs(np.asarray(res[res_i]["max w R4 with tipcorr"]) - y_ana)) / (100 * y_ana), c=(0., n2, n1), marker="o")

    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in the transversal direction')
    plt.ylabel('rel. err. w max [%]')
    plt.legend(('R0 - aspect ratio 10',
                'R0 with tip corr - aspect ratio 10',
                'R4 - aspect ratio 10',
                'R4 with tip corr - aspect ratio 10',
                'R0 - aspect ratio 20',
                'R0 with tip corr - aspect ratio 20',
                'R4 - aspect ratio 20',
                'R4 with tip corr - aspect ratio 20',
                'R0 - aspect ratio 30',
                'R0 with tip corr - aspect ratio 30',
                'R4 - aspect ratio 30',
                'R4 with tip corr - aspect ratio 30'
                ), loc='lower left', shadow=True, title='tip corr as Ryder & Napier 1985' )
    plt.xscale('log')
    plt.yscale('log')

    ####################
    # SIF              #
    ####################

    fig1 = plt.figure()
    plt.suptitle('Rectangular crack test')

    plt.plot(results3["nx"], results3["KI R0"], c='r', marker="+")
    plt.plot(results3["nx"], results3["KI R0 with tipcorr"], c='r', marker=".")
    plt.plot(results3["nx"], results3["KI R4"], c='b', marker="+")
    plt.plot(results3["nx"], results3["KI R4 with tipcorr"], c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in x direction crack')
    plt.ylabel('rel. err. KI [%]')
    plt.legend(('R0 - NO tip corr',  'R0 - tip corr as Ryder & Napier 1985',
                'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985'), loc='lower left', shadow=True, title='Aspect ratio = 30')
    plt.xscale('log')
    plt.yscale('log')

    ####################
    # w(x) adim        #
    ####################
    res = [results1, results2, results3]
    ar = [10,20,30]
    for res_i in range(3):
        fig1 = plt.figure()
        plt.suptitle('Rectangular crack test')
        xadim = np.asarray(res[res_i]["x_center_section"][-1])/((res[res_i]["H"][-1])/2.)
        xsol = np.sqrt(1 - xadim * xadim)
        w_R0 = np.asarray(res[res_i]["w_R0"][-1]) / np.asarray(res[res_i]["w_R0"][-1]).max()
        w_R0_tipcorr = np.asarray(res[res_i]["w_R0_tipcorr"][-1]) / np.asarray(res[res_i]["w_R0_tipcorr"][-1]).max()
        w_R4 = np.asarray(res[res_i]["w_R4"][-1]) / np.asarray(res[res_i]["w_R4"][-1]).max()
        w_R4_tipcorr = np.asarray(res[res_i]["w_R4_tipcorr"][-1]) / np.asarray(res[res_i]["w_R4_tipcorr"][-1]).max()

        re_w_R0 = 100 * np.abs(w_R0-xsol) / (xsol)
        re_w_R0_tc = 100 * np.abs(w_R0_tipcorr - xsol) / (xsol)
        re_w_R4 = 100 * np.abs(w_R4-xsol) / (xsol)
        re_w_R4_tc = 100 * np.abs(w_R4_tipcorr - xsol) / (xsol)

        plt.plot(xadim, re_w_R0, c='r', marker="+")
        plt.plot(xadim, re_w_R0_tc, c='r', marker=".")
        plt.plot(xadim, re_w_R4, c='b', marker="+")
        plt.plot(xadim, re_w_R4_tc, c='b', marker=".")
        plt.tick_params(labeltop=True, labelright=True)
        plt.grid(True, which="both", ls="-")

        plt.ylabel('rel error w [%]')
        plt.xlabel('x/(H/2)')
        plt.legend(('R0 - NO tip corr',  'R0 - tip corr as Ryder & Napier 1985',
                    'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985'), loc='upper center', shadow=True, title='Aspect ratio = '+str(ar[res_i]))
        plt.xlim([0,1])

    ####################
    # w(x)             #
    ####################
    res = [results1, results2, results3]
    ar = [10, 20, 30]
    for res_i in range(3):
        fig1 = plt.figure()
        plt.suptitle('Rectangular crack test')
        xadim = np.asarray(res[res_i]["x_center_section"][-1]) / ((res[res_i]["H"][-1]) / 2.)
        xsol = np.sqrt(1 - xadim * xadim) * wmax_plane_strain_solution(res[res_i]["youngs mod"], res[res_i]["nu"],res[res_i]["p"], res[res_i]["H"][H_i])
        w_R0 = np.asarray(res[res_i]["w_R0"][-1])
        w_R0_tipcorr = np.asarray(res[res_i]["w_R0_tipcorr"][-1])
        w_R4 = np.asarray(res[res_i]["w_R4"][-1])
        w_R4_tipcorr = np.asarray(res[res_i]["w_R4_tipcorr"][-1])

        re_w_R0 = 100 * np.abs(w_R0 - xsol) / (xsol)
        re_w_R0_tc = 100 * np.abs(w_R0_tipcorr - xsol) / (xsol)
        re_w_R4 = 100 * np.abs(w_R4 - xsol) / (xsol)
        re_w_R4_tc = 100 * np.abs(w_R4_tipcorr - xsol) / (xsol)

        plt.plot(xadim, re_w_R0, c='r', marker="+")
        plt.plot(xadim, re_w_R0_tc, c='r', marker=".")
        plt.plot(xadim, re_w_R4, c='b', marker="+")
        plt.plot(xadim, re_w_R4_tc, c='b', marker=".")
        plt.tick_params(labeltop=True, labelright=True)
        plt.grid(True, which="both", ls="-")

        plt.ylabel('rel error w [%]')
        plt.xlabel('x/(H/2)')
        plt.legend(('R0 - NO tip corr', 'R0 - tip corr as Ryder & Napier 1985',
                    'R4 - NO tip corr', 'R4 - tip corr as Ryder & Napier 1985'), loc='upper center', shadow=True,
                   title='Aspect ratio = ' + str(ar[res_i]))
        plt.xlim([0, 1])


    ####################
    # SIF by volume    #
    ####################
    Ep = results3["youngs mod"] / (1 - results3["nu"]**2)
    hx = 2. * results3["Lx"][-1] / (results3["nx"][-1] - 1)
    KI_ana = KI_2DPS_solution(results3["p"], results3["H"][-1])
    w1 = results3["w_R0_tipcorr"][-1][-1]
    w2 = results3["w_R0_tipcorr"][-1][-2]

    # A: from tip to h/2
    KI_A = 3. * Ep * np.sqrt(np.pi) * w1 / (8. * np.sqrt(hx))
    KI_A_relerr = 100 * np.abs(KI_A - KI_ana)/KI_ana

    # B: from tip to h
    KI_B = 3. * Ep * np.sqrt(np.pi*0.5) * w1 / (8. * np.sqrt(hx))
    KI_B_relerr = 100 * np.abs(KI_B - KI_ana)/KI_ana

    # C: from tip to 3 h/2
    KI_C =  Ep * np.sqrt(np.pi/3.) * (2*w1 + w2) / (8. * np.sqrt(hx))
    KI_C_relerr = 100 * np.abs(KI_C - KI_ana)/KI_ana

    # D: from tip to 2 h
    KI_D = 3. * Ep * np.sqrt(np.pi) * (w1 + w2) / (32. * np.sqrt(hx))
    KI_D_relerr = 100 * np.abs(KI_D - KI_ana)/KI_ana

    # E: from h/2 to 3 h/2
    KI_E =  3. * (4. + 3. * np.sqrt(6.)) * Ep * np.sqrt(np.pi/2.) * (w2) / (152. * np.sqrt(hx))
    KI_E_relerr = 100 * np.abs(KI_E - KI_ana)/KI_ana

    # F: from h/2 to 2 h
    KI_F = 3. * (4. + np.sqrt(2.)) * Ep * np.sqrt(np.pi) * (w2) / (112. * np.sqrt(hx))
    KI_F_relerr = 100 * np.abs(KI_F - KI_ana)/KI_ana

    plt.show()
print(" <<<< DONE >>>>")

