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
import glob

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


run = False


if run:

    ## --- Setting the different parameters --- ##
    Nx_set_rad = [101, 119, 165, 229, 269, 317, 399, 431, 465, 585, 737, 795, 859, 927, 1001]   # Number of elements
                                                                                                # for Ar = 1
    Nx_set_10 = [33, 45, 53, 73, 101, 109, 127, 137, 159, 201, 317]    # Number of elements # for Ar = 10
    aspect_ratio_set = [1, 10] # Aspect ratios
    eta_set = [5, 10, 15] # distance threshold
    epsilon_set = [1e-3, 5e-4, 1e-4] # limit on final accuracy

    # solid properties
    sim_info = {"nu": 0.4 , "youngs mod": 3.3e4} # Poissons'ratio and young's modulus
    sim_info["Eprime"] = sim_info["youngs mod"] / (1 - sim_info["nu"] ** 2)  # plain strain modulus

    # set the domain size
    sim_info["domain x"] = [-100, 100]

    # uniform opening
    sim_info["wset"] = 1

    for ar_i in aspect_ratio_set:
        if ar_i == 1:
            Nx_set = Nx_set_rad
        else:
            Nx_set = Nx_set_10

        for Nx_i in Nx_set:
            new_ar = True
            ## --- Define the Infos on the simulations --- ##
            # aspect ratio and number of elements
            sim_info["aspect ratio"] = ar_i
            sim_info["Nx"] = Nx_i
            sim_info["Ny"] = Nx_i * ar_i

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

            # nel = Nx_i ** 2 * ar_i
            # uplim = max([min([500, int(nel/10)]), 100])
            # lowlim = min([max([50, int(nel/100)]), 100])
            # leaf_size_set = np.around(np.linspace(lowlim, uplim, 4, endpoint=True), -1).astype(int) # leaf size
            if int(len(EltCrack)) >= 5e5:
                leaf_size_set = [350, 500, 750]
            elif int(len(EltCrack)) >= 1e5:
                leaf_size_set = [200, 350, 500]
            else:
                leaf_size_set = [100, 150, 250]

            print(f"Leaf sizes: {leaf_size_set}")
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
                        results["epsilon"].append(epsilon_i)

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
            print("Saving to file")
            if Nx_i < 10:
                xstr = "000" + str(Nx_i)
            elif Nx_i < 100:
                xstr = "00" + str(Nx_i)
            elif Nx_i < 1000:
                xstr = "0" + str(Nx_i)
            else:
                xstr = str(Nx_i)

            if ar_i < 10:
                arstr = "00" + str(ar_i)
            elif ar_i < 100:
                arstr = "0" + str(ar_i)
            else:
                arstr = str(ar_i)

            content = results
            action = 'dump_this_dictionary'
            append_to_json_file("DoF_HMatConvergence_Nx_" + xstr + "_Ar_" + arstr, [content], action,
                                delete_existing_filename=True)

            ## --- State that one Nx is done --- ##
            print(F" <<<< FINISHED Ar = {ar_i} , Nx = {Nx_i}  >>>>")

        ## --- State that one Nx is done --- ##
        print(F" <<<< FINISHED Ar = {ar_i} completely  >>>>")


# ----------------------------------------------
# ----------------------------------------------
# POSTPROCESS
# ----------------------------------------------
# ----------------------------------------------
post = True
if post:
    #################
    # What to plot? #
    #################
    plot_ar = False
    plot_dof = True

    old_data_set = True
    new_data_set = False

    ###############################
    # Prepare some plotting stuff #
    ###############################

    ## --- Define colorsmap marker etc. --- ##
    cmap = EPFLcolor()
    markers = ["+", "o", "s", "d", "v"]

    ## --- load the data --- ##
    if old_data_set:
        print("Loading old data_set")
        file_name = 'HMatConvergence'
        files = sorted(glob.glob(file_name + '*'))
        check = True
        for file in files:
            with open(file, "r+") as json_file:
                results_loaded_int = json.load(json_file)[0]  # get the data
            if check:
                results_loaded = results_loaded_int
                check = False
            else:
                results_loaded = {key: results_loaded[key] + results_loaded_int[key] for key in results_loaded}

    if new_data_set:
        print("Loading new data_set")
        file_name = 'DoF_HMatConvergence'
        files = sorted(glob.glob(file_name + '*'))
        check = True
        for file in files:
            with open(file, "r+") as json_file:
                results_loaded_int = json.load(json_file)[0]  # get the data
            if check:
                results_loaded = results_loaded_int
                check = False
            else:
                results_loaded = {key: results_loaded[key] + results_loaded_int[key] for key in results_loaded}

    print("Plotting results")

    ## --- prepare all the unique parts --- ##
    # Nx --> color
    unique_Nx = np.unique(results_loaded["nx"])
    Nx_indexes = []
    for Nx_i in unique_Nx:
        Nx_indexes.append(np.where(results_loaded["nx"] == Nx_i)[0])

    # eta
    unique_eta = np.unique(results_loaded["eta"])
    eta_indexes = []
    for eta_i in unique_eta:
        eta_indexes.append(np.where(results_loaded["eta"] == eta_i)[0])

    # epsilon
    if np.max(np.asarray(results_loaded["epsilon"])) == 0:
        results_loaded["epsilon"] = [1e-3, 1e-4, 1e-5] * int(len(results_loaded["epsilon"]) / 3)
    unique_epsilon = np.unique(results_loaded["epsilon"])
    epsilon_indexes = []
    for epsilon_i in unique_epsilon:
        epsilon_indexes.append(np.where(results_loaded["epsilon"] == epsilon_i)[0])

    ## --- define the variables to plot, y labels and the leaf size legend --- ##
    vars = ["max rel_err", "compression ratio", "t_Hmat", "t_Dot"]
    ylabels = ['Rel err on pressure for uniform opening DD [%]',
               'Compression ratio [%]',
               'Computation time for HMat [s]',
               'Computation time for Hmat dot prodcut [s]']

    lslegend = [', smallest ls', ', small ls', ', big ls', ', largest ls']

    if old_data_set:
        if plot_ar:
            colors = cmap(np.linspace(0.1, 1., len(unique_Nx), endpoint=True))
            ##############################
            # Plotting varying leaf size #
            ##############################
            for plotvar in range(len(vars)):
                ## --- prepare the plot --- ##
                locals()['fig' + str(plotvar)] = plt.figure()
                ax = locals()['fig' + str(plotvar)].add_subplot(1, 1, 1)
                plt.suptitle('Rectangular crack test: leaf size')
                legend = []
                for ind in range(len(unique_Nx)):
                    # find unique As number then for this generate the number indexes as always 4 niner packages per as
                    nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[ind]]))
                    start_ls = [y * 36 for y in range(nAs)]
                    for ind2 in range(4):
                        ls_indexes = [[y + ls for y in range(ind2 * 9, (ind2 + 1) * 9)] for ls in start_ls]
                        ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                        indexes = reduce(np.intersect1d, (Nx_indexes[ind][ls_indexes], eta_indexes[1], epsilon_indexes[1]))
                        sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["aspect ratio"])[indexes])]
                        legend.append('Nx = ' + str(unique_Nx[ind]) + lslegend[ind2])
                        plt.plot(np.asarray(results_loaded["aspect ratio"])[sorted_indexes],
                                 (np.asarray(results_loaded[vars[plotvar]])[sorted_indexes]),
                                 c=colors[ind], marker=markers[ind2])

                plt.xlabel('Fracture aspect ratio')
                plt.ylabel(ylabels[plotvar])
                plt.xscale('log')
                plt.yscale('log')
                # Add a grid
                ax.grid(which='both')
                locals()['fig' + str(plotvar)].legend(tuple(legend), title='eta = ' + str(unique_eta[1]) + ', eps = ' +
                                                                           str(unique_epsilon[1]), loc=7)
                locals()['fig' + str(plotvar)].tight_layout()
                plt.subplots_adjust(right=0.675)

            ############################
            # Plotting varying epsilon #
            ############################
            for lind in [0, 3]:
                for plotvar in range(len(vars)):
                    ## --- prepare the plot --- ##
                    locals()['fig' + str(plotvar)] = plt.figure()
                    ax = locals()['fig' + str(plotvar)].add_subplot(1, 1, 1)
                    plt.suptitle('Rectangular crack test: epsilon')
                    legend = []
                    for ind in range(len(unique_Nx)):
                        # find unique As number then for this generate the number indexes as always 4 niner packages per as
                        nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[ind]]))
                        start_ls = [y * 36 for y in range(nAs)]
                        ls_indexes = [[y + ls for y in range(lind * 9, (lind + 1) * 9)] for ls in start_ls]
                        ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                        for ind2 in range(len(unique_epsilon)):
                            indexes = reduce(np.intersect1d, (epsilon_indexes[ind2], Nx_indexes[ind][ls_indexes],
                                                              eta_indexes[1]))
                            sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["aspect ratio"])[indexes])]
                            legend.append('Nx = ' + str(unique_Nx[ind]) + ', eps = ' + str(unique_epsilon[ind2]))
                            plt.plot(np.asarray(results_loaded["aspect ratio"])[sorted_indexes],
                                     (np.asarray(results_loaded[vars[plotvar]])[sorted_indexes]),
                                     c=colors[ind], marker=markers[ind2])

                    plt.xlabel('Fracture aspect ratio')
                    plt.ylabel(ylabels[plotvar])
                    plt.xscale('log')
                    plt.yscale('log')
                    # Add a grid
                    ax.grid(which='both')
                    locals()['fig' + str(plotvar)].legend(tuple(legend), title='eta = ' + str(unique_eta[1]) + lslegend[lind]
                                                          , loc=7)
                    locals()['fig' + str(plotvar)].tight_layout()
                    plt.subplots_adjust(right=0.675)

            ############################
            # Plotting varying eta #
            ############################
            for lind in [0, 3]:
                for plotvar in range(len(vars)):
                    ## --- prepare the plot --- ##
                    locals()['fig' + str(plotvar)] = plt.figure()
                    ax = locals()['fig' + str(plotvar)].add_subplot(1, 1, 1)
                    plt.suptitle('Rectangular crack test: eta')
                    legend = []
                    for ind in range(len(unique_Nx)):
                        # find unique As number then for this generate the number indexes as always 4 niner packages per as
                        nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[ind]]))
                        start_ls = [y * 36 for y in range(nAs)]
                        ls_indexes = [[y + ls for y in range(lind * 9, (lind + 1) * 9)] for ls in start_ls]
                        ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                        for ind2 in range(len(unique_eta)):
                            indexes = reduce(np.intersect1d, (epsilon_indexes[1], Nx_indexes[ind][ls_indexes],
                                                              eta_indexes[ind2]))
                            sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["aspect ratio"])[indexes])]
                            legend.append('Nx = ' + str(unique_Nx[ind]) + ', eta = ' + str(unique_eta[ind2]))
                            plt.plot(np.asarray(results_loaded["aspect ratio"])[sorted_indexes],
                                     (np.asarray(results_loaded[vars[plotvar]])[sorted_indexes]),
                                     c=colors[ind], marker=markers[ind2])

                    plt.xlabel('Fracture aspect ratio')
                    plt.ylabel(ylabels[plotvar])
                    plt.xscale('log')
                    plt.yscale('log')
                    # Add a grid
                    ax.grid(which='both')
                    locals()['fig' + str(plotvar)].legend(tuple(legend), title='eps = ' + str(unique_epsilon[1]) +
                                                                               lslegend[lind], loc=7)
                    locals()['fig' + str(plotvar)].tight_layout()
                    plt.subplots_adjust(right=0.675)

        if plot_dof:
            ##############################
            # Plotting varying leaf size #
            ##############################
            for plotvar in range(len(vars)):
                ## --- prepare the plot --- ##
                locals()['fig' + str(plotvar + 1)] = plt.figure()
                ax = locals()['fig' + str(plotvar + 1)].add_subplot(1, 1, 1)
                plt.suptitle('Rectangular crack test: leaf size')
                legend = []
                colors = cmap(np.linspace(0.1, 1., 4, endpoint=True))
                for indls in range(4):
                    varvals = np.array([])
                    Dofvals = np.array([])
                    for indNx in range(len(unique_Nx)):
                        # find unique As number then for this generate the number indexes as always 4 niner packages per as
                        nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[indNx]]))
                        start_ls = [y * 36 for y in range(nAs)]
                        ls_indexes = [[y + ls for y in range(indls * 9, (indls + 1) * 9)] for ls in start_ls]
                        ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                        indexes = reduce(np.intersect1d, (Nx_indexes[indNx][ls_indexes], eta_indexes[1],
                                                          epsilon_indexes[1]))
                        sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["n. of Elts"])[indexes])]
                        varvals = np.append(varvals, np.asarray(results_loaded[vars[plotvar]])[sorted_indexes])
                        Dofvals = np.append(Dofvals, np.asarray(results_loaded["n. of Elts"])[sorted_indexes])
                    legend.append(lslegend[indls])
                    sorted_indexes = np.argsort(Dofvals)
                    plt.plot(Dofvals[sorted_indexes], varvals[sorted_indexes], c=colors[indls], marker=markers[indls])

                plt.xlabel('#DoF')
                plt.ylabel(ylabels[plotvar])
                plt.xscale('log')
                plt.yscale('log')
                # Add a grid
                ax.grid(which='both')
                locals()['fig' + str(plotvar + 1)].legend(tuple(legend), title='eta = ' + str(unique_eta[1]) + ', eps = ' +
                                                                           str(unique_epsilon[1]), loc=7)
                locals()['fig' + str(plotvar + 1)].tight_layout()
                plt.subplots_adjust(right=0.675)


            ############################
            # Plotting varying eta #
            ############################
            for lind in [0, 3]:
                for plotvar in range(len(vars)):
                    ## --- prepare the plot --- ##
                    locals()['fig' + str(plotvar + 1)] = plt.figure()
                    ax = locals()['fig' + str(plotvar + 1)].add_subplot(1, 1, 1)
                    plt.suptitle('Rectangular crack test: eta')
                    legend = []
                    for indEta in range(len(unique_eta)):
                        varvals = np.array([])
                        Dofvals = np.array([])
                        for indNx in range(len(unique_Nx)):
                            # find unique As number then for this generate the number indexes as always 4 niner packages per as
                            nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[indNx]]))
                            start_ls = [y * 36 for y in range(nAs)]
                            ls_indexes = [[y + ls for y in range(lind * 9, (lind + 1) * 9)] for ls in start_ls]
                            ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                            indexes = reduce(np.intersect1d, (epsilon_indexes[1], Nx_indexes[indNx][ls_indexes],
                                                              eta_indexes[indEta]))
                            sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["n. of Elts"])[indexes])]
                            varvals = np.append(varvals, np.asarray(results_loaded[vars[plotvar]])[sorted_indexes])
                            Dofvals = np.append(Dofvals, np.asarray(results_loaded["n. of Elts"])[sorted_indexes])
                        legend.append("Eta = " + str(unique_eta[indEta]))
                        sorted_indexes = np.argsort(Dofvals)
                        plt.plot(Dofvals[sorted_indexes], varvals[sorted_indexes], c=colors[indEta],
                                 marker=markers[indEta])

                    plt.xlabel('#DoF')
                    plt.ylabel(ylabels[plotvar])
                    plt.xscale('log')
                    plt.yscale('log')
                    # Add a grid
                    ax.grid(which='both')
                    locals()['fig' + str(plotvar + 1)].legend(tuple(legend), title='eps = ' + str(unique_epsilon[1]) +
                                                                               lslegend[lind], loc=7)
                    locals()['fig' + str(plotvar + 1)].tight_layout()
                    plt.subplots_adjust(right=0.675)

            ############################
            # Plotting varying epsilon #
            ############################
            for lind in [0, 3]:
                for plotvar in range(len(vars)):
                    ## --- prepare the plot --- ##
                    locals()['fig' + str(plotvar + 1)] = plt.figure()
                    ax = locals()['fig' + str(plotvar + 1)].add_subplot(1, 1, 1)
                    plt.suptitle('Rectangular crack test: epsilon')
                    legend = []
                    for indEps in range(len(unique_epsilon)):
                        varvals = np.array([])
                        Dofvals = np.array([])
                        for indNx in range(len(unique_Nx)):
                            # find unique As number then for this generate the number indexes as always 4 niner packages per as
                            nAs = len(np.unique(np.asarray(results_loaded["aspect ratio"])[Nx_indexes[indNx]]))
                            start_ls = [y * 36 for y in range(nAs)]
                            ls_indexes = [[y + ls for y in range(lind * 9, (lind + 1) * 9)] for ls in start_ls]
                            ls_indexes = np.sort(np.asarray([item for sublist in ls_indexes for item in sublist]).astype(int))
                            indexes = reduce(np.intersect1d, (epsilon_indexes[indEps], Nx_indexes[indNx][ls_indexes],
                                                              eta_indexes[1]))
                            sorted_indexes = indexes[np.argsort(np.asarray(results_loaded["n. of Elts"])[indexes])]
                            varvals = np.append(varvals, np.asarray(results_loaded[vars[plotvar]])[sorted_indexes])
                            Dofvals = np.append(Dofvals, np.asarray(results_loaded["n. of Elts"])[sorted_indexes])
                        legend.append("Epsilon = " + str(unique_epsilon[indEps]))
                        sorted_indexes = np.argsort(Dofvals)
                        plt.plot(Dofvals[sorted_indexes], varvals[sorted_indexes], c=colors[indEps],
                                 marker=markers[indEps])

                    plt.xlabel('#DoF')
                    plt.ylabel(ylabels[plotvar])
                    plt.xscale('log')
                    plt.yscale('log')
                    # Add a grid
                    ax.grid(which='both')
                    locals()['fig' + str(plotvar + 1)].legend(tuple(legend), title='eta = ' + str(unique_eta[1]) +
                                                                               lslegend[lind], loc=7)
                    locals()['fig' + str(plotvar + 1)].tight_layout()
                    plt.subplots_adjust(right=0.675)

print(" <<<< FINISHED with All >>>>")

