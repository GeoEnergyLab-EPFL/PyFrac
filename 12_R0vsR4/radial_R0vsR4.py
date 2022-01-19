# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
from fracture_obj.fracture_initialization import Geometry, generate_footprint, get_survey_points
import time
import json

# local imports
from common_rect_and_radial_tests import *
from tip.tip_inversion import StressIntensityFactor
from utilities.postprocess_fracture import append_to_json_file
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from level_set.continuous_front_reconstruction import plot_two_fronts

from src.mesh_obj.mesh import CartesianMesh


def get_mix_err(var_ana, var_num):
    """
     if the analytical value is >1:
        get the relative error
     else:
        return the absolute one

    """
    if var_ana < 1.:
        return (np.abs(var_ana - var_num))
    else:
        return (100 * (np.abs(var_ana - var_num) / var_ana))


def get_rel_err(var_ana, var_num):
    """
     if the analytical value is >1:
        get the relative error
     else:
        return 0.

    """
    if var_ana > 1.:
        return (100 * (np.abs(var_ana - var_num) / var_ana))
    else:
        return (0.)


def get_abs_err(var_ana, var_num):
    """
     get the absolute error

    """
    return np.abs(var_ana - var_num)


def get_err_w(w_num, sim_info, geterr):
    """
     considers a radial crack.
     for eac cell in the crack return the error according to the given function "geterr".
     it returns the error and the radius.
    """

    _e_ = []
    r_ = []
    for i in range(len(EltCrack)):
        e_i = EltCrack[i]
        x_i = Mesh.CenterCoor[e_i][0]
        y_i = Mesh.CenterCoor[e_i][1]
        r_i = np.sqrt((x_i) ** 2 + (y_i) ** 2) / sim_info["R"]
        w_ana = w_radial_solution(x_i, y_i, sim_info["youngs mod"], sim_info["nu"], sim_info["p"], sim_info["R"])
        _e_.append(geterr(w_ana, w_num[i]))
        r_.append(r_i)
    return _e_, r_


def get_err_sig_zz(sigz_num, xy_obs,sim_info, geterr):
    """
     considers a radial crack.
     for each given point in xy_obs, return the error according to the given function "geterr".
     it returns the error and the radius.
    """

    _e_ = []
    r_ = []
    for i in range(len(xy_obs)):
        x_i = xy_obs[i,0]
        y_i = xy_obs[i,1]
        r_i = np.sqrt((x_i) ** 2 + (y_i) ** 2) / sim_info["R"]
        sigz_ana = sig_zz_radial_solution(x_i, y_i, sim_info["p"], sim_info["R"])
        _e_.append(geterr(sigz_ana, sigz_num[i]))
        r_.append(r_i)
    return _e_  #, r_

# ----------------------------------------------
# ----------------------------------------------
# RUN
# ----------------------------------------------
# ----------------------------------------------
run =True
file_name = "results_radial.json"

if run:
    sim_info = {}

    # deciding the aspect ratio
    # - it is Ly/Lx
    sim_info["aspect ratio"] = 1

    # number of mesh refinements
    #   - along x and y
    param = 25
    sim_info["n. of refinements x"] = param
    sim_info["n. of refinements y"] = param
    sim_info["max. n. of refinements"] = np.maximum(sim_info["n. of refinements x"], sim_info["n. of refinements y"])

    # the coarsest mesh (the y direction will be a function of x)
    sim_info["nx min"] = 11
    #sim_info["nx min"] = 5

    # set the domain size (the y direction will be a function of x)
    sim_info["domain x"] = [-60.,60.]
    #sim_info["domain x"] = [-4., 4.]

    # solid properties
    sim_info["nu"] = 0.4                                                    # Poisson's ratio
    #sim_info["nu"] = 0.1                                                    # Poisson's ratio

    sim_info["youngs mod"] = 3.3e1                                          # Young's modulus
    #sim_info["youngs mod"] = 2 * 1.0 * (1.0 + 0.1)

    sim_info["Eprime"] = sim_info["youngs mod"] / (1 - sim_info["nu"]**2)   # plain strain modulus

    # geometry
    sim_info["R"] = 41
    #sim_info["R"] = 2.

    # uniform load
    sim_info["p"] =1.10**12

    maxref = sim_info["max. n. of refinements"]

    results = {"nx" : [],
               "ny" : [],
               "Lx" : [],
               "Ly" : [],
               "r_": [],
               "rel e w R0": [],
               "rel e w R4": [],
               "rel e w R0 with tipcorr": [],
               "rel e w R4 with tipcorr": [],
               "abs e w R0": [],
               "abs e w R4": [],
               "abs e w R0 with tipcorr": [],
               "abs e w R4 with tipcorr": [],
               "mix e w R0": [],
               "mix e w R4": [],
               "mix e w R0 with tipcorr": [],
               "mix e w R4 with tipcorr": [],
               "max w R0" : [],
               "max w R4" : [],
               "max w R0 with tipcorr": [],
               "max w R4 with tipcorr": [],
               "frac volume R0": [],
               "frac volume R4": [],
               "frac volume R0 with tipcorr" : [ ],
               "frac volume R4 with tipcorr" : [ ],
               "frac volume R0 with ff": [],
               "frac volume R4 with ff": [],
               "frac volume R0 with tipcorr and ff": [],
               "frac volume R4 with tipcorr and ff": [],
               "KI R0": [],
               "KI R4": [],
               "KI R0 with tipcorr": [],
               "KI R4 with tipcorr": [],
               "KI R0 av": [],
               "KI R4 av": [],
               "KI R0 with tipcorr av": [],
               "KI R4 with tipcorr av": [],
               "n. of Elts" : [],
               "nu": sim_info["nu"],  # Poisson's ratio
               "youngs mod" : sim_info["youngs mod"],
               "Eprime": sim_info["Eprime"],
               "R": sim_info["R"],
               "p": sim_info["p"],
               "w_R0" : [],
               "w_R4" : [],
               "w_R0_tipcorr" : [],
               "w_R4_tipcorr" : [],
               "eltcrack" : []
                }


    # loop over the different meshes
    for refinement_ID in np.arange(1, maxref, 1):

        print(f"Step {refinement_ID} of {maxref} :")
        st = 0

        # creating mesh & plotting
        Mesh = get_mesh(sim_info, refinement_ID)

        # defining the geometry
        Fr_geometry = Geometry('radial', radius=sim_info["R"])
        surv_cells, surv_dist, inner_cells = get_survey_points(Fr_geometry, Mesh)
        EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, \
        l, alpha, FillF, sgndDist, Ffront, number_of_fronts, fronts_dictionary = generate_footprint(Mesh, surv_cells, inner_cells, surv_dist,
                                                                         'LS_continousfront')

        # A = np.zeros(Mesh.NumberOfElts)
        # A[EltRibbon] = 1
        # A[EltTip] = 2
        # plot_as_matrix(A, Mesh)
        if refinement_ID == 1 or  refinement_ID == maxref -1:
            plot_two_fronts(Mesh, newfront=Ffront, oldfront=None, fig=None, grid=True, cells=EltCrack)
        #   plot_two_fronts(Mesh, newfront=None, oldfront=None , fig=None, grid=True, cells = EltCrack, my_marker = " ")

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
        #rel_err_num = 100 * np.abs(sol_R0 - sol_R4) / sol_R4
        #plot_as_matrix(rel_err_num, mesh=Mesh) # 2D plot
        #plot_3d_scatter(sol_R4, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1]) # 3D plot

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

        all_w = np.zeros(Mesh.NumberOfElts)

        all_w[EltCrack] = sol_R0
        v_R0_ff = np.sum(all_w[EltTip] * FillF) * Mesh.hx * Mesh.hy + \
                  np.sum(all_w[EltChannel]) * Mesh.hx * Mesh.hy

        all_w[EltCrack] = sol_R4
        v_R4_ff = np.sum(all_w[EltTip] * FillF) * Mesh.hx * Mesh.hy + \
                  np.sum(all_w[EltChannel]) * Mesh.hx * Mesh.hy

        all_w[EltCrack] = sol_R0_tipcorr
        v_R0_tc_ff = np.sum(all_w[EltTip] * FillF) * Mesh.hx * Mesh.hy + \
                  np.sum(all_w[EltChannel]) * Mesh.hx * Mesh.hy

        all_w[EltCrack] = sol_R4_tipcorr
        v_R4_tc_ff = np.sum(all_w[EltTip] * FillF) * Mesh.hx * Mesh.hy + \
                  np.sum(all_w[EltChannel]) * Mesh.hx * Mesh.hy

        results["frac volume R0 with ff"].append(v_R0_ff)
        results["frac volume R4 with ff"].append(v_R4_ff)
        results["frac volume R0 with tipcorr and ff"].append(v_R0_tc_ff)
        results["frac volume R4 with tipcorr and ff"].append(v_R4_tc_ff)

        # > SIF estimation <

        KI_ana = KI_radial_solution(sim_info["p"], sim_info["R"])
        #
        all_w[EltCrack] = sol_R0
        KIPrime_R0 = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                           sgndDist,
                                           EltTip,
                                           EltRibbon,
                                           np.full(len(EltTip), True),
                                           Mesh,
                                           Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))

        relerr_KIPrime_R0 = 100. * (np.abs(KIPrime_R0 - KI_ana) / KI_ana).max()
        relerr_KIPrime_R0_av = 100. * np.mean(np.abs(KIPrime_R0 - KI_ana) / KI_ana)
        results["KI R0"].append(relerr_KIPrime_R0)
        results["KI R0 av"].append(relerr_KIPrime_R0_av)
        #
        all_w[EltCrack] = sol_R4
        KIPrime_R4 = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                           sgndDist,
                                           EltTip,
                                           EltRibbon,
                                           np.full(len(EltTip), True),
                                           Mesh,
                                           Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))
        relerr_KIPrime_R4 = 100 * (np.abs(KIPrime_R4 - KI_ana) / KI_ana).max()
        relerr_KIPrime_R4_av = 100 * np.mean(np.abs(KIPrime_R4 - KI_ana) / KI_ana)
        results["KI R4"].append(relerr_KIPrime_R4)
        results["KI R4 av"].append(relerr_KIPrime_R4_av)
        #
        all_w[EltCrack] = sol_R0_tipcorr
        KIPrime_R0_tipcorr = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                        sgndDist,
                                        EltTip,
                                        EltRibbon,
                                        np.full(len(EltTip),True),
                                        Mesh,
                                        Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))
        relerr_KIPrime_R0_tipcorr = 100 * (np.abs(KIPrime_R0_tipcorr - KI_ana) / KI_ana).max()
        relerr_KIPrime_R0_tipcorr_av = 100 * np.mean(np.abs(KIPrime_R0_tipcorr - KI_ana) / KI_ana)
        results["KI R0 with tipcorr"].append(relerr_KIPrime_R0_tipcorr)
        results["KI R0 with tipcorr av"].append(relerr_KIPrime_R0_tipcorr_av)
        #
        all_w[EltCrack] = sol_R4_tipcorr
        KIPrime_R4_tipcorr = np.sqrt(np.pi / 32.) * StressIntensityFactor(all_w,
                                        sgndDist,
                                        EltTip,
                                        EltRibbon,
                                        np.full(len(EltTip), True),
                                        Mesh,
                                        Eprime=np.full(Mesh.NumberOfElts,sim_info["Eprime"]))
        relerr_KIPrime_R4_tipcorr = 100 * (np.abs(KIPrime_R4_tipcorr - KI_ana) / KI_ana).max()
        relerr_KIPrime_R4_tipcorr_av = 100 * np.mean(np.abs(KIPrime_R4_tipcorr - KI_ana) / KI_ana)
        results["KI R4 with tipcorr"].append(relerr_KIPrime_R4_tipcorr)
        results["KI R4 with tipcorr av"].append(relerr_KIPrime_R4_tipcorr_av)

        # get the relative error w
        rel_e_R0, _ = get_err_w(sol_R0, sim_info, get_rel_err)
        rel_e_R4, _ = get_err_w(sol_R4, sim_info, get_rel_err)
        rel_e_R0_tipcorr, _ = get_err_w(sol_R0_tipcorr, sim_info, get_rel_err)
        rel_e_R4_tipcorr, r_ = get_err_w(sol_R4_tipcorr, sim_info, get_rel_err)

        results["rel e w R0"].append(rel_e_R0)
        results["rel e w R4"].append(rel_e_R4)
        results["rel e w R0 with tipcorr"].append(rel_e_R0_tipcorr)
        results["rel e w R4 with tipcorr"].append(rel_e_R4_tipcorr)
        results["r_"].append(r_)

        # get the abs error w
        abs_e_R0, _ = get_err_w(sol_R0, sim_info, get_abs_err)
        abs_e_R4, _ = get_err_w(sol_R4, sim_info, get_abs_err)
        abs_e_R0_tipcorr, _ = get_err_w(sol_R0_tipcorr, sim_info, get_abs_err)
        abs_e_R4_tipcorr, _ = get_err_w(sol_R4_tipcorr, sim_info, get_abs_err)

        results["abs e w R0"].append(abs_e_R0)
        results["abs e w R4"].append(abs_e_R4)
        results["abs e w R0 with tipcorr"].append(abs_e_R0_tipcorr)
        results["abs e w R4 with tipcorr"].append(abs_e_R4_tipcorr)

        # get the mixed error w
        mix_e_R0, _ = get_err_w(sol_R0, sim_info, get_mix_err)
        mix_e_R4, _ = get_err_w(sol_R4, sim_info, get_mix_err)
        mix_e_R0_tipcorr, _ = get_err_w(sol_R0_tipcorr, sim_info, get_mix_err)
        mix_e_R4_tipcorr, _ = get_err_w(sol_R4_tipcorr, sim_info, get_mix_err)

        results["mix e w R0"].append(mix_e_R0)
        results["mix e w R4"].append(mix_e_R4)
        results["mix e w R0 with tipcorr"].append(mix_e_R0_tipcorr)
        results["mix e w R4 with tipcorr"].append(mix_e_R4_tipcorr)

        # - store nonzero w and elements index
        results["w_R0"].append(sol_R0.tolist())
        results["w_R4"].append(sol_R4.tolist())
        results["w_R0_tipcorr"].append(sol_R0_tipcorr.tolist())
        results["w_R4_tipcorr"].append(sol_R4_tipcorr.tolist())
        results["eltcrack"].append(EltCrack.tolist())
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


if not run:
    with open(file_name, "r+") as json_file:
        results = json.load(json_file)[0]  # get the data

print("Plotting results")

sig_ahead_tip = False
if sig_ahead_tip:

    # ---- sigma ahead of the tip ----
    R_end = 1.25 * results["R"]
    n_of_obs = 50
    ampl_f = 1.007
    y_obs = np.arange(results["R"] * ampl_f,R_end,(R_end-results["R"] * ampl_f)/n_of_obs)
    x_obs = np.zeros(len(y_obs))

    xy_obs = np.vstack((x_obs, y_obs)).transpose()

    C_R4_tract = []
    C_R0_tract = []
    C_R4R0_tract = []

    C_R4_tract_err = []
    C_R0_tract_err = []
    C_R4R0_tract_err = []

    xy_obs_dim_hy = []
    for ind in [10,15,20]:#range(np.minimum(len(results["nx"]),21) ):
        # ---- mesh plot ----
        # creating mesh & plotting
        nx_coarse = results["nx"][ind]
        ny_coarse = results["ny"][ind]
        Lx_coarse = results["Lx"][ind]
        Ly_coarse = results["Ly"][ind]

        Mesh_ind = CartesianMesh(Lx_coarse, Ly_coarse, nx_coarse, ny_coarse)


        # defining the geometry
        Fr_geometry = Geometry('radial', radius=results["R"])
        surv_cells, surv_dist, inner_cells = get_survey_points(Fr_geometry, Mesh_ind)
        EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, \
        l, alpha, FillF, sgndDist, Ffront, number_of_fronts, fronts_dictionary = generate_footprint(Mesh_ind, surv_cells,
                                                                                                    inner_cells, surv_dist,
                                                                                                    'LS_continousfront')
        #plot_two_fronts(Mesh_ind, newfront=Ffront, oldfront=None, fig=None, grid=True, cells=EltCrack)

        xy_crack = Mesh_ind.CenterCoor[EltCrack,:]
        C_R4 = load_isotropic_elasticity_matrix_toepliz(Mesh_ind, results["Eprime"], Kernel='R4')
        C_R4_tract.append( - C_R4.get_normal_traction_at(xy_obs, xy_crack, results["w_R4"][ind]))
        C_R0 = load_isotropic_elasticity_matrix_toepliz(Mesh_ind, results["Eprime"], Kernel='R0')
        C_R0_tract.append( - C_R0.get_normal_traction_at(xy_obs, xy_crack, results["w_R0"][ind]))
        C_R4R0_tract.append( - C_R0.get_normal_traction_at(xy_obs, xy_crack, results["w_R4"][ind]))

        C_R4_tract_err.append(get_err_sig_zz(C_R4_tract[-1], xy_obs, results, get_rel_err))
        C_R0_tract_err.append(get_err_sig_zz(C_R0_tract[-1], xy_obs, results, get_rel_err))
        C_R4R0_tract_err.append(get_err_sig_zz(C_R4R0_tract[-1], xy_obs, results, get_rel_err))
        xy_obs_dim_hy.append(100. * (xy_obs - results["R"])/Mesh_ind.hy)
        print(f' Iter: {ind} of {len(results["nx"])}')

    fig1 = plt.figure()
    plt.suptitle('Radial crack test')

    paper_r4_res = np.asarray([
    [0.02092464962959268, 1.6763990168866765,],
    [0.04210679236660522, 1.121707248894925,],
    [0.0843774176225035, 0.6718308339876984,],
    [0.12589615140594718, 0.4699481186184915,],
    [0.16804310457815785, 0.3568512304574911,],
    [0.2511110189651793, 0.28049060556115446]])


    paper_ana_sig = np.asarray([
    [0.00774596101183192, 4.961567413899291],
    [0.008173521357429415, 4.559266330224471],
    [0.008458271618112204, 4.28794746676657],
    [0.009052129595585039, 4.007233584524432],
    [0.010712641171756021, 3.529934733616421],
    [0.01177755495216079, 3.3146409189024295],
    [0.013459814256998999, 3.0712026435959023],
    [0.01621568643370886, 2.706014783815978],
    [0.019444354231644055, 2.4249558376123197],
    [0.023910322805886147, 2.1156712392957684],
    [0.02899711654114845, 1.815659872482918],
    [0.034554966079342664, 1.5810685731982153],
    [0.04166487852008735, 1.369660352672767],
    [0.04707687329135345, 1.2333107930207774],
    [0.05295687918353806, 1.1296408203127264],
    [0.06069718578591651, 1.030404484555632],
    [0.06658023636011455, 0.9594749924317902],
    [0.07416642405867131, 0.8696133775788448],
    [0.08423199781014869, 0.7747497851324399],
    [0.09243901066972565, 0.6941614017831981],
    [0.0989450611778106, 0.6558911987232054],
    [0.10669102218964252, 0.6174586126224977],
    [0.11381746290417427, 0.5837844295541776],
    [0.12032394836683258, 0.5501914380062143],
    [0.13085927305752187, 0.5067162785518677],
    [0.13922605923037398, 0.47755692395486093],
    [0.14635249994490576, 0.4438827408865409],
    [0.1544101784009678, 0.4241184050737745],
    [0.16029801347547262, 0.4046382395822583],
    [0.16789072549262957, 0.3849347974097581],
    [0.18075262219613786, 0.35986401580218885],
    [0.1885007579808365, 0.3448174872616301],
    [0.19686797910826193, 0.3203353441766543],
    [0.2083362813587749, 0.3094788780259776],
    [0.217635175182304, 0.3035837937085839],
    [0.226934069005833, 0.2976887093911902],
    [0.23623339778393546, 0.29647083658582574],
    [0.2442919461491442, 0.286060923797117]])

    xy_obs_dim = (xy_obs - results["R"])/results["R"]
    plt.plot(xy_obs_dim[:,1], C_R0_tract[0]/results["p"], c='r', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R0_tract[1]/results["p"], c='b', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R0_tract[2]/results["p"], c='g', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R4_tract[0]/results["p"], c='r', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4_tract[1]/results["p"], c='b', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4_tract[2]/results["p"], c='g', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract[0]/results["p"], c='r', marker="x")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract[1]/results["p"], c='b', marker="x")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract[2]/results["p"], c='g', marker="x")
    sig_zz_ana = np.zeros(len(xy_obs_dim))
    for pnt in range(len(xy_obs_dim)):
        sig_zz_ana[pnt] = sig_zz_radial_solution(xy_obs[pnt,0], xy_obs[pnt,1], results["p"], results["R"])
    plt.plot(xy_obs_dim[:,1], sig_zz_ana/results["p"], c='black', marker=" ")
    plt.plot(paper_r4_res[:,0], paper_r4_res[:,1], c='black', marker="*")
    plt.plot(paper_ana_sig[:,0], paper_ana_sig[:,1], c='black', marker="o")


    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")
    plt.xlabel('(r-R)/R')
    plt.ylabel('sig z/p')
    plt.legend(('R0 - 1',
                'R0 - 2',
                'R0 - 3',
                'R4 - 1',
                'R4 - 2',
                'R4 - 3',
                'R4/R0 - 1',
                'R4/R0 - 2',
                'R4/R0 - 3','analytical'),loc='upper right', shadow=True)



    fig1 = plt.figure()
    plt.suptitle('Radial crack test')
    plt.plot(xy_obs_dim[:,1], C_R0_tract_err[0], c='r', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R0_tract_err[1], c='b', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R0_tract_err[2], c='g', marker="+")
    plt.plot(xy_obs_dim[:,1], C_R4_tract_err[0], c='r', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4_tract_err[1], c='b', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4_tract_err[2], c='g', marker="o")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract_err[0], c='r', marker="x")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract_err[1], c='b', marker="x")
    plt.plot(xy_obs_dim[:,1], C_R4R0_tract_err[2], c='g', marker="x")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")
    plt.xlabel('(r-R)/R')
    plt.ylabel('relative error sig_z')
    plt.legend(('R0 - 1',
                'R0 - 2',
                'R0 - 3',
                'R4 - 1',
                'R4 - 2',
                'R4 - 3',
                'R4/R0 - 1',
                'R4/R0 - 2',
                'R4/R0 - 3','analytical'),loc='upper right', shadow=True)


    fig1 = plt.figure()
    plt.suptitle('Radial crack test')
    plt.plot(xy_obs_dim_hy[0][:,1], C_R0_tract_err[0], c='r', marker="+")
    plt.plot(xy_obs_dim_hy[1][:,1], C_R0_tract_err[1], c='b', marker="+")
    plt.plot(xy_obs_dim_hy[2][:,1], C_R0_tract_err[2], c='g', marker="+")
    plt.plot(xy_obs_dim_hy[0][:,1], C_R4_tract_err[0], c='r', marker="o")
    plt.plot(xy_obs_dim_hy[1][:,1], C_R4_tract_err[1], c='b', marker="o")
    plt.plot(xy_obs_dim_hy[2][:,1], C_R4_tract_err[2], c='g', marker="o")
    plt.plot(xy_obs_dim_hy[0][:,1], C_R4R0_tract_err[0], c='r', marker="x")
    plt.plot(xy_obs_dim_hy[1][:,1], C_R4R0_tract_err[1], c='b', marker="x")
    plt.plot(xy_obs_dim_hy[2][:,1], C_R4R0_tract_err[2], c='g', marker="x")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")
    plt.xlabel('(r-R)/(elem_size)')
    plt.ylabel('relative error sig_z')
    plt.legend(('R0 - 1',
                'R0 - 2',
                'R0 - 3',
                'R4 - 1',
                'R4 - 2',
                'R4 - 3',
                'R4/R0 - 1',
                'R4/R0 - 2',
                'R4/R0 - 3','analytical'),loc='upper right', shadow=True)


# ---- w max -----
# fig1 = plt.figure()
# plt.suptitle('Radial crack test')
# plt.plot(results["n. of Elts"], results["max w R0"], c='r', marker="+")
# plt.plot(results["n. of Elts"], results["max w R4"], c='b', marker="+")
#
# y_ana = np.full(2,w_radial_solution(0.,0.,results["youngs mod"],results["nu"],results["p"],results["R"]))
# plt.plot([results["n. of Elts"][0],results["n. of Elts"][-1]], y_ana, c='black', marker=" ")
# plt.tick_params(labeltop=True, labelright=True)
# plt.grid(True, which="both", ls="-")
# plt.xlabel('# of DOF in the crack')
# plt.ylabel('w max')
# plt.legend(('R0 - no tip corr', 'R4 - no tip corr', 'analytical'),loc='upper center', shadow=True)
# plt.xscale('log')
# plt.yscale('log')




# ---- volume -----
# fig1 = plt.figure()
# plt.suptitle('Radial crack test')
# plt.plot(results["n. of Elts"], results["frac volume R0"], c='r', marker="+")
# plt.plot(results["n. of Elts"], results["frac volume R4"], c='b', marker="+")
#
# y_ana = np.full(2, Volume_radial_solution(results["youngs mod"],results["nu"],results["p"],results["R"]))
# plt.plot([results["n. of Elts"][0],results["n. of Elts"][-1]], y_ana, c='black', marker=" ")
# plt.tick_params(axis='both')
# plt.grid(True, which="both", ls="-")
# plt.xlabel('# of DOF')
# plt.ylabel('frac. volume')
# plt.legend(('R0 - no tip corr', 'R4 - no tip corr', 'analytical'),loc='upper center', shadow=True)
# plt.xscale('log')
# plt.yscale('log')



other_plots = False
if other_plots:
    # --------------------
    # volume - rel err
    # --------------------
    fig1 = plt.figure()
    plt.suptitle('Radial crack test')
    y_ana = Volume_radial_solution(results["youngs mod"],results["nu"],results["p"],results["R"])

    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R0"])-y_ana)/y_ana, c='r', marker="+")
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R4"])-y_ana)/y_ana, c='b', marker="+")
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R0 with tipcorr"])-y_ana)/y_ana, c='r', marker=".")
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R4 with tipcorr"])-y_ana)/y_ana, c='b', marker=".")

    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R0 with ff"])-y_ana)/y_ana, c='r', marker="+", linestyle='dashed')
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R4 with ff"])-y_ana)/y_ana, c='b', marker="+", linestyle='dashed')
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R0 with tipcorr and ff"])-y_ana)/y_ana, c='r', marker=".", linestyle='dashed')
    plt.plot(results["n. of Elts"], 100. * np.abs(np.asarray(results["frac volume R4 with tipcorr and ff"])-y_ana)/y_ana, c='b', marker=".", linestyle='dashed')

    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in the crack')
    plt.ylabel('rel. err. volume [%]')
    plt.legend(('R0 - NO tip corr',
                'R4 - NO tip corr',
                'R0 - tip corr as Ryder & Napier 1985',
                'R4 - tip corr as Ryder & Napier 1985',
                'R0 - NO tip corr - filling f',
                'R4 - NO tip corr - filling f',
                'R0 - tip corr as Ryder & Napier 1985 - filling f',
                'R4 - tip corr as Ryder & Napier 1985 - filling f'),loc='upper right', shadow=True)
    plt.xscale('log')
    plt.yscale('log')

    # --------------------
    # rel err w max
    # --------------------
    fig1 = plt.figure()
    plt.suptitle('Radial crack test')

    y_ana = w_radial_solution(0.,0.,results["youngs mod"],results["nu"],results["p"],results["R"])
    plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R0"]) - y_ana)/y_ana, c='r', marker="+")
    plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R4"]) - y_ana)/y_ana, c='b', marker="+")
    plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R0 with tipcorr"]) - y_ana)/y_ana, c='r', marker=".")
    plt.plot(results["n. of Elts"], 100 * np.abs(np.asarray(results["max w R4 with tipcorr"]) - y_ana)/y_ana, c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in the crack')
    plt.ylabel('rel. err. w max [%]')
    plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='lower left', shadow=True)
    plt.xscale('log')
    plt.yscale('log')

    # --------------------
    # rel err KI max
    # --------------------
    fig1 = plt.figure()
    plt.suptitle('Radial crack test')

    plt.plot(results["n. of Elts"], results["KI R0"], c='r', marker="+")
    plt.plot(results["n. of Elts"], results["KI R4"], c='b', marker="+")
    plt.plot(results["n. of Elts"], results["KI R0 with tipcorr"], c='r', marker=".")
    plt.plot(results["n. of Elts"], results["KI R4 with tipcorr"], c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in the crack')
    plt.ylabel('rel. err. KI max [%]')
    plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='lower left', shadow=True)
    plt.xscale('log')
    plt.yscale('log')

    # --------------------
    # rel err KI mean
    # --------------------
    fig1 = plt.figure()
    plt.suptitle('Radial crack test')

    plt.plot(results["n. of Elts"], results["KI R0 av"], c='r', marker="+")
    plt.plot(results["n. of Elts"], results["KI R4 av"], c='b', marker="+")
    plt.plot(results["n. of Elts"], results["KI R0 with tipcorr av"], c='r', marker=".")
    plt.plot(results["n. of Elts"], results["KI R4 with tipcorr av"], c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('# of DOF in the crack')
    plt.ylabel('<rel. err. KI> [%]')
    plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='lower left', shadow=True)
    plt.xscale('log')
    plt.yscale('log')

    # --------------------------
    # rel err w(r) (fine mesh)
    # --------------------
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks_y = np.arange(0, 101, 20)
    minor_ticks_y = np.arange(0, 101, 5)
    major_ticks_x = np.arange(0, 1.01, 0.2)
    minor_ticks_x = np.arange(0, 1.01, 0.05)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.suptitle('Radial crack test')
    # saving the solution
    indx = -1
    plt.scatter(results["r_"][indx], results["rel e w R0"][indx], c='r', marker="+")
    plt.scatter(results["r_"][indx], results["rel e w R4"][indx], c='g', marker="+")
    plt.scatter(results["r_"][indx], results["rel e w R0 with tipcorr"][indx], c='orange', marker=".")
    plt.scatter(results["r_"][indx], results["rel e w R4 with tipcorr"][indx], c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('r/R')
    plt.ylabel('rel. err. w [%]')
    plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='upper left', shadow=True)

    # --------------------------
    # rel err w(r) (coarse mesh)
    # --------------------
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks_y = np.arange(0, 101, 20)
    minor_ticks_y = np.arange(0, 101, 5)
    major_ticks_x = np.arange(0, 1.01, 0.2)
    minor_ticks_x = np.arange(0, 1.01, 0.05)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.suptitle('Radial crack test')
    # saving the solution
    indx = 0
    plt.scatter(results["r_"][indx], results["rel e w R0"][indx], c='r', marker="+")
    plt.scatter(results["r_"][indx], results["rel e w R4"][indx], c='g', marker="+")
    plt.scatter(results["r_"][indx], results["rel e w R0 with tipcorr"][indx], c='orange', marker=".")
    plt.scatter(results["r_"][indx], results["rel e w R4 with tipcorr"][indx], c='b', marker=".")
    plt.tick_params(labeltop=True, labelright=True)
    plt.grid(True, which="both", ls="-")

    plt.xlabel('r/R')
    plt.ylabel('rel. err. w [%]')
    plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='upper left', shadow=True)

# # --------------------------
# # abs err w(r) (fine mesh)
# fig1 = plt.figure()
# ax = fig1.add_subplot(1, 1, 1)
#
# # Major ticks every 20, minor ticks every 5
# major_ticks_y = np.arange(0, 5.01, 0.2)
# minor_ticks_y = np.arange(0, 5.01, 0.05)
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
# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
# plt.suptitle('Radial crack test')
# # saving the solution
# indx = -1
# plt.scatter(results["r_"][indx], results["abs e w R0"][indx], c='r', marker="+")
# plt.scatter(results["r_"][indx], results["abs e w R4"][indx], c='g', marker="+")
# plt.scatter(results["r_"][indx], results["abs e w R0 with tipcorr"][indx], c='orange', marker=".")
# plt.scatter(results["r_"][indx], results["abs e w R4 with tipcorr"][indx], c='b', marker=".")
# plt.tick_params(labeltop=True, labelright=True)
# plt.grid(True, which="both", ls="-")
#
# plt.xlabel('r/R')
# plt.ylabel('abs. err. w')
# plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='upper left', shadow=True)

# --------------------------
# mix err w(r)
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
# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
#
# plt.suptitle('Radial crack test')
# # saving the solution
# indx = -1
# plt.scatter(results["r_"][indx], results["mix e w R0"][indx], c='r', marker="+")
# plt.scatter(results["r_"][indx], results["mix e w R4"][indx], c='g', marker="+")
# plt.scatter(results["r_"][indx], results["mix e w R0 with tipcorr"][indx], c='orange', marker=".")
# plt.scatter(results["r_"][indx], results["mix e w R4 with tipcorr"][indx], c='b', marker=".")
# plt.tick_params(labeltop=True, labelright=True)
# plt.grid(True, which="both", ls="-")
#
# plt.xlabel('r/R')
# plt.ylabel('mix. err. w [% and abs]')
# plt.legend(('R0 - NO tip corr', 'R4 - NO tip corr','R0 - tip corr as Ryder & Napier 1985', 'R4 - tip corr as Ryder & Napier 1985', 'analytical'),loc='upper left', shadow=True)

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
