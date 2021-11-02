# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# External imports
import numpy as np
import logging
import time
from scipy.sparse.linalg import gmres



# Internal imports
from systems.sys_volume_and_load_control import MakeEquationSystem_volumeControl, MakeEquationSystem_volumeControl_double_fracture, \
    MakeEquationSystem_volumeControl_symmetric
from systems.sys_back_subst_EHL import MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse, \
    MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP, ADot, \
    MakeEquationSystem_ViscousFluid_pressure_substituted_sparse, MakeEquationSystem_ViscousFluid_pressure_substituted
from systems.explicit_RKL import solve_width_pressure_RKL2
from systems.systems_functions import velocity
from systems import Hdot

from mesh.symmetry import get_symetric_elements

from non_linear_solvers.picard_newton import Picard_Newton
from non_linear_solvers.anderson import Anderson

from properties import instrument_start, instrument_close


# ----------------------------------------------------------------------------------------------------------------------



def solve_width_pressure(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,Boundary,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon,
                         doublefracturedictionary = None):
    """
    This function evaluates the width and pressure by constructing and solving the coupled elasticity and fluid flow
    equations. The system of equations are formed according to the type of solver given in the simulation properties.
    """
    log = logging.getLogger('PyFrac.solve_width_pressure')
    if sim_properties.get_volumeControl():
        if sim_properties.volumeControlGMRES:

            #time_beg = time.time()
            # C is is the Hmat object
            D_i = np.reciprocal(C.diag_val)  # Only 1 value of the elasticity matrix
            S_i = -np.reciprocal(Fr_lstTmStp.EltChannel.size * D_i)  # Inverse Schur complement

            counter = Hdot.gmres_counter()  # to obtain the number of iteration and residual

            total_vol = (sum(Fr_lstTmStp.w)+ sum(Qin[EltCrack]) * (timeStep) / Fr_lstTmStp.mesh.EltArea)  # - something

            # building the right hand side of the system premultiplied by a left preconditioner
            C._set_domain_IDX(EltTip)
            C._set_codomain_IDX(Fr_lstTmStp.EltChannel)
            if wTip.size == 0:
                g1 = D_i * (mat_properties.SigmaO[Fr_lstTmStp.EltChannel]) + D_i * S_i * (total_vol) * np.ones(Fr_lstTmStp.EltChannel.size)  # D_e^-1 * sigma - vol_incr * S^-1 * D_e^-1 *[1...1](vertical)
                g2 = S_i * (total_vol)  # S^-1 * vol_incr --> change
            else:
                g1 = D_i * (mat_properties.SigmaO[Fr_lstTmStp.EltChannel] - C._matvec(wTip)) + D_i * S_i * (total_vol - np.sum(wTip)) * np.ones(Fr_lstTmStp.EltChannel.size)  # D_e^-1 * sigma - vol_incr * S^-1 * D_e^-1 *[1...1](vertical)
                g2 = S_i * (total_vol - np.sum(wTip))  # S^-1 * vol_incr --> change
            rhs_prec = np.concatenate((g1, np.asarray([g2])))  # preconditionned b (Ax=b)

            # solving the system using a left preconditioner
            data = C, Fr_lstTmStp.EltChannel, D_i, S_i
            system_dot_prod = Hdot.Volume_Control(data)
            #begtime_gmres=time.time()
            sol_GMRES = gmres(system_dot_prod, rhs_prec, tol=sim_properties.gmres_tol,
                              maxiter=sim_properties.gmres_maxiter, callback=counter)
            #endtime_gmres=time.time()
            # check convergence
            # todo assess the convergence against the true residual (not the one with respect to the preconditioned rhs)
            if sol_GMRES[1] > 0:
                log.warning(
                    "Volume Control system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
                rel_err = np.linalg.norm(system_dot_prod._matvec(sol_GMRES[0]) - (rhs_prec)) / np.linalg.norm(
                    rhs_prec)
                log.warning("         error of the solution: " + str(rel_err))
            elif sol_GMRES[1] == 0:
                rel_err = np.linalg.norm(system_dot_prod._matvec(sol_GMRES[0]) - (rhs_prec)) / np.linalg.norm(
                    rhs_prec)
                log.debug(
                    " --> GMRES Volume Control converged after " + str(counter.niter) + " iter. & rel err is " + str(
                        rel_err))

            # update the solution vectors w and p
            sol = sol_GMRES[0]
            w = np.copy(Fr_lstTmStp.w)
            w[Fr_lstTmStp.EltChannel] = sol[np.arange(Fr_lstTmStp.EltChannel.size)]
            w[EltTip] = wTip

            # from utility import plot_as_matrix
            # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            # K[Fr_lstTmStp.EltChannel] = sol[np.arange(Fr_lstTmStp.EltChannel.size)]
            # plot_as_matrix(K, Fr_lstTmStp.mesh)

            p = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
            p[EltCrack] = sol[-1]

            return_data_solve = [None, None, None]
            return_data = [return_data_solve, np.asarray([]), False]

            #compute_time = time.time()-time_beg
            #append_new_line('./Data/radial_VC_gmres/timing.txt', str(Fr_lstTmStp.EltChannel.size)+"  "+str(compute_time))
            #compute_time_gmres=endtime_gmres-begtime_gmres
            #append_new_line('./Data/radial_VC_gmres/timing_gmres.txt',
            #                str(Fr_lstTmStp.EltChannel.size) + "  " + str(compute_time_gmres) )
            return w, p, return_data

        else:
            if sim_properties.symmetric and not sim_properties.useBlockToeplizCompression:
                try:
                    Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel]
                except AttributeError:
                    raise SystemExit("Symmetric fracture needs symmetric mesh. Set symmetric flag to True\n"
                                     "while initializing the mesh")

                EltChannel_sym = Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel]
                EltChannel_sym = np.unique(EltChannel_sym)

                EltTip_sym = Fr_lstTmStp.mesh.corresponding[EltTip]
                EltTip_sym = np.unique(EltTip_sym)

                # todo: make tip correction while injecting in the same footprint
                ##########################################################################################
                #                                                                                        #
                #  when we inject on the same footprint we should make the tip correction at the tip     #
                #  this is not done, but it will not affect the accuracy, only the speed of convergence  #
                #  since we are never accessing the diagonal of the elasticity matrix when iterating on  #
                #  the position of the front.                                                            #
                #                                                                                        #
                ##########################################################################################

                # CARLO: we can remove it because the diagonal terms of C are never accessed
                # FillF_mesh = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
                # FillF_mesh[EltTip] = FillFrac
                # FillF_sym = FillF_mesh[Fr_lstTmStp.mesh.activeSymtrc[EltTip_sym]]
                # partlyFilledTip_sym = np.where(FillF_sym <= 1)[0]

                # C_EltTip = np.copy(C[np.ix_(EltTip_sym[partlyFilledTip_sym],
                #                             EltTip_sym[
                #                                 partlyFilledTip_sym])])  # keeping the tip element entries to restore current
                #
                # # filling fraction correction for element in the tip region
                # FillF = FillF_sym[partlyFilledTip_sym]
                # for e in range(len(partlyFilledTip_sym)):
                #     r = FillF[e] - .25
                #     if r < 0.1:
                #         r = 0.1
                #     ac = (1 - r) / r
                #     self_infl = self_influence(Fr_lstTmStp.mesh, mat_properties.Eprime)
                #     C[EltTip_sym[partlyFilledTip_sym[e]], EltTip_sym[partlyFilledTip_sym[e]]] += \
                #         ac * np.pi / 4. * self_infl

                wTip_sym = np.zeros((len(EltTip_sym),), dtype=np.float64)
                wTip_sym_elts = Fr_lstTmStp.mesh.activeSymtrc[EltTip_sym]
                for i in range(len(EltTip_sym)):
                    if len(np.where(EltTip == wTip_sym_elts[i])[0]) != 1:
                        other_corr = get_symetric_elements(Fr_lstTmStp.mesh, [wTip_sym_elts[i]])
                        for j in range(4):
                            in_tip = np.where(EltTip == other_corr[0][j])[0]
                            if len(in_tip) > 0:
                                wTip_sym[i] = wTip[in_tip]
                                break
                    else:
                        wTip_sym[i] = wTip[np.where(EltTip == wTip_sym_elts[i])[0]]

                dwTip = wTip - Fr_lstTmStp.w[EltTip]
                A, b = MakeEquationSystem_volumeControl_symmetric(Fr_lstTmStp.w,
                                                                  wTip_sym,
                                                                  EltChannel_sym,
                                                                  EltTip_sym,
                                                                  C,
                                                                  timeStep,
                                                                  Qin,
                                                                  mat_properties.SigmaO,
                                                                  Fr_lstTmStp.mesh.EltArea,
                                                                  LkOff,
                                                                  Fr_lstTmStp.mesh.volWeights,
                                                                  Fr_lstTmStp.mesh.activeSymtrc,
                                                                  dwTip)
                # CARLO: we can remove it because the diagonal terms of C are never accessed
                # C[np.ix_(EltTip_sym[partlyFilledTip_sym], EltTip_sym[partlyFilledTip_sym])] = C_EltTip
            else:
                # todo: make tip correction while injecting in the same footprint
                ##########################################################################################
                #                                                                                        #
                #  when we inject on the same footprint we should make the tip correction at the tip     #
                #  this is not done, but it will not affect the accuracy, only the speed of convergence  #
                #  since we are never accessing the diagonal of the elasticity matrix when iterating on  #
                #  the position of the front.                                                            #
                #                                                                                        #
                ##########################################################################################
                # CARLO: we can remove it because the diagonal terms of C are never accessed
                # C_EltTip = np.copy(C[np.ix_(EltTip[partlyFilledTip],
                #                             EltTip[partlyFilledTip])])  # keeping the tip element entries to restore current
                # #  tip correction. This is done to avoid copying the full elasticity matrix.
                #
                # # filling fraction correction for element in the tip region
                # FillF = FillFrac[partlyFilledTip]
                # for e in range(0, len(partlyFilledTip)):
                #     r = FillF[e] - .25
                #     if r < 0.1:
                #         r = 0.1
                #     ac = (1 - r) / r
                #     C[EltTip[partlyFilledTip[e]], EltTip[partlyFilledTip[e]]] *= (1. + ac * np.pi / 4.)

                if sim_properties.doublefracture and doublefracturedictionary['number_of_fronts']==2:
                    #compute the channel from the last time step for the two fractures
                    EltChannelFracture0 = np.setdiff1d(Fr_lstTmStp.fronts_dictionary['crackcells_0'],Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_0'])
                    EltChannelFracture1 = np.setdiff1d(Fr_lstTmStp.fronts_dictionary['crackcells_1'],Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_1'])
                    # from utility import plot_as_matrix
                    # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
                    # K[EltChannelFracture1] = 1
                    # K[EltChannelFracture0] = 2
                    # K[Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_1']] = 3
                    # K[Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_0']] = 4
                    # plot_as_matrix(K, Fr_lstTmStp.mesh)
                    if EltTip.size == 0 :
                        EltTipFracture0 = EltTip
                        EltTipFracture1 = EltTip
                    else:
                        EltTipFracture0 = doublefracturedictionary['TIPcellsANDfullytrav_0']
                        EltTipFracture1 = doublefracturedictionary['TIPcellsANDfullytrav_1']
                    wtipindexFR0 = np.where(np.in1d(EltTip, EltTipFracture0))[0]
                    wtipindexFR1 = np.where(np.in1d(EltTip, EltTipFracture1))[0]
                    wTipFR0 = wTip[wtipindexFR0]
                    wTipFR1 = wTip[wtipindexFR1]
                    QinFR0=Qin[EltChannelFracture0]
                    QinFR1=Qin[EltChannelFracture1]

                    # CARLO: I check if can be possible to have a Channel to be tip
                    if np.any(np.isin(np.concatenate((EltChannelFracture0, EltChannelFracture1)),np.concatenate((EltTipFracture0, EltTipFracture1)),assume_unique=True)):
                        SystemExit("Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")

                    A, b = MakeEquationSystem_volumeControl_double_fracture(Fr_lstTmStp.w,
                                                                            wTipFR0,
                                                                            wTipFR1,
                                                                            EltChannelFracture0,
                                                                            EltChannelFracture1,
                                                                            EltTipFracture0,
                                                                            EltTipFracture1,
                                                                            mat_properties.SigmaO,
                                                                            C,
                                                                            timeStep,
                                                                            QinFR0,
                                                                            QinFR1,
                                                                            Fr_lstTmStp.mesh.EltArea,
                                                                            LkOff)
                else:
                    # CARLO: I check if can be possible to have a Channel to be tip
                    if np.any(np.isin(Fr_lstTmStp.EltChannel,EltTip,assume_unique=True)):
                        SystemExit("Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")
                    #time_beg=time.time()
                    A, b = MakeEquationSystem_volumeControl(Fr_lstTmStp.w,
                                                        wTip,
                                                        Fr_lstTmStp.EltChannel,
                                                        EltTip,
                                                        mat_properties.SigmaO,
                                                        C,
                                                        timeStep,
                                                        Qin,
                                                        Fr_lstTmStp.mesh.EltArea,
                                                        LkOff)
                # CARLO: we can remove it because the diagonal terms of C are never accessed
                # # regain original C (without filling fraction correction)
                # C[np.ix_(EltTip[partlyFilledTip], EltTip[partlyFilledTip])] = C_EltTip

            perfNode_nonLinSys = instrument_start('nonlinear system solve', perfNode)
            perfNode_widthConstrItr = instrument_start('width constraint iteration', perfNode_nonLinSys)
            perfNode_linSys = instrument_start('linear system solve', perfNode_widthConstrItr)
            status = True
            fail_cause = None
            #begtime_sol=time.time()
            try:
                sol = np.linalg.solve(A, b)

            except np.linalg.linalg.LinAlgError:
                status = False
                fail_cause = 'sigular matrix'
            #fintime_sol=time.time()
            #compute_time_sol=fintime_sol-begtime_sol
            #compute_time = fintime_sol - time_beg
            # append_new_line('./Data/radial_VC_gmres/timing_direct.txt',
            #                  str(Fr_lstTmStp.EltChannel.size)+ "  " + str(compute_time))
            # append_new_line('./Data/radial_VC_gmres/timing_direct_sol.txt',
            #                 str(Fr_lstTmStp.EltChannel.size) + "  " + str(compute_time_sol))
            if perfNode is not None:
                instrument_close(perfNode_widthConstrItr, perfNode_linSys, None,
                                 len(b), status, fail_cause, Fr_lstTmStp.time)
                perfNode_widthConstrItr.linearSolve_data.append(perfNode_linSys)

                instrument_close(perfNode_nonLinSys, perfNode_widthConstrItr, None,
                                 len(b), status, fail_cause, Fr_lstTmStp.time)
                perfNode_nonLinSys.widthConstraintItr_data.append(perfNode_widthConstrItr)

                instrument_close(perfNode, perfNode_nonLinSys, None, len(b), status, fail_cause, Fr_lstTmStp.time)
                perfNode.nonLinSolve_data.append(perfNode_nonLinSys)

            # equate other three quadrants to the evaluated quadrant
            if sim_properties.symmetric:
                del_w = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
                for i in range(len(sol) - 1):
                    del_w[Fr_lstTmStp.mesh.symmetricElts[Fr_lstTmStp.mesh.activeSymtrc[EltChannel_sym[i]]]] = sol[i]
                w = np.copy(Fr_lstTmStp.w)
                w[Fr_lstTmStp.EltChannel] += del_w[Fr_lstTmStp.EltChannel]
                for i in range(len(wTip_sym_elts)):
                    w[Fr_lstTmStp.mesh.symmetricElts[wTip_sym_elts[i]]] = wTip_sym[i]
            else:
                w = np.copy(Fr_lstTmStp.w)
                if sim_properties.doublefracture and doublefracturedictionary['number_of_fronts'] == 2:
                    w[EltChannelFracture0] += sol[np.arange(EltChannelFracture0.size)]
                    w[EltChannelFracture1] += sol[np.arange(EltChannelFracture0.size,EltChannelFracture0.size+EltChannelFracture1.size)]
                    w[EltTipFracture0] = wTipFR0
                    w[EltTipFracture1] = wTipFR1
                else:
                    w[Fr_lstTmStp.EltChannel] += sol[np.arange(Fr_lstTmStp.EltChannel.size)]
                    w[EltTip] = wTip

            p = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
            if sim_properties.doublefracture and doublefracturedictionary['number_of_fronts'] == 2:
                p[doublefracturedictionary['crackcells_0']] = sol[-2]
                p[doublefracturedictionary['crackcells_1']] = sol[-1]
            else:
                p[EltCrack] = sol[-1]

            p[EltCrack] = sol[-1]

            return_data_solve = [None, None, None]
            return_data = [return_data_solve, np.asarray([]), False]

            return w, p, return_data

    if sim_properties.get_viscousInjection():

        # velocity at the cell edges evaluated with the guess width. Used as guess
        # values for the implicit velocity solver.
        vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
        if fluid_properties.turbulence:
            wguess = np.copy(Fr_lstTmStp.w)
            wguess[EltTip] = wTip

            vk = velocity(wguess,
                          EltCrack,
                          Fr_lstTmStp.mesh,
                          InCrack,
                          Fr_lstTmStp.muPrime,
                          C,
                          mat_properties.SigmaO)

        perfNode_nonLinSys = instrument_start('nonlinear system solve', perfNode)

        neg = np.array([], dtype=int)
        new_neg = np.array([], dtype=int)
        active_contraint = True
        to_solve = np.setdiff1d(EltCrack, EltTip)  # only taking channel elements to solve

        # adding stagnant tip cells to the cells which are solved. This adds stability as the elasticity is also
        # solved for the stagnant tip cells as compared to tip cells which are moving.
        if sim_properties.solveStagnantTip:
            stagnant_tip = np.where(Vel < 1e-10)[0]
        else:
            stagnant_tip = []
        to_impose = np.delete(EltTip, stagnant_tip)
        imposed_val = np.delete(wTip, stagnant_tip)
        to_solve = np.append(to_solve, EltTip[stagnant_tip])

        wc_to_impose = []
        fully_closed = False
        corr_ribb_flag = False
        # Making and solving the system of equations. The width constraint is checked. If active, system is remade with
        # the constraint imposed and is resolved.

        while active_contraint:

            perfNode_widthConstrItr = instrument_start('width constraint iteration', perfNode_nonLinSys)

            to_solve_k = np.setdiff1d(to_solve, neg, assume_unique=True)
            to_impose_k = to_impose
            imposed_val_k = imposed_val
            # the code below finds the tip cells with corresponding closed ribbon cells and add them in the list
            # of elements to be solved.
            if len(neg) > 0 and len(to_impose) > 0:
                if sim_properties.solveTipCorrRib and corr_ribbon.size != 0:
                    if not corr_ribb_flag:
                        # do it once
                        tip_sorted = np.argsort(EltTip)
                        to_impose_pstn = np.searchsorted(EltTip[tip_sorted], to_impose)
                        ind_toImps_tip = tip_sorted[to_impose_pstn]
                        corr_ribbon_TI = corr_ribbon[ind_toImps_tip]
                        corr_ribb_flag = True

                    toImp_neg_rib = np.asarray([], dtype=np.int)
                    for i, elem in enumerate(to_impose):
                        if corr_ribbon_TI[i] in neg:
                            toImp_neg_rib = np.append(toImp_neg_rib, i)
                    to_solve_k = np.append(to_solve_k, np.setdiff1d(to_impose[toImp_neg_rib], neg))
                    to_impose_k = np.delete(to_impose, toImp_neg_rib)
                    imposed_val_k = np.delete(imposed_val, toImp_neg_rib)

            EltCrack_k = np.concatenate((to_solve_k, neg))
            EltCrack_k = np.concatenate((EltCrack_k, to_impose_k))

            # The code below finds the indices(in the EltCrack list) of the neighbours of all the cells in the crack.
            # This is done to avoid costly slicing of the large numpy arrays while making the linear system during the
            # fixed point iterations. For neighbors that are outside the fracture, len(EltCrack) + 1 is returned.
            corr_nei = np.full((len(EltCrack_k), 4), len(EltCrack_k), dtype=np.int)
            for i, elem in enumerate(EltCrack_k):
                corresponding = np.where(EltCrack_k == Fr_lstTmStp.mesh.NeiElements[elem, 0])[0]
                if len(corresponding) > 0:
                    corr_nei[i, 0] = corresponding
                corresponding = np.where(EltCrack_k == Fr_lstTmStp.mesh.NeiElements[elem, 1])[0]
                if len(corresponding) > 0:
                    corr_nei[i, 1] = corresponding
                corresponding = np.where(EltCrack_k == Fr_lstTmStp.mesh.NeiElements[elem, 2])[0]
                if len(corresponding) > 0:
                    corr_nei[i, 2] = corresponding
                corresponding = np.where(EltCrack_k == Fr_lstTmStp.mesh.NeiElements[elem, 3])[0]
                if len(corresponding) > 0:
                    corr_nei[i, 3] = corresponding

            lst_edgeInCrk = None
            if fluid_properties.rheology in ["Herschel-Bulkley", "HBF", 'power law', 'PLF']:
                lst_edgeInCrk = [np.where(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 0]])[0],
                              np.where(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 1]])[0],
                              np.where(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 2]])[0],
                              np.where(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 3]])[0]]

            arg = (
                EltCrack_k,
                to_solve_k,
                to_impose_k,
                imposed_val_k,
                wc_to_impose,
                Fr_lstTmStp,
                fluid_properties,
                mat_properties,
                sim_properties,
                timeStep,
                Qin,
                C,
                Boundary,
                InCrack,
                LkOff,
                neg,
                corr_nei,
                lst_edgeInCrk)

            w_guess = np.zeros(Fr_lstTmStp.mesh.NumberOfElts, dtype=np.float64)
            avg_dw = (sum(Qin) * timeStep / Fr_lstTmStp.mesh.EltArea - sum(
                    imposed_val_k - Fr_lstTmStp.w[to_impose_k])) / len(to_solve_k)
            w_guess[to_solve_k] = Fr_lstTmStp.w[to_solve_k] #+ avg_dw
            w_guess[to_impose_k] = imposed_val_k
            if Boundary is not None:
                traction_guess =  Boundary.getTraction(w_guess, EltCrack)
                pf_guess_neg = np.dot(C[np.ix_(neg, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[neg]  + traction_guess[neg]
                pf_guess_tip = np.dot(C[np.ix_(to_impose_k, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[to_impose_k] + traction_guess[to_impose_k]
            else:
                pf_guess_neg = np.dot(C[np.ix_(neg, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[neg]
                pf_guess_tip = np.dot(C[np.ix_(to_impose_k, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[to_impose_k]
            if sim_properties.elastohydrSolver == 'implicit_Picard' or sim_properties.elastohydrSolver == 'implicit_Anderson':
                if sim_properties.solveDeltaP:
                    sys_size = len(to_solve_k) + len(pf_guess_neg) + len(pf_guess_tip)
                    if sim_properties.solveSparse:
                        if sim_properties.EHL_GMRES:
                            sys_fun = ADot(sys_size, dtype=np.float64)
                        else:
                            sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse
                    else:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP
                    #guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
                    guess = np.concatenate((np.full(len(to_solve_k), 0., dtype=np.float64),
                                            pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                                            pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k]))
                else:
                    if sim_properties.solveSparse:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_sparse
                    else:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted
                    #guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
                    guess = np.concatenate((np.full(len(to_solve_k), 0., dtype=np.float64),
                                            pf_guess_neg,
                                            pf_guess_tip))

                # guess = 1e5 * np.ones((len(EltCrack), ), float)
                # guess[np.arange(len(to_solve_k))] = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size \
                                                        # * np.ones((len(to_solve_k),), float)

                inter_itr_init = [vk, np.array([], dtype=int), None]

                if sim_properties.elastohydrSolver == 'implicit_Picard':

                    typValue = np.copy(guess)

                    sol, data_nonLinSolve = Picard_Newton(None,
                                           sys_fun,
                                           guess,
                                           typValue,
                                           inter_itr_init,
                                           sim_properties,
                                           *arg,
                                           perf_node=perfNode_widthConstrItr)
                elif sim_properties.elastohydrSolver == 'implicit_Anderson':
                    #Ander_time = -time.time()
                    sol, data_nonLinSolve = Anderson(sys_fun,
                                             guess,
                                             inter_itr_init,
                                             sim_properties,
                                             *arg,
                                             perf_node=perfNode_widthConstrItr)
                    #Ander_time = Ander_time + time.time()
                    #file_name = '/Users/carloperuzzo/Desktop/Pyfrac_formulation/_gmres_dev/_preconditioner/_data&performances/Gmres_with_random_E/iterT_25.csv'
                    #append_new_line(file_name, str(sys_size)+','+str(Ander_time))
                elif sim_properties.elastohydrSolver == 'JacobianFreeNewton':
                    log.error("NOT YET IMPLEMENTED!")
                    # another option is scipy.optimize.newton_krylov
            elif sim_properties.elastohydrSolver == 'RKL2':
                sol, data_nonLinSolve = solve_width_pressure_RKL2(mat_properties.Eprime,
                                                          sim_properties.enableGPU,
                                                          sim_properties.nThreads,
                                                          perfNode_widthConstrItr,
                                                          *arg)
            else:
                raise SystemExit("The given elasto-hydrodynamic solver is not supported!")


            failed_sol = np.isnan(sol).any()

            if perfNode_widthConstrItr is not None:
                fail_cause = None
                norm = None
                if len(neg) > 0:
                    norm = len(new_neg) / len(neg)
                if failed_sol:
                    if len(perfNode_widthConstrItr.linearSolve_data) >= sim_properties.maxSolverItrs:
                        fail_cause = 'did not converge after max iterations'
                    else:
                        fail_cause = 'singular matrix'

                instrument_close(perfNode_nonLinSys, perfNode_widthConstrItr, norm, len(sol),
                                 not failed_sol, fail_cause, Fr_lstTmStp.time)
                perfNode_nonLinSys.widthConstraintItr_data.append(perfNode_widthConstrItr)

            if failed_sol:
                if perfNode_nonLinSys is not None:
                    instrument_close(perfNode, perfNode_nonLinSys, None, len(sol), not failed_sol,
                                     fail_cause, Fr_lstTmStp.time)
                    perfNode.nonLinSolve_data.append(perfNode_nonLinSys)
                return np.nan, np.nan, (np.nan, np.nan)

            w = np.copy(Fr_lstTmStp.w)
            w[to_solve_k] += sol[:len(to_solve_k)]
            w[to_impose_k] = imposed_val_k
            w[neg] = wc_to_impose

            neg_km1 = np.copy(neg)
            wc_km1 = np.copy(wc_to_impose)
            below_wc_k = np.where(w[to_solve_k] < mat_properties.wc)[0]
            if len(below_wc_k) > 0:
                # for cells where max width in w history is greater than wc
                wHst_above_wc = np.where(Fr_lstTmStp.wHist[to_solve_k] >= mat_properties.wc)[0]
                impose_wc_at = np.intersect1d(wHst_above_wc, below_wc_k)

                # for cells with max width in w history less than wc
                wHst_below_wc = np.where(Fr_lstTmStp.wHist[to_solve_k] < mat_properties.wc)[0]
                dwdt_neg = np.where(w[to_solve_k] <= Fr_lstTmStp.w[to_solve_k])[0]
                impose_wHist_at = np.intersect1d(wHst_below_wc, dwdt_neg)

                neg_k = to_solve_k[np.concatenate((impose_wc_at, impose_wHist_at))]
                # the corresponding values of width to be imposed in cells where width constraint is active
                wc_k = np.full((len(impose_wc_at) + len(impose_wHist_at),), mat_properties.wc, dtype=np.float64)
                wc_k[len(impose_wc_at):] = Fr_lstTmStp.wHist[to_solve_k[impose_wHist_at]]

                new_neg = np.setdiff1d(neg_k, neg)
                if len(new_neg) == 0:
                    active_contraint = False
                else:
                    # if sim_properties.frontAdvancing is not 'implicit':
                    #     log.warning('Changing front advancing scheme to implicit due to width going negative...')
                    #     sim_properties.frontAdvancing = 'implicit'
                    #     return np.nan, np.nan, (np.nan, np.nan)

                    # cumulatively add the cells with active width constraint
                    neg = np.hstack((neg_km1, new_neg))
                    new_wc = []
                    for i in new_neg:
                        new_wc.append(wc_k[np.where(neg_k == i)[0]][0])
                    wc_to_impose = np.hstack((wc_km1, np.asarray(new_wc)))
                    log.debug('Iterating on cells with active width constraint...')
            else:
                active_contraint = False


        pf = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
        # pressure evaluated by dot product of width and elasticity matrix
        if Boundary is not None:
            pf[to_solve_k] = np.dot(C[np.ix_(to_solve_k, EltCrack)], w[EltCrack]) +  mat_properties.SigmaO[to_solve_k] + Boundary.last_traction[to_solve_k]
        else:
            pf[to_solve_k] = np.dot(C[np.ix_(to_solve_k, EltCrack)], w[EltCrack]) +  mat_properties.SigmaO[to_solve_k]
        if sim_properties.solveDeltaP:
            pf[neg_km1] = Fr_lstTmStp.pFluid[neg_km1] + sol[len(to_solve_k):len(to_solve_k) + len(neg_km1)]
            pf[to_impose_k] = Fr_lstTmStp.pFluid[to_impose_k] + sol[len(to_solve_k) + len(neg_km1):]
        else:
            pf[neg_km1] = sol[len(to_solve_k):len(to_solve_k) + len(neg_km1)]
            pf[to_impose_k] = sol[len(to_solve_k) + len(neg_km1):]


        if perfNode_nonLinSys is not None:
            instrument_close(perfNode, perfNode_nonLinSys, None, len(sol), True, None, Fr_lstTmStp.time)
            perfNode.nonLinSolve_data.append(perfNode_nonLinSys)

        if len(neg) == len(to_solve):
            fully_closed = True

        return_data = [data_nonLinSolve, neg_km1, fully_closed]
        return w, pf, return_data