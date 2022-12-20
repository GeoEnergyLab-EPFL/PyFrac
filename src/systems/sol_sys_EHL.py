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

from systems.make_sys_back_subst_EHL import MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse, \
    MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP, EHL_sys_obj, \
    MakeEquationSystem_ViscousFluid_pressure_substituted_sparse, MakeEquationSystem_ViscousFluid_pressure_substituted, \
    MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse_injection_line
from systems.explicit_RKL import solve_width_pressure_RKL2
from systems.make_sys_common_fun import velocity
from systems.make_sys_monolithic_EHL import Monolithic_EHL_sys_obj, MakeEquationSystem_Monolithic_precond
from non_linear_solvers.picard_newton import Picard_Newton
from non_linear_solvers.anderson import Anderson
from properties import instrument_start, instrument_close
from linear_solvers.linear_direct_solver import Direct_linear_solver
from linear_solvers.linear_iterative_solver import Iterative_linear_solver
from linear_solvers.preconditioners.prec_back_subst_EHL import EHL_iLU_Prec


def sol_sys_EHL(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, inj_properties, EltTip, partlyFilledTip, C, Boundary,
                FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon, stagnant_tip,
                doublefracturedictionary = None, inj_same_footprint = False):

        log = logging.getLogger('PyFrac.solve_width_pressure.sol_sys_EHL')


        # velocity at the cell edges evaluated with the guess width. Used as guess
        # values for the implicit velocity solver.

        if fluid_properties.turbulence:
            vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
            wguess = np.copy(Fr_lstTmStp.w)
            if not inj_same_footprint: wguess[EltTip] = wTip

            vk = velocity(wguess,
                          EltCrack,
                          Fr_lstTmStp.mesh,
                          InCrack,
                          Fr_lstTmStp.muPrime,
                          C,
                          mat_properties.SigmaO)
        else:
            vk = None


        perfNode_nonLinSys = instrument_start('nonlinear system solve', perfNode)

        # ---- the following has been taken out because it seems to be unphysical ---
        # adding stagnant tip cells to the cells which are solved. This adds stability as the elasticity is also
        # solved for the stagnant tip cells as compared to tip cells which are moving.
        #if sim_properties.solveStagnantTip:
        #    stagnant_tip = np.where(Vel < 1e-10)[0]
        #else:
        #    stagnant_tip = []
        # ---------------------------------------

        # Initialising the arrays
        neg = np.array([], dtype=int)
        new_neg = np.array([], dtype=int)
        active_contraint = True

        if inj_same_footprint:
            to_solve = EltCrack
            to_impose = np.array([], dtype=int)
            imposed_val = np.array([], dtype=int)
        else:
            to_solve = np.setdiff1d(EltCrack, EltTip)  # only taking channel elements to solve

            # adding stagnant tip cells to the cells which are solved. This adds stability as the elasticity is also
            # solved for the stagnant tip cells as compared to tip cells which are moving.
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

            # define the strategy for the current loop, remove the cells with negative opening
            to_solve_k = np.setdiff1d(to_solve, neg, assume_unique=True)
            to_impose_k = to_impose
            imposed_val_k = imposed_val

            # ------
            # If a ribbon cell gets negative, then solve for opening in the corresponding tip if the latter was not
            # already in the list
            #    (the code below finds the tip cells with corresponding closed ribbon cells and add them in the list
            #    of elements to be solved.)
            if len(neg) > 0 and len(to_impose) > 0:
                if sim_properties.solveTipCorrRib and corr_ribbon.size != 0:
                    if not corr_ribb_flag:
                        # do it once
                        if inj_same_footprint:
                            EltTip_loc = np.array([], dtype=int)
                        else:
                            EltTip_loc = EltTip
                        #   1) Returns the indices that would sort EltTip
                        tip_sorted = np.argsort(EltTip_loc)
                        #   2) Returns the positions of "to_impose" in "EltTip[tip_sorted]" ordered as to_impose
                        to_impose_pstn = np.searchsorted(EltTip_loc[tip_sorted], to_impose)
                        #   3) indexes of "to_impose" in EltTip ordered as to_impose
                        ind_toImps_tip = tip_sorted[to_impose_pstn]
                        #   4) list of "corr_ribbon" of "to_impose" ordered as to_impose
                        corr_ribbon_TI = corr_ribbon[ind_toImps_tip]
                        corr_ribb_flag = True

                    # make a list of the ind of corresponding ribbon (and to_impose) that are in neg
                    toImp_neg_rib = np.asarray([], dtype=np.int)
                    for i, elem in enumerate(to_impose):
                        if corr_ribbon_TI[i] in neg:
                            toImp_neg_rib = np.append(toImp_neg_rib, i)

                    # find the tip that are not in neg and whose ribbon are in neg and solve for opening there
                    to_solve_k = np.append(to_solve_k, np.setdiff1d(to_impose[toImp_neg_rib], neg))
                    to_impose_k = np.delete(to_impose, toImp_neg_rib)
                    imposed_val_k = np.delete(imposed_val, toImp_neg_rib)
            # ------

            # set all what is in crack
            EltCrack_k = np.concatenate((to_solve_k, neg))
            EltCrack_k = np.concatenate((EltCrack_k, to_impose_k))


            # ------
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


            # Prepare the arguments
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


            # Define the guess opening
            w_guess = np.zeros(Fr_lstTmStp.mesh.NumberOfElts, dtype=np.float64)
            avg_dw = (sum(Qin) * timeStep / Fr_lstTmStp.mesh.EltArea -
                      sum(imposed_val_k - Fr_lstTmStp.w[to_impose_k])) / len(to_solve_k)
            w_guess[to_solve_k] = Fr_lstTmStp.w[to_solve_k] #+ avg_dw
            w_guess[to_impose_k] = imposed_val_k

            # Define the guess pressure
            pf_guess_neg = np.dot(C[np.ix_(neg, EltCrack_k)], w_guess[EltCrack_k]) + mat_properties.SigmaO[neg]
            pf_guess_tip = np.dot(C[np.ix_(to_impose_k, EltCrack_k)], w_guess[EltCrack_k]) + mat_properties.SigmaO[to_impose_k]

            # Add the boundary effect
            if Boundary is not None:
                traction_guess = Boundary.getTraction(w_guess, EltCrack)
                pf_guess_neg = pf_guess_neg + traction_guess[neg]
                pf_guess_tip = pf_guess_tip + traction_guess[to_impose_k]

            # Define the size of the system to be solved:
            sys_size = len(to_solve_k) + len(pf_guess_neg) + len(pf_guess_tip)

            # Based on the chosen elasto-hydrodynamic solver:
            #     - set the function/class that builds the system of equations to be solve and the rhs
            #     - set the gess value for the solution of the system
            if sim_properties.elastohydrSolver == 'implicit_Picard' or sim_properties.elastohydrSolver == 'implicit_Anderson':
                if inj_properties.modelInjLine:
                    sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse_injection_line
                    guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
                                            pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                                            pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k]))

                elif sim_properties.solveDeltaP:
                    if sim_properties.solveSparse:
                        if sim_properties.EHL_iter_lin_solve:
                            if not sim_properties.solve_monolithic:
                                sys_fun = EHL_sys_obj(sys_size, dtype=np.float64)
                            else:
                                sys_fun = Monolithic_EHL_sys_obj(len(to_solve_k) + sys_size, dtype=np.float64, *arg)
                        else:
                            if not sim_properties.solve_monolithic:
                                sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse
                            else:
                                sys_fun = MakeEquationSystem_Monolithic_precond
                    else:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP
                    if not sim_properties.solve_monolithic:
                        #guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
                        guess = np.concatenate((np.full(len(to_solve_k), 0., dtype=np.float64),
                                                pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                                                pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k]))
                    else:
                        guess = np.concatenate((np.full(len(to_solve_k), 0., dtype=np.float64),
                                                np.full(len(to_solve_k), 0., dtype=np.float64),
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

                if inj_properties.modelInjLine:
                    #inj_cells = np.where(abs(Qin) > 0)[0]
                    inj_cells = np.intersect1d(inj_properties.sourceElem, Fr_lstTmStp.EltChannel)
                    sink_cells = np.intersect1d(inj_properties.sourceElem, Fr_lstTmStp.EltCrack)

                    sink = np.zeros(Fr_lstTmStp.mesh.NumberOfElts)
                    if inj_properties.sinkElem:
                        sink[inj_properties.sinkElem] = inj_properties.sinkVel * Fr_lstTmStp.mesh.EltArea

                    indxCurTime = max(np.where(Fr_lstTmStp.time + timeStep >= inj_properties.injectionRate[0, :])[0])
                    currentRate = inj_properties.injectionRate[1, indxCurTime]  # current injection rate

                    inj_ch = np.intersect1d(inj_cells, to_solve_k)
                    inj_act = np.intersect1d(inj_cells, neg)
                    inj_in_ch = []
                    inj_in_act = []
                    for m in inj_ch:
                        inj_in_ch.append(np.where(to_solve_k == m)[0][0])
                    for m in inj_act:
                        inj_in_act.append(np.where(neg == m)[0][0])

                    arg = (arg, inj_properties, inj_ch, inj_act, sink_cells, Fr_lstTmStp.pInjLine,
                           np.asarray(inj_in_ch, dtype=int), np.asarray(inj_in_act, dtype=int), currentRate, sink)

                    guess_il = np.zeros(len(inj_cells) + 1)
                    guess_il[1:] = Qin[inj_cells]
                    guess = np.concatenate((guess, guess_il))

                # -Instantiate a direct or iterative linear solver object
                if sim_properties.EHL_iter_lin_solve:
                    if not sim_properties.solve_monolithic:
                        if inj_same_footprint: rcmp_prec_before2ndIter = True
                        else: rcmp_prec_before2ndIter = False

                        linear_solver = Iterative_linear_solver(sys_fun, sim_properties.gmres_tol,
                                                                sim_properties.gmres_maxiter, sim_properties.gmres_Restart,
                                                                prec_func=EHL_iLU_Prec,
                                                                rcmp_prec_before2ndIter= rcmp_prec_before2ndIter)
                    else:
                        linear_solver = Iterative_linear_solver(sys_fun, sim_properties.gmres_tol,
                                                                sim_properties.gmres_maxiter, sim_properties.gmres_Restart)
                else:
                    linear_solver = Direct_linear_solver(sys_fun)

                # Call the solver of the non linear system or go with an RKL method
                if sim_properties.elastohydrSolver == 'implicit_Picard':

                    typValue = np.copy(guess)

                    sol, data_nonLinSolve = Picard_Newton(None,
                                           linear_solver,
                                           guess,
                                           typValue,
                                           inter_itr_init,
                                           sim_properties,
                                           *arg,
                                           perf_node=perfNode_widthConstrItr)
                elif sim_properties.elastohydrSolver == 'implicit_Anderson':
                    # Ander_time = -time.time()
                    sol, data_nonLinSolve = Anderson(linear_solver,
                                             guess,
                                             inter_itr_init,
                                             sim_properties,
                                             *arg,
                                             perf_node=perfNode_widthConstrItr)
                    # Ander_time = Ander_time + time.time()

                    # file_name = '/home/carlo/Desktop/test_EHL_direct_vs_iter/directT.csv'
                    # append_new_line(file_name,
                    #                 str(sys_size) + ','
                    #                 + str(Ander_time) + ','
                    #                 + str(linear_solver.A_creation) + ','
                    #                 + str(linear_solver.call_ID))
                    # file_name = '/home/carlo/Desktop/test_EHL_direct_vs_iter/iterT.csv'
                    # file_name = '/home/carlo/Desktop/test_EHL_direct_vs_iter/iterHMATT.csv'
                    # append_new_line(file_name,
                    #                  str(sys_size)+','
                    #                 +str(Ander_time)+','
                    #                 +str(linear_solver.A_creation)+','
                    #                 +str(linear_solver.ILU_comp)+','
                    #                 + str(linear_solver.cumulativeITERsSOLVER) + ','
                    #                 + str(linear_solver.call_ID))
                elif sim_properties.elastohydrSolver == 'JacobianFreeNewton':
                    log.error("NOT YET IMPLEMENTED!") # another option is scipy.optimize.newton_krylov

            elif sim_properties.elastohydrSolver == 'RKL2':
                sol, data_nonLinSolve = solve_width_pressure_RKL2(mat_properties.Eprime,
                                                          sim_properties.enableGPU,
                                                          sim_properties.nThreads,
                                                          perfNode_widthConstrItr,
                                                          *arg)
            else:
                raise SystemExit("The given elasto-hydrodynamic solver is not supported!")

            # Check if just the iterative linear solve did not go well
            if (sol == None).any():
                print('Could not solve the linear system. Retry with a monolithic version')
                sol, data_nonLinSolve = Anderson(Direct_linear_solver(Monolithic_EHL_sys_obj(len(to_solve_k) +
                                                                                             sys_size,
                                                                                             dtype=np.float64,
                                                                                             *arg)),
                                                 guess,
                                                 inter_itr_init,
                                                 sim_properties,
                                                 *arg,
                                                 perf_node=perfNode_widthConstrItr)

            # Check if the solution is NaN at any point and why it failed
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

            # Set the solution for the current iteration on the width constraints
            # update "to solve"
            w = np.copy(Fr_lstTmStp.w)
            w[to_solve_k] += sol[:len(to_solve_k)]
            # update "to impose"
            w[to_impose_k] = imposed_val_k
            # update "neg" or active width constraints
            w[neg] = wc_to_impose


            neg_km1 = np.copy(neg)
            wc_km1 = np.copy(wc_to_impose)

            # find new cells with an opening below the minimum
            below_wc_k = np.where(w[to_solve_k] < mat_properties.wc)[0]

            if len(below_wc_k) > 0:
                # (1)
                # find the cells where the max width in w history is greater than wc AND, at the same time, now is w < wc
                # in those cells the opening needs to be imposed
                wHst_above_wc = np.where(Fr_lstTmStp.wHist[to_solve_k] >= mat_properties.wc)[0]
                impose_wc_at = np.intersect1d(wHst_above_wc, below_wc_k)

                # (2)
                # find the cells where the max width in w history is less than wc AND, at the same time, now is dw/dt<0
                # these are the cells that are closing with w<wc therefore we MUST impose w = max width in w history
                wHst_below_wc = np.where(Fr_lstTmStp.wHist[to_solve_k] < mat_properties.wc)[0]
                dwdt_neg = np.where(w[to_solve_k] <= Fr_lstTmStp.w[to_solve_k])[0]
                impose_wHist_at = np.intersect1d(wHst_below_wc, dwdt_neg)

                # (3)
                # prepare the list of cells where the constraints needs to be imposed at the next iteration
                neg_k = to_solve_k[np.concatenate((impose_wc_at, impose_wHist_at))]

                # (4)
                # prepare the corresponding values of width to be imposed in cells where width constraint is active
                N_of_impose_wc_at = len(impose_wc_at)
                N_of_impose_wHist_at = len(impose_wHist_at)
                N_of_constraints = N_of_impose_wc_at + N_of_impose_wHist_at
                wc_k = np.full((N_of_constraints,), mat_properties.wc, dtype=np.float64)
                wc_k[len(impose_wc_at):] = Fr_lstTmStp.wHist[to_solve_k[impose_wHist_at]]

                # (5)
                # check if you are imposing wc and w_max_hist on the same cells as the previous iteration
                new_neg = np.setdiff1d(neg_k, neg)
                if len(new_neg) == 0:
                    active_contraint = False
                else:
                    # if sim_properties.frontAdvancing is not 'implicit':
                    #     log.warning('Changing front advancing scheme to implicit due to width going negative...')
                    #     sim_properties.frontAdvancing = 'implicit'
                    #     return np.nan, np.nan, (np.nan, np.nan)

                    # (6)
                    # cumulatively add the cells with active width constraint
                    neg = np.hstack((neg_km1, new_neg))
                    new_wc = []
                    for new_neg_i in new_neg:
                        new_wc.append(wc_k[np.where(neg_k == new_neg_i)[0]][0])
                    wc_to_impose = np.hstack((wc_km1, np.asarray(new_wc)))
                    log.debug('Iterating on cells with active width constraint...')
            else:
                active_contraint = False
        #
        # END OF THE LOOP ON THE active_contraint


        # from utilities.utility import plot_as_matrix
        # K = np.full(Fr_lstTmStp.mesh.NumberOfElts, np.NaN)
        # K[to_solve] = w[to_solve]
        # plot_as_matrix(K, Fr_lstTmStp.mesh)

        # prepare the  array containing the pressure
        pf = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
        if sim_properties.solve_monolithic and sim_properties.solveDeltaP:
            ch_act_toimpose = np.concatenate((to_solve_k, to_impose_k, neg_km1))
            pf[ch_act_toimpose] = Fr_lstTmStp.pFluid[ch_act_toimpose] + sol[len(to_solve_k):]
        elif not sim_properties.solve_monolithic:
            if sim_properties.useBlockToeplizCompression:
                C._set_domain_and_codomain_IDX(EltCrack, to_solve_k)
                # Fluid pressure in the channel evaluated by dot product between elasticity matrix and w
                pf[to_solve_k] = C._matvec(w[EltCrack]) + mat_properties.SigmaO[to_solve_k]
            else:
                # Fluid pressure in the channel evaluated by dot product between elasticity matrix and w
                pf[to_solve_k] = np.dot(C[np.ix_(to_solve_k, EltCrack)], w[EltCrack]) + mat_properties.SigmaO[to_solve_k]

            # Effect of the finite domain
            if Boundary is not None: pf[to_solve_k] = pf[to_solve_k] + Boundary.last_traction[to_solve_k]

            if sim_properties.solveDeltaP:
                pf[neg_km1] = Fr_lstTmStp.pFluid[neg_km1] + sol[len(to_solve_k):len(to_solve_k) + len(neg_km1)]
                #pf[to_impose_k] = Fr_lstTmStp.pFluid[to_impose_k] + sol[len(to_solve_k) + len(neg_km1):]
                # modified for injection line
                pf[to_impose_k] = Fr_lstTmStp.pFluid[to_impose_k] + sol[len(to_solve_k) + len(neg_km1):
                                                                    len(to_solve_k) + len(neg_km1) + len(to_impose_k)]
            else:
                pf[neg_km1] = sol[len(to_solve_k):len(to_solve_k) + len(neg_km1)]
                #pf[to_impose_k] = sol[len(to_solve_k) + len(neg_km1):]
                # modified for injection line
                pf[to_impose_k] = sol[len(to_solve_k) + len(neg_km1):
                                        len(to_solve_k) + len(neg_km1) + len(to_impose_k)]


        if perfNode_nonLinSys is not None:
            instrument_close(perfNode, perfNode_nonLinSys, None, len(sol), True, None, Fr_lstTmStp.time)
            perfNode.nonLinSolve_data.append(perfNode_nonLinSys)

        # check the fracture status
        if len(neg) == len(to_solve):
            fully_closed = True

        if inj_properties.modelInjLine:
            pil_indx = len(to_solve_k) + len(neg_km1) + len(to_impose_k)
            dp_il = sol[pil_indx]
            Q_ch = sol[pil_indx + 1: pil_indx + 1 + len(inj_ch)]
            Q_act = sol[pil_indx + 1 + len(inj_ch): pil_indx + 1 + len(inj_ch) + len(inj_act)]
        else:
            dp_il = None
            Q_ch = None
            Q_act = None

        if inj_properties.modelInjLine:
            return_data = [data_nonLinSolve, neg_km1, fully_closed, dp_il, (Q_ch, inj_ch), (Q_act, inj_act)]
        else:
            return_data = [data_nonLinSolve, neg_km1, fully_closed]

        return w, pf, return_data