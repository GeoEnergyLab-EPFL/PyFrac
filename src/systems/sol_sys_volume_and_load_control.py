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
from systems.make_sys_volume_and_load_control import MakeEquationSystem_volumeControl, MakeEquationSystem_volumeControl_double_fracture, \
    MakeEquationSystem_volumeControl_symmetric, Volume_Control_4_gmres
from mesh_obj.symmetry import get_symetric_elements
from properties import instrument_start, instrument_close
from linear_solvers.linear_iterative_solver import iteration_counter

def sol_sys_volume_and_load_control(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, inj_properties, EltTip, partlyFilledTip, C,Boundary,
                                     FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon,
                                     doublefracturedictionary = None, inj_same_footprint = False):
    log = logging.getLogger('PyFrac.solve_width_pressure.sol_sys_volume_and_load_control')
    if sim_properties.volumeControlGMRES:

        # time_beg = time.time()

        counter = iteration_counter(log)  # to obtain the number of iteration and residual
        # todo: include leakoff
        total_vol = (sum(Fr_lstTmStp.w) + sum(Qin[EltCrack]) * (timeStep) / Fr_lstTmStp.mesh.EltArea)  # - something


        # C can be either the Hmat object or a toeplitz matrix
        D_i = np.reciprocal(C.diag_val)  # Only 1 value of the elasticity matrix
        if inj_same_footprint: # this is the case where you do not impose the tip opening

            S_i = -np.reciprocal(Fr_lstTmStp.EltCrack.size * D_i)  # Inverse Schur complement

            g1 = D_i * (mat_properties.SigmaO[Fr_lstTmStp.EltCrack]) + D_i * S_i * (total_vol) * np.ones(
                Fr_lstTmStp.EltCrack.size)  # D_e^-1 * sigma - vol_incr * S^-1 * D_e^-1 *[1...1](vertical)
            g2 = S_i * (total_vol)  # S^-1 * vol_incr --> change

            data = C, Fr_lstTmStp.EltCrack, D_i, S_i
        else: # this is the case where you do impose the tip opening

            S_i = -np.reciprocal(Fr_lstTmStp.EltChannel.size * D_i)  # Inverse Schur complement

            C._set_domain_and_codomain_IDX(EltTip, Fr_lstTmStp.EltChannel)
            g1 = D_i * (mat_properties.SigmaO[Fr_lstTmStp.EltChannel] - C._matvec(wTip)) + D_i * S_i * (
                        total_vol - np.sum(wTip)) * np.ones(
                Fr_lstTmStp.EltChannel.size)  # D_e^-1 * sigma - vol_incr * S^-1 * D_e^-1 *[1...1](vertical)
            g2 = S_i * (total_vol - np.sum(wTip))  # S^-1 * vol_incr --> change
            data = C, Fr_lstTmStp.EltChannel, D_i, S_i

        rhs_prec = np.concatenate((g1, np.asarray([g2])))  # preconditionned b (Ax=b)

        system_dot_prod = Volume_Control_4_gmres(data)
        # begtime_gmres=time.time
        sol_GMRES = gmres(system_dot_prod, rhs_prec, tol=sim_properties.gmres_tol,
                          maxiter=sim_properties.gmres_maxiter, callback=counter)
        # endtime_gmres=time.time()
        # (C._matvec(sol_GMRES[0][0:-1]) - sol_GMRES[0][-1]) useful to check
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
        if inj_same_footprint:
            w[Fr_lstTmStp.EltCrack] = sol[np.arange(Fr_lstTmStp.EltCrack.size)]
        else:
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

        # compute_time = time.time()-time_beg
        # append_new_line('./Data/radial_VC_gmres/timing.txt', str(Fr_lstTmStp.EltChannel.size)+"  "+str(compute_time))
        # compute_time_gmres=endtime_gmres-begtime_gmres
        # append_new_line('./Data/radial_VC_gmres/timing_gmres.txt',
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

            if sim_properties.doublefracture and doublefracturedictionary['number_of_fronts'] == 2:
                # compute the channel from the last time step for the two fractures
                EltChannelFracture0 = np.setdiff1d(Fr_lstTmStp.fronts_dictionary['crackcells_0'],
                                                   Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_0'])
                EltChannelFracture1 = np.setdiff1d(Fr_lstTmStp.fronts_dictionary['crackcells_1'],
                                                   Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_1'])
                # from utility import plot_as_matrix
                # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
                # K[EltChannelFracture1] = 1
                # K[EltChannelFracture0] = 2
                # K[Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_1']] = 3
                # K[Fr_lstTmStp.fronts_dictionary['TIPcellsONLY_0']] = 4
                # plot_as_matrix(K, Fr_lstTmStp.mesh)
                if EltTip.size == 0:
                    EltTipFracture0 = EltTip
                    EltTipFracture1 = EltTip
                else:
                    EltTipFracture0 = doublefracturedictionary['TIPcellsANDfullytrav_0']
                    EltTipFracture1 = doublefracturedictionary['TIPcellsANDfullytrav_1']
                wtipindexFR0 = np.where(np.in1d(EltTip, EltTipFracture0))[0]
                wtipindexFR1 = np.where(np.in1d(EltTip, EltTipFracture1))[0]
                wTipFR0 = wTip[wtipindexFR0]
                wTipFR1 = wTip[wtipindexFR1]
                QinFR0 = Qin[EltChannelFracture0]
                QinFR1 = Qin[EltChannelFracture1]

                # CARLO: I check if can be possible to have a Channel to be tip
                if np.any(np.isin(np.concatenate((EltChannelFracture0, EltChannelFracture1)),
                                  np.concatenate((EltTipFracture0, EltTipFracture1)), assume_unique=True)):
                    SystemExit(
                        "Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")

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
                if np.any(np.isin(Fr_lstTmStp.EltChannel, EltTip, assume_unique=True)):
                    SystemExit(
                        "Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")
                # time_beg=time.time()
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
        # begtime_sol=time.time()
        try:
            sol = np.linalg.solve(A, b)

        except np.linalg.linalg.LinAlgError:
            status = False
            fail_cause = 'sigular matrix'
        # fintime_sol=time.time()
        # compute_time_sol=fintime_sol-begtime_sol
        # compute_time = fintime_sol - time_beg
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
                w[EltChannelFracture1] += sol[
                    np.arange(EltChannelFracture0.size, EltChannelFracture0.size + EltChannelFracture1.size)]
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