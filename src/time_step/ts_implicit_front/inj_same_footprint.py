# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import copy
import logging
import numpy as np

# Internal Imports
from fluid.reyNumb import turbulence_check_tip
from systems.make_sys_common_fun import calculate_fluid_flow_characteristics_laminar
from systems.sol_sys_dispatcher import solve_width_pressure
from tip.volume_integral import leak_off_stagnant_tip


def injection_same_footprint(Fr_lstTmStp, C, Boundary, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             perfNode=None):
    """
    This function solves the ElastoHydrodynamic equations to get the fracture width. The fracture footprint is taken
    to be the same as in the fracture from the last time step.

    Args:
        Fr_lstTmStp (Fracture):                     -- fracture object from the last time step.
        C (ndarray):                                -- the elasticity matrix.
        timeStep (float):                           -- time step.
        Qin (ndarray):                              -- current injection rate.
        mat_properties (MaterialProperties):        -- material properties.
        fluid_properties (FluidProperties):         -- fluid properties.
        sim_properties (SimulationProperties):      -- simulation parameters.
        perfNode (IterationProperties):             -- a performance node to store performance data.

    Returns:
        - exitstatus (int)          -- exit status (see the function description below for the possibilities).
        - Fr_kplus1 (Fracture)      -- the fracture after injection with the same footprint.

    """
    log = logging.getLogger('PyFrac.injection_same_footprint')

    if len(Fr_lstTmStp.InCrack[np.where(Fr_lstTmStp.InCrack == 1)]) > sim_properties.maxElementIn and \
            sim_properties.meshReductionPossible:
        exitstatus = 16
        return exitstatus, Fr_lstTmStp

    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if sum(mat_properties.Cprime[Fr_lstTmStp.EltCrack]) > 0.:
        # the tip cells are assumed to be stagnant in same footprint evaluation
        LkOff[Fr_lstTmStp.EltTip] = leak_off_stagnant_tip(Fr_lstTmStp.EltTip,
                                                          Fr_lstTmStp.l,
                                                          Fr_lstTmStp.alpha,
                                                          Fr_lstTmStp.TarrvlZrVrtx[Fr_lstTmStp.EltTip],
                                                          Fr_lstTmStp.time + timeStep,
                                                          mat_properties.Cprime,
                                                          timeStep,
                                                          Fr_lstTmStp.mesh)

        # Calculate leak-off term for the channel cell
        t_lst_min_t0 = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_lst_min_t0[t_lst_min_t0 < 0.] = 0.
        t_min_t0 = t_lst_min_t0 + timeStep
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * (t_min_t0 ** 0.5 -
                                                                                             t_lst_min_t0 ** 0.5) * Fr_lstTmStp.mesh.EltArea

    LkOff[Fr_lstTmStp.pFluid <= mat_properties.porePressure] = 0.

    if np.isnan(LkOff[Fr_lstTmStp.EltCrack]).any():
        exitstatus = 13
        return exitstatus, None

    # solve for width. All of the fracture cells are solved (tip values imposed from the last time step)
    empty = np.array([], dtype=int)
    if sim_properties.doublefracture and Fr_lstTmStp.fronts_dictionary['number_of_fronts'] == 2: # here we save the cells in the two cracks
        doublefracturedictionary = {"number_of_fronts": Fr_lstTmStp.fronts_dictionary['number_of_fronts'],
                                    "crackcells_0": Fr_lstTmStp.fronts_dictionary['crackcells_0'],
                                    "crackcells_1": Fr_lstTmStp.fronts_dictionary['crackcells_1']}
    elif sim_properties.projMethod != 'LS_continousfront':
        doublefracturedictionary = {"number_of_fronts": 1}
    else:
        doublefracturedictionary = {"number_of_fronts": Fr_lstTmStp.fronts_dictionary['number_of_fronts']}

    # set the Tip correction as Rider & Napier, 1985.
    # it can be shown that tip correction and R0 kernel are performing better than R4 kernel and this type of tip correction

    C._set_tipcorr(Fr_lstTmStp.FillF, Fr_lstTmStp.EltTip)
    C._set_kerneltype_as_R0()
    w_k, p_k, return_data = solve_width_pressure(Fr_lstTmStp, #Fr_lstTmStp
                                                 sim_properties,
                                                 fluid_properties,
                                                 mat_properties,
                                                 Fr_lstTmStp.EltTip, #empty, sending Eltip to set tip correction
                                                 empty, #partlyFilledTip
                                                 C,
                                                 Boundary,
                                                 Fr_lstTmStp.FillF[empty],
                                                 Fr_lstTmStp.EltCrack,
                                                 Fr_lstTmStp.InCrack,
                                                 LkOff,
                                                 empty,#Fr_lstTmStp.w[Fr_lstTmStp.EltTip],#empty, #wTip
                                                 timeStep,
                                                 Qin,
                                                 perfNode,
                                                 empty, #Vel
                                                 empty, #corr_ribbon
                                                 empty, #stagnant
                                                 doublefracturedictionary= doublefracturedictionary,
                                                 inj_same_footprint = True)
    C.enable_tip_corr = False
    C._set_kerneltype_as_it_used_to_be()


    # from utility import plot_as_matrix
    # K = w_k
    # plot_as_matrix(K, Fr_lstTmStp.mesh)

    # check if the solution is valid
    if np.isnan(w_k).any() or np.isnan(p_k).any():
        exitstatus = 5
        return exitstatus, None

    if (w_k < 0).any():
        log.warning('Neg width encountered!')

    # from utility import plot_as_matrix
    # plot_as_matrix(w_k, Fr_lstTmStp.mesh)

    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w = w_k
    Fr_kplus1.pFluid = p_k
    Fr_kplus1.pNet = np.zeros((Fr_kplus1.mesh.NumberOfElts,))
    if Boundary is not None:
        Fr_kplus1.boundEffTraction = Boundary.last_traction
        Fr_kplus1.pNet[Fr_lstTmStp.EltCrack] = p_k[Fr_lstTmStp.EltCrack] - mat_properties.SigmaO[Fr_lstTmStp.EltCrack] - Fr_kplus1.boundEffTraction[Fr_lstTmStp.EltCrack]
    else:
        Fr_kplus1.pNet[Fr_lstTmStp.EltCrack] = p_k[Fr_lstTmStp.EltCrack] - mat_properties.SigmaO[Fr_lstTmStp.EltCrack]
    Fr_kplus1.closed = return_data[1]
    Fr_kplus1.v = np.zeros((len(Fr_kplus1.EltTip), ), dtype=np.float64)
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += np.sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    # Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - sum(Fr_kplus1.LkOffTotal[Fr_kplus1.EltCrack])) \
    #                        / Fr_kplus1.injectedVol
    # Fr_kplus1.efficiency = (Fr_kplus1.injectedVol -Fr_kplus1.LkOffTotal) \
    #                        / Fr_kplus1.injectedVol
    Fr_kplus1.efficiency = Fr_kplus1.mesh.EltArea * (np.sum(Fr_kplus1.w[Fr_kplus1.EltTip] * Fr_kplus1.FillF) +
                                                     np.sum(Fr_kplus1.w[Fr_kplus1.EltChannel])) / Fr_kplus1.injectedVol
    Fr_kplus1.source = Fr_lstTmStp.EltCrack[np.where(Qin[Fr_lstTmStp.EltCrack] != 0)[0]]


    if return_data[0]!=None:
        Fr_kplus1.effVisc = return_data[0][1]
        fluidVel = return_data[0][0]
    if fluid_properties.turbulence:
        if sim_properties.saveReynNumb or sim_properties.saveFluidFlux:
            ReNumb, check = turbulence_check_tip(fluidVel, Fr_kplus1, fluid_properties, return_ReyNumb=True)
            if sim_properties.saveReynNumb:
                Fr_kplus1.ReynoldsNumber = ReNumb
            if sim_properties.saveFluidFlux:
                Fr_kplus1.fluidFlux = ReNumb * 3 / 4 / fluid_properties.density * fluid_properties.viscosity
        if sim_properties.saveFluidVel:
            Fr_kplus1.fluidVelocity = fluidVel
        if sim_properties.saveFluidVelAsVector:  raise SystemExit('saveFluidVelAsVector Not yet implemented')
        if sim_properties.saveFluidFluxAsVector: raise SystemExit('saveFluidFluxAsVector Not yet implemented')
    else:
        if sim_properties.saveFluidFlux or sim_properties.saveFluidVel or sim_properties.saveReynNumb or sim_properties.saveFluidFluxAsVector or sim_properties.saveFluidVelAsVector:
            ###todo: re-evaluating these parameters is highly inefficient. They have to be stored if neccessary when
            # the solution is evaluated.
            fluid_flux, fluid_vel, Rey_num, fluid_flux_components, fluid_vel_components = calculate_fluid_flow_characteristics_laminar(Fr_kplus1.w,
                                                                                          Fr_kplus1.pFluid,
                                                                                          mat_properties.SigmaO,
                                                                                          Fr_kplus1.mesh,
                                                                                          Fr_kplus1.EltCrack,
                                                                                          Fr_kplus1.InCrack,
                                                                                          fluid_properties.muPrime,
                                                                                          fluid_properties.density)

            if sim_properties.saveFluidFlux:
                fflux = np.zeros((4, Fr_kplus1.mesh.NumberOfElts), dtype=np.float32)
                fflux[:, Fr_kplus1.EltCrack] = fluid_flux
                Fr_kplus1.fluidFlux = fflux

            if sim_properties.saveFluidFluxAsVector:
                fflux_components = np.zeros((8, Fr_kplus1.mesh.NumberOfElts), dtype=np.float32)
                fflux_components[:, Fr_kplus1.EltCrack] = fluid_flux_components
                Fr_kplus1.fluidFlux_components = fflux_components

            if sim_properties.saveFluidVel:
                fvel = np.zeros((4, Fr_kplus1.mesh.NumberOfElts), dtype=np.float32)
                fvel[:, Fr_kplus1.EltCrack] = fluid_vel
                Fr_kplus1.fluidVelocity = fvel

            if sim_properties.saveFluidVelAsVector:
                fvel_components = np.zeros((8, Fr_kplus1.mesh.NumberOfElts), dtype=np.float32)
                fvel_components[:, Fr_kplus1.EltCrack] = fluid_vel_components
                Fr_kplus1.fluidVelocity_components = fvel_components

            if sim_properties.saveReynNumb:
                Rnum = np.zeros((4, Fr_kplus1.mesh.NumberOfElts), dtype=np.float32)
                Rnum[:, Fr_kplus1.EltCrack] = Rey_num
                Fr_kplus1.ReynoldsNumber = Rnum

    Fr_lstTmStp.closed = return_data[1]
    # check if the solution is valid

    if return_data[2]:
        return 14, Fr_kplus1

    exitstatus = 1
    return exitstatus, Fr_kplus1