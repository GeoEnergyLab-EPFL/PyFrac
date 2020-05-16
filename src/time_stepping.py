# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import copy

# local imports
from volume_integral import leak_off_stagnant_tip, find_corresponding_ribbon_cell
from symmetry import get_symetric_elements, self_influence
from utility import find_regime
from tip_inversion import TipAsymInversion, StressIntensityFactor
from elastohydrodynamic_solver import *
from level_set import SolveFMM, reconstruct_front, reconstruct_front_LS_gradient, UpdateLists
from continuous_front_reconstruction import reconstruct_front_continuous, UpdateListsFromContinuousFrontRec
from properties import IterationProperties, instrument_start, instrument_close
from anisotropy import *
from labels import TS_errorMessages
from explicit_RKL import solve_width_pressure_RKL2
from postprocess_fracture import append_to_json_file

def attempt_time_step(Frac, C, mat_properties, fluid_properties, sim_properties, inj_properties,
                      timeStep, perfNode=None):
    """
    This function attempts to propagate fracture with the given time step. The function injects fluid and propagates
    the fracture front according to the front advancing scheme given in the simulation properties.

    Args:
        Frac (Fracture):                        -- fracture object from the last time step.
        C (ndarray):                            -- the elasticity matrix.
        mat_properties (MaterialProperties):    -- material properties.
        fluid_properties (FluidProperties):     -- fluid properties.
        sim_properties (SimulationProperties):  -- simulation parameters.
        inj_properties (InjectionProperties):   -- injection properties.
        timeStep (float):                       -- time step.
        perfNode (IterationProperties):         -- a performance node to store performance data.

    Returns:
        - exitstatus (int)      -- see documentation for possible values.
        - Fr_k (Fracture)       -- fracture after advancing time step.
    """

    Qin = inj_properties.get_injection_rate(Frac.time, Frac)
    if inj_properties.sinkLocFunc is not None:
        Qin[inj_properties.sinkElem] -= inj_properties.sinkVel * Frac.mesh.EltArea

    if sim_properties.frontAdvancing == 'explicit':

        perfNode_explFront = instrument_start('extended front', perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    perfNode_explFront)

        if perfNode_explFront is not None:
            instrument_close(perfNode, perfNode_explFront, None,
                             len(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus], Frac.time)
            perfNode.extendedFront_data.append(perfNode_explFront)

        return exitstatus, Fr_k

    elif sim_properties.frontAdvancing == 'predictor-corrector':
        if sim_properties.verbosity > 1:
            print('Advancing front with velocity from last time-step...')

        perfNode_explFront = instrument_start('extended front', perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    perfNode_explFront)

        if perfNode_explFront is not None:
            instrument_close(perfNode, perfNode_explFront, None,
                             len(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus], Frac.time)
            perfNode.extendedFront_data.append(perfNode_explFront)

    elif sim_properties.frontAdvancing == 'implicit':
        if sim_properties.verbosity > 1:
            print('Solving ElastoHydrodynamic equations with same footprint...')

        perfNode_sameFP = instrument_start('same front', perfNode)

        # width by injecting the fracture with the same foot print (balloon like inflation)
        exitstatus, Fr_k = injection_same_footprint(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    perfNode_sameFP)
        if perfNode_sameFP is not None:
            instrument_close(perfNode, perfNode_sameFP, None,
                             len(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus], Frac.time)
            perfNode.sameFront_data.append(perfNode_sameFP)

    else:
        raise ValueError("Provided front advancing type not supported")

    if exitstatus != 1:
        # failed
        return exitstatus, Fr_k

    # Check for the propagation condition with the new width. If the all of the front is stagnant, return fracture as
    # final without front iteration.
    stagnant_crt = np.full((len(Fr_k.EltRibbon),), False, dtype=bool)
    # stagnant cells where propagation criteria is not met
    stagnant_crt[np.where(mat_properties.Kprime[Fr_k.EltRibbon] * (-Fr_k.sgndDist[Fr_k.EltRibbon]) ** 0.5 / (
            mat_properties.Eprime * Fr_k.w[Fr_k.EltRibbon]) > 1)[0]] = True
    # stagnant cells where fracture is closed
    stagnant_closed = np.full((len(Fr_k.EltRibbon),), False, dtype=bool)
    for i in range(len(Fr_k.EltRibbon)):
        stagnant_closed[i] = Fr_k.EltRibbon[i] in Fr_k.closed
    stagnant = np.bitwise_or(stagnant_closed, stagnant_crt)

    if np.all(stagnant):
        return 1, Fr_k

    if sim_properties.verbosity > 1:
        print('Starting Fracture Front loop...')

    norm = 10.
    k = 0

    # Fracture front loop to find the correct front location
    while norm > sim_properties.tolFractFront:
        k = k + 1
        if sim_properties.verbosity > 1:
            print('\nIteration ' + repr(k))
        fill_frac_last = np.copy(Fr_k.FillF)

        perfNode_extFront = instrument_start('extended front', perfNode)
        # find the new footprint and solve the elastohydrodynamic equations to to get the new fracture
        (exitstatus, Fr_k) = injection_extended_footprint(Fr_k.w,
                                                          Frac,
                                                          C,
                                                          timeStep,
                                                          Qin,
                                                          mat_properties,
                                                          fluid_properties,
                                                          sim_properties,
                                                          perfNode_extFront)

        if exitstatus == 1:
            # norm is evaluated by dividing the difference in the area of the tip cells between two successive
            # iterations with the number of tip cells.
            norm = abs((sum(Fr_k.FillF) - sum(fill_frac_last)) / len(Fr_k.FillF))
        else:
            norm = np.nan

        if perfNode_extFront is not None:
            instrument_close(perfNode, perfNode_extFront, norm,
                             len(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus], Frac.time)
            perfNode.extendedFront_data.append(perfNode_extFront)

        if exitstatus != 1:
            return exitstatus, Fr_k

        if sim_properties.verbosity > 1:
            print('Norm of subsequent filling fraction estimates = ' + repr(norm))

        if k == sim_properties.maxFrontItrs:
            exitstatus = 6
            return exitstatus, None

    if sim_properties.verbosity > 1:
        print("Fracture front converged after " + repr(k) + " iterations with norm = " + repr(norm))

    return exitstatus, Fr_k


# ----------------------------------------------------------------------------------------------------------------------

def injection_same_footprint(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
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
    
    w_k, p_k, return_data = solve_width_pressure(Fr_lstTmStp,
                                         sim_properties,
                                         fluid_properties,
                                         mat_properties,
                                         empty,
                                         empty,
                                         C,
                                         Fr_lstTmStp.FillF[empty],
                                         Fr_lstTmStp.EltCrack,
                                         Fr_lstTmStp.InCrack,
                                         LkOff,
                                         empty,
                                         timeStep,
                                         Qin,
                                         perfNode,
                                         empty,
                                         empty)

    # check if the solution is valid
    if np.isnan(w_k).any() or np.isnan(p_k).any():
        exitstatus = 5
        return exitstatus, None

    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w = w_k
    Fr_kplus1.pFluid = p_k
    Fr_kplus1.pNet = np.zeros((Fr_kplus1.mesh.NumberOfElts,))
    Fr_kplus1.pNet[Fr_lstTmStp.EltCrack] = p_k[Fr_lstTmStp.EltCrack] - mat_properties.SigmaO[Fr_lstTmStp.EltCrack]
    Fr_kplus1.closed = return_data[1]
    Fr_kplus1.v = np.zeros((len(Fr_kplus1.EltTip), ), dtype=np.float64)
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += LkOff
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - sum(Fr_kplus1.LkOffTotal[Fr_kplus1.EltCrack])) \
                           / Fr_kplus1.injectedVol
    Fr_kplus1.source = Fr_lstTmStp.EltCrack[np.where(Qin[Fr_lstTmStp.EltCrack] != 0)[0]]
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


# -----------------------------------------------------------------------------------------------------------------------


def injection_extended_footprint(w_k, Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties,
                                 sim_properties, perfNode=None):
    """
    This function takes the fracture width from the last iteration of the fracture front loop, calculates the level set
    (fracture front position) by inverting the tip asymptote and then solves the ElastoHydrodynamic equations to obtain
    the new fracture width.

    Args:
        w_k (ndarray):                          -- the width from last iteration of fracture front.
        Fr_lstTmStp (Fracture):                 -- fracture object from the last time step.
        C (ndarray):                            -- the elasticity matrix.
        timeStep (float):                       -- time step.
        Qin (ndarray):                          -- current injection rate.
        mat_properties (MaterialProperties):    -- material properties.
        fluid_properties (FluidProperties ):    -- fluid properties.
        sim_properties (SimulationProperties):  -- simulation parameters.
        perfNode (IterationProperties):         -- the IterationProperties object passed to be populated with data.

    Returns:
        - exitstatus (int)  possible values are

        | 0       -- not propagated
        | 1       -- iteration successful
        | 2       -- evaluated level set is not valid
        | 3       -- front is not tracked correctly
        | 4       -- evaluated tip volume is not valid
        | 5       -- solution of elastohydrodynamic solver is not valid
        | 6       -- did not converge after max iterations
        | 7       -- tip inversion not successful
        | 8       -- ribbon element not found in the enclosure of a tip cell
        | 9       -- filling fraction not correct
        | 10      -- toughness iteration did not converge
        | 11      -- projection could not be found
        | 12      -- reached end of grid
        | 13      -- leak off can't be evaluated
        | 14      -- fracture fully closed

        - Fracture:            fracture after advancing time step.

    """

    itr = 0
    sgndDist_k = np.copy(Fr_lstTmStp.sgndDist)

    # toughness iteration loop
    while itr < sim_properties.maxProjItrs:

        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.TI_elasticity:
            if sim_properties.projMethod == 'ILSA_orig':
                projection_method = projection_from_ribbon
            elif sim_properties.projMethod == 'LS_grad':
                projection_method = projection_from_ribbon_LS_gradient
            elif sim_properties.projMethod == 'LS_continousfront': #todo: test this case!!!
                projection_method = projection_from_ribbon_LS_gradient
            if itr == 0 :
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   Fr_lstTmStp.EltChannel,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k)
                alpha_ribbon_km1 = np.zeros(Fr_lstTmStp.EltRibbon.size, )
            else:
                alpha_ribbon_k = 0.3 * alpha_ribbon_k + 0.7 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                Fr_lstTmStp.EltChannel,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k)
            if np.isnan(alpha_ribbon_k).any():
                exitstatus = 11
                return exitstatus, None

        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c:

            Kprime_k = get_toughness_from_cellCenter(alpha_ribbon_k,
                                                     sgndDist_k,
                                                     Fr_lstTmStp.EltRibbon,
                                                     mat_properties,
                                                     Fr_lstTmStp.mesh) * (32 / np.pi) ** 0.5

            if np.isnan(Kprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Kprime_k = None

        if mat_properties.TI_elasticity:
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if np.isnan(Eprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Eprime_k = None

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely
        # large float value. (algorithm requires inf)

        perfNode_tipInv = instrument_start('tip inversion', perfNode)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(w_k,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               fluid_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k,
                                                               Eprime_k=Eprime_k,
                                                               perfNode=perfNode_tipInv)

        status = True
        fail_cause = None
        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            status = False
            fail_cause = 'tip inversion failed'
            exitstatus = 7

        if perfNode_tipInv is not None:
            instrument_close(perfNode, perfNode_tipInv, None, len(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            perfNode.tipInv_data.append(perfNode_tipInv)

        if not status:
            return exitstatus, None

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
        front_region = np.where(abs(Fr_lstTmStp.sgndDist) < current_prefactor * 6.66 * (
                Fr_lstTmStp.mesh.hx ** 2 + Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
        # the search region outwards from the front position at last time step
        pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx ** 2 +
                                                                       Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
        # the search region inwards from the front position at last time step
        ngtv_region = np.where(Fr_lstTmStp.sgndDist[front_region] < 0)[0]

        # SOLVE EIKONAL eq via Fast Marching Method to get the distance from tip for each cell.
        SolveFMM(sgndDist_k,
                 Fr_lstTmStp.EltRibbon,
                 Fr_lstTmStp.EltChannel,
                 Fr_lstTmStp.mesh,
                 front_region[pstv_region],
                 front_region[ngtv_region])

        # do it only once if not anisotropic
        if not (sim_properties.paramFromTip or mat_properties.anisotropic_K1c
                or mat_properties.TI_elasticity) or sim_properties.explicitProjection:
            break

        norm = np.linalg.norm(abs(alpha_ribbon_k - alpha_ribbon_km1) / np.pi * 2)
        if norm < sim_properties.toleranceProjection:
            if sim_properties.verbosity > 1:
                print("projection iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break

        alpha_ribbon_km1 = np.copy(alpha_ribbon_k)
        if sim_properties.verbosity > 1:
            print("iterating on projection... norm " + repr(norm))
        itr += 1

    # if itr == sim_properties.maxProjItrs:
    #     exitstatus = 10
    #     return exitstatus, None

    if sim_properties.saveRegime:
        regime = find_regime(w_k, Fr_lstTmStp, mat_properties, sim_properties, timeStep, Kprime_k,
                             -sgndDist_k[Fr_lstTmStp.EltRibbon])

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    if sim_properties.projMethod == 'ILSA_orig':
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front(sgndDist_k,
                                                                 front_region,
                                                                 Fr_lstTmStp.EltChannel,
                                                                 Fr_lstTmStp.mesh)
    elif sim_properties.projMethod == 'LS_grad':
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front_LS_gradient(sgndDist_k,
                                                                             front_region,
                                                                             Fr_lstTmStp.EltChannel,
                                                                             Fr_lstTmStp.mesh)
    elif sim_properties.projMethod == 'LS_continousfront':
        correct_size_of_pstv_region = [False,False]
        recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = False
        while not correct_size_of_pstv_region[0]:
            EltsTipNew, \
            listofTIPcellsONLY, \
            l_k, \
            alpha_k, \
            CellStatus, \
            newRibbon, \
            zrVertx_k_with_fully_traversed, \
            zrVertx_k_without_fully_traversed, \
            correct_size_of_pstv_region, \
            sgndDist_k_temp, Ffront,number_of_fronts  = reconstruct_front_continuous(sgndDist_k,
                                                           front_region[pstv_region],
                                                           Fr_lstTmStp.EltRibbon,
                                                           Fr_lstTmStp.EltChannel,
                                                           Fr_lstTmStp.mesh,
                                                           recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge)
            if correct_size_of_pstv_region[1]:
                exitstatus = 7 #You are here because the level set has negative values until the end of the mesh
                return exitstatus, None

            if not correct_size_of_pstv_region[0]:
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = np.unique(np.ndarray.flatten(Fr_lstTmStp.mesh.NeiElements[front_region]))

                # the search region outwards from the front position at last time step
                pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx ** 2 +
                                                                               Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
                # the search region inwards from the front position at last time step
                ngtv_region = np.where(Fr_lstTmStp.sgndDist[front_region] < 0)[0]

                #sgndDist_k_temp2 = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,),float)  # Initializing the cells with extremely
                # large float value. (algorithm requires inf)
                #sgndDist_k_temp2[Fr_lstTmStp.EltRibbon] = sgndDist_k[Fr_lstTmStp.EltRibbon]
                #sgndDist_k = sgndDist_k_temp2
                #sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                #                                               Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

                # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
                SolveFMM(sgndDist_k,
                         Fr_lstTmStp.EltTip,
                         Fr_lstTmStp.EltCrack,
                         Fr_lstTmStp.mesh,
                         front_region[pstv_region],
                         front_region[ngtv_region])
        sgndDist_k = sgndDist_k_temp

        del correct_size_of_pstv_region
    else:
        raise SystemExit("projection method not supported")


    if not np.in1d(EltsTipNew, front_region).any():
        raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                         " level set.")

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    if len(np.intersect1d(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0:
        exitstatus = 12
        return exitstatus, None

    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1  #EltsTipNew is new tip + fully traversed

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = Integral_over_cell(EltsTipNew,
                                    alpha_k,
                                    l_k,
                                    Fr_lstTmStp.mesh,
                                    'A',
                                    projMethod=sim_properties.projMethod) / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-4)] = 1.0

    # if filling fraction is below zero or above 1+1e-4
    if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
        exitstatus = 9
        return exitstatus, None

    if sim_properties.projMethod != 'LS_continousfront':
        # todo: some of the list are redundant to calculate on each iteration
        # Evaluate the element lists for the trial fracture front
        (EltChannel_k,
         EltTip_k,
         EltCrack_k,
         EltRibbon_k,
         zrVertx_k,
         CellStatus_k) = UpdateLists(Fr_lstTmStp.EltChannel,
                                     EltsTipNew,
                                     FillFrac_k,
                                     sgndDist_k,
                                     Fr_lstTmStp.mesh)
    elif sim_properties.projMethod == 'LS_continousfront':
        zrVertx_k = zrVertx_k_without_fully_traversed
        (EltChannel_k,
         EltTip_k,
         EltCrack_k,
         EltRibbon_k,
         CellStatus_k) = UpdateListsFromContinuousFrontRec(newRibbon,
                                                           sgndDist_k,
                                                           Fr_lstTmStp.EltChannel,
                                                           EltsTipNew,
                                                           listofTIPcellsONLY,
                                                           Fr_lstTmStp.mesh)
                                                           
        if np.isnan(EltChannel_k).any():
            exitstatus = 3
            return exitstatus, None
                                                           
    # EletsTipNew may contain fully filled elements also. Identifying only the partially filled elements
    partlyFilledTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]
    if sim_properties.verbosity > 1:
        print('Solving the EHL system with the new trial footprint')

    if sim_properties.projMethod != 'LS_continousfront':
    # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    else: zrVrtx_newTip = zrVertx_k_with_fully_traversed.transpose()

    # finding ribbon cells corresponding to tip cells
    corr_ribbon = find_corresponding_ribbon_cell(EltsTipNew,
                                                 alpha_k,
                                                 zrVrtx_newTip,
                                                 Fr_lstTmStp.mesh)
    Cprime_tip = mat_properties.Cprime[corr_ribbon]

    # Calculating toughness at tip to be used to calculate the volume integral in the tip cells
    if sim_properties.paramFromTip or mat_properties.anisotropic_K1c:
        if sim_properties.projMethod != 'LS_continousfront':
            # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
            zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        else: zrVrtx_newTip = zrVertx_k_with_fully_traversed.transpose()

        # get toughness from tip in case of anisotropic or
        Kprime_tip = (32 / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                         Fr_lstTmStp.mesh,
                                                                         mat_properties,
                                                                         alpha_k,
                                                                         l_k,
                                                                         zrVrtx_newTip)
    else:
        Kprime_tip = mat_properties.Kprime[corr_ribbon]

    if mat_properties.TI_elasticity:
        Eprime_tip = TI_plain_strain_modulus(alpha_k,
                                             mat_properties.Cij)
    else:
        Eprime_tip = np.full((EltsTipNew.size,), mat_properties.Eprime, dtype=np.float64)

    if perfNode is not None:
        perfNode_wTip = instrument_start('nonlinear system solve', perfNode)

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = (-(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) /
                (Fr_lstTmStp.mesh.hx**2 + Fr_lstTmStp.mesh.hy**2)**0.5 < sim_properties.toleranceVStagnant)
    # we need to remove it:
    # if stagnant.any() and not ((sim_properties.get_tipAsymptote() == 'U') or (sim_properties.get_tipAsymptote() == 'U1')):
    #     if sim_properties.verbosity > 1:
    #         print("Stagnant front is only supported with universal tip asymptote. continuing...")
    #     stagnant = np.full((EltsTipNew.size,), False, dtype=bool)

    if perfNode is not None:
        perfNode_tipWidth = instrument_start('tip width', perfNode)
        #todo close tip width instrumentation

    if stagnant.any():
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(w_k,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        Eprime=Eprime_tip)

        # todo: Find the right cause of failure
        # if the stress Intensity factor cannot be found. The most common reason is wiggles in the front resulting
        # in isolated tip cells.
        if np.isnan(KIPrime).any():
            exitstatus = 8
            return exitstatus, None

        # Calculate average width in the tip cells by integrating tip asymptote. Width of stagnant cells are calculated
        # using the stress intensity factor (see Dontsov and Peirce, JFM RAPIDS, 2017)
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  stagnant=stagnant,
                                  KIPrime=KIPrime,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip) / Fr_lstTmStp.mesh.EltArea
    else:
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  Kprime=Kprime_tip,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip,
                                  stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea

    # check if the tip volume has gone into negative
    smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        #  warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])

    if (wTip < 0).any() or sum(wTip) == 0.:
        exitstatus = 4
        return exitstatus, None

    if perfNode is not None:
        pass
        # todo close tip width instrumentation

    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if sum(mat_properties.Cprime[EltsTipNew]) > 0:
        # Calculate leak-off term for the tip cell
        LkOff[EltsTipNew] = 2 * mat_properties.Cprime[EltsTipNew] * Integral_over_cell(EltsTipNew,
                                                                                       alpha_k,
                                                                                       l_k,
                                                                                       Fr_lstTmStp.mesh,
                                                                                       'Lk',
                                                                                       mat_prop=mat_properties,
                                                                                       frac=Fr_lstTmStp,
                                                                                       Vel=Vel_k,
                                                                                       dt=timeStep,
                                                                                       arrival_t=
                                                                                       Fr_lstTmStp.TarrvlZrVrtx[
                                                                                           EltsTipNew])

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0:
        # todo: no need to evaluate on each iteration. Need to decide. Evaluating here for now for better readability
        t_lst_min_t0 = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_lst_min_t0[t_lst_min_t0 < 0.] = 0.
        t_min_t0 = t_lst_min_t0 + timeStep
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * (
                t_min_t0 ** 0.5 - t_lst_min_t0 ** 0.5) * Fr_lstTmStp.mesh.EltArea
        if stagnant.any():
            LkOff[EltsTipNew[stagnant]] = leak_off_stagnant_tip(EltsTipNew[stagnant],
                                                                l_k[stagnant],
                                                                alpha_k[stagnant],
                                                                Fr_lstTmStp.TarrvlZrVrtx[EltsTipNew[stagnant]],
                                                                Fr_lstTmStp.time + timeStep,
                                                                mat_properties.Cprime,
                                                                timeStep,
                                                                Fr_lstTmStp.mesh)

    # set leak off to zero if pressure below pore pressure
    LkOff[Fr_lstTmStp.pFluid <= mat_properties.porePressure] = 0.

    if np.isnan(LkOff[EltsTipNew]).any():
        exitstatus = 13
        return exitstatus, None

    w_n_plus1, pf_n_plus1, data = solve_width_pressure(Fr_lstTmStp,
                                                       sim_properties,
                                                       fluid_properties,
                                                       mat_properties,
                                                       EltsTipNew,
                                                       partlyFilledTip,
                                                       C,
                                                       FillFrac_k,
                                                       EltCrack_k,
                                                       InCrack_k,
                                                       LkOff,
                                                       wTip,
                                                       timeStep,
                                                       Qin,
                                                       perfNode,
                                                       Vel_k,
                                                       corr_ribbon)

    # check if the new width is valid
    if np.isnan(w_n_plus1).any():
        exitstatus = 5
        return exitstatus, None

    fluidVel = data[0][0]
    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = np.copy(Fr_lstTmStp.Tarrival)
    max_Tarrival = np.nanmax(Tarrival_k)
    nc = np.setdiff1d(EltChannel_k, Fr_lstTmStp.EltChannel)
    new_channel = np.array([], dtype=int)
    for i in nc:
        new_channel = np.append(new_channel, np.where(EltsTipNew == i)[0])
    t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] / Vel_k[new_channel]
    max_l = Fr_lstTmStp.mesh.hx * np.cos(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * np.sin(alpha_k[new_channel])
    t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) / Vel_k[new_channel]
    Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2
    to_correct = np.where(Tarrival_k[EltsTipNew[new_channel]] < max_Tarrival)[0]
    Tarrival_k[EltsTipNew[new_channel[to_correct]]] = max_Tarrival

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w = w_n_plus1
    Fr_kplus1.pFluid = pf_n_plus1
    Fr_kplus1.pNet = np.zeros((Fr_kplus1.mesh.NumberOfElts,))
    Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = Vel_k[partlyFilledTip]
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.InCrack = InCrack_k
    if sim_properties.projMethod != 'LS_continousfront':
        Fr_kplus1.process_fracture_front()
    else :
        Fr_kplus1.Ffront = Ffront
        Fr_kplus1.number_of_fronts = number_of_fronts
        if sim_properties.saveToDisk and sim_properties.saveStatisticsPostCoalescence and Fr_lstTmStp.number_of_fronts != Fr_kplus1.number_of_fronts:
            myJsonName = sim_properties.set_outputFolder+"_mesh_study.json"
            append_to_json_file(myJsonName, Fr_kplus1.mesh.nx, 'append2keyAND2list', key='nx')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.ny, 'append2keyAND2list', key='ny')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hx, 'append2keyAND2list', key='hx')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hy, 'append2keyAND2list', key='hy')
            append_to_json_file(myJsonName, Fr_kplus1.EltCrack.size, 'append2keyAND2list', key='elements_in_crack')
            append_to_json_file(myJsonName, Fr_kplus1.EltTip.size, 'append2keyAND2list', key='elements_in_tip')
            append_to_json_file(myJsonName, Fr_kplus1.time, 'append2keyAND2list', key='coalescence_time')

    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.Tarrival = Tarrival_k
    new_tip = np.where(np.isnan(Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip]))[0]
    Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] / Fr_kplus1.v[new_tip]
    Fr_kplus1.wHist = np.maximum(Fr_kplus1.w, Fr_lstTmStp.wHist)
    Fr_kplus1.closed = data[1]
    tip_neg_rib = np.asarray([], dtype=np.int)
    # adding tip cells with closed corresponding ribbon cells to the list of closed cells
    for i, elem in enumerate(Fr_kplus1.EltTip):
        if corr_ribbon[i] in Fr_kplus1.closed and elem not in Fr_kplus1.closed:
            tip_neg_rib = np.append(tip_neg_rib, elem)
    Fr_kplus1.closed = np.append(Fr_kplus1.closed, tip_neg_rib)
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += np.sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol
    if sim_properties.saveRegime:
        Fr_kplus1.regime = regime
    Fr_kplus1.source = Fr_lstTmStp.EltCrack[np.where(Qin[Fr_lstTmStp.EltCrack] != 0)[0]]
    Fr_kplus1.effVisc = data[0][1]
    Fr_kplus1.yieldRatio = data[0][2]

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
            fluid_flux, fluid_vel, Rey_num, fluid_flux_components, fluid_vel_components= calculate_fluid_flow_characteristics_laminar(Fr_kplus1.w,
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

    if data[2]:
        return 14, Fr_kplus1

    exitstatus = 1
    return exitstatus, Fr_kplus1


# -----------------------------------------------------------------------------------------------------------------------

def solve_width_pressure(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon):
    """
    This function evaluates the width and pressure by constructing and solving the coupled elasticity and fluid flow
    equations. The system of equations are formed according to the type of solver given in the simulation properties.
    """

    if sim_properties.get_volumeControl():

        if sim_properties.symmetric:
            try:
                Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel]
            except AttributeError:
                raise SystemExit("Symmetric fracture needs symmetric mesh. Set symmetric flag to True\n"
                                 "while initializing the mesh")

            EltChannel_sym = Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel]
            EltChannel_sym = np.unique(EltChannel_sym)

            EltTip_sym = Fr_lstTmStp.mesh.corresponding[EltTip]
            EltTip_sym = np.unique(EltTip_sym)

            FillF_mesh = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            FillF_mesh[EltTip] = FillFrac
            FillF_sym = FillF_mesh[Fr_lstTmStp.mesh.activeSymtrc[EltTip_sym]]
            partlyFilledTip_sym = np.where(FillF_sym <= 1)[0]

            C_EltTip = np.copy(C[np.ix_(EltTip_sym[partlyFilledTip_sym],
                                        EltTip_sym[
                                            partlyFilledTip_sym])])  # keeping the tip element entries to restore current

            # filling fraction correction for element in the tip region
            FillF = FillF_sym[partlyFilledTip_sym]
            for e in range(len(partlyFilledTip_sym)):
                r = FillF[e] - .25
                if r < 0.1:
                    r = 0.1
                ac = (1 - r) / r
                self_infl = self_influence(Fr_lstTmStp.mesh, mat_properties.Eprime)
                C[EltTip_sym[partlyFilledTip_sym[e]], EltTip_sym[partlyFilledTip_sym[e]]] += \
                    ac * np.pi / 4. * self_infl

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

            C[np.ix_(EltTip_sym[partlyFilledTip_sym], EltTip_sym[partlyFilledTip_sym])] = C_EltTip
        else:
            C_EltTip = np.copy(C[np.ix_(EltTip[partlyFilledTip],
                                        EltTip[partlyFilledTip])])  # keeping the tip element entries to restore current
            #  tip correction. This is done to avoid copying the full elasticity matrix.

            # filling fraction correction for element in the tip region
            FillF = FillFrac[partlyFilledTip]
            for e in range(0, len(partlyFilledTip)):
                r = FillF[e] - .25
                if r < 0.1:
                    r = 0.1
                ac = (1 - r) / r
                C[EltTip[partlyFilledTip[e]], EltTip[partlyFilledTip[e]]] *= (1. + ac * np.pi / 4.)

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

            # regain original C (without filling fraction correction)
            C[np.ix_(EltTip[partlyFilledTip], EltTip[partlyFilledTip])] = C_EltTip

        perfNode_nonLinSys = instrument_start('nonlinear system solve', perfNode)
        perfNode_widthConstrItr = instrument_start('width constraint iteration', perfNode_nonLinSys)
        perfNode_linSys = instrument_start('linear system solve', perfNode_widthConstrItr)
        status = True
        fail_cause = None
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.linalg.LinAlgError:
            status = False
            fail_cause = 'sigular matrix'

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
            w[Fr_lstTmStp.EltChannel] += sol[np.arange(Fr_lstTmStp.EltChannel.size)]
            w[EltTip] = wTip

        p = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
        p[EltCrack] = sol[-1]

        return_data = (None, np.asarray([]), False)
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
                InCrack,
                LkOff,
                neg,
                corr_nei,
                lst_edgeInCrk)
            
            w_guess = np.zeros(Fr_lstTmStp.mesh.NumberOfElts, dtype=np.float64)
            avg_dw = (sum(Qin) * timeStep / Fr_lstTmStp.mesh.EltArea - sum(
                    imposed_val_k - Fr_lstTmStp.w[to_impose_k])) / len(to_solve_k)
            w_guess[to_solve_k] = Fr_lstTmStp.w[to_solve_k] + avg_dw
            w_guess[to_impose_k] = imposed_val_k
            pf_guess_neg = np.dot(C[np.ix_(neg, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[neg]
            pf_guess_tip = np.dot(C[np.ix_(to_impose_k, EltCrack_k)], w_guess[EltCrack_k]) +  mat_properties.SigmaO[to_impose_k]
            if sim_properties.elastohydrSolver == 'implicit_Picard' or sim_properties.elastohydrSolver == 'implicit_Anderson':
                if sim_properties.solveDeltaP:
                    if sim_properties.solveSparse:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse
                    else:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP
                    guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
                                            pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                                            pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k]))
                else:
                    if sim_properties.solveSparse:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_sparse
                    else:
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted
                    guess = np.concatenate((np.full(len(to_solve_k), avg_dw, dtype=np.float64),
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
                else:
                    sol, data_nonLinSolve = Anderson(sys_fun,
                                             guess,
                                             inter_itr_init,
                                             sim_properties,
                                             *arg,
                                             perf_node=perfNode_widthConstrItr)

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
                    #     print('Changing front advancing scheme to implicit due to width going negative...')
                    #     sim_properties.frontAdvancing = 'implicit'
                    #     return np.nan, np.nan, (np.nan, np.nan)

                    # cumulatively add the cells with active width constraint
                    neg = np.hstack((neg_km1, new_neg))
                    new_wc = []
                    for i in new_neg:
                        new_wc.append(wc_k[np.where(neg_k == i)[0]][0])
                    wc_to_impose = np.hstack((wc_km1, np.asarray(new_wc)))
                    if sim_properties.verbosity > 1:
                        print('Iterating on cells with active width constraint...')
            else:
                active_contraint = False


        pf = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
        # pressure evaluated by dot product of width and elasticity matrix
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


# -----------------------------------------------------------------------------------------------------------------------


def turbulence_check_tip(vel, Fr, fluid, return_ReyNumb=False):
    """
    This function calculate the Reynolds number at the cell edges and check if any to the edge between the ribbon cells
    and the tip cells are turbulent (i.e. the Reynolds number is greater than 2100).

    Arguments:
        vel (ndarray-float):            -- the array giving velocity of each edge of the cells in domain
        Fr (Fracture object):           -- the fracture object to be checked
        fluid (FluidProperties):        -- fluid properties object
        return_ReyNumb (boolean):       -- if True, Reynolds number at all cell edges will also be returned

    Returns:
        - Re (ndarray)     -- Reynolds number of all the cells in the domain; row-wise in the following order, 0--left,\
                              1--right, 2--bottom, 3--top.
        - boolean          -- True if any of the edge between the ribbon and tip cells is turbulent (i.e. Reynolds \
                               number is more than 2100).
    """
    # width at the adges by averaging
    wLftEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 0]]) / 2
    wRgtEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 1]]) / 2
    wBtmEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 2]]) / 2
    wTopEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 3]]) / 2

    Re = np.zeros((4, Fr.EltRibbon.size,), dtype=np.float64)
    Re[0, :] = 4 / 3 * fluid.density * wLftEdge * vel[0, Fr.EltRibbon] / fluid.viscosity
    Re[1, :] = 4 / 3 * fluid.density * wRgtEdge * vel[1, Fr.EltRibbon] / fluid.viscosity
    Re[2, :] = 4 / 3 * fluid.density * wBtmEdge * vel[2, Fr.EltRibbon] / fluid.viscosity
    Re[3, :] = 4 / 3 * fluid.density * wTopEdge * vel[3, Fr.EltRibbon] / fluid.viscosity

    ReNum_Ribbon = []
    # adding Reynolds number of the edges between the ribbon and tip cells to a list
    for i in range(0, Fr.EltRibbon.size):
        for j in range(0, 4):
            # if the current neighbor (j) of the ribbon cells is in the tip elements list
            if np.where(Fr.mesh.NeiElements[Fr.EltRibbon[i], j] == Fr.EltTip)[0].size > 0:
                ReNum_Ribbon = np.append(ReNum_Ribbon, Re[j, i])

    if return_ReyNumb:
        wLftEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 0]]) / 2
        wRgtEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 1]]) / 2
        wBtmEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 2]]) / 2
        wTopEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 3]]) / 2

        Re = np.zeros((4, Fr.mesh.NumberOfElts,), dtype=np.float64)
        Re[0, Fr.EltCrack] = 4 / 3 * fluid.density * wLftEdge * vel[0, Fr.EltCrack] / fluid.viscosity
        Re[1, Fr.EltCrack] = 4 / 3 * fluid.density * wRgtEdge * vel[1, Fr.EltCrack] / fluid.viscosity
        Re[2, Fr.EltCrack] = 4 / 3 * fluid.density * wBtmEdge * vel[2, Fr.EltCrack] / fluid.viscosity
        Re[3, Fr.EltCrack] = 4 / 3 * fluid.density * wTopEdge * vel[3, Fr.EltCrack] / fluid.viscosity

        return Re, (ReNum_Ribbon > 2100.).any()
    else:
        return (ReNum_Ribbon > 2100.).any()


# -----------------------------------------------------------------------------------------------------------------------


def time_step_explicit_front(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             perfNode=None):
    """
    This function advances the fracture front in an explicit manner by propagating it with the velocity from the last
    time step (see Zia and Lecampion 2019 for details).

    Args:
        Fr_lstTmStp (Fracture):                 -- fracture object from the last time step.
        C (ndarray):                            -- the elasticity matrix.
        timeStep (float):                       -- time step.
        Qin (ndarray):                          -- current injection rate.
        mat_properties (MaterialProperties):    -- material properties.
        fluid_properties (FluidProperties ):    -- fluid properties.
        sim_properties (SimulationProperties):  -- simulation parameters.
        perfNode (IterationProperties):         -- a performance node to store performance data.

    Returns:
        - exitstatus (int)  possible values are

            | 0       -- not propagated
            | 1       -- iteration successful
            | 2       -- evaluated level set is not valid
            | 3       -- front is not tracked correctly
            | 4       -- evaluated tip volume is not valid
            | 5       -- solution of elastohydrodynamic solver is not valid
            | 6       -- did not converge after max iterations
            | 7       -- tip inversion not successful
            | 8       -- ribbon element not found in the enclosure of a tip cell
            | 9       -- filling fraction not correct
            | 10      -- toughness iteration did not converge
            | 11      -- projection could not be found
            | 12      -- reached end of grid
            | 13      -- leak off can't be evaluated
            | 14      -- fracture fully closed

        - Fracture:            fracture after advancing time step.

    """

    sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with maximum
                                                                          # float value. (algorithm requires inf)
    sgndDist_k[Fr_lstTmStp.EltChannel] = 0  # for cells inside the fracture

    sgndDist_k[Fr_lstTmStp.EltTip] = Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep *
                                                                                 Fr_lstTmStp.v)
    current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
    cell_diag = (Fr_lstTmStp.mesh.hx ** 2 + Fr_lstTmStp.mesh.hy ** 2) ** 0.5
    expected_range = max(current_prefactor * 6.66 * cell_diag, 1.5 * cell_diag) # expected range of possible propagation
    front_region = np.where(abs(Fr_lstTmStp.sgndDist) < expected_range)[0]
    # the search region outwards from the front position at last time step
    pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx ** 2 +
                                                                   Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
    # the search region inwards from the front position at last time step
    ngtv_region = np.where(Fr_lstTmStp.sgndDist[front_region] < 0)[0]

    # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
    SolveFMM(sgndDist_k,
             Fr_lstTmStp.EltTip,
             Fr_lstTmStp.EltCrack,
             Fr_lstTmStp.mesh,
             front_region[pstv_region],
             front_region[ngtv_region])

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    if sim_properties.projMethod == 'ILSA_orig':
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front(sgndDist_k,
                                                                 front_region,
                                                                 Fr_lstTmStp.EltChannel,
                                                                 Fr_lstTmStp.mesh)
    elif sim_properties.projMethod == 'LS_grad':
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front_LS_gradient(sgndDist_k,
                                                                             front_region,
                                                                             Fr_lstTmStp.EltChannel,
                                                                             Fr_lstTmStp.mesh)

    elif sim_properties.projMethod == 'LS_continousfront':
        correct_size_of_pstv_region = [False,False]
        recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = False
        while not correct_size_of_pstv_region[0]:
            EltsTipNew, \
            listofTIPcellsONLY, \
            l_k, \
            alpha_k, \
            CellStatus, \
            newRibbon, \
            zrVertx_k_with_fully_traversed, \
            zrVertx_k_without_fully_traversed, \
            correct_size_of_pstv_region,\
            sgndDist_k_temp, Ffront,number_of_fronts      = reconstruct_front_continuous(sgndDist_k,
                                                                       front_region[pstv_region],
                                                                       Fr_lstTmStp.EltRibbon,
                                                                       Fr_lstTmStp.EltChannel,
                                                                       Fr_lstTmStp.mesh,
                                                                       recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge)
            if correct_size_of_pstv_region[1]:
                exitstatus = 7 #You are here because the level set has negative values until the end of the mesh
                return exitstatus, None

            if not correct_size_of_pstv_region[0]:
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = np.unique(np.ndarray.flatten(Fr_lstTmStp.mesh.NeiElements[front_region]))

                # the search region outwards from the front position at last time step
                pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx ** 2 +
                                                                               Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
                # the search region inwards from the front position at last time step
                ngtv_region = np.where(Fr_lstTmStp.sgndDist[front_region] < 0)[0]

                #sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,),float)  # Initializing the cells with extremely
                                                                                     # large float value. (algorithm requires inf)
                #sgndDist_k[Fr_lstTmStp.EltChannel] = 0  # for cells inside the fracture

                #sgndDist_k[Fr_lstTmStp.EltTip] = Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep *
                #                                                                             Fr_lstTmStp.v)

                # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
                SolveFMM(sgndDist_k,
                         Fr_lstTmStp.EltTip,
                         Fr_lstTmStp.EltCrack,
                         Fr_lstTmStp.mesh,
                         front_region[pstv_region],
                         front_region[ngtv_region])

        sgndDist_k = sgndDist_k_temp
        del correct_size_of_pstv_region
    else:
        raise SystemExit("projection method not supported")

    if not np.in1d(EltsTipNew, front_region).any():
        raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                         " level set.")

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    # tipNeighb = Fr_lstTmStp.mesh.NeiElements[EltsTipNew, :]
    # for i in range(0, len(EltsTipNew)):
    #     if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
    #         exitstatus = 12
    #         return exitstatus, None
    if len(np.intersect1d(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0:
        exitstatus = 12
        return exitstatus, None

    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = Integral_over_cell(EltsTipNew,
                                    alpha_k,
                                    l_k,
                                    Fr_lstTmStp.mesh,
                                    'A',
                                    projMethod=sim_properties.projMethod) / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1.0 + 1e-4)] = 1.0

    # if filling fraction is below zero or above 1+1e-6
    if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
        exitstatus = 9
        return exitstatus, None

    if sim_properties.projMethod != 'LS_continousfront':
        # todo: some of the list are redundant to calculate on each iteration
        # Evaluate the element lists for the trial fracture front
        # new tip elements contain only the partially filled elements
        (EltChannel_k,
         EltTip_k,
         EltCrack_k,
         EltRibbon_k,
         zrVertx_k,
         CellStatus_k) = UpdateLists(Fr_lstTmStp.EltChannel,
                                     EltsTipNew,
                                     FillFrac_k,
                                     sgndDist_k,
                                     Fr_lstTmStp.mesh)

    elif sim_properties.projMethod == 'LS_continousfront':
        # new tip elements contain only the partially filled elements
        zrVertx_k = zrVertx_k_without_fully_traversed
        (EltChannel_k,
         EltTip_k,
         EltCrack_k,
         EltRibbon_k,
         CellStatus_k) = UpdateListsFromContinuousFrontRec(newRibbon,
                                                           sgndDist_k,
                                                           Fr_lstTmStp.EltChannel,
                                                           EltsTipNew,
                                                           listofTIPcellsONLY,
                                                           Fr_lstTmStp.mesh)
        if np.isnan(EltChannel_k).any():
            exitstatus = 3
            return exitstatus, None

    # EletsTipNew may contain fully filled elements also. Identifying only the partially filled elements
    partlyFilledTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]

    if sim_properties.verbosity > 1:
        print('Solving the EHL system with the new trial footprint')

    if sim_properties.projMethod != 'LS_continousfront':
    # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    else: zrVrtx_newTip = zrVertx_k_with_fully_traversed.transpose()
    # finding ribbon cells corresponding to tip cells
    corr_ribbon = find_corresponding_ribbon_cell(EltsTipNew,
                                                 alpha_k,
                                                 zrVrtx_newTip,
                                                 Fr_lstTmStp.mesh)
    Cprime_tip = mat_properties.Cprime[corr_ribbon]

    if sim_properties.paramFromTip or mat_properties.anisotropic_K1c:
        Kprime_tip = (32 / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                         Fr_lstTmStp.mesh,
                                                                         mat_properties,
                                                                         alpha_k,
                                                                         l_k,
                                                                         zrVrtx_newTip)
    else:
        Kprime_tip = mat_properties.Kprime[corr_ribbon]

    if mat_properties.TI_elasticity:
        Eprime_tip = TI_plain_strain_modulus(alpha_k,
                                             mat_properties.Cij)
    else:
        Eprime_tip = np.full((EltsTipNew.size,), mat_properties.Eprime, dtype=np.float64)

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    if perfNode is not None:
        perfNode_tipWidth = instrument_start('tip width', perfNode)
        # todo close tip width instrumentation

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = (-(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) /
                (Fr_lstTmStp.mesh.hx**2 + Fr_lstTmStp.mesh.hy**2)**0.5 < sim_properties.toleranceVStagnant)
    # we need to remove it:
    # if stagnant.any() and not ((sim_properties.get_tipAsymptote() == 'U') or (sim_properties.get_tipAsymptote() == 'U1')):
    #     if sim_properties.verbosity > 1:
    #         print("Stagnant front is only supported with universal tip asymptote. Continuing...")
    #     stagnant = np.full((EltsTipNew.size,), False, dtype=bool)

    if stagnant.any():
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(Fr_lstTmStp.w,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        Eprime_tip)

        # todo: Find the right cause of failure
        # if the stress Intensity factor cannot be found. The most common reason is wiggles in the front resulting
        # in isolated tip cells.
        if np.isnan(KIPrime).any():
            exitstatus = 8
            return exitstatus, None

        # Calculate average width in the tip cells by integrating tip asymptote. Width of stagnant cells are calculated
        # using the stress intensity factor (see Dontsov and Peirce, JFM RAPIDS, 2017)

        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  stagnant=stagnant,
                                  KIPrime=KIPrime,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip) / Fr_lstTmStp.mesh.EltArea
    else:
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  Kprime=Kprime_tip,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip,
                                  stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea

    # check if the tip volume has gone into negative
    smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])

    if (wTip < 0).any() or sum(wTip) == 0.:
        exitstatus = 4
        return exitstatus, None

    if perfNode is not None:
        pass
        # todo close tip width instrumentation

    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if sum(mat_properties.Cprime[EltsTipNew]) > 0:
        # Calculate leak-off term for the tip cell
        LkOff[EltsTipNew] = 2 * mat_properties.Cprime[EltsTipNew] * Integral_over_cell(EltsTipNew,
                                                                                       alpha_k,
                                                                                       l_k,
                                                                                       Fr_lstTmStp.mesh,
                                                                                       'Lk',
                                                                                       mat_prop=mat_properties,
                                                                                       frac=Fr_lstTmStp,
                                                                                       Vel=Vel_k,
                                                                                       dt=timeStep,
                                                                                       arrival_t=
                                                                                       Fr_lstTmStp.TarrvlZrVrtx[
                                                                                           EltsTipNew])
        if np.isnan(LkOff[EltsTipNew]).any():
            exitstatus = 13
            return exitstatus, None

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0:
        t_since_arrival = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_since_arrival[t_since_arrival < 0.] = 0.
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * ((t_since_arrival
                                                                                              + timeStep) ** 0.5 - t_since_arrival ** 0.5) * Fr_lstTmStp.mesh.EltArea
        if np.isnan(LkOff[Fr_lstTmStp.EltChannel]).any():
            exitstatus = 13
            return exitstatus, None

        if stagnant.any():
            LkOff[EltsTipNew[stagnant]] = leak_off_stagnant_tip(EltsTipNew[stagnant],
                                                                l_k[stagnant],
                                                                alpha_k[stagnant],
                                                                Fr_lstTmStp.TarrvlZrVrtx[EltsTipNew[stagnant]],
                                                                Fr_lstTmStp.time + timeStep,
                                                                mat_properties.Cprime,
                                                                timeStep,
                                                                Fr_lstTmStp.mesh)

    # set leak off to zero if pressure below pore pressure
    LkOff[Fr_lstTmStp.pFluid <= mat_properties.porePressure] = 0.

    w_n_plus1, pf_n_plus1, data = solve_width_pressure(Fr_lstTmStp,
                                                       sim_properties,
                                                       fluid_properties,
                                                       mat_properties,
                                                       EltsTipNew,
                                                       partlyFilledTip,
                                                       C,
                                                       FillFrac_k,
                                                       EltCrack_k,
                                                       InCrack_k,
                                                       LkOff,
                                                       wTip,
                                                       timeStep,
                                                       Qin,
                                                       perfNode,
                                                       Vel_k,
                                                       corr_ribbon)

    # check if the new width is valid
    if np.isnan(w_n_plus1).any():
        exitstatus = 5
        return exitstatus, None


    fluidVel = data[0][0]
    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = np.copy(Fr_lstTmStp.Tarrival)
    max_Tarrival = np.nanmax(Tarrival_k)
    nc = np.setdiff1d(EltChannel_k, Fr_lstTmStp.EltChannel)
    new_channel = np.array([], dtype=int)
    for i in nc:
        new_channel = np.append(new_channel, np.where(EltsTipNew == i)[0])
    if np.any(Vel_k[new_channel]==0):
        print("why we have zeros?")
    t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] / Vel_k[new_channel]
    max_l = Fr_lstTmStp.mesh.hx * np.cos(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * np.sin(alpha_k[new_channel])
    t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) / Vel_k[new_channel]
    Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2
    to_correct = np.where(Tarrival_k[EltsTipNew[new_channel]] < max_Tarrival)[0]
    Tarrival_k[EltsTipNew[new_channel[to_correct]]] = max_Tarrival

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.w = w_n_plus1
    Fr_kplus1.pFluid = pf_n_plus1
    Fr_kplus1.pNet = np.zeros((Fr_kplus1.mesh.NumberOfElts,))
    Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.time += timeStep
    Fr_kplus1.closed = data[1]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.InCrack = InCrack_k
    if sim_properties.projMethod != 'LS_continousfront':
        Fr_kplus1.process_fracture_front()
    else :
        Fr_kplus1.Ffront = Ffront
        Fr_kplus1.number_of_fronts = number_of_fronts
        if sim_properties.saveToDisk and sim_properties.saveStatisticsPostCoalescence and Fr_lstTmStp.number_of_fronts != Fr_kplus1.number_of_fronts:
            myJsonName = sim_properties.set_outputFolder+"_mesh_study.json"
            append_to_json_file(myJsonName, Fr_kplus1.mesh.nx, 'append2keyAND2list', key='nx')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.ny, 'append2keyAND2list', key='ny')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hx, 'append2keyAND2list', key='hx')
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hy, 'append2keyAND2list', key='hy')
            append_to_json_file(myJsonName, Fr_kplus1.EltCrack.size, 'append2keyAND2list', key='elements_in_crack')
            append_to_json_file(myJsonName, Fr_kplus1.EltTip.size, 'append2keyAND2list', key='elements_in_tip')
            append_to_json_file(myJsonName, Fr_kplus1.time, 'append2keyAND2list', key='coalescence_time')
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.Tarrival = Tarrival_k
    Fr_kplus1.wHist = np.maximum(Fr_kplus1.w, Fr_lstTmStp.wHist)
    Fr_kplus1.effVisc = data[0][1]
    Fr_kplus1.yieldRatio = data[0][2]

    if sim_properties.verbosity > 1:
        print("Solved...\nFinding velocity of front...")

    itr = 0
    # toughness iteration loop
    while itr < sim_properties.maxProjItrs:

        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.TI_elasticity:
            if sim_properties.projMethod == 'ILSA_orig':
                projection_method = projection_from_ribbon
            elif sim_properties.projMethod == 'LS_grad':
                projection_method = projection_from_ribbon_LS_gradient
            elif sim_properties.projMethod == 'LS_continousfront':
                projection_method = projection_from_ribbon_LS_gradient #todo: test this case!!!

            if itr == 0 :
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   Fr_lstTmStp.EltChannel,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k)
                alpha_ribbon_km1 = np.zeros(Fr_lstTmStp.EltRibbon.size, )
            else:
                alpha_ribbon_k = 0.3 * alpha_ribbon_k + 0.7 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                Fr_lstTmStp.EltChannel,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k)
            if np.isnan(alpha_ribbon_k).any():
                exitstatus = 11
                return exitstatus, None

        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c:

            Kprime_k = get_toughness_from_cellCenter(alpha_ribbon_k,
                                                     sgndDist_k,
                                                     Fr_lstTmStp.EltRibbon,
                                                     mat_properties,
                                                     Fr_lstTmStp.mesh) * (32 / np.pi) ** 0.5

            if np.isnan(Kprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Kprime_k = None

        if mat_properties.TI_elasticity:
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if np.isnan(Eprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Eprime_k = None

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely
        # large float value. (algorithm requires inf)

        perfNode_tipInv = instrument_start('tip inversion', perfNode)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(Fr_kplus1.w,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               fluid_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k,
                                                               Eprime_k=Eprime_k)

        status, fail_cause = True, None
        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            status = False
            fail_cause = 'tip inversion failed'
            exitstatus = 7

        if perfNode_tipInv is not None:
            instrument_close(perfNode, perfNode_tipInv, None, len(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            perfNode.tipInv_data.append(perfNode_tipInv)

        if not status:
            return exitstatus, None

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
        front_region = np.where(abs(Fr_lstTmStp.sgndDist) < current_prefactor * 6.66 * (
                Fr_lstTmStp.mesh.hx ** 2 + Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]

        if not np.in1d(Fr_kplus1.EltTip, front_region).any():
            raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                             " level set.")
        # the search region outwards from the front position at last time step
        pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx ** 2 +
                                                                       Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
        # the search region inwards from the front position at last time step
        ngtv_region = np.where(Fr_lstTmStp.sgndDist[front_region] < 0)[0]

        # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
        SolveFMM(sgndDist_k,
                 Fr_lstTmStp.EltRibbon,
                 Fr_lstTmStp.EltChannel,
                 Fr_lstTmStp.mesh,
                 front_region[pstv_region],
                 front_region[ngtv_region])

        # do it only once if not anisotropic
        if not (sim_properties.paramFromTip or mat_properties.anisotropic_K1c
                or mat_properties.TI_elasticity) or sim_properties.explicitProjection:
            break

        norm = np.linalg.norm(abs(alpha_ribbon_k - alpha_ribbon_km1) / np.pi * 2)
        if norm < sim_properties.toleranceProjection:
            if sim_properties.verbosity > 1:
                print("Projection iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break
        alpha_ribbon_km1 = np.copy(alpha_ribbon_k)
        if sim_properties.verbosity > 1:
            print("iterating on projection... norm = " + repr(norm))
        itr += 1

    # todo Hack!!! keep going if projection does not converge
    # if itr == sim_properties.maxProjItrs:
    #     exitstatus = 10
    #     return exitstatus, None

    Fr_kplus1.v = -(sgndDist_k[Fr_kplus1.EltTip] - Fr_lstTmStp.sgndDist[Fr_kplus1.EltTip]) / timeStep
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    new_tip = np.where(np.isnan(Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip]))[0]
    if np.any(Fr_kplus1.v[new_tip]==0):
        print('why')
    Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] / Fr_kplus1.v[new_tip]
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += np.sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol
    Fr_kplus1.source = np.where(Qin != 0)[0]

    if sim_properties.saveRegime:
        regime = np.full((Fr_lstTmStp.mesh.NumberOfElts,), np.nan, dtype=np.float32)
        regime[Fr_lstTmStp.EltRibbon] = find_regime(Fr_kplus1.w,
                                                    Fr_lstTmStp,
                                                    mat_properties,
                                                    sim_properties,
                                                    timeStep,
                                                    Kprime_k,
                                                    -sgndDist_k[Fr_lstTmStp.EltRibbon])
        Fr_kplus1.regime = regime

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

    if data[2]:
        return 14, Fr_kplus1

    exitstatus = 1
    return exitstatus, Fr_kplus1

