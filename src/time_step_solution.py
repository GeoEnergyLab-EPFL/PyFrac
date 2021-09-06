# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
#import time
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
# local imports
import logging
from volume_integral import leak_off_stagnant_tip, find_corresponding_ribbon_cell
from symmetry import get_symetric_elements, self_influence
from tip_inversion import TipAsymInversion, StressIntensityFactor
from elastohydrodynamic_solver import *
from level_set import SolveFMM, reconstruct_front, reconstruct_front_LS_gradient, UpdateLists, get_front_region
from continuous_front_reconstruction import reconstruct_front_continuous, UpdateListsFromContinuousFrontRec, you_advance_more_than_2_cells
from properties import IterationProperties, instrument_start, instrument_close
from anisotropy import *
from labels import TS_errorMessages
from explicit_RKL import solve_width_pressure_RKL2
from postprocess_fracture import append_to_json_file
import Hdot
from utility import append_new_line


def attempt_time_step(Frac, C, Boundary, mat_properties, fluid_properties, sim_properties, inj_properties,
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
    log = logging.getLogger('PyFrac.attempt_time_step')
    Qin = inj_properties.get_injection_rate(Frac.time, Frac)
    if inj_properties.sinkLocFunc is not None:
        Qin[inj_properties.sinkElem] -= inj_properties.sinkVel * Frac.mesh.EltArea

    if inj_properties.delayed_second_injpoint_elem is not None:
        if inj_properties.rate_delayed_inj_pt_func is None:
            if Frac.time >= inj_properties.injectionTime_delayed_second_injpoint:
                Qin[inj_properties.delayed_second_injpoint_elem]= inj_properties.injectionRate_delayed_second_injpoint/len(inj_properties.delayed_second_injpoint_elem)
            else:
                Qin[inj_properties.delayed_second_injpoint_elem] = inj_properties.init_rate_delayed_second_injpoint/len(inj_properties.delayed_second_injpoint_elem)
        else:
            Qin[inj_properties.delayed_second_injpoint_elem] = inj_properties.rate_delayed_inj_pt_func(Frac.time)/len(inj_properties.delayed_second_injpoint_elem)
        log.debug("\n  max value of the array Q(x,y) =   " + str(Qin.max()))
        log.debug("\n  Q at the delayed inj point    =   " + str(Qin[inj_properties.delayed_second_injpoint_elem]))

    if sim_properties.frontAdvancing == 'explicit':

        perfNode_explFront = instrument_start('extended front', perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    Boundary,
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

        # check if we advanced more than two cells
        if exitstatus == 1:
            if you_advance_more_than_2_cells(Fr_k.fully_traversed, Frac.EltTip, Frac.mesh.NeiElements, Frac.Ffront, Fr_k.Ffront, Fr_k.mesh) and \
                sim_properties.limitAdancementTo2cells:
                exitstatus = 17
                return exitstatus, Frac

        return exitstatus, Fr_k

    elif sim_properties.frontAdvancing == 'predictor-corrector':
        log.debug('Advancing front with velocity from last time-step...')

        perfNode_explFront = instrument_start('extended front', perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    Boundary,
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
        log.debug('Solving ElastoHydrodynamic equations with same footprint...')

        perfNode_sameFP = instrument_start('same front', perfNode)

        # width by injecting the fracture with the same footprint (balloon like inflation)
        exitstatus, Fr_k = injection_same_footprint(Frac,
                                                    C,
                                                    Boundary,
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
    if mat_properties.inv_with_heter_K1c:
        front_region = get_front_region(Fr_k.mesh, Fr_k.EltRibbon, Fr_k.sgndDist[Fr_k.EltRibbon])
        alpha_ribbon = projection_from_ribbon_LS_gradient_at_tip(Fr_k.EltRibbon,
                                           front_region,
                                           Fr_k.mesh,
                                           Fr_k.sgndDist,
                                           global_alpha=mat_properties.inv_with_heter_K1c)
        Kprime_k = get_toughness_from_cellCenter_iter(alpha_ribbon, Fr_k.mesh.CenterCoor[Fr_k.EltRibbon],
                                                      mat_properties)
        stagnant_crt[np.where(Kprime_k.of(Fr_k.sgndDist[Fr_k.EltRibbon], mesh = Fr_k.mesh, ribbon = Fr_k.EltRibbon) * (-Fr_k.sgndDist[Fr_k.EltRibbon]) ** 0.5 / (
                mat_properties.Eprime * Fr_k.w[Fr_k.EltRibbon]) > 1)[0]] = True
        # from utility import plot_as_matrix
        # K = np.zeros((Fr_k.mesh.NumberOfElts,), )
        # K[Fr_k.EltRibbon] = Kprime_k.of(Fr_k.sgndDist[Fr_k.EltRibbon])
        # plot_as_matrix(K, Fr_k.mesh)
    else:
        stagnant_crt[np.where(mat_properties.Kprime[Fr_k.EltRibbon] * (-Fr_k.sgndDist[Fr_k.EltRibbon]) ** 0.5 / (
                mat_properties.Eprime * Fr_k.w[Fr_k.EltRibbon]) > 1)[0]] = True
    # stagnant cells where fracture is closed
    stagnant_closed = np.full((len(Fr_k.EltRibbon),), False, dtype=bool)
    for i in range(len(Fr_k.EltRibbon)):
        stagnant_closed[i] = Fr_k.EltRibbon[i] in Fr_k.closed
    stagnant = np.bitwise_or(stagnant_closed, stagnant_crt)

    if np.all(stagnant):
        delta_w = np.abs(Fr_k.w[Fr_k.EltRibbon]-Frac.w[Fr_k.EltRibbon])
        if np.sum(Qin) != 0 and np.max(delta_w) < sim_properties.tolerancewIncr:
            log.warning('The time step is too small to induce a significant change in opening')
            return 18, Fr_k
        else:
            return 1, Fr_k

    log.debug('Starting Fracture Front loop...')

    norm = 10.
    k = 0
    previous_norm = 100 # initially set with a big value
    loop_already_forced = False
    force_1loop = False
    # Fracture front loop to find the correct front location
    while norm > sim_properties.tolFractFront or force_1loop:
        k = k + 1
        log.debug(' ')
        log.debug('Iteration ' + repr(k))
        fill_frac_last = np.copy(Fr_k.FillF)

        # update the confining stress
        # if mat_properties.boundaryEffect.active:
        #     mat_properties.updateConfiningStress(Fr_k.w, Fr_k.EltCrack)

        perfNode_extFront = instrument_start('extended front', perfNode)
        # find the new footprint and solve the elastohydrodynamic equations to to get the new fracture
        (exitstatus, Fr_k) = injection_extended_footprint(Fr_k.w,
                                                          Frac,
                                                          C,
                                                          Boundary,
                                                          timeStep,
                                                          Qin,
                                                          mat_properties,
                                                          fluid_properties,
                                                          sim_properties,
                                                          perfNode_extFront,
                                                          front_previous_iter=Fr_k.Ffront)

        if exitstatus == 1:
            # norm is evaluated by dividing the difference in the area of the tip cells between two successive
            # iterations with the number of tip cells.
            if not k == 1:
                norm = abs((sum(Fr_k.FillF) - sum(fill_frac_last)) / len(Fr_k.FillF))
                norm_first_it = None
            else:
                norm_first_it = abs((sum(Fr_k.FillF) - sum(fill_frac_last)) / len(Fr_k.FillF))
        else:
            norm = np.nan

        if perfNode_extFront is not None:
            instrument_close(perfNode, perfNode_extFront, norm,
                             len(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus], Frac.time)
            perfNode.extendedFront_data.append(perfNode_extFront)

        if exitstatus != 1:
            return exitstatus, Fr_k
        if not k == 2:
            log.debug('Norm of subsequent filling fraction estimates = ' + repr(norm))
        else:
            if norm_first_it is None:
                log.debug('Norm of subsequent filling fraction estimates = ' + repr(norm))
            else:
                log.debug('Norm of subsequent filling fraction estimates = ' + repr(norm_first_it) + ' forcing 2 iter min.')


        # sometimes the code is going to fail because of the max number of iterations due to the lack of
        # improvement of the norm
        if norm is not np.nan:
            if abs((previous_norm-norm)/norm) < 0.0001 and k > 35:
                log.debug( 'Norm of subsequent Norms of subsequent filling fraction estimates = ' + str(abs((previous_norm-norm)/norm)) + ' < 0.0001')
                exitstatus = 15
                return exitstatus, None

        # preventing infinite or not effective loops
        if (k >= sim_properties.maxFrontItrs and norm > 0.026) or k > 100:
            if norm > 10*sim_properties.tolFractFront or  k>200:
                exitstatus = 6
                return exitstatus, None

        if norm < sim_properties.tolFractFront and loop_already_forced:
            if norm < previous_norm:
                # the convergence has been achieved
                loop_already_forced = False
                force_1loop = False
            else:
                # forcing another loop
                loop_already_forced = True
                force_1loop = True
        elif norm < sim_properties.tolFractFront and not loop_already_forced:
            log.debug(' --> Forcing one more loop to see if the convergence has been really achieved')
            loop_already_forced = True
            force_1loop = True
        elif norm >= sim_properties.tolFractFront and loop_already_forced:
            log.debug(' --> convergence not achieved, iterate more if other conditions allows for that')
            loop_already_forced = False
            force_1loop = False
        elif norm >= sim_properties.tolFractFront and not loop_already_forced:
            # normal case
            loop_already_forced = False
            force_1loop = False

        previous_norm = norm

    # check if we advanced more than two cells
    if exitstatus == 1:
        if you_advance_more_than_2_cells(Fr_k.fully_traversed, Frac.EltTip, Frac.mesh.NeiElements, Frac.Ffront, Fr_k.Ffront, Fr_k.mesh) and \
                sim_properties.limitAdancementTo2cells:
            exitstatus = 17
            return exitstatus, Frac

    log.debug("Fracture front converged after " + repr(k) + " iterations with norm = " + repr(norm))

    return exitstatus, Fr_k


# ----------------------------------------------------------------------------------------------------------------------

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

    # todo: make tip correction while injecting in the same footprint
    ##########################################################################################
    #                                                                                        #
    #  when we inject on the same footprint we should make the tip correction at the tip     #
    #  this is not done, but it will not affect the accuracy, only the speed of convergence  #
    #  since we are never accessing the diagonal of the elasticity matrix when iterating on  #
    #  the position of the front.                                                            #
    #                                                                                        #
    ##########################################################################################
    w_k, p_k, return_data = solve_width_pressure(Fr_lstTmStp, #Fr_lstTmStp
                                                 sim_properties,
                                                 fluid_properties,
                                                 mat_properties,
                                                 Fr_lstTmStp.EltTip, #empty, #EltTip
                                                 empty, #partlyFilledTip
                                                 C,
                                                 Boundary,
                                                 Fr_lstTmStp.FillF[empty],
                                                 Fr_lstTmStp.EltCrack,
                                                 Fr_lstTmStp.InCrack,
                                                 LkOff,
                                                 Fr_lstTmStp.w[Fr_lstTmStp.EltTip],#empty, #wTip
                                                 timeStep,
                                                 Qin,
                                                 perfNode,
                                                 empty, #Vel
                                                 empty, #corr_ribbon
                                                 doublefracturedictionary= doublefracturedictionary)
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
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol -Fr_kplus1.LkOffTotal) \
                           / Fr_kplus1.injectedVol
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


# -----------------------------------------------------------------------------------------------------------------------


def injection_extended_footprint(w_k, Fr_lstTmStp, C, Boundary, timeStep, Qin, mat_properties, fluid_properties,
                                 sim_properties, perfNode=None, front_previous_iter = None):
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
        | 15      -- iterations on front will not converge (continuous front)
        | 16      -- max number of cells achieved. Reducing the number of cells
        | 17      -- you advanced more than two cells in a row. Repeating with a smaller time step
        | 18      -- the max abs increment in fracture opening is smaller than a given threshold despite a positive injection rate. Try larger time step
        - Fracture:            fracture after advancing time step.

    """
    log = logging.getLogger('PyFrac.injection_extended_footprint')
    itr = 0
    sgndDist_k = np.copy(Fr_lstTmStp.sgndDist)

    # from utility import plot_as_matrix
    # plot_as_matrix(w_k, Fr_lstTmStp.mesh)

    # toughness iteration loop
    while itr < sim_properties.maxProjItrs:
        # get the current direction of propagation
        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.TI_elasticity or mat_properties.inv_with_heter_K1c:
            if sim_properties.projMethod == 'ILSA_orig':
                projection_method = projection_from_ribbon
                second_arg = Fr_lstTmStp.EltChannel
            elif sim_properties.projMethod == 'LS_grad':
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.front_region
            elif sim_properties.projMethod == 'LS_continousfront': #todo: test this case!!!
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.front_region
            if itr == 0 :
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   second_arg,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k,
                                                   global_alpha=mat_properties.inv_with_heter_K1c)
                alpha_ribbon_km1 = np.zeros(Fr_lstTmStp.EltRibbon.size, )
            elif not mat_properties.inv_with_heter_K1c:
                alpha_ribbon_k = 0.25 * alpha_ribbon_k + 0.75 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                second_arg,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k, global_alpha=mat_properties.inv_with_heter_K1c)
            else :
                alpha_ribbon_k = 0.0 * alpha_ribbon_k + 1. * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                second_arg,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k, global_alpha=mat_properties.inv_with_heter_K1c)
                #alpha_ribbon_k[np.where(alpha_ribbon_k>2.*np.pi)[0]] = alpha_ribbon_k[np.where(alpha_ribbon_k>2.*np.pi)[0]] - 2. * np.pi
            # from utility import plot_as_matrix
            # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            # K[Fr_lstTmStp.EltRibbon] = alpha_ribbon_k
            # plot_as_matrix(K, Fr_lstTmStp.mesh)
            if np.isnan(alpha_ribbon_k).any():
                exitstatus = 11
                return exitstatus, None

        if (sim_properties.paramFromTip or mat_properties.anisotropic_K1c) and not mat_properties.inv_with_heter_K1c:
            Kprime_k = get_toughness_from_cellCenter(alpha_ribbon_k,
                                                     sgndDist_k,
                                                     Fr_lstTmStp.EltRibbon,
                                                     mat_properties,
                                                     Fr_lstTmStp.mesh) * (32 / np.pi) ** 0.5
            if np.isnan(Kprime_k).any():
                exitstatus = 11
                return exitstatus, None
        elif mat_properties.inv_with_heter_K1c:
            Kprime_k =  get_toughness_from_cellCenter_iter(alpha_ribbon_k, Fr_lstTmStp.mesh.CenterCoor[Fr_lstTmStp.EltRibbon], mat_properties)
        else:
            Kprime_k = None

        # ----- plot to check -----
        #K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
        #K[Fr_lstTmStp.EltRibbon] = Kprime_k.of(sgndDist_k[Fr_lstTmStp.EltRibbon],mesh=Fr_lstTmStp.mesh, ribbon=Fr_lstTmStp.EltRibbon)
        # from utility import plot_as_matrix
        # plot_as_matrix(K, Fr_lstTmStp.mesh)

        if mat_properties.TI_elasticity and not mat_properties.inv_with_heter_K1c:
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if np.isnan(Eprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Eprime_k = None

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        # large float value. (algorithm requires inf)
        sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely

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
            # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
            # K[Fr_lstTmStp.EltRibbon] = sgndDist_k[Fr_lstTmStp.EltRibbon]
            # from utility import plot_as_matrix
            # plot_as_matrix(K, Fr_lstTmStp.mesh)
            # np.argwhere(np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon])).flatten()

        if perfNode_tipInv is not None:
            instrument_close(perfNode, perfNode_tipInv, None, len(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            perfNode.tipInv_data.append(perfNode_tipInv)

        if not status:
            return exitstatus, None

        # # Check for positive
        # if np.any(sgndDist_k[Fr_lstTmStp.EltRibbon]>0):
        #     log.debug("found a positive signed distance: it must not happen ")
        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        # current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
        # front_region = np.where(abs(Fr_lstTmStp.sgndDist) < current_prefactor * 22.66 * Fr_lstTmStp.mesh.cellDiag)[0]

        front_region = get_front_region(Fr_lstTmStp.mesh, Fr_lstTmStp.EltRibbon, sgndDist_k[Fr_lstTmStp.EltRibbon])

        #front_region = np.arange(Fr_lstTmStp.mesh.NumberOfElts)
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
                or mat_properties.TI_elasticity or mat_properties.inv_with_heter_K1c) or sim_properties.explicitProjection:
            break

        norm = np.linalg.norm(np.abs(np.sin(alpha_ribbon_k) - np.sin(alpha_ribbon_km1)) + np.abs(np.cos(alpha_ribbon_k) - np.cos(alpha_ribbon_km1)))
        if norm < sim_properties.toleranceProjection:
            log.debug("projection iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break

        alpha_ribbon_km1 = np.copy(alpha_ribbon_k)
        log.debug("iterating on projection... norm " + repr(norm))
        itr += 1

    # if itr == sim_properties.maxProjItrs:
    #     exitstatus = 10
    #     return exitstatus, None

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
        correct_size_of_pstv_region = [False, False, False]
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
            sgndDist_k_temp, Ffront,number_of_fronts, fronts_dictionary = reconstruct_front_continuous(sgndDist_k,
                                                                          front_region[pstv_region],
                                                                          Fr_lstTmStp.EltRibbon,
                                                                          Fr_lstTmStp.EltChannel,
                                                                          Fr_lstTmStp.mesh,
                                                                          recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                          lstTmStp_EltCrack0=Fr_lstTmStp.fronts_dictionary['crackcells_0'], oldfront=Fr_lstTmStp.Ffront)
            if correct_size_of_pstv_region[2]:
                exitstatus = 7 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, None

            if correct_size_of_pstv_region[1]:
                Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
                Fr_kplus1.EltTipBefore = Fr_lstTmStp.EltTip
                Fr_kplus1.EltTip = EltsTipNew  # !!! EltsTipNew are the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                # (in case of anisotropic remeshing)
                exitstatus = 12 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, Fr_kplus1

            if not correct_size_of_pstv_region[0]:
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = np.unique(np.ndarray.flatten(Fr_lstTmStp.mesh.NeiElements[front_region]))
                #front_region = np.arange(Fr_lstTmStp.mesh.NumberOfElts)
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
        sgndDist_k = sgndDist_k_temp

        del correct_size_of_pstv_region
    else:
        raise SystemExit("projection method not supported")

    # from continuous_front_reconstruction import get_xy_from_Ffront, plot_xy_points
    # x,y = get_xy_from_Ffront(Ffront)
    # plot_xy_points(front_region, Fr_lstTmStp.mesh, sgndDist_k, Fr_lstTmStp.EltRibbon, x, y, fig=None, annotate_cellName=False,
    #                annotate_edgeName=False, annotatePoints=False, grid=True, oldfront=front_previous_iter, joinPoints=True,
    #                disregard_plus=False) # or oldfront=Fr_lstTmStp.Ffront

    if not np.in1d(EltsTipNew, front_region).any():
        raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                         " level set.")

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        # from utility import plot_as_matrix
        # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
        # K[EltsTipNew] = alpha_k
        # plot_as_matrix(K, Fr_lstTmStp.mesh)
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    if len(np.intersect1d(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0:
        Fr_lstTmStp.EltTipBefore = Fr_lstTmStp.EltTip
        Fr_lstTmStp.EltTip = EltsTipNew
        exitstatus = 12
        return exitstatus, Fr_lstTmStp

    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1  #EltsTipNew is new tip + fully traversed

    if len(InCrack_k[np.where(InCrack_k == 1)]) > sim_properties.maxElementIn and \
            sim_properties.meshReductionPossible:
        exitstatus = 16
        return exitstatus, Fr_lstTmStp


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
         CellStatus_k,
         fully_traversed_k) = UpdateLists(Fr_lstTmStp.EltChannel,
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
         CellStatus_k,
         fully_traversed_k) = UpdateListsFromContinuousFrontRec(newRibbon,
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
    log.debug('Solving the EHL system with the new trial footprint')

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
    if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.inv_with_heter_K1c:
        if sim_properties.projMethod != 'LS_continousfront':
            # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
            zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
            # get toughness from tip in case of anisotropic or
            Kprime_tip = (32. / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                             Fr_lstTmStp.mesh,
                                                                             mat_properties,
                                                                             alpha_k,
                                                                             l_k,
                                                                             zrVrtx_newTip)
        else:
            toughness_from_zeroVertex = False
            zrVrtx_newTip = zrVertx_k_with_fully_traversed.transpose()
            if toughness_from_zeroVertex:
                # get toughness from tip in case of anisotropic or
                Kprime_tip = (32. / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                                 Fr_lstTmStp.mesh,
                                                                                 mat_properties,
                                                                                 alpha_k,
                                                                                 l_k,
                                                                                 zrVrtx_newTip)
            else:
                Kprime_tip = (32. / np.pi) ** 0.5 * get_toughness_from_Front(Ffront,
                                                                               EltsTipNew,
                                                                               EltTip_k,
                                                                               fully_traversed_k,
                                                                               Fr_lstTmStp.mesh,
                                                                               mat_properties,
                                                                               alpha_k,
                                                                               get_from_mid_front = False)


    # from utility import plot_as_matrix
    # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
    # K[EltsTipNew] = Kprime_tip
    # plot_as_matrix(K, Fr_lstTmStp.mesh)
    elif not mat_properties.inv_with_heter_K1c:
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
    #     log.warning("Stagnant front is only supported with universal tip asymptote. continuing...")
    #     stagnant = np.full((EltsTipNew.size,), False, dtype=bool)

    # from utility import plot_as_matrix
    # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
    # K[EltsTipNew[np.where(stagnant)[0]]] = 2
    # plot_as_matrix(K, Fr_lstTmStp.mesh)
    # EltsTipNew[np.where(stagnant)[0]]
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
                                  Kprime = Kprime_tip,
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
    smallNgtvWTip = np.where(np.logical_and(wTip < 0., wTip > -1.e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        #  warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])

    if (wTip < 0.).any() or sum(wTip) == 0.:
        exitstatus = 4
        return exitstatus, None

    if perfNode is not None:
        pass
        # todo close tip width instrumentation

    LkOff = np.zeros((len(CellStatus),), dtype=np.float64)
    if sum(mat_properties.Cprime[EltsTipNew]) > 0.:
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

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0.:
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
    if sim_properties.doublefracture and fronts_dictionary['number_of_fronts'] == 2:
        doublefracturedictionary = {"number_of_fronts": fronts_dictionary['number_of_fronts'],
                                    "crackcells_0": fronts_dictionary['crackcells_0'],
                                    "crackcells_1": fronts_dictionary['crackcells_1'],
                                    "TIPcellsANDfullytrav_0": fronts_dictionary['TIPcellsANDfullytrav_0'],
                                    "TIPcellsANDfullytrav_1": fronts_dictionary['TIPcellsANDfullytrav_1']}
    elif sim_properties.projMethod != 'LS_continousfront':
        doublefracturedictionary = {"number_of_fronts": 1}
    else:
        doublefracturedictionary = {"number_of_fronts":fronts_dictionary['number_of_fronts']}

    w_n_plus1, pf_n_plus1, data = solve_width_pressure(Fr_lstTmStp,
                                                       sim_properties,
                                                       fluid_properties,
                                                       mat_properties,
                                                       EltsTipNew,
                                                       partlyFilledTip,
                                                       C,
                                                       Boundary,
                                                       FillFrac_k,
                                                       EltCrack_k,
                                                       InCrack_k,
                                                       LkOff,
                                                       wTip,
                                                       timeStep,
                                                       Qin,
                                                       perfNode,
                                                       Vel_k,
                                                       corr_ribbon,
                                                       doublefracturedictionary=doublefracturedictionary)
    # from utility import plot_as_matrix
    # K = pf_n_plus1
    # plot_as_matrix(K, Fr_lstTmStp.mesh)
    # check if the new width is valid
    if np.isnan(w_n_plus1).any():
        exitstatus = 5
        return exitstatus, None
    if data[0] != None:
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
    if Boundary is not None:
        Fr_kplus1.boundEffTraction = Boundary.last_traction
        Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k] - Boundary.last_traction[EltCrack_k]
    else:
        Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.front_region = front_region
    Fr_kplus1.ZeroVertex = zrVertx_k
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.fully_traversed = fully_traversed_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = Vel_k[partlyFilledTip]
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.InCrack = InCrack_k
    if sim_properties.projMethod != 'LS_continousfront':
        Fr_kplus1.process_fracture_front()
    else :
        Fr_kplus1.fronts_dictionary = fronts_dictionary
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

    if sim_properties.saveRegime and not sim_properties.get_volumeControl():
        # regime = find_regime(w_k, Fr_lstTmStp, mat_properties, fluid_properties, sim_properties, timeStep, Kprime_k,
        #                      -sgndDist_k[Fr_lstTmStp.EltRibbon])
        Fr_kplus1.update_tip_regime(mat_properties, fluid_properties, timeStep)
        # Fr_kplus1.regime =regime

    Fr_kplus1.source = Fr_lstTmStp.EltCrack[np.where(Qin[Fr_lstTmStp.EltCrack] != 0)[0]]
    if data[0] != None:
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
                    "WARNING: Volume Control system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
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
                    if sim_properties.solveSparse:
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


def time_step_explicit_front(Fr_lstTmStp, C, Boundary, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
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
            | 15      -- iterations on front will not converge (continuous front)
            | 16      -- max number of cells achieved. Reducing the number of cells
            | 17      -- you advanced more than two cells in a row. Repeating with a smaller time step
            | 18      -- the max abs increment in fracture opening is smaller than a given threshold despite a positive injection rate. Try larger time step
        - Fracture:            fracture after advancing time step.

    """
    log = logging.getLogger('PyFrac.time_step_explicit_front')
    sgndDist_k = 1e50 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with maximum
                                                                          # float value. (algorithm requires inf)
    sgndDist_k[Fr_lstTmStp.EltChannel] = 0  # for cells inside the fracture

    sgndDist_k[Fr_lstTmStp.EltTip] = Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep *
                                                                                 Fr_lstTmStp.v)
    current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
    cell_diag = (Fr_lstTmStp.mesh.hx ** 2 + Fr_lstTmStp.mesh.hy ** 2) ** 0.5
    expected_range = max(current_prefactor * 22.66 * cell_diag, 1.5 * cell_diag) # expected range of possible propagation
    front_region = np.where(abs(Fr_lstTmStp.sgndDist) < expected_range)[0]
    #front_region = np.arange(Fr_lstTmStp.mesh.NumberOfElts)
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
        correct_size_of_pstv_region = [False, False, False]
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
            sgndDist_k_temp, Ffront,number_of_fronts, fronts_dictionary = reconstruct_front_continuous(sgndDist_k,
                                                                          front_region[pstv_region],
                                                                          Fr_lstTmStp.EltRibbon,
                                                                          Fr_lstTmStp.EltChannel,
                                                                          Fr_lstTmStp.mesh,
                                                                          recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                          lstTmStp_EltCrack0=Fr_lstTmStp.fronts_dictionary['crackcells_0'], oldfront=Fr_lstTmStp.Ffront)
            if correct_size_of_pstv_region[2]:
                exitstatus = 7 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, None

            if correct_size_of_pstv_region[1]:
                Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
                Fr_kplus1.EltTipBefore = Fr_lstTmStp.EltTip
                Fr_kplus1.EltTip = EltsTipNew  # !!! EltsTipNew are the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                # (in case of anisotropic remeshing)
                exitstatus = 12 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, Fr_kplus1


            if not correct_size_of_pstv_region[0]:
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = np.unique(np.ndarray.flatten(Fr_lstTmStp.mesh.NeiElements[front_region]))
                #front_region = np.arange(Fr_lstTmStp.mesh.NumberOfElts)
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
    if len(np.intersect1d(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0:
        Fr_lstTmStp.EltTipBefore = Fr_lstTmStp.EltTip
        Fr_lstTmStp.EltTip = EltsTipNew
        exitstatus = 12
        return exitstatus, Fr_lstTmStp

    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1

    if len(InCrack_k[np.where(InCrack_k == 1)]) > sim_properties.maxElementIn and \
            sim_properties.meshReductionPossible:
        exitstatus = 16
        return exitstatus, Fr_lstTmStp

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
         CellStatus_k,
         fully_traversed_k) = UpdateLists(Fr_lstTmStp.EltChannel,
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
         CellStatus_k,
         fully_traversed_k) = UpdateListsFromContinuousFrontRec(newRibbon,
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

    log.debug('Solving the EHL system with the new trial footprint')

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
    #     log.warning("Stagnant front is only supported with universal tip asymptote. Continuing...")
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

    LkOff = np.zeros((len(CellStatus),), dtype=np.float64)
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
    if sim_properties.doublefracture and fronts_dictionary['number_of_fronts'] == 2:
        doublefracturedictionary = {"number_of_fronts": fronts_dictionary['number_of_fronts'],
                                    "crackcells_0": fronts_dictionary['crackcells_0'],
                                    "crackcells_1": fronts_dictionary['crackcells_1'],
                                    "TIPcellsANDfullytrav_0": fronts_dictionary['TIPcellsANDfullytrav_0'],
                                    "TIPcellsANDfullytrav_1": fronts_dictionary['TIPcellsANDfullytrav_1']}
    elif sim_properties.projMethod != 'LS_continousfront':
        doublefracturedictionary = {"number_of_fronts": 1}
    else:
         doublefracturedictionary = {"number_of_fronts":fronts_dictionary['number_of_fronts']}
    w_n_plus1, pf_n_plus1, data = solve_width_pressure(Fr_lstTmStp,
                                                       sim_properties,
                                                       fluid_properties,
                                                       mat_properties,
                                                       EltsTipNew,
                                                       partlyFilledTip,
                                                       C,
                                                       Boundary,
                                                       FillFrac_k,
                                                       EltCrack_k,
                                                       InCrack_k,
                                                       LkOff,
                                                       wTip,
                                                       timeStep,
                                                       Qin,
                                                       perfNode,
                                                       Vel_k,
                                                       corr_ribbon,
                                                       doublefracturedictionary = doublefracturedictionary)

    # check if the new width is valid
    if np.isnan(w_n_plus1).any():
        exitstatus = 5
        return exitstatus, None

    if data[0] != None: #todo: Check why we need this if condition in the case of volume control
        fluidVel = data[0][0]
    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = np.copy(Fr_lstTmStp.Tarrival)
    max_Tarrival = np.nanmax(Tarrival_k)
    nc = np.setdiff1d(EltChannel_k, Fr_lstTmStp.EltChannel)
    new_channel = np.array([], dtype=int)
    for i in nc:
        new_channel = np.append(new_channel, np.where(EltsTipNew == i)[0])
    if np.any(Vel_k[new_channel]==0):
        log.debug("why we have zeros?")
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
    if Boundary is not None:
        Fr_kplus1.boundEffTraction = Boundary.last_traction
        Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k] - Fr_kplus1.boundEffTraction[EltCrack_k]
    else:
        Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.time += timeStep
    Fr_kplus1.closed = data[1]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.fully_traversed = fully_traversed_k
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
        Fr_kplus1.fronts_dictionary = fronts_dictionary
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
    if data[0] != None: #todo: Check why we need  this if condition in the case of volume control
        Fr_kplus1.effVisc = data[0][1]
        Fr_kplus1.yieldRatio = data[0][2]


    log.debug("Solved...")
    log.debug("Finding velocity of front...")

    itr = 0
    # toughness iteration loop
    while itr < sim_properties.maxProjItrs:
        if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.TI_elasticity:
            if sim_properties.projMethod == 'ILSA_orig':
                projection_method = projection_from_ribbon
                second_arg = Fr_lstTmStp.EltChannel
            elif sim_properties.projMethod == 'LS_grad':
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.EltChannel #this is inefficient, the band region should be given instead (look at the implicit case)
            elif sim_properties.projMethod == 'LS_continousfront':
                projection_method = projection_from_ribbon_LS_gradient_at_tip
                second_arg = Fr_lstTmStp.EltChannel #this is inefficient, the band region should be given instead (look at the implicit case)

            if itr == 0 :
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   second_arg,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k)
                alpha_ribbon_km1 = np.zeros(Fr_lstTmStp.EltRibbon.size, )
            else:
                alpha_ribbon_k = 0.25 * alpha_ribbon_k + 0.75 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                second_arg,
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
        front_region = np.where(abs(Fr_lstTmStp.sgndDist) < current_prefactor * 22.66 * Fr_lstTmStp.mesh.cellDiag)[0]
        #front_region = np.arange(Fr_lstTmStp.mesh.NumberOfElts)

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
            log.debug("Projection iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break
        alpha_ribbon_km1 = np.copy(alpha_ribbon_k)
        log.debug("iterating on projection... norm = " + repr(norm))
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
    Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] / Fr_kplus1.v[new_tip]
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += np.sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol
    Fr_kplus1.source = np.where(Qin != 0)[0]

    if sim_properties.saveRegime:
        # regime = np.full((Fr_lstTmStp.mesh.NumberOfElts,), np.nan, dtype=np.float32)
        # regime[Fr_lstTmStp.EltRibbon] = find_regime(Fr_kplus1.w,
        #                                             Fr_lstTmStp,
        #                                             mat_properties,
        #                                             fluid_properties,
        #                                             sim_properties,
        #                                             timeStep,
        #                                             Kprime_k,
        #                                             -sgndDist_k[Fr_lstTmStp.EltRibbon])
        # Fr_kplus1.regime = regime
        Fr_kplus1.update_tip_regime(mat_properties, fluid_properties, timeStep)

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

