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

# Internal imports
from level_set.continuous_front_reconstruction import you_advance_more_than_2_cells
from utilities.labels import TS_errorMessages
from time_step.ts_explicit_front import time_step_explicit_front
from time_step.ts_implicit_front import injection_same_footprint, injection_extended_footprint
from level_set.level_set import get_front_region
from level_set.anisotropy import projection_from_ribbon_LS_gradient_at_tip, get_toughness_from_cellCenter_iter
from properties import IterationProperties, instrument_start, instrument_close

# ----------------------------------------------------------------------------------------------------------------------

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
        stagnant_crt[np.where(Kprime_k.of(Fr_k.sgndDist[Fr_k.EltRibbon], mesh = Fr_k.mesh, ribbon = Fr_k.EltRibbon) *
                              (-Fr_k.sgndDist[Fr_k.EltRibbon]) ** 0.5 /
                              (mat_properties.Eprime * Fr_k.w[Fr_k.EltRibbon]) > 1)[0]] = True

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
                log.debug('Norm of subsequent filling fraction estimates = ' + repr(norm_first_it)
                          + ' forcing 2 iter min.')

        # sometimes the code is going to fail because of the max number of iterations due to the lack of
        # improvement of the norm
        if norm is not np.nan:
            if abs((previous_norm-norm)/norm) < 0.0001 and k > 35:
                log.debug('Norm of subsequent Norms of subsequent filling fraction estimates = ' +
                          str(abs((previous_norm-norm)/norm)) + ' < 0.0001')
                exitstatus = 15
                return exitstatus, None

        # preventing infinite or not effective loops
        if (k >= sim_properties.maxFrontItrs and norm > 0.026) or k > 100:
            if norm > 10 * sim_properties.tolFractFront or k > 200:
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
        if you_advance_more_than_2_cells(Fr_k.fully_traversed,
                                         Frac.EltTip,
                                         Frac.mesh.NeiElements,
                                         Frac.Ffront,
                                         Fr_k.Ffront,
                                         Fr_k.mesh) and \
                                         sim_properties.limitAdancementTo2cells:
            exitstatus = 17
            return exitstatus, Frac

    log.debug("Fracture front converged after " + repr(k) + " iterations with norm = " + repr(norm))

    return exitstatus, Fr_k
