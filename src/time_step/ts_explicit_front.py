# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
import copy
import logging

# Internal imports
from level_set.FMM import fmm
from level_set.level_set_utils import get_front_region
from level_set.discontinuous_front_reconstruction import reconstruct_front, UpdateLists
from level_set.continuous_front_reconstruction import reconstruct_front_continuous, UpdateListsFromContinuousFrontRec
from level_set.anisotropy import reconstruct_front_LS_gradient, get_toughness_from_Front
from level_set.anisotropy import find_zero_vertex
from level_set.anisotropy import get_toughness_from_zeroVertex
from level_set.anisotropy import get_fracture_size_dependent_toughness, get_fracture_velocity_dependent_toughness
from time_step.ts_toughess_direction_loop import toughness_direction_loop

from tip.volume_integral import Integral_over_cell
from tip.volume_integral import leak_off_stagnant_tip, find_corresponding_ribbon_cell
from tip.tip_inversion import StressIntensityFactor

from systems.sol_sys_dispatcher import solve_width_pressure
from systems.make_sys_common_fun import calculate_fluid_flow_characteristics_laminar
from properties import IterationProperties, instrument_start
from fluid.reyNumb import turbulence_check_tip
from utilities.postprocess_fracture import append_to_json_file
from solid.elasticity_Transv_Isotropic import TI_plain_strain_modulus
# ----------------------------------------------------------------------------------------------------------------------


def time_step_explicit_front(Fr_lstTmStp, C, Boundary, timeStep, Qin, mat_properties, fluid_properties, inj_properties,
                             sim_properties, perfNode=None):
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

    # We get a front region around the Riboon elements. We expect the front to be in there
    front_region = get_front_region(Fr_lstTmStp.mesh, Fr_lstTmStp.EltRibbon,
                                    Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

    # the search region outwards from the front position at last time step is the front region not being in the channel
    pstv_region = np.setdiff1d(front_region, Fr_lstTmStp.EltChannel)

    # the search region inwards from the front position at last time step is the remaining elements
    ngtv_region = np.setdiff1d(front_region, pstv_region)

    # Creating a fmm structure to solve the level set
    fmmStruct = fmm(Fr_lstTmStp.mesh)

    # We define the tip elements as the known elements and solve from there outwards to the domain boundary.
    fmmStruct.solveFMM((Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v), Fr_lstTmStp.EltTip),
                       np.unique(np.hstack((pstv_region, Fr_lstTmStp.EltTip))), Fr_lstTmStp.mesh)

    # We define the tip elements as the known elements and solve from there inwards (inside the fracture). To do so,
    # we need a sign change on the level set (positive inside)
    toEval = np.unique(np.hstack((ngtv_region, Fr_lstTmStp.EltTip)))
    fmmStruct.solveFMM((-(Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v)), Fr_lstTmStp.EltTip),
                    toEval, Fr_lstTmStp.mesh)

    # The solution stored in the object is the calculated level set. we need however to change the sign as to have
    # negative inside and positive outside.
    sgndDist_k = fmmStruct.LS
    sgndDist_k[toEval] = -sgndDist_k[toEval]

    # In the case of explicit time steps and particular geometries one can get a sign change in the channel for these
    # configurations we implement a front loop to fix the level set there as well
    # recession = True
    # if len(np.where(sgndDist_k[ngtv_region] >= 0)[0]) != 0:
    #     log.warning('We get a recession of the front. We change the scheme to implicit, and re-attempt.')
    #     exitstatus = 19
    #     return exitstatus, None
        # newFixedElts = ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]
    #     newFixedSgndDist = Fr_lstTmStp.sgndDist[newFixedElts]
    #     fixedElts = np.hstack((Fr_lstTmStp.EltTip, newFixedElts))
    #     fixedSgndDist = np.hstack((Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v),
    #                                newFixedSgndDist))
    #     recession = False
    #     iterator = 0
    #
    # while not recession:
    #     log.warning('The channel elements with a negative level set are ' +
    #                 str(ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]))
    #     iterator = iterator + 1
    #     fmmStruct = fmm(Fr_lstTmStp.mesh)
    #
    #     # We define the tip elements as the known elements and solve from there outwards to the domain boundary.
    #     fmmStruct.solveFMM((fixedSgndDist, fixedElts), np.unique(np.hstack((pstv_region, fixedElts))), Fr_lstTmStp.mesh)
    #
    #     # We define the tip elements as the known elements and solve from there inwards (inside the fracture). To do so,
    #     # we need a sign change on the level set (positive inside)
    #     toEval = np.unique(np.hstack((ngtv_region, fixedElts)))
    #     fmmStruct.solveFMM((-fixedSgndDist, fixedElts), toEval, Fr_lstTmStp.mesh)
    #
    #     # The solution stored in the object is the calculated level set. we need however to change the sign as to have
    #     # negative inside and positive outside.
    #     sgndDist_k = fmmStruct.LS
    #     sgndDist_k[toEval] = -sgndDist_k[toEval]
    #
    #     if len(np.where(sgndDist_k[ngtv_region] >= 0)[0]) == 0 and iterator < 100:
    #         recession = True
    #         log.warning('Required iterations on the level set were ' + str(iterator))
    #     elif iterator >= 100:
    #         recession = True
    #         sgndDist_k[ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]] = \
    #             Fr_lstTmStp.sgndDist[ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]]
    #         log.warning('Required iterations exceeded ' + str(iterator) +
    #                     ', we simply fix the cells and accept it like this')
    #     else:
    #         newFixedElts = ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]
    #         fixedElts = np.hstack((fixedElts, newFixedElts))
    #         newFixedSgndDist = Fr_lstTmStp.sgndDist[newFixedElts]
    #         fixedSgndDist = np.hstack((fixedSgndDist, newFixedSgndDist))

    eval_region = np.where(sgndDist_k[front_region] >= -Fr_lstTmStp.mesh.cellDiag)[0]

    # Gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
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
                                                                          front_region[eval_region],
                                                                          Fr_lstTmStp.EltRibbon,
                                                                          Fr_lstTmStp.EltChannel,
                                                                          Fr_lstTmStp.mesh,
                                                                          recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                          lstTmStp_EltCrack0=Fr_lstTmStp.fronts_dictionary['crackcells_0'], oldfront=Fr_lstTmStp.Ffront)
            if correct_size_of_pstv_region[2]:
                exitstatus = 7 # You are here because the level set has negative values until the end of the mesh
                               # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, None

            if correct_size_of_pstv_region[3]:
                exitstatus = 20 # You are here because the level set has negative values in one of the channel cells
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

                # Extend the front region with the neighbours (by one cell)
                front_region = np.unique(np.hstack((front_region,
                                                    np.ndarray.flatten(Fr_lstTmStp.mesh.NeiElements[front_region]))))

                # the search region outwards from the front position at last time step
                pstv_region = np.setdiff1d(front_region, Fr_lstTmStp.EltChannel)

                # the search region inwards from the front position at last time step
                ngtv_region = np.setdiff1d(front_region, pstv_region)

                # Creating a fmm structure to solve the level set
                fmmStruct = fmm(Fr_lstTmStp.mesh)

                # We define the tip elements as the known elements and solve from there outwards to the domain boundary.
                fmmStruct.solveFMM(
                    (Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v), Fr_lstTmStp.EltTip),
                    np.unique(np.hstack((pstv_region, Fr_lstTmStp.EltTip))), Fr_lstTmStp.mesh)

                # We define the tip elements as the known elements and solve from there inwards (inside the fracture). To do so,
                # we need a sign change on the level set (positive inside)
                toEval = np.unique(np.hstack((ngtv_region, Fr_lstTmStp.EltTip)))
                fmmStruct.solveFMM(
                    (-(Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v)), Fr_lstTmStp.EltTip),
                    toEval, Fr_lstTmStp.mesh)

                # The solution stored in the object is the calculated level set. we need however to change the sign as to have
                # negative inside and positive outside.
                sgndDist_k = fmmStruct.LS
                sgndDist_k[toEval] = -sgndDist_k[toEval]

                # In the case of explicit time steps and particular geometries one can get a sign change in the channel for these
                # configurations we implement a front loop to fix the level set there as well
                # recession = True
                # if len(np.where(sgndDist_k[ngtv_region] >= 0)[0]) != 0:
                #     log.warning('We get a recession of the front. Changing to the classical front reconstruction.')
                #     exitstatus = 20
                #     return exitstatus, None
                    # fixedElts = np.hstack((Fr_lstTmStp.EltTip, ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]))
                    # fixedSgndDist = np.hstack((Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep * Fr_lstTmStp.v),
                    #                            Fr_lstTmStp.sgndDist[
                    #                                ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]]))
                    # recession = False
                    # iterator = 0

                # while not recession:
                #     log.warning('The channel elements with a negative level set are ' +
                #                 str(ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]))
                #     iterator = iterator + 1
                #     fmmStruct = fmm(Fr_lstTmStp.mesh)
                #
                #     # We define the tip elements as the known elements and solve from there outwards to the domain boundary.
                #     fmmStruct.solveFMM((fixedSgndDist, fixedElts), np.unique(np.hstack((pstv_region, fixedElts))),
                #                        Fr_lstTmStp.mesh)
                #
                #     # We define the tip elements as the known elements and solve from there inwards (inside the fracture). To do so,
                #     # we need a sign change on the level set (positive inside)
                #     toEval = np.unique(np.hstack((ngtv_region, fixedElts)))
                #     fmmStruct.solveFMM((-fixedSgndDist, fixedElts), toEval, Fr_lstTmStp.mesh)
                #
                #     # The solution stored in the object is the calculated level set. we need however to change the sign as to have
                #     # negative inside and positive outside.
                #     sgndDist_k = fmmStruct.LS
                #     sgndDist_k[toEval] = -sgndDist_k[toEval]
                #
                #     if len(np.where(sgndDist_k[ngtv_region] >= 0)[0]) == 0 and iterator < 100:
                #         recession = True
                #         log.warning('Required iterations on the level set were ' + str(iterator))
                #     elif iterator >= 100:
                #         recession = True
                #         sgndDist_k[ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]] = \
                #             Fr_lstTmStp.sgndDist[ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]]
                #         log.warning('Required iterations exceeded ' + str(iterator) +
                #                     ', we simply fix the cells and accept it like this')
                #     else:
                #         newFixedElts = ngtv_region[np.where(sgndDist_k[ngtv_region] >= 0)[0]]
                #         fixedElts = np.hstack((fixedElts, newFixedElts))
                #         newFixedSgndDist = Fr_lstTmStp.sgndDist[newFixedElts]
                #         fixedSgndDist = np.hstack((fixedSgndDist, newFixedSgndDist))

                eval_region = np.where(sgndDist_k[front_region] >= -Fr_lstTmStp.mesh.cellDiag)[0]

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

    if sim_properties.paramFromTip or mat_properties.anisotropic_K1c or mat_properties.inv_with_heter_K1c:
        if not mat_properties.sizeDependentToughness[0] and not mat_properties.velocityDependentToughness[0]:
            if sim_properties.projMethod != 'LS_continousfront':
                # get toughness from zero vertex
                Kprime_tip = (32 / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                                 Fr_lstTmStp.mesh,
                                                                                 mat_properties,
                                                                                 alpha_k,
                                                                                 l_k,
                                                                                 zrVrtx_newTip)
            else:
                # toughness_from_zeroVertex = False
                # The if is redundant if we set it to false just before: I remove
                # zrVrtx_newTip = zrVertx_k_with_fully_traversed.transpose()
                # if toughness_from_zeroVertex:
                #     # get toughness from zero vertex
                #     Kprime_tip = (32. / np.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                #                                                                      Fr_lstTmStp.mesh,
                #                                                                      mat_properties,
                #                                                                      alpha_k,
                #                                                                      l_k,
                #                                                                      zrVrtx_newTip)
                # else:
                # get toughness from Ffront
                Kprime_tip = (32. / np.pi) ** 0.5 * get_toughness_from_Front(Ffront,
                                                                             EltsTipNew,
                                                                             EltTip_k,
                                                                             fully_traversed_k,
                                                                             Fr_lstTmStp.mesh,
                                                                             mat_properties,
                                                                             alpha_k,
                                                                             get_from_mid_front = False)
        else:
            Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep
            Vel_k[Vel_k < 0] = 0
            if mat_properties.sizeDependentToughness[0]:
                Kprime_tip = (32 / np.pi) ** 0.5 * get_fracture_size_dependent_toughness(mat_properties, Ffront,
                                                                                         EltsTipNew, Vel_k,
                                                                                         Fr_lstTmStp.mesh,
                                                                                         mat_properties.sizeDependentToughness[1],
                                                                                         mat_properties.sizeDependentToughness[2])
            elif mat_properties.velocityDependentToughness[0]:
                Kprime_tip = (32 / np.pi) ** 0.5 * get_fracture_velocity_dependent_toughness(mat_properties, EltsTipNew,
                                                                                             Vel_k,
                                                                                             mat_properties.velocityDependentToughness[1],
                                                                                             mat_properties.velocityDependentToughness[2],
                                                                                             mat_properties.velocityDependentToughness[3])

    elif not mat_properties.inv_with_heter_K1c:
                Kprime_tip = mat_properties.Kprime[corr_ribbon]

    if mat_properties.TI_elasticity:
        Eprime_tip = TI_plain_strain_modulus(alpha_k,
                                             mat_properties.Cij)
    else:
        Eprime_tip = np.full((EltsTipNew.size,), mat_properties.Eprime, dtype=np.float64)

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep
    Vel_k[Vel_k < 0] = 0

    if perfNode is not None:
        perfNode_tipWidth = instrument_start('tip width', perfNode)
        # todo close tip width instrumentation

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = (-(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew])
                 / Fr_lstTmStp.mesh.cellDiag < sim_properties.toleranceVStagnant)
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
        if np.isnan(t_since_arrival).any():
            t_since_arrival[np.argwhere(np.isnan(t_since_arrival))] = 0.
        if (np.abs(t_since_arrival) == np.inf).any():
            t_since_arrival[np.argwhere(np.abs(t_since_arrival) == np.inf)] = 0.
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * ((t_since_arrival
                                                                                              + timeStep) ** 0.5
                                                                                             - t_since_arrival ** 0.5) \
                                        * Fr_lstTmStp.mesh.EltArea

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

    C._set_tipcorr(Fr_lstTmStp.FillF, Fr_lstTmStp.EltTip)
    C._set_kerneltype_as_R0()
    w_n_plus1, pf_n_plus1, data = solve_width_pressure(Fr_lstTmStp,
                                                       sim_properties,
                                                       fluid_properties,
                                                       mat_properties,
                                                       inj_properties,
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
                                                       stagnant,
                                                       doublefracturedictionary = doublefracturedictionary)
    C.enable_tip_corr = False
    C._set_kerneltype_as_it_used_to_be()

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
    if np.any(Vel_k[new_channel] <= 0.):  # negative velocitz set to zero...
        Vel_k[new_channel[np.where(Vel_k[new_channel] <= 0.)]] = 1e-6
        log.debug("Some zero velocities are set to a minimum velocity to evaluate the leak-off.")
    t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] / Vel_k[new_channel]
    max_l = Fr_lstTmStp.mesh.hx * np.cos(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * np.sin(alpha_k[new_channel])
    t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) / Vel_k[new_channel]
    Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2
    to_correct = np.where(Tarrival_k[EltsTipNew[new_channel]] < max_Tarrival)[0]
    Tarrival_k[EltsTipNew[new_channel[to_correct]]] = max_Tarrival
    if np.isnan(Tarrival_k[EltChannel_k]).any():
        Tarrival_k[EltChannel_k[np.isnan(Tarrival_k[EltChannel_k])]] = \
            Fr_lstTmStp.Tarrival[EltChannel_k[np.isnan(Tarrival_k[EltChannel_k])]]
    if (np.abs(Tarrival_k[EltChannel_k]) == np.inf).any():
        Tarrival_k[EltChannel_k[np.argwhere(np.abs(Tarrival_k[EltChannel_k]) == np.inf)]] = max_Tarrival

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
    else:
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

    if len(data) > 3:
        Fr_kplus1.injectionRate = np.zeros(Fr_kplus1.mesh.NumberOfElts, dtype=np.float64)
        Fr_kplus1.pInjLine = Fr_lstTmStp.pInjLine + data[3]
        Fr_kplus1.injectionRate[data[4][1]] = data[4][0]
        Fr_kplus1.injectionRate[data[5][1]] = data[5][0]
        Q_indx = np.where(abs(Qin) > 0)[0]
        print('dPIL ' + repr(data[3]))
        print('PIL ' + repr(Fr_kplus1.pInjLine))
        print("Qil " + repr(sum(Fr_kplus1.injectionRate[Q_indx])))
        # print("pressure drop " + repr(
        #     inj_properties.perforationFriction * Fr_kplus1.injectionRate[Q_indx] ** 2))
        # print("pressure at injection " + repr(Fr_kplus1.pFluid[Q_indx]))

    log.debug("Solved...")
    log.debug("Finding velocity of front...")

    front_region, eval_region, sgndDist_k = toughness_direction_loop(Fr_kplus1.w, sgndDist_k, Fr_lstTmStp,
                                                                     sim_properties, mat_properties, fluid_properties,
                                                                     timeStep, log, perfNode)

    if eval_region is None:
        return front_region, None

    Fr_kplus1.v = -(sgndDist_k[Fr_kplus1.EltTip] - Fr_lstTmStp.sgndDist[Fr_kplus1.EltTip]) / timeStep
    Fr_kplus1.v[Fr_kplus1.v < 0] = 0
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.Ffront_last = Fr_lstTmStp.Ffront
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    new_tip = np.where(np.isnan(Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip]))[0]
    Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] / Fr_kplus1.v[new_tip]
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += np.sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = Fr_kplus1.mesh.EltArea * (np.sum(Fr_kplus1.w[Fr_kplus1.EltTip] * Fr_kplus1.FillF) +
                                                     np.sum(Fr_kplus1.w[Fr_kplus1.EltChannel])) / Fr_kplus1.injectedVol
    # Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol
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
                                                                                          fluid_properties.density,
                                                                                          sim_properties,
                                                                                          mat_properties)

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