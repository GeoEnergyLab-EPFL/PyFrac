#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 11.07.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights
# reserved. See the LICENSE.TXT file for more details.
#

import numpy as np
from src.TipInversion import *
from src.ElastoHydrodynamicSolver import *
from src.LevelSet import *
import copy
from src.VolIntegral import *

def attempt_time_step_volumeControl(Frac, C, Material_properties, Simulation_Parameters, Injection_Parameters, TimeStep,
                                    Mesh):
    """ Propagate fracture one time step. The function injects fluid into the fracture, first by keeping the same
    footprint. This gives the first trial value of the width. The ElastoHydronamic system is then solved iteratively
    until convergence is achieved.

    Arguments:
        Frac (Fracture object):                             fracture object from the last time step
        C (ndarray-float):                                  the elasticity matrix
        Material_properties (MaterialProperties object):    material properties
        Fluid_properties (FluidProperties object):          fluid properties
        Simulation_Parameters (SimulationParameters object): simulation parameters
        Injection_Parameters (InjectionProperties object):  injection properties
        TimeStep (float):                                   time step

    Return:
        int:   possible values:
                                    0       -- not propagated
                                    1       -- iteration successful
                                    2       -- evaluated level set is not valid
                                    3       -- front is not tracked correctly
                                    4       -- evaluated tip volume is not valid
                                    5       -- solution of elastohydrodynamic solver is not valid
                                    6       -- did not converge after max iterations
                                    7       -- tip inversion not successful
                                    8       -- Ribbon element not found in the enclosure of a tip cell
                                    9       -- Filling fraction not correct

        Fracture object:            fracture after advancing time step.
    """
    indxCurTime = max(np.where(Frac.time >= Injection_Parameters.injectionRate[0, :])[0])
    Qin = Injection_Parameters.injectionRate[1, indxCurTime]  # current injection rate
    # todo : write log file
    # f = open('log', 'a')

    print('Solving ElastoHydrodynamic equations with same footprint...')
    # width by injecting the fracture with the same foot print (balloon like inflation)
    exitstatus, w_k = injection_same_footprint_volumeControl(Frac,
                                                            C,
                                                            TimeStep,
                                                            Qin,
                                                            Mesh)

    if exitstatus != 1:
        # failed
        return exitstatus, None

    print('Starting Fracture Front loop...')

    norm = 10.
    k = 0
    Fr_k = Frac

    # Fracture front loop to find the correct front location
    while norm > Simulation_Parameters.tolFractFront:
        k = k + 1
        print('\nIteration ' + repr(k))
        Fr_kminus1 = copy.deepcopy(Fr_k)

        # find the new footprint and solve the elastohydrodynamic equations to to get the new fracture
        (exitstatus, Fr_k) = injection_extended_footprint_volumeControl(w_k,
                                                              Frac,
                                                              C,
                                                              TimeStep,
                                                              Qin,
                                                              Material_properties,
                                                              Simulation_Parameters)
        if exitstatus != 1:
            return exitstatus, None

        # the new fracture width (notably the new width in the ribbon cells).
        w_k = np.copy(Fr_k.w)

        # norm is evaluated by dividing the difference in the area of the tip cells between two successive iterations
        # with the number of tip cells.
        norm = abs((sum(Fr_k.FillF) - sum(Fr_kminus1.FillF)) / len(Fr_k.FillF))
        print('Norm of subsequent filling fraction estimates = ' + repr(norm))

        if k == Simulation_Parameters.maxFrontItr:
            exitstatus = 6
            return exitstatus, None

    return exitstatus, Fr_k

#-----------------------------------------------------------------------------------------------------------------------

def injection_same_footprint_volumeControl(Fr_lstTmStp, C, timeStep, Q, mesh):
    """
    This function solves the ElastoHydrodynamic equations to get the fracture width. The fracture footprint is taken
    to be the same as in the fracture from the last time step.
    Arguments:
        Fr_lstTmStp (Fracture object):                      fracture object from the last time step
        C (ndarray-float):                                  the elasticity matrix
        timeStep (float):                                   time step
        Qin (ndarray-float):                                current injection rate
        mat_properties (MaterialProperties object):         material properties
        Fluid_properties (FluidProperties object):          fluid properties
        Simulation_Parameters (SimulationParameters object): simulation parameters

    Returns:
        int:            exit status
        ndarray-float:  width of the fracture after injection with the same footprint

    """
    C_EltTip = C[np.ix_(Fr_lstTmStp.EltTip, Fr_lstTmStp.EltTip)]  # keeping the tip element entries to restore current
    #  tip correction. This is done to avoid copying the full elasticity matrix.

    # filling fraction correction for element in the tip region
    for e in range(0, len(Fr_lstTmStp.EltTip)):
        r = Fr_lstTmStp.FillF[e] - .25
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r
        C[Fr_lstTmStp.EltTip[e], Fr_lstTmStp.EltTip[e]] = C[Fr_lstTmStp.EltTip[e], Fr_lstTmStp.EltTip[e]] * (1.
                                                                                                             + ac * np.pi / 4.)

    # index of current time in the time series (first row) of the injection rate array



    (A, b) = MakeEquationSystem_volumeControl_sameFP(Fr_lstTmStp.w, Fr_lstTmStp.EltCrack, C, timeStep, Q, mesh.EltArea)
    sol = np.linalg.solve(A, b)

    # getting new width by adding the change in width solution to the width from last time step
    w_k = np.copy(Fr_lstTmStp.w)
    w_k[Fr_lstTmStp.EltCrack] += sol[np.arange(Fr_lstTmStp.EltCrack.size)]


    # regain original C (without filling fraction correction)
    C[np.ix_(Fr_lstTmStp.EltTip, Fr_lstTmStp.EltTip)] = C_EltTip

    # # check if the width has gone into negative
    # # todo: !!! Hack: if the width is negative but greater than some factor times the mean width, it is ignored. This
    # #  usually happens when high stress is applied forcing small widths. This will not effect the results as its done
    # # in the ballooning of the fracture to get the guess width for the next iteration.
    # smallNgtvWTip = np.where(np.logical_and(w_k < 0, w_k > -1 * np.mean(w_k)))
    # if np.asarray(smallNgtvWTip).size > 0:
    #     # warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
    #     w_k[smallNgtvWTip] = 0.01 * abs(w_k[smallNgtvWTip])

    # check if the solution is valid
    if np.isnan(w_k).any() or (w_k < 0).any():
        exitstatus = 5
        return exitstatus, None
    else:
        exitstatus = 1
        return exitstatus, w_k

#-----------------------------------------------------------------------------------------------------------------------

def injection_extended_footprint_volumeControl(w_k, Fr_lstTmStp, C, timeStep, Qin, Material_properties, sim_parameters):
    """
    This function takes the fracture width from the last iteration of the fracture front loop, calculates the level set
    (fracture front position) by inverting the tip asymptote and then solves the ElastoHydrodynamic equations to obtain
    the new fracture width.

    Arguments:
        w_k (ndarray-float);                                fracture width from the last iteration
        Fr_lstTmStp (Fracture object):                      fracture object from the last time step
        C (ndarray-float):                                  the elasticity matrix
        timeStep (float):                                   time step
        Qin (ndarray-float):                                current injection rate
        Material_properties (MaterialProperties object):    material properties
        Fluid_properties (FluidProperties object):          fluid properties
        sim_Parameters (SimulationParameters object):       simulation parameters

    Returns:
        int:   possible values:
                                    0       -- not propagated
                                    1       -- iteration successful
                                    2       -- evaluated level set is not valid
                                    3       -- front is not tracked correctly
                                    4       -- evaluated tip volume is not valid
                                    5       -- solution of elastohydrodynamic solver is not valid
                                    6       -- did not converge after max iterations
                                    7       -- tip inversion not successful
                                    8       -- Ribbon element not found in the enclosure of a tip cell
                                    9       -- Filling fraction not correct

        Fracture object:            fracture after advancing time step.
    """

    itr = 0
    sgndDist_k = np.copy(Fr_lstTmStp.sgndDist)

    # toughness iteration loop
    while itr < sim_parameters.maxToughnessItr:

        sgndDist_km1 = np.copy(sgndDist_k)
        l_m1 = sgndDist_km1[Fr_lstTmStp.EltRibbon]

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e10 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely
        # large float value. (algorithm requires inf)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion_hetrogenous_toughness(w_k,
                                                                                     Fr_lstTmStp,
                                                                                     Material_properties,
                                                                                     sgndDist_km1)

        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            exitstatus = 7
            return exitstatus, None

        # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
        SolveFMM(sgndDist_k,
                 Fr_lstTmStp.EltRibbon,
                 Fr_lstTmStp.EltChannel,
                 Fr_lstTmStp.mesh)

        # if some elements remain unevaluated by fast marching method. It happens with unrealistic fracture geometry.
        # todo: not satisfied with why this happens. need re-examining
        if max(sgndDist_k) == 1e10:
            exitstatus = 2
            return exitstatus, None

        norm = np.linalg.norm(1 - abs(l_m1 / sgndDist_k[Fr_lstTmStp.EltRibbon]))
        if norm < sim_parameters.toleranceToughness:
            print("toughness iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                  repr(norm))
            break

        # do it only once if KprimeFunc
        if Material_properties.KprimeFunc is None:
            break
        print("toughness iteration " + repr(itr) + "norm " + repr(norm))
        itr += 1

    if itr == sim_parameters.maxToughnessItr:
        exitstatus = 10
        return exitstatus, None
    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    (EltsTipNew, l_k, alpha_k, CellStatus) = reconstruct_front(sgndDist_k,
                                                               Fr_lstTmStp.EltChannel,
                                                               Fr_lstTmStp.mesh)

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    tipNeighb = Fr_lstTmStp.mesh.NeiElements[EltsTipNew, :]
    for i in range(0, len(EltsTipNew)):
        if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
            Fr_lstTmStp.plot_fracture('complete', 'footPrint')
            raise SystemExit('Reached end of the grid. exiting....')

    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = VolumeIntegral(EltsTipNew,
                                alpha_k,
                                l_k,
                                Fr_lstTmStp.mesh,
                                'A',
                                Material_properties,
                                Fr_lstTmStp.muPrime,
                                Vel_k) / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-6 (up to 6 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-6)] = 1.0

    # if filling fraction is below zero or above 1+1e-6
    if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
        exitstatus = 9
        return exitstatus, None

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

    # EletsTipNew may contain fully filled elements also. Identifying only the partially filled elements
    partlyFilledTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]

    print('Solving the EHL system with the new trial footprint')

    # Calculating toughness at tip to be used to calculate the volume integral in the tip cells
    zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    Kprime_tip = toughness_at_tip_zeroVertex(EltsTipNew,
                                             Fr_lstTmStp.mesh,
                                             Material_properties,
                                             alpha_k,
                                             l_k,
                                             zrVrtx_newTip)

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = abs(1 - sgndDist_k[EltsTipNew] / Fr_lstTmStp.sgndDist[EltsTipNew]) < 1e-5
    if stagnant.any():
        # if any tip cell with stagnant front
        # calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(w_k,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        Material_properties.Eprime)

        # todo: Find the right cause of failure
        # if the stress Intensity factor cannot be found. The most common reason is wiggles in the front resulting
        # in isolated tip cells.
        if np.isnan(KIPrime).any():
            exitstatus = 8
            return exitstatus, None

        # Calculate average width in the tip cells by integrating tip asymptote. Width of stagnant cells are calculated
        # using the stress intensity factor (see Dontsov and Peirce, JFM RAPIDS, 2017)
        wTip = VolumeIntegral(EltsTipNew,
                              alpha_k,
                              l_k,
                              Fr_lstTmStp.mesh,
                              sim_parameters.tipAsymptote,
                              Material_properties,
                              Fr_lstTmStp.muPrime,
                              Vel_k,
                              Kprime=Kprime_tip,
                              stagnant=stagnant,
                              KIPrime=KIPrime
                              ) / Fr_lstTmStp.mesh.EltArea
    else:
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = VolumeIntegral(EltsTipNew,
                              alpha_k,
                              l_k,
                              Fr_lstTmStp.mesh,
                              sim_parameters.tipAsymptote,
                              Material_properties,
                              Fr_lstTmStp.muPrime,
                              Vel_k,
                              Kprime=Kprime_tip) / Fr_lstTmStp.mesh.EltArea

    # check if the tip volume has gone into negative
    smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        #                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])


    if (wTip < 0).any():
        exitstatus = 4
        return exitstatus, None

    A, b = MakeEquationSystem_volumeControl_extendedFP(Fr_lstTmStp.w,
                                                wTip,
                                                Fr_lstTmStp.EltChannel,
                                                EltsTipNew,
                                                C,
                                                timeStep,
                                                Qin,
                                                Fr_lstTmStp.mesh.EltArea)

    sol = np.linalg.solve(A, b)

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)

    Fr_kplus1.time += timeStep

    Fr_kplus1.w[Fr_lstTmStp.EltChannel] += sol[np.arange(Fr_lstTmStp.EltChannel.size)]
    Fr_kplus1.w[EltsTipNew] = wTip

    # check if the new width is valid
    if np.isnan(Fr_kplus1.w).any():
        exitstatus = 5
        return exitstatus, None

    if (
        Fr_kplus1.w < 0).any():  # todo: clean this up as it might blow up !    -> we need a linear solver with constraint to handle pinch point properly.
        print(repr(np.where((Fr_kplus1.w < 0))))
        print(repr(Fr_kplus1.w[np.where((Fr_kplus1.w < 0))[0]]))
    # exitstatus = 5
    #        return exitstatus, None

    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k

    # pressure evaluated by dot product of width and elasticity matrix
    Fr_kplus1.p = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    Fr_kplus1.p[EltCrack_k] = sol[-1]
    Fr_kplus1.sgndDist = sgndDist_k

    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = Vel_k[partlyFilledTip]

    Fr_kplus1.InCrack = InCrack_k

    Fr_kplus1.process_fracture_front()
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * (Fr_kplus1.mesh.EltArea)

    # if Fr_kplus1.time > 4200:
    #     Fr_kplus1.plot_fracture("complete", "footPrint")
    exitstatus = 1
    return exitstatus, Fr_kplus1


#-----------------------------------------------------------------------------------------------------------------------

def toughness_at_tip_zeroVertex(elts, mesh, mat_prop, alpha, l, zero_vrtx):

    if mat_prop.anisotropic:
        return mat_prop.KprimeFunc(alpha)
    else:
        x = np.zeros((len(elts),), )
        y = np.zeros((len(elts),), )
        for i in range(0, len(elts)):
            if zero_vrtx[i] == 0:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 0], 0] + l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 0], 1] + l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 1:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 0] - l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 1] + l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 2:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 0] - l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 1] - l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 3:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 0] + l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 1] - l[i] * np.sin(alpha[i])

        return mat_prop.KprimeFunc(x, y)