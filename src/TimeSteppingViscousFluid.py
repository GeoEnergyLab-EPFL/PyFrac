#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#


import copy
import sys

# local imports
from src.VolIntegral import *
from src.Utility import *
from src.TipInversion import *
from src.ElastoHydrodynamicSolver import *
from src.LevelSet import *
from src.HFAnalyticalSolutions import *
from src.TimeSteppingMechLoading import *
from src.TimeSteppingVolumeControl import *
from src.Properties import IterationProperties
import time


def attempt_time_step_viscousFluid(Frac, C, mat_properties, fluid_properties, sim_properties,
                                   inj_properties, timeStep, perfNode=None):
    """ Propagate fracture one time step. The function injects fluid into the fracture, first by keeping the same
    footprint. This gives the first trial value of the width. The ElastoHydronamic system is then solved iteratively
    until convergence is achieved.
    
    Arguments:
        Frac (Fracture object):                             fracture object from the last time step 
        C (ndarray-float):                                  the elasticity matrix 
        mat_properties (MaterialProperties object):    material properties
        fluid_properties (FluidProperties object):          fluid properties 
        sim_properties (SimulationParameters object): simulation parameters
        inj_properties (InjectionProperties object):  injection properties
        timeStep (float):                                   time step 
    
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
                                    10      -- Toughness iteration did not converge
                                    
        Fracture object:            fracture after advancing time step. 
    """

    exitstatus = 0  # exit code to be returned

    # index of current time in the time series (first row) of the injection rate array
    indxCurTime = max(np.where(Frac.time >= inj_properties.injectionRate[0, :])[0])
    CurrentRate = inj_properties.injectionRate[1, indxCurTime]  # current injection rate

    Qin = np.zeros((Frac.mesh.NumberOfElts), float)
    Qin[inj_properties.source_location] = CurrentRate # current injection over the domain

    # todo : write log file
    # f = open('log', 'a')

    if sim_properties.frontAdvancing == 'explicit':

        # make a new performance collection node to collect data about the explicit time step advancement
        if perfNode is not None:
            perfNode_explFront = IterationProperties(itr_type="explicit front")
            perfNode_explFront.subIterations = [[], [], []]
        else:
            perfNode_explFront = None

        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                      C,
                                                      timeStep,
                                                      Qin,
                                                      mat_properties,
                                                      fluid_properties,
                                                      sim_properties,
                                                      perfNode_explFront)

        if perfNode_explFront is not None:
            perfNode_explFront.CpuTime_end = time.time()
            if Fr_k is not None:
                perfNode.NumbOfElts = Fr_k.EltCrack.size
            perfNode.iterations += 1
            perfNode.normList.append(np.nan)
            perfNode.subIterations[0].append(perfNode_explFront)
            if exitstatus != 1:
                perfNode.time = Frac.time + timeStep
                perfNode.failure_cause = exitstatus
            else:
                perfNode.time = Fr_k.time

        return exitstatus, Fr_k

    elif sim_properties.frontAdvancing == 'semi-implicit':
        if sim_properties.verbosity > 1:
            print('Advancing front with velocity from last time-step...')

        if perfNode is not None:
            perfNode_explFront = IterationProperties(itr_type="explicit front")
            perfNode_explFront.subIterations = [[], [], []]
        else:
            perfNode_explFront = None

        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    perfNode_explFront)

        if perfNode_explFront is not None:
            perfNode_explFront.CpuTime_end = time.time()
            if Fr_k is not None:
                perfNode.NumbOfElts = Fr_k.EltCrack.size
            perfNode.iterations += 1
            perfNode.normList.append(np.nan)
            perfNode.subIterations[0].append(perfNode_explFront)
            if exitstatus != 1:
                perfNode.time = Frac.time + timeStep
                perfNode.failure_cause = exitstatus

        if exitstatus == 1:
            w_k = np.copy(Fr_k.w)

    elif sim_properties.frontAdvancing == 'implicit':
        if sim_properties.verbosity > 1:
            print('Solving ElastoHydrodynamic equations with same footprint...')

        if perfNode is not None:
            perfNode_sameFP = IterationProperties(itr_type="same footprint injection")
            perfNode_sameFP.subIterations = []
        else:
            perfNode_sameFP = None

        # width by injecting the fracture with the same foot print (balloon like inflation)
        exitstatus, w_k = injection_same_footprint(Frac,
                                                   C,
                                                   timeStep,
                                                   Qin,
                                                   mat_properties,
                                                   fluid_properties,
                                                   sim_properties,
                                                   perfNode_sameFP)
        if perfNode_sameFP is not None:
            perfNode_sameFP.CpuTime_end = time.time()
            if Frac is not None:
                perfNode.NumbOfElts = Frac.EltCrack.size
            perfNode.iterations += 1
            perfNode.normList.append(np.nan)
            perfNode.subIterations[1].append(perfNode_sameFP)
            if exitstatus != 1:
                perfNode.time = Frac.time + timeStep
                perfNode.failure_cause = exitstatus
    else:
        raise ValueError("Provided front advancing type not supported")

    if exitstatus != 1:
        # failed
        return exitstatus, None

    if sim_properties.verbosity > 1:
        print('Starting Fracture Front loop...')

    norm = 10.
    k = 0
    Fr_k = Frac

    # Fracture front loop to find the correct front location
    while norm > sim_properties.tolFractFront:
        k = k + 1
        if sim_properties.verbosity > 1:
            print('\nIteration ' + repr(k))
        fill_frac_last = np.copy(Fr_k.FillF)

        if perfNode is not None:
            perfNode_extendedFP = IterationProperties(itr_type="extended footprint injection")
            perfNode_extendedFP.subIterations = [[], [], []]
        else:
            perfNode_extendedFP = None

        # find the new footprint and solve the elastohydrodynamic equations to to get the new fracture
        (exitstatus, Fr_k) = injection_extended_footprint(w_k,
                                                          Frac,
                                                          C,
                                                          timeStep,
                                                          Qin,
                                                          mat_properties,
                                                          fluid_properties,
                                                          sim_properties,
                                                          perfNode_extendedFP)

        if exitstatus == 1:
            # the new fracture width (notably the new width in the ribbon cells).
            w_k = np.copy(Fr_k.w)

            # norm is evaluated by dividing the difference in the area of the tip cells between two successive iterations
            # with the number of tip cells.
            norm = abs((sum(Fr_k.FillF) - sum(fill_frac_last)) / len(Fr_k.FillF))
        else:
            norm = np.nan

        if perfNode_extendedFP is not None:
            perfNode_extendedFP.CpuTime_end = time.time()
            if Fr_k is not None:
                perfNode.NumbOfElts = Fr_k.EltCrack.size
            perfNode.iterations += 1
            perfNode.normList.append(norm)
            perfNode.subIterations[2].append(perfNode_extendedFP)
            if exitstatus != 1:
                perfNode.time = Frac.time + timeStep
                perfNode.failure_cause = exitstatus

        if exitstatus != 1:
            return exitstatus, None

        if sim_properties.verbosity > 1:
            print('Norm of subsequent filling fraction estimates = ' + repr(norm))

        if k == sim_properties.maxFrontItr:
            exitstatus = 6
            if perfNode_extendedFP is not None:
                perfNode.time = Frac.time + timeStep
                perfNode.failure_cause = exitstatus
            return exitstatus, None

    if sim_properties.verbosity > 1:
        print("Fracture front converged after " + repr(k) + " iterations with norm = " + repr(norm))

    if perfNode is not None:
        perfNode.time = Fr_k.time

    return exitstatus, Fr_k


# ----------------------------------------------------------------------------------------------------------------------

def injection_same_footprint(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             perfNode=None):
    """
    This function solves the ElastoHydrodynamic equations to get the fracture width. The fracture footprint is taken
    to be the same as in the fracture from the last time step.
    Arguments:
        Fr_lstTmStp (Fracture object):                      fracture object from the last time step 
        C (ndarray-float):                                  the elasticity matrix 
        timeStep (float):                                   time step 
        Qin (ndarray-float):                                current injection rate
        mat_properties (MaterialProperties object):         material properties
        fluid_properties (FluidProperties object):          fluid properties 
        sim_properties (SimulationParameters object): simulation parameters
        
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

    # average injected fluid over footprint taken as [\delta] W guess for the iterative solver
    delwGuess = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size * np.ones((Fr_lstTmStp.EltCrack.size,), float)


    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if sum(mat_properties.Cprime[Fr_lstTmStp.EltCrack]) > 0.:
        # Calculate leak-off term for the tip cell
        LkOff[Fr_lstTmStp.EltTip] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltTip] * Integral_over_cell(Fr_lstTmStp.EltTip,
                                                                               Fr_lstTmStp.alpha,
                                                                               Fr_lstTmStp.l,
                                                                               Fr_lstTmStp.mesh,
                                                                               'Lk',
                                                                               mat_prop=mat_properties,
                                                                               frac=Fr_lstTmStp,
                                                                               Vel=Fr_lstTmStp.v,
                                                                               dt=timeStep)

        # Calculate leak-off term for the channel cell
        t_lst_min_t0 = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_min_t0 = t_lst_min_t0 + timeStep

        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * (t_min_t0 ** 0.5 -
                                        t_lst_min_t0 ** 0.5) * Fr_lstTmStp.mesh.EltArea

    # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
    vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if fluid_properties.turbulence:

        wguess = np.copy(Fr_lstTmStp.w)
        wguess[Fr_lstTmStp.EltCrack] = wguess[Fr_lstTmStp.EltCrack] + delwGuess

        # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
        vk = velocity(wguess,
                      Fr_lstTmStp.EltCrack,
                      Fr_lstTmStp.mesh,
                      Fr_lstTmStp.InCrack,
                      Fr_lstTmStp.muPrime,
                      C,
                      mat_properties.SigmaO)

    argSameFP = (
        Fr_lstTmStp.w,
        Fr_lstTmStp.EltCrack,
        Qin,
        C,
        timeStep,
        Fr_lstTmStp.muPrime,
        Fr_lstTmStp.mesh,
        Fr_lstTmStp.InCrack,
        LkOff,
        mat_properties.SigmaO,
        fluid_properties.density,
        fluid_properties.turbulence,
        mat_properties.grainSize)

    # typical values of the variable. Used to calculate Jacobian (see Piccard_Newton function documentation)
    # todo: guess is taken as typical values. Needs to be reconsidered
    typclValue = delwGuess

    if perfNode is not None:
        perfNode.iterations += 1
        perfNode_Picard = IterationProperties(itr_type="Picard iteration")
        perfNode_Picard.subIterations = []
    else:
        perfNode_Picard = None

    # solving the system
    sol, vel = Picard_Newton(Elastohydrodynamic_ResidualFun_sameFP,
                               MakeEquationSystem_viscousFluid_sameFP,
                               delwGuess,
                               typclValue,
                               vk,
                               sim_properties.toleranceEHL,
                               sim_properties.maxSolverItr,
                               *argSameFP,
                               perf_node=perfNode_Picard)

    if perfNode_Picard is not None:
        perfNode_Picard.CpuTime_end = time.time()
        perfNode.subIterations.append(perfNode_Picard)

    # getting new width by adding the change in width solution to the width from last time step
    w_k = np.copy(Fr_lstTmStp.w)
    w_k[Fr_lstTmStp.EltCrack] = w_k[Fr_lstTmStp.EltCrack] + sol

    # regain original C (without filling fraction correction)
    C[np.ix_(Fr_lstTmStp.EltTip, Fr_lstTmStp.EltTip)] = C_EltTip


    # check if the width has gone into negative
    # todo: !!! Hack: if the width is negative but greater than some factor times the mean width, it is ignored. This
    #  usually happens when high stress is applied forcing small widths. This will not effect the results as its done
    # in the ballooning of the fracture to get the guess width for the next iteration.
    smallNgtvWTip = np.where(np.logical_and(w_k < 0, w_k > -1 * np.mean(w_k)))
    if np.asarray(smallNgtvWTip).size > 0:
        # warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        w_k[smallNgtvWTip] = 0.01*abs(w_k[smallNgtvWTip])


    # check if the solution is valid
    if np.isnan(w_k).any() or (w_k < 0).any():
        exitstatus = 5
        return exitstatus, None
    else:
        exitstatus = 1
        return exitstatus, w_k


# -----------------------------------------------------------------------------------------------------------------------

def injection_extended_footprint(w_k, Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties,
                                 sim_properties, perfNode=None):
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
        mat_properties (MaterialProperties object):    material properties
        fluid_properties (FluidProperties object):          fluid properties 
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
                                    10      -- Toughness iteration did not converge
                                    11      -- Projection could not be found
                                    12      -- Reached end of grid

        Fracture object:            fracture after advancing time step. 
    """

    itr = 0
    sgndDist_k = np.copy(Fr_lstTmStp.sgndDist)
    if sim_properties.precise_tipParam or mat_properties.anisotropic:
        alpha_ribbon = projection_from_ribbon(Fr_lstTmStp.EltRibbon,
                                                 Fr_lstTmStp.EltChannel,
                                                 Fr_lstTmStp.mesh,
                                                 sgndDist_k)
        Kprime_k = (32 / math.pi) ** 0.5 * get_toughness_from_cellCenter(alpha_ribbon,
                                                sgndDist_k,
                                                Fr_lstTmStp.EltRibbon,
                                                mat_properties,
                                                Fr_lstTmStp.mesh)
    # Kprime from last iteration; starts with zero
        Kprime_km1 = 0 * np.copy(Kprime_k)

    # toughness iteration loop
    while itr < sim_properties.maxToughnessItr:

        sgndDist_km1 = np.copy(sgndDist_k)
        l_m1 = sgndDist_km1[Fr_lstTmStp.EltRibbon]

        if sim_properties.precise_tipParam or mat_properties.anisotropic:
            alpha_ribbon = projection_from_ribbon(Fr_lstTmStp.EltRibbon,
                                                  Fr_lstTmStp.EltChannel,
                                                  Fr_lstTmStp.mesh,
                                                  sgndDist_k)
            if np.isnan(alpha_ribbon).any():
                exitstatus = 11
                return exitstatus, None
            # under relaxing toughnesss
            Kprime_k = 0.3 * Kprime_k + 0.7 * get_toughness_from_cellCenter(alpha_ribbon,
                                                            sgndDist_k,
                                                            Fr_lstTmStp.EltRibbon,
                                                            mat_properties,
                                                            Fr_lstTmStp.mesh) * (32 / math.pi) ** 0.5

            if np.isnan(Kprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Kprime_k = None



        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e10 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely
                                                                        # large float value. (algorithm requires inf)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(w_k,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k)


        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            exitstatus = 7
            return exitstatus, None

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        front_region = np.where(abs(Fr_lstTmStp.sgndDist) < sim_properties.tmStpPrefactor * 6.66 * (
                                            Fr_lstTmStp.mesh.hx**2 + Fr_lstTmStp.mesh.hy**2)**0.5)[0]
        # the search region outwards from the front position at last time step
        pstv_region = np.where(Fr_lstTmStp.sgndDist[front_region] >= -(Fr_lstTmStp.mesh.hx**2 +
                                                                  Fr_lstTmStp.mesh.hy**2)**0.5)[0]
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
        if not mat_properties.anisotropic and not sim_properties.precise_tipParam:
            break

        # norm = np.linalg.norm(1 - abs(l_m1/sgndDist_k[Fr_lstTmStp.EltRibbon]))
        norm = np.linalg.norm(1 - abs(Kprime_k / Kprime_km1))
        if norm < sim_properties.toleranceToughness:
            if sim_properties.verbosity > 1:
                print("toughness iteration converged after " + repr(itr-1) + " iterations; exiting norm " +
                  repr(norm))
            break

        Kprime_km1 = np.copy(Kprime_k)
        if sim_properties.verbosity > 1:
            print("iterating on toughness... norm " + repr(norm))
        itr += 1

    # if itr == sim_properties.maxToughnessItr:
    #     exitstatus = 10
    #     return exitstatus, None

    if sim_properties.saveRegime:
        regime_t = find_regime(w_k, Fr_lstTmStp, mat_properties, sim_properties, timeStep, Kprime_k,
                           -sgndDist_k[Fr_lstTmStp.EltRibbon])

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    (EltsTipNew, l_k, alpha_k, CellStatus) = reconstruct_front(sgndDist_k,
                                                                Fr_lstTmStp.EltChannel,
                                                                Fr_lstTmStp.mesh)

    if not np.in1d(EltsTipNew, front_region).any():
        raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                         " level set.")

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    tipNeighb = Fr_lstTmStp.mesh.NeiElements[EltsTipNew, :]
    for i in range(0, len(EltsTipNew)):
        if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
            exitstatus = 12
            return exitstatus, None


    # generate the InCrack array for the current front position
    InCrack_k = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = Integral_over_cell(EltsTipNew,
                                alpha_k,
                                l_k,
                                Fr_lstTmStp.mesh,
                                'A') / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-4)] = 1.0

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
    if sim_properties.verbosity > 1:
        print('Solving the EHL system with the new trial footprint')

    # Calculating toughness at tip to be used to calculate the volume integral in the tip cells
    if sim_properties.precise_tipParam or mat_properties.anisotropic:
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        Kprime_tip = (32 / math.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                   Fr_lstTmStp.mesh,
                                                   mat_properties,
                                                   alpha_k,
                                                   l_k,
                                                   zrVrtx_newTip)
    else:
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        Kprime_tip = mat_properties.Kprime[tip_neighbor_in_ribbon(EltsTipNew, zrVrtx_newTip, Fr_lstTmStp.mesh)]

    if perfNode is not None:
        perfNode.iterations += 1
        perfNode_wTip = IterationProperties(itr_type="tip volume")
    else:
        perfNode_wTip = None

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = abs(1 - sgndDist_k[EltsTipNew] / Fr_lstTmStp.sgndDist[EltsTipNew]) < 1e-5
    if stagnant.any() and not sim_properties.get_tipAsymptote() is 'U':
        print("Stagnant front is only supported with universal tip asymptote. continuing...")
        stagnant = np.full((EltsTipNew.size, ), False, dtype=bool)

    if stagnant.any():
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(w_k,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        mat_properties.Eprime)

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
                              KIPrime=KIPrime) / Fr_lstTmStp.mesh.EltArea
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
                              stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea

    # check if the tip volume has gone into negative
    smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        #                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])


    if (wTip < 0).any() or sum(wTip)==0.:
        exitstatus = 4
        return exitstatus, None

    guess = np.zeros((Fr_lstTmStp.EltChannel.size + EltsTipNew.size,), float)
    # pguess = Fr_lstTmStp.p[EltsTipNew]

    guess[np.arange(Fr_lstTmStp.EltChannel.size)] = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size \
                                                    * np.ones((Fr_lstTmStp.EltChannel.size,), float)

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
                                                                    dt=timeStep)
    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0:
        #todo: no need to evaluate on each iteration. Need to decide. Evaluating here for now for better readability
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * ((Fr_lstTmStp.time +
                        timeStep - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel])**0.5 - (Fr_lstTmStp.time -
                        Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel])**0.5) * Fr_lstTmStp.mesh.EltArea

    # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
    vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if fluid_properties.turbulence:
        wguess = np.copy(Fr_lstTmStp.w)
        wguess[Fr_lstTmStp.EltChannel] = wguess[Fr_lstTmStp.EltChannel] + guess[np.arange(Fr_lstTmStp.EltChannel.size)]
        wguess[EltsTipNew] = wTip

        vk = velocity(wguess,
                      EltCrack_k,
                      Fr_lstTmStp.mesh,
                      InCrack_k,
                      Fr_lstTmStp.muPrime,
                      C,
                      mat_properties.SigmaO)

    # typical value for pressure
    typValue = np.copy(guess)
    typValue[Fr_lstTmStp.EltChannel.size + np.arange(EltsTipNew.size)] = 1e5

    # todo too many arguments; properties class needs to be utilized
    arg = (
        Fr_lstTmStp.EltChannel,
        EltsTipNew,
        Fr_lstTmStp.w,
        wTip,
        EltCrack_k,
        Fr_lstTmStp.mesh,
        timeStep,
        Qin,
        C,
        Fr_lstTmStp.muPrime,
        fluid_properties.density,
        InCrack_k,
        LkOff,
        mat_properties.SigmaO,
        fluid_properties.turbulence,
        mat_properties.grainSize
        )

    if perfNode is not None:
        perfNode.iterations += 1
        perfNode_Picard = IterationProperties(itr_type="Picard iteration")
        perfNode_Picard.subIterations = []
    else:
        perfNode_Picard = None

    # sloving the system of equations for the change in width in the channel elements and pressure in the tip elements
    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_ExtendedFP,
                               MakeEquationSystem_viscousFluid_extendedFP,
                               guess,
                               typValue,
                               vk,
                               sim_properties.toleranceEHL,
                               sim_properties.maxSolverItr,
                               *arg,
                               perf_node=perfNode_Picard)

    if perfNode_Picard is not None:
        perfNode_Picard.CpuTime_end = time.time()
        perfNode.subIterations[2].append(perfNode_Picard)

    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = np.copy(Fr_lstTmStp.Tarrival)
    new_channel = np.where(FillFrac_k>0.9999)[0]
    t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] / Vel_k[new_channel]
    max_l = Fr_lstTmStp.mesh.hx * np.cos(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * np.sin(alpha_k[new_channel])
    t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) / Vel_k[new_channel]
    Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave)/2

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w[Fr_lstTmStp.EltChannel] += sol[np.arange(Fr_lstTmStp.EltChannel.size)]
    Fr_kplus1.w[EltsTipNew] = wTip

    # check if the new width is valid
    if np.isnan(Fr_kplus1.w).any()  :
        exitstatus = 5
        return exitstatus, None

    if (Fr_kplus1.w < 0).any():  #todo: clean this up as it might blow up !    -> we need a linear solver with constraint to handle pinch point properly.
        print("found negative width, ignoring...")
        # print(repr(np.where((Fr_kplus1.w < 0))))
        # print(repr(Fr_kplus1.w[np.where((Fr_kplus1.w < 0))[0]]))
        Fr_kplus1.w[np.where(Fr_kplus1.w < 1e-10)[0]] = 1e-10
        # exitstatus = 5
        # return exitstatus, None

    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k

    # pressure evaluated by dot product of width and elasticity matrix
    Fr_kplus1.p[Fr_kplus1.EltCrack] = np.dot(C[np.ix_(Fr_kplus1.EltCrack, Fr_kplus1.EltCrack)],
                                             Fr_kplus1.w[Fr_kplus1.EltCrack])
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = Vel_k[partlyFilledTip]
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.InCrack = InCrack_k
    Fr_kplus1.process_fracture_front()
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * (Fr_kplus1.mesh.EltArea)
    Fr_kplus1.Tarrival = Tarrival_k

    # setting leak off
    Fr_kplus1.LkOff_vol[Fr_kplus1.EltChannel] = 2 * mat_properties.Cprime[Fr_kplus1.EltChannel] * (
                            Fr_kplus1.time - Fr_kplus1.Tarrival[Fr_kplus1.EltChannel])**0.5 * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff_vol[Fr_kplus1.EltTip] = 2 * mat_properties.Cprime[Fr_kplus1.EltTip] * Integral_over_cell(
                                                                Fr_kplus1.EltTip,
                                                                Fr_kplus1.alpha,
                                                                Fr_kplus1.l,
                                                                Fr_kplus1.mesh,
                                                                'Lk',
                                                                mat_prop=mat_properties,
                                                                frac=Fr_kplus1,
                                                                Vel=Fr_kplus1.v,
                                                                dt=1.e20)
    injected_vol = sum(Qin) * Fr_kplus1.time
    Fr_kplus1.efficiency = (injected_vol - sum(Fr_kplus1.LkOff_vol[Fr_kplus1.EltCrack])) / injected_vol

    if sim_properties.saveRegime:
        Fr_kplus1.regime = np.vstack((regime_t, Fr_lstTmStp.EltRibbon))

    if sim_properties.saveReynNumb:
        ReNumb, check = turbulence_check_tip(vel, Fr_kplus1, fluid_properties, return_ReyNumb=True)
        Fr_kplus1.ReynoldsNumber = ReNumb

    exitstatus = 1
    return exitstatus, Fr_kplus1

#-----------------------------------------------------------------------------------------------------------------------


def turbulence_check_tip(vel, Fr, fluid, return_ReyNumb=False):
    """
    This function calculate the Reynolds number at the cell edges and check if any to the edge between the ribbon cells
    and the tip cells are turbulent (i.e. the Reynolds number is greater than 2100).
    
    Arguments:
        vel (ndarray-float):                    the array giving velocity of each edge of the cells in domain 
        Fr (Fracture object):                   the fracture object to be checked
        fluid (FluidProperties object):         fluid properties object 
        return_ReyNumb (boolean, default False): if true, Reynolds number at all cell edges will be returned 
    
    Returns:
        ndarray-float:      Reynolds number of all the cells in the domain; row-wise in the following order : 0--left,
                            1--right, 2--bottom, 3--top
        boolean             true if any of the edge between the ribbon and tip cells is turbulent (i.e. Reynolds number
                            is more than 2100)
    """
    # width at the adges by averaging
    wLftEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 0]]) / 2
    wRgtEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 1]]) / 2
    wBtmEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 2]]) / 2
    wTopEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 3]]) / 2

    Re = np.zeros((4, Fr.EltRibbon.size, ), dtype=np.float64)
    Re[0, :] = 4 / 3 * fluid.density * wLftEdge * vel[0, Fr.EltRibbon] / fluid.viscosity
    Re[1, :] = 4 / 3 * fluid.density * wRgtEdge * vel[1, Fr.EltRibbon] / fluid.viscosity
    Re[2, :] = 4 / 3 * fluid.density * wBtmEdge * vel[2, Fr.EltRibbon] / fluid.viscosity
    Re[3, :] = 4 / 3 * fluid.density * wTopEdge * vel[3, Fr.EltRibbon] / fluid.viscosity

    ReNum_Ribbon = []
    # adding Reynolds number of the edges between the ribbon and tip cells to a list
    for i in range(0,Fr.EltRibbon.size):
        for j in range(0,4):
            # if the current neighbor (j) of the ribbon cells is in the tip elements list
            if np.where(Fr.mesh.NeiElements[Fr.EltRibbon[i], j] == Fr.EltTip)[0].size>0:
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


def time_step_explicit_front(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             perfNode=None):
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
        mat_properties (MaterialProperties object):    material properties
        fluid_properties (FluidProperties object):          fluid properties
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
                                    10      -- Toughness iteration did not converge
                                    11      -- Projection could not be found
                                    12      -- Reached end of grid
                                    13      -- Leak off can't be evaluated

        Fracture object:            fracture after advancing time step.
    """

    sgndDist_k = 1e10 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with maximum
    # float value. (algorithm requires inf)
    sgndDist_k[Fr_lstTmStp.EltChannel] = 0  # for cells inside the fracture

    sgndDist_k[Fr_lstTmStp.EltTip] = Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep *
                                                                                 Fr_lstTmStp.v)

    front_region = np.where(abs(Fr_lstTmStp.sgndDist) < sim_properties.tmStpPrefactor * 6.66 *(
                Fr_lstTmStp.mesh.hx ** 2 + Fr_lstTmStp.mesh.hy ** 2) ** 0.5)[0]
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
    (EltsTipNew, l_k, alpha_k, CellStatus) = reconstruct_front(sgndDist_k,
                                                               Fr_lstTmStp.EltChannel,
                                                               Fr_lstTmStp.mesh)

    if not np.in1d(EltsTipNew, front_region).any():
        raise SystemExit("The tip elements are not in the band. Increase the size of the band for FMM to evaluate"
                         " level set.")

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    tipNeighb = Fr_lstTmStp.mesh.NeiElements[EltsTipNew, :]
    for i in range(0, len(EltsTipNew)):
        if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
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
                                    'A') / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-4)] = 1.0

    # if filling fraction is below zero or above 1+1e-6
    if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
        exitstatus = 9
        return exitstatus, None

    if Fr_lstTmStp.time + timeStep > 70:
        print()
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

    if sim_properties.verbosity > 1:
        print('Solving the EHL system with the new trial footprint')

    # Calculating toughness at tip to be used to calculate the volume integral in the tip cells
    if sim_properties.precise_tipParam or mat_properties.anisotropic:
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        Kprime_tip = (32 / math.pi) ** 0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                           Fr_lstTmStp.mesh,
                                                                           mat_properties,
                                                                           alpha_k,
                                                                           l_k,
                                                                           zrVrtx_newTip)
    else:
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        Kprime_tip = mat_properties.Kprime[tip_neighbor_in_ribbon(EltsTipNew, zrVrtx_newTip, Fr_lstTmStp.mesh)]

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    # create a performance node for the root finding to get tip volume
    if perfNode is not None:
        perfNode.iterations += 1
        perfNode_wTip = IterationProperties(itr_type="tip volume")
    else:
        perfNode_wTip = None

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = Vel_k < 1e-14
    if stagnant.any() and not sim_properties.get_tipAsymptote() is 'U':
        print("Stagnant front is only supported with universal tip asymptote. Continuing...")
        stagnant = np.full((EltsTipNew.size,), False, dtype=bool)

    if stagnant.any():
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(Fr_lstTmStp.w,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        mat_properties.Eprime)

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
                                  KIPrime=KIPrime) / Fr_lstTmStp.mesh.EltArea
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
                                  stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea

    # check if the tip volume has gone into negative
    smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    if np.asarray(smallNgtvWTip).size > 0:
        #                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
        wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])

    if (wTip < 0).any() or sum(wTip) == 0.:
        exitstatus = 4
        return exitstatus, None

    guess = np.zeros((Fr_lstTmStp.EltChannel.size + EltsTipNew.size,), float)
    # pguess = Fr_lstTmStp.p[EltsTipNew]

    guess[np.arange(Fr_lstTmStp.EltChannel.size)] = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size \
                                                    * np.ones((Fr_lstTmStp.EltChannel.size,), float)

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
                                                                                        dt=timeStep)
        if np.isnan(LkOff[EltsTipNew]).any():
            exitstatus = 13
            return exitstatus, None

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0:
        t_since_arrival = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        # t_since_arrival[np.where(t_since_arrival < 0)[0]] = 0
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * ((t_since_arrival
                                            + timeStep)**0.5 - t_since_arrival**0.5) * Fr_lstTmStp.mesh.EltArea
        if np.isnan(LkOff[Fr_lstTmStp.EltChannel]).any():
            exitstatus = 13
            return exitstatus, None

    # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
    vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if fluid_properties.turbulence:
        wguess = np.copy(Fr_lstTmStp.w)
        wguess[Fr_lstTmStp.EltChannel] = wguess[Fr_lstTmStp.EltChannel] + guess[np.arange(Fr_lstTmStp.EltChannel.size)]
        wguess[EltsTipNew] = wTip

        vk = velocity(wguess,
                      EltCrack_k,
                      Fr_lstTmStp.mesh,
                      InCrack_k,
                      Fr_lstTmStp.muPrime,
                      C,
                      mat_properties.SigmaO)

    # typical value for pressure
    typValue = np.copy(guess)
    typValue[Fr_lstTmStp.EltChannel.size + np.arange(EltsTipNew.size)] = 1e5

    # todo too many arguments; properties class needs to be utilized
    arg = (
        Fr_lstTmStp.EltChannel,
        EltsTipNew,
        Fr_lstTmStp.w,
        wTip,
        EltCrack_k,
        Fr_lstTmStp.mesh,
        timeStep,
        Qin,
        C,
        Fr_lstTmStp.muPrime,
        fluid_properties.density,
        InCrack_k,
        LkOff,
        mat_properties.SigmaO,
        fluid_properties.turbulence,
        mat_properties.grainSize
    )
    if perfNode is not None:
        perfNode.iterations += 1
        perfNode_Picard = IterationProperties(itr_type="Picard iteration")
        perfNode_Picard.subIterations = []
    else:
        perfNode_Picard = None

    # sloving the system of equations for the change in width in the channel elements and pressure in the tip elements
    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_ExtendedFP,
                               MakeEquationSystem_viscousFluid_extendedFP,
                               guess,
                               typValue,
                               vk,
                               sim_properties.toleranceEHL,
                               sim_properties.maxSolverItr,
                               *arg,
                               perf_node=perfNode_Picard)

    if perfNode_Picard is not None:
        perfNode_Picard.CpuTime_end = time.time()
        perfNode.subIterations[2].append(perfNode_Picard)

    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = np.copy(Fr_lstTmStp.Tarrival)
    new_channel = np.where(FillFrac_k > 0.9999)[0]
    t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] / Vel_k[new_channel]
    max_l = Fr_lstTmStp.mesh.hx * np.cos(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * np.sin(alpha_k[new_channel])
    t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) / Vel_k[new_channel]
    Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = copy.deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w[Fr_lstTmStp.EltChannel] += sol[np.arange(Fr_lstTmStp.EltChannel.size)]
    Fr_kplus1.w[EltsTipNew] = wTip

    # check if the new width is valid
    if np.isnan(Fr_kplus1.w).any():
        exitstatus = 5
        return exitstatus, None

    if (Fr_kplus1.w < 0).any():  # todo: clean this up as it might blow up !    -> we need a linear solver with constraint to handle pinch point properly.
        print("found negative width. Ignoring...")
        # print(repr(np.where((Fr_kplus1.w < 0))))
        # print(repr(Fr_kplus1.w[np.where((Fr_kplus1.w < 0))[0]]))
        Fr_kplus1.w[np.where(Fr_kplus1.w <= 1e-10)[0]] = 1e-10
        # exitstatus = 5
        # return exitstatus, None

    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k

    # pressure evaluated by dot product of width and elasticity matrix
    Fr_kplus1.p[Fr_kplus1.EltCrack] = np.dot(C[np.ix_(Fr_kplus1.EltCrack, Fr_kplus1.EltCrack)],
                                             Fr_kplus1.w[Fr_kplus1.EltCrack])
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.InCrack = InCrack_k
    Fr_kplus1.process_fracture_front()
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * (Fr_kplus1.mesh.EltArea)
    Fr_kplus1.Tarrival = Tarrival_k

    # setting leak off
    Fr_kplus1.LkOff_vol[Fr_kplus1.EltChannel] = 2 * mat_properties.Cprime[Fr_kplus1.EltChannel] * (
            Fr_kplus1.time - Fr_kplus1.Tarrival[Fr_kplus1.EltChannel]) ** 0.5 * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff_vol[Fr_kplus1.EltTip] = 2 * mat_properties.Cprime[Fr_kplus1.EltTip] * Integral_over_cell(
                                                    Fr_kplus1.EltTip,
                                                    Fr_kplus1.alpha,
                                                    Fr_kplus1.l,
                                                    Fr_kplus1.mesh,
                                                    'Lk',
                                                    mat_prop=mat_properties,
                                                    frac=Fr_kplus1,
                                                    Vel=Vel_k,
                                                    dt=1.e20)
    injected_vol = sum(Qin) * Fr_kplus1.time
    Fr_kplus1.efficiency = (injected_vol - sum(Fr_kplus1.LkOff_vol[Fr_kplus1.EltCrack])) / injected_vol

    if sim_properties.verbosity > 1:
        print("Solved...\nFinding velocity of front...")
    itr = 0
    if sim_properties.precise_tipParam or mat_properties.anisotropic:
        alpha_ribbon = projection_from_ribbon(Fr_lstTmStp.EltRibbon,
                                              Fr_lstTmStp.EltChannel,
                                              Fr_lstTmStp.mesh,
                                              sgndDist_k)
        Kprime_k = (32 / math.pi) ** 0.5 * get_toughness_from_cellCenter(alpha_ribbon,
                                                                         sgndDist_k,
                                                                         Fr_lstTmStp.EltRibbon,
                                                                         mat_properties,
                                                                         Fr_lstTmStp.mesh)
        # Kprime from last iteration; starts with zero
        Kprime_km1 = 0 * np.copy(Kprime_k)

    # toughness iteration loop
    while itr < sim_properties.maxToughnessItr:

        sgndDist_km1 = np.copy(sgndDist_k)
        l_m1 = sgndDist_km1[Fr_lstTmStp.EltRibbon]

        if sim_properties.precise_tipParam or mat_properties.anisotropic:
            alpha_ribbon = projection_from_ribbon(Fr_lstTmStp.EltRibbon,
                                                  Fr_lstTmStp.EltChannel,
                                                  Fr_lstTmStp.mesh,
                                                  sgndDist_k)
            if np.isnan(alpha_ribbon).any():
                exitstatus = 11
                return exitstatus, None
            # under relaxing toughnesss
            Kprime_k = 0.3 * Kprime_k + 0.7 * get_toughness_from_cellCenter(alpha_ribbon,
                                                                            sgndDist_k,
                                                                            Fr_lstTmStp.EltRibbon,
                                                                            mat_properties,
                                                                            Fr_lstTmStp.mesh) * (32 / math.pi) ** 0.5

            if np.isnan(Kprime_k).any():
                exitstatus = 11
                return exitstatus, None
        else:
            Kprime_k = None

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e10 * np.ones((Fr_lstTmStp.mesh.NumberOfElts,), float)  # Initializing the cells with extremely
        # large float value. (algorithm requires inf)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(Fr_kplus1.w,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k)

        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            exitstatus = 7
            return exitstatus, None

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = np.minimum(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                       Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        front_region =  np.where(abs(Fr_lstTmStp.sgndDist) < sim_properties.tmStpPrefactor * 6.66 * (
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

        # # if some elements remain unevaluated by fast marching method. It happens with unrealistic fracture geometry.
        # # todo: not satisfied with why this happens. need re-examining
        # if max(sgndDist_k) == 1e10:
        #     exitstatus = 2
        #     return exitstatus, None

        # do it only once if not anisotropic
        if not mat_properties.anisotropic and not sim_properties.precise_tipParam:
            break

        # norm = np.linalg.norm(1 - abs(l_m1/sgndDist_k[Fr_lstTmStp.EltRibbon]))
        norm = np.linalg.norm(1 - abs(Kprime_k / Kprime_km1))
        if norm < sim_properties.toleranceToughness:
            if sim_properties.verbosity > 1:
                print("toughness iteration converged after " + repr(itr - 1) + " iterations; exiting norm " +
                      repr(norm))
            break

        Kprime_km1 = np.copy(Kprime_k)
        if sim_properties.verbosity > 1:
            print("iterating on toughness... norm " + repr(norm))
        itr += 1

    # if itr == sim_properties.maxToughnessItr:
    #     exitstatus = 10
    #     return exitstatus, None

    Fr_kplus1.v = -(sgndDist_k[Fr_kplus1.EltTip] - Fr_lstTmStp.sgndDist[Fr_kplus1.EltTip]) / timeStep
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    if sim_properties.saveRegime:
        regime_t = find_regime(Fr_kplus1.w, Fr_lstTmStp, mat_properties, sim_properties, timeStep, Kprime_k,
                               -sgndDist_k[Fr_lstTmStp.EltRibbon])
        Fr_kplus1.regime = np.vstack((regime_t, Fr_lstTmStp.EltRibbon))

    # # check if the tip has laminar flow, to be consistent with tip asymptote.
    # ReNumb, check = turbulence_check_tip(vel, Fr_kplus1, fluid_properties, return_ReyNumb=True)
    # # plot Reynold's number
    # plot_Reynolds_number(Fr_kplus1, ReNumb, 1)

    exitstatus = 1
    return exitstatus, Fr_kplus1

# -----------------------------------------------------------------------------------------------------------------------

def tip_neighbor_in_ribbon(tip_elts, tip_zero_vrtx, mesh):

    coresp_neighbor = np.empty((len(tip_elts), ) , dtype=int)
    neighbors = mesh.NeiElements[tip_elts]
    for i in range(len(tip_elts)):
        if tip_zero_vrtx[i] == 0:
            coresp_neighbor[i] = neighbors[i, 2] - 1
        elif tip_zero_vrtx[i] == 1:
            coresp_neighbor[i] = neighbors[i, 2] + 1
        elif tip_zero_vrtx[i] == 2:
            coresp_neighbor[i] = neighbors[i, 3] + 1
        elif tip_zero_vrtx[i] == 3:
            coresp_neighbor[i] = neighbors[i, 3] - 1

    return coresp_neighbor