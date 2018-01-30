#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

from src.VolIntegral import *
from src.Utility import *
from src.TipInversion import *
from src.ElastoHydrodynamicSolver import *
from src.LevelSet import *
from src.HFAnalyticalSolutions import *
from src.TimeSteppingMechLoading import *
from src.TimeSteppingVolumeControl import *
from scipy.optimize import least_squares
import copy
import warnings
import sys

def attempt_time_step_viscousFluid(Frac, C, Material_properties, Fluid_properties, Simulation_Parameters, Injection_Parameters,
                      TimeStep):
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
                                    10      -- Toughness iteration did not converge
                                    
        Fracture object:            fracture after advancing time step. 
    """

    exitstatus = 0  # exit code to be returned

    # index of current time in the time series (first row) of the injection rate array
    indxCurTime = max(np.where(Frac.time >= Injection_Parameters.injectionRate[0, :])[0])
    CurrentRate = Injection_Parameters.injectionRate[1, indxCurTime]  # current injection rate

    Qin = np.zeros((Frac.mesh.NumberOfElts), float)
    Qin[Injection_Parameters.source_location] = CurrentRate # current injection over the domain

    # todo : write log file
    # f = open('log', 'a')

    print('Solving ElastoHydrodynamic equations with same footprint...')
    # width by injecting the fracture with the same foot print (balloon like inflation)
    exitstatus, w_k = injection_same_footprint(Frac,
                                               C,
                                               TimeStep,
                                               Qin,
                                               Material_properties,
                                               Fluid_properties,
                                               Simulation_Parameters)

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
        (exitstatus, Fr_k) = injection_extended_footprint(w_k,
                                                          Frac,
                                                          C,
                                                          TimeStep,
                                                          Qin,
                                                          Material_properties,
                                                          Fluid_properties,
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


# ----------------------------------------------------------------------------------------------------------------------

def injection_same_footprint(Fr_lstTmStp, C, timeStep, Qin, mat_properties, Fluid_properties, Simulation_Parameters):
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

    # average injected fluid over footprint taken as [\delta] W guess for the iterative solver
    delwGuess = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size * np.ones((Fr_lstTmStp.EltCrack.size,), float)


    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
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
    if Fluid_properties.turbulence:

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
        Fluid_properties.density,
        Fluid_properties.turbulence,
        mat_properties.grainSize)

    # typical values of the variable. Used to calculate Jacobian (see Piccard_Newton function documentation)
    # todo: guess is taken as typical values. Needs to be reconsidered
    typclValue = delwGuess

    # solving the system
    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_sameFP,
                               MakeEquationSystem_viscousFluid_sameFP,
                               delwGuess,
                               typclValue,
                               vk,
                               Simulation_Parameters.toleranceEHL,
                               Simulation_Parameters.maxSolverItr,
                               *argSameFP)

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

def injection_extended_footprint(w_k, Fr_lstTmStp, C, timeStep, Qin, Material_properties, Fluid_properties,
                                 sim_parameters):
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
                                    10      -- Toughness iteration did not converge
                                    11      -- Projection could not be found
                                    12      -- Reached end of grid

        Fracture object:            fracture after advancing time step. 
    """

    itr = 0
    sgndDist_k = np.copy(Fr_lstTmStp.sgndDist)
    if Material_properties.anisotropic:
        Kprime_k = toughness_at_tip_CellCenter(Fr_lstTmStp.EltRibbon,
                                           Fr_lstTmStp.EltChannel,
                                           Fr_lstTmStp.mesh,
                                           Material_properties,
                                           sgndDist_k)
        Kprime_km1 = 0 * np.copy(Kprime_k)
    # toughness iteration loop
    while itr < sim_parameters.maxToughnessItr:

        sgndDist_km1 = np.copy(sgndDist_k)
        l_m1 = sgndDist_km1[Fr_lstTmStp.EltRibbon]

        #todo: Only done for anistropic. Has to be done for heterogenous toughness
        if Material_properties.anisotropic:
            Kprime_k = 0.7 * Kprime_k + 0.3 * toughness_at_tip_CellCenter(Fr_lstTmStp.EltRibbon,
                                                                          Fr_lstTmStp.EltChannel,
                                                                          Fr_lstTmStp.mesh,
                                                                          Material_properties,
                                                                          sgndDist_k)
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
                                                               Material_properties,
                                                               sim_parameters,
                                                               timeStep)


        # if tip inversion returns nan
        if np.isnan(sgndDist_k[Fr_lstTmStp.EltRibbon]).any():
            exitstatus = 7
            return exitstatus, None

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        front_region = np.where(abs(Fr_lstTmStp.sgndDist) < 2 * (Fr_lstTmStp.mesh.hx**2 + Fr_lstTmStp.mesh.hy**2)**0.5)[0]
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

        # # if some elements remain unevaluated by fast marching method. It happens with unrealistic fracture geometry.
        # # todo: not satisfied with why this happens. need re-examining
        # if max(sgndDist_k) == 1e10:
        #     exitstatus = 2
        #     return exitstatus, None

        # do it only once if not anisotropic
        if not Material_properties.anisotropic:
            break

        # norm = np.linalg.norm(1 - abs(l_m1/sgndDist_k[Fr_lstTmStp.EltRibbon]))
        norm = np.linalg.norm(1 - abs(Kprime_k / Kprime_km1))
        if norm < sim_parameters.toleranceToughness:
            print("toughness iteration converged after " + repr(itr-1) + " iterations; exiting norm " +
                  repr(norm))
            break



        Kprime_km1 = np.copy(Kprime_k)
        print("iterating on toughness... norm " + repr(norm))
        itr += 1

    # if itr == sim_parameters.maxToughnessItr:
    #     exitstatus = 10
    #     return exitstatus, None

    # regime_t = find_regime(w_k, Fr_lstTmStp, Material_properties, sim_parameters, timeStep, Kprime_k,
    #                        -sgndDist_k[Fr_lstTmStp.EltRibbon])

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
            exitstatus = 12
            return exitstatus, None
            # Fr_lstTmStp.plot_fracture('complete', 'footPrint')
            # raise SystemExit('Reached end of the grid. exiting....')

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
    # zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    # Kprime_tip = toughness_at_tip_zeroVertex(EltsTipNew,
    #                                        Fr_lstTmStp.mesh,
    #                                        Material_properties,
    #                                        alpha_k,
    #                                        l_k,
    #                                        zrVrtx_newTip)

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

        wTip = Integral_over_cell(EltsTipNew,
                              alpha_k,
                              l_k,
                              Fr_lstTmStp.mesh,
                              sim_parameters.tipAsymptote,
                              frac=Fr_lstTmStp,
                              mat_prop=Material_properties,
                              fluid_prop=Fluid_properties,
                              Vel=Vel_k,
                              stagnant=stagnant,
                              KIPrime=KIPrime) / Fr_lstTmStp.mesh.EltArea
    else:
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = Integral_over_cell(EltsTipNew,
                              alpha_k,
                              l_k,
                              Fr_lstTmStp.mesh,
                              sim_parameters.tipAsymptote,
                              frac=Fr_lstTmStp,
                              mat_prop=Material_properties,
                              fluid_prop=Fluid_properties,
                              Vel=Vel_k,
                              stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea

    # # check if the tip volume has gone into negative
    # smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -1e-4 * np.mean(wTip)))
    # if np.asarray(smallNgtvWTip).size > 0:
    #     #                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
    #     wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])


    if (wTip < 0).any():
        exitstatus = 4
        return exitstatus, None

    guess = np.zeros((Fr_lstTmStp.EltChannel.size + EltsTipNew.size,), float)
    # pguess = Fr_lstTmStp.p[EltsTipNew]

    guess[np.arange(Fr_lstTmStp.EltChannel.size)] = timeStep * sum(Qin) / Fr_lstTmStp.EltCrack.size \
                                                    * np.ones((Fr_lstTmStp.EltCrack.size,), float)

    LkOff = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    # Calculate leak-off term for the tip cell
    LkOff[EltsTipNew] = 2 * Material_properties.Cprime[EltsTipNew] * Integral_over_cell(EltsTipNew,
                                                                alpha_k,
                                                                l_k,
                                                                Fr_lstTmStp.mesh,
                                                                'Lk',
                                                                mat_prop=Material_properties,
                                                                frac=Fr_lstTmStp,
                                                                Vel=Vel_k,
                                                                dt=timeStep)

    #todo: no need to evaluate on each iteration. Need to decide. Evaluating here for now for better readability
    LkOff[Fr_lstTmStp.EltChannel] = 2 * Material_properties.Cprime[Fr_lstTmStp.EltChannel] * ((Fr_lstTmStp.time +
                    timeStep - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel])**0.5 - (Fr_lstTmStp.time -
                    Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel])**0.5) * Fr_lstTmStp.mesh.EltArea

    # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
    vk = np.zeros((4, Fr_lstTmStp.mesh.NumberOfElts,), dtype=np.float64)
    if Fluid_properties.turbulence:
        wguess = np.copy(Fr_lstTmStp.w)
        wguess[Fr_lstTmStp.EltChannel] = wguess[Fr_lstTmStp.EltChannel] + guess[np.arange(Fr_lstTmStp.EltChannel.size)]
        wguess[EltsTipNew] = wTip

        vk = velocity(wguess,
                      EltCrack_k,
                      Fr_lstTmStp.mesh,
                      InCrack_k,
                      Fr_lstTmStp.muPrime,
                      C,
                      Material_properties.SigmaO)

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
        Fluid_properties.density,
        InCrack_k,
        LkOff,
        Material_properties.SigmaO,
        Fluid_properties.turbulence,
        Material_properties.grainSize
        )

    # sloving the system of equations for the change in width in the channel elements and pressure in the tip elements
    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_ExtendedFP,
                               MakeEquationSystem_viscousFluid_extendedFP,
                               guess,
                               typValue,
                               vk,
                               sim_parameters.toleranceEHL,
                               sim_parameters.maxSolverItr,
                               *arg)

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
        print(repr(np.where((Fr_kplus1.w < 0))))
        print(repr(Fr_kplus1.w[np.where((Fr_kplus1.w < 0))[0]]))
#        exitstatus = 5
#        return exitstatus, None

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
    Fr_kplus1.InCrack = InCrack_k
    Fr_kplus1.process_fracture_front()
    Fr_kplus1.FractureVolume = np.sum(Fr_kplus1.w) * (Fr_kplus1.mesh.EltArea)
    Fr_kplus1.Tarrival = Tarrival_k

    Fr_kplus1.LkOff_vol[Fr_kplus1.EltChannel] = 2 * Material_properties.Cprime[Fr_kplus1.EltChannel] * (
                            Fr_kplus1.time - Fr_kplus1.Tarrival[Fr_kplus1.EltChannel])**0.5 * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff_vol[Fr_kplus1.EltTip]= 2 * Material_properties.Cprime[Fr_kplus1.EltTip] * Integral_over_cell(
                                                                Fr_kplus1.EltTip,
                                                                Fr_kplus1.alpha,
                                                                Fr_kplus1.l,
                                                                Fr_kplus1.mesh,
                                                                'Lk',
                                                                mat_prop=Material_properties,
                                                                frac=Fr_kplus1,
                                                                Vel=Fr_kplus1.v,
                                                                dt=1.e20)
    injected_vol = sum(Qin) * Fr_kplus1.time
    Fr_kplus1.efficiency = (injected_vol - sum(Fr_kplus1.LkOff_vol[Fr_kplus1.EltCrack])) / injected_vol
    # Fig = Fr_kplus1.plot_fracture('complete','footPrint')
    # plot_as_matrix(Tarrival_k, Fr_kplus1.mesh)

    # Fr_kplus1.regime = np.vstack((regime_t, Fr_lstTmStp.EltRibbon))

    # # check if the tip has laminar flow, to be consistent with tip asymptote.
    # ReNumb, check = turbulence_check_tip(vel, Fr_kplus1, Fluid_properties, return_ReyNumb=True)
    # # plot Reynold's number
    # plot_Reynolds_number(Fr_kplus1, ReNumb, 1)

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

#-----------------------------------------------------------------------------------------------------------------------

def toughness_at_tip_CellCenter(ribbon_elts, channel_elts, mesh, mat_prop, sgnd_dist):
    """
    This function gives the scaled toughness(Kprime) at the closest tip point from the cell centers of the ribbon cells.
    The function is different from the toughness_at_tip as it calculates the closest tip from cell centers and not from
    the zero vertex.
    Arguments:
        ribbon_elts (ndarray-int): list of ribbon elements
        mesh (CartesianMesh object): The cartesian mesh object
        mat_prop (MaterialProperties object):    Material properties:
        sgnd_dist (ndarray-float): level set data

    Returns:
        ndarray-float : Kprime at the closest tip point from the center of the given ribbon cells
    """

    # dist = -sgnd_dist
    # alpha = np.zeros((ribbon_elts.size,), dtype=np.float64)

    neighbors = mesh.NeiElements[ribbon_elts]
    (elt_tip, l_tip, alpha_tip, CellStatus) = reconstruct_front(sgnd_dist,
                                                               channel_elts,
                                                               mesh)
    FillFrac = Integral_over_cell(elt_tip,
                                alpha_tip,
                                l_tip,
                                mesh,
                                'A') / mesh.EltArea


    partly_filled = np.where(FillFrac<0.999999)[0]
    # inside = np.logical_and(l_tip*np.cos(alpha_tip)<=mesh.hx*1.01, l_tip*np.sin(alpha_tip)<=mesh.hy*1.01)
    # partly_filled = np.where(inside)[0]
    elt_tip = elt_tip[partly_filled]
    l_tip = l_tip[partly_filled]
    alpha_tip = alpha_tip[partly_filled]

    # K = np.zeros((mesh.NumberOfElts,), )
    # K[elt_tip] = FillFrac[partly_filled]
    # plot_as_matrix(K, mesh)

    to_delete = np.array([], dtype=int)
    zero_alpha = np.where(alpha_tip == 0.)[0]
    for i in zero_alpha:
        lftneigb_in_zero = np.where(elt_tip[zero_alpha] == mesh.NeiElements[elt_tip[i], 0])[0]
        if lftneigb_in_zero.size > 0:
            if l_tip[zero_alpha[lftneigb_in_zero]] > l_tip[i]:
                to_delete = np.append(to_delete, zero_alpha[lftneigb_in_zero])
            else:
                to_delete = np.append(to_delete, i)
        rgtneigb_in_zero = np.where(elt_tip[zero_alpha] == mesh.NeiElements[elt_tip[i], 1])[0]
        if rgtneigb_in_zero.size > 0:
            if l_tip[zero_alpha[rgtneigb_in_zero]] > l_tip[i]:
                to_delete = np.append(to_delete, zero_alpha[rgtneigb_in_zero])
            else:
                to_delete = np.append(to_delete, i)


    ninety_alpha = np.where(alpha_tip == np.pi/2)[0]
    for i in ninety_alpha:
        btmneigb_in_ninety = np.where(elt_tip[ninety_alpha] == mesh.NeiElements[elt_tip[i], 2])[0]
        if btmneigb_in_ninety.size > 0:
            if l_tip[ninety_alpha[btmneigb_in_ninety]] > l_tip[i]:
                to_delete = np.append(to_delete, ninety_alpha[btmneigb_in_ninety])
            else:
                to_delete = np.append(to_delete, i)
        topneigb_in_ninety = np.where(elt_tip[ninety_alpha] == mesh.NeiElements[elt_tip[i], 3])[0]
        if topneigb_in_ninety.size > 0:
            if l_tip[ninety_alpha[topneigb_in_ninety]] > l_tip[i]:
                to_delete = np.append(to_delete, ninety_alpha[topneigb_in_ninety])
            else:
                to_delete = np.append(to_delete, i)
    to_delete = np.unique(to_delete)


    elt_tip = np.delete(elt_tip, to_delete)
    l_tip = np.delete(l_tip, to_delete)
    alpha_tip = np.delete(alpha_tip, to_delete)



    if np.isnan(alpha_tip).any():
        is_nan = np.where(np.isnan(alpha_tip))[0]
        for i in is_nan:
            enclosing = mesh.NeiElements[elt_tip[i]]
            enclosing = np.array([enclosing[0],enclosing[2]-1,enclosing[3]-1,enclosing[2],enclosing[3],enclosing[2]+1,enclosing[3]+1,enclosing[1]])
            for j in range(5):
                lft_in_tip = np.where(elt_tip == enclosing[j])[0]
                if lft_in_tip.size>0 and not np.isnan(alpha_tip[lft_in_tip]):
                    break
            for j in range(7,2,-1):
                rgt_in_tip = np.where(elt_tip == enclosing[j])[0]
                if rgt_in_tip.size>0 and not np.isnan(alpha_tip[rgt_in_tip]):
                    break
            if rgt_in_tip.size>1 or lft_in_tip.size>1:
                print("found double")
            alpha_tip[i] = (alpha_tip[rgt_in_tip]+alpha_tip[lft_in_tip])/2
            print("corrected alpha tip")

    zero_vertex_tip = find_zero_vertex(elt_tip, sgnd_dist, mesh)
    smthed_tip, a, b, c, pnt_lft, pnt_rgt, neig_lft, neig_rgt = construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip)
    if np.isnan(smthed_tip).any():
        return np.nan
    zr_vrtx_smthed_tip = find_zero_vertex(smthed_tip, sgnd_dist, mesh)
    alpha = find_projection(ribbon_elts, smthed_tip, zr_vrtx_smthed_tip, a, b, c, pnt_lft[:,0], pnt_lft[:,1], pnt_rgt[:,0], pnt_rgt[:,1], neig_lft, neig_rgt, mesh)
    # alpha = get_alpha_at_cell_Center_reconstructed_front(ribbon_elts, elt_tip, alpha_tip, l_tip, zero_vertex_tip, mesh, mat_prop)
    # K = np.zeros((mesh.NumberOfElts,), )
    # K[ribbon_elts] = alpha
    # plot_as_matrix(K, mesh)

    if mat_prop.anisotropic:
        return mat_prop.KprimeFunc(alpha)
    else:
        dist = -sgnd_dist
        x = np.zeros((len(ribbon_elts),),)
        y = np.zeros((len(ribbon_elts),), )

        # evaluating the closest tip points
        for i in range(0, len(ribbon_elts)):
            if zero_vertex[i]==0:

                x[i] = mesh.CenterCoor[ribbon_elts[i],0] + dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i],1] + dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i]==1:

                x[i] = mesh.CenterCoor[ribbon_elts[i],0] - dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i],1] + dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i]==2:

                x[i] = mesh.CenterCoor[ribbon_elts[i],0] - dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i],1] - dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i]==3:

                x[i] = mesh.CenterCoor[ribbon_elts[i],0] + dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i],1] - dist[ribbon_elts[i]] * np.sin(alpha[i])

            if abs(dist[mesh.NeiElements[ribbon_elts[i],0]]/dist[mesh.NeiElements[ribbon_elts[i],1]]-1) < 1e-7:
                if sgnd_dist[neighbors[i,2]] < sgnd_dist[neighbors[i,3]]:
                    x[i] = mesh.CenterCoor[ribbon_elts[i], 0]
                    y[i] = mesh.CenterCoor[ribbon_elts[i], 1] + dist[ribbon_elts[i]]
                elif sgnd_dist[neighbors[i,2]] > sgnd_dist[neighbors[i,3]]:
                    x[i] = mesh.CenterCoor[ribbon_elts[i], 0]
                    y[i] = mesh.CenterCoor[ribbon_elts[i], 1] - dist[ribbon_elts[i]]

        # returning the Kprime according to the given function
        return mat_prop.KprimeFunc(x, y)

#-----------------------------------------------------------------------------------------------------------------------
def find_zero_vertex(Elts, level_set, mesh):

    zero_vertex = np.zeros((len(Elts),),dtype=int)
    for i in range(0, len(Elts)):
        neighbors = mesh.NeiElements[Elts]

        if level_set[neighbors[i, 0]] <= level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] <= level_set[
                                                                                                neighbors[i, 3]]:
            zero_vertex[i] = 0
        elif level_set[neighbors[i, 0]] > level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] <= level_set[
                                                                                                neighbors[i, 3]]:
            zero_vertex[i] = 1
        elif level_set[neighbors[i, 0]] > level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] > level_set[
                                                                                                neighbors[i, 3]]:
            zero_vertex[i] = 2
        elif level_set[neighbors[i, 0]] <= level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] > level_set[
                                                                                                neighbors[i, 3]]:
            zero_vertex[i] = 3

        # neighbors = mesh.NeiElements[Elts]
        #
        # if level_set[Elts[i]] <= level_set[neighbors[i, 1]] and level_set[Elts[i]] <= level_set[
        #     neighbors[i, 3]]:
        #     zero_vertex[i] = 0
        # elif level_set[neighbors[i, 0]] > level_set[Elts[i]] and level_set[Elts[i]] <= level_set[
        #     neighbors[i, 3]]:
        #     zero_vertex[i] = 1
        # elif level_set[neighbors[i, 0]] > level_set[Elts[i]] and level_set[neighbors[i, 2]] > level_set[
        #     Elts[i]]:
        #     zero_vertex[i] = 2
        # elif level_set[Elts[i]] <= level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] > level_set[
        #     Elts[i]]:
        #     zero_vertex[i] = 3



    return zero_vertex

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


def get_alpha_at_cell_Center(elems, mesh, sgnd_dist, zero_vertex):

    dist = -sgnd_dist
    alpha = np.zeros((elems.size,), dtype=np.float64)
    for i in range(0, len(elems)):
        if zero_vertex[i]==0:
            # north-east direction of propagation
            cos_alpha = (dist[elems[i]] - dist[mesh.NeiElements[elems[i], 1]]) / mesh.hx
            alpha[i] = np.arccos(cos_alpha)
            if cos_alpha<0 or cos_alpha>1.:
                cos_alpha = (dist[mesh.NeiElements[elems[i], 0]] - dist[elems[i]]) / mesh.hx
                alpha[i] = np.arccos(cos_alpha)


        elif zero_vertex[i]==1:
            # north-west direction of propagation
            cos_alpha = (dist[elems[i]] - dist[mesh.NeiElements[elems[i], 0]]) / mesh.hx
            alpha[i] = np.arccos(cos_alpha)
            if cos_alpha<0 or cos_alpha>1.:
                cos_alpha = (dist[mesh.NeiElements[elems[i], 1]] - dist[elems[i]]) / mesh.hx
                alpha[i] = np.arccos(cos_alpha)

        elif zero_vertex[i]==2:
            # south-west direction of propagation
            cos_alpha = (dist[elems[i]] - dist[mesh.NeiElements[elems[i], 0]]) / mesh.hx
            alpha[i] = np.arccos(cos_alpha)
            if cos_alpha<0 or cos_alpha>1.:
                cos_alpha = (dist[mesh.NeiElements[elems[i], 1]] - dist[elems[i]] ) / mesh.hx
                alpha[i] = np.arccos(cos_alpha)

        elif zero_vertex[i]==3:
            # south-east direction of propagation
            cos_alpha = (dist[elems[i]] - dist[mesh.NeiElements[elems[i], 1]]) / mesh.hx
            alpha[i] = np.arccos(cos_alpha)
            if cos_alpha<0 or cos_alpha>1.:
                cos_alpha = (dist[mesh.NeiElements[elems[i], 0]] - dist[elems[i]] ) / mesh.hx
                alpha[i] = np.arccos(cos_alpha)


        warnings.filterwarnings("ignore")
        if abs(dist[mesh.NeiElements[elems[i], 0]] / dist[mesh.NeiElements[elems[i], 1]] - 1) < 1e-12:
            # if the angle is 90 degrees
            alpha[i] = np.pi / 2
        if abs(dist[mesh.NeiElements[elems[i], 2]] / dist[mesh.NeiElements[elems[i], 3]] - 1) < 1e-12:
            # if the angle is 0 degrees
            alpha[i] = 0

    # alpha_lst_sqrs = np.zeros((elems.size,), dtype=np.float64)
    # for i in range(0, len(elems)):
    #
    #     neighbors = mesh.NeiElements[elems[i]]
    #     enclosing = np.array([neighbors[2] - 1,
    #                           neighbors[0],
    #                           neighbors[3] - 1,
    #                           neighbors[2],
    #                           neighbors[3],
    #                           neighbors[2] + 1,
    #                           neighbors[1],
    #                           neighbors[3] + 1
    #                           ])
    #
    #     left_neigh = np.asarray([], int)  # find neighbors in Ribbon cells
    #     for e in range(5):
    #         lft_indx = np.where(elems == enclosing[e])[0]
    #         if lft_indx.size > 0:
    #             left_neigh = elems[lft_indx[0]]
    #             break
    #     for e in range(7, 2, -1):
    #         rght_indx = np.where(elems == enclosing[e])[0]
    #         if rght_indx.size > 0:
    #             right_neigh = elems[rght_indx[0]]
    #             break
    #
    #     if zero_vertex[i] == 0:
    #         left = -mesh.hx
    #         dist_lft = dist[left_neigh]
    #
    #
    #     guess = np.array([-mesh.hx + dist[left_neigh] * np.cos(alpha[lft_indx]),
    #                       dist[elems[i]] * np.cos(alpha[i]),
    #                       mesh.hx + dist[right_neigh] * np.cos(alpha[rght_indx]),
    #                       dist[left_neigh] * np.sin(alpha[lft_indx]),
    #                       dist[elems[i]] * np.sin(alpha[i]),
    #                       dist[right_neigh] * np.sin(alpha[rght_indx])
    #                       ])
    #
    #     args = (dist[left_neigh],
    #             dist[elems[i]],
    #             dist[right_neigh],
    #             [-mesh.hx, 0.],
    #             [0., 0.],
    #             [mesh.hx, 0.]
    #             )
    #
    #     sol = least_squares(fun, guess, args=args)
    #     slope = (sol.x[4]-sol.x[3])/(sol[1].x-sol.x[0])
    #     intrcpt = sol[4].x-slope*sol.x[1]
    #
    #     alpha_lst_sqrs[i] = np.pi/2 - np.arccos(intrcpt/dist[elems[i]])

    alpha_lst_sqrs = np.zeros((elems.size,), dtype=np.float64)
    for i in range(0, len(elems)):

        neighbors = mesh.NeiElements[elems[i]]

        if zero_vertex[i] == 0:
            l_left = dist[neighbors[0]]
            l_rght = dist[neighbors[1]]
        if zero_vertex[i] == 1:
            l_left = dist[neighbors[1]]
            l_rght = dist[neighbors[0]]
        if zero_vertex[i] == 2:
            l_left = dist[neighbors[1]]
            l_rght = dist[neighbors[0]]
        if zero_vertex[i] == 3:
            l_left = dist[neighbors[0]]
            l_rght = dist[neighbors[1]]


        guess = np.array([-mesh.hx + l_left * np.cos(alpha[i]),
                          dist[elems[i]] * np.cos(alpha[i]),
                          mesh.hx + l_rght * np.cos(alpha[i]),
                          l_left * np.sin(alpha[i]),
                          dist[elems[i]] * np.sin(alpha[i]),
                          l_rght * np.sin(alpha[i])
                          ])

        args = (l_left,
                dist[elems[i]],
                l_rght,
                [-mesh.hx, 0.],
                [0., 0.],
                [mesh.hx, 0.],
                [dist[elems[i]] * np.cos(alpha[i]),dist[elems[i]] * np.sin(alpha[i])])

        bound = (guess-mesh.hx/4, guess+mesh.hx/4)

        sol = least_squares(fun, guess, args=args, bounds=bound)
        fun(sol.x, *args)

        slope = (sol.x[1]-sol.x[0])/(sol.x[4]-sol.x[3])
        intrcpt = sol.x[1]-slope*sol.x[4]

        alpha_lst_sqrs[i] = np.pi/2 - np.arccos((sol.x[4]**2+sol.x[1]**2)**0.5/intrcpt)


    return alpha


def fun(x, *arg):
    ylu = x[0]
    y0u = x[1]
    yru = x[2]
    xlu = x[3]
    x0u = x[4]
    xru = x[5]
    slp1 = (y0u-ylu)/(x0u-xlu)
    slp2 = (yru-y0u)/(xru-x0u)
    slp3 = (yru - ylu) / (xru - xlu)
    (ll,l0,lr, coorl,coor0,coorr,coorr_guess) = arg

    res = np.zeros((7,),dtype=np.float64)
    res[0]=yru-ylu- slp1 * (xru-xlu)
    res[1]=ylu-y0u- slp2 * (xlu-x0u)
    res[2]=yru-y0u- slp3 * (xru-x0u)

    res[3] = ll - ((ylu - coorl[1]) ** 2 + (xlu - coorl[0]) ** 2)**0.5
    res[4] = l0 - ((y0u - coor0[1]) ** 2 + (x0u - coor0[0]) ** 2)**0.5
    res[5] = lr - ((yru - coorr[1]) ** 2 + (xru - coorr[0]) ** 2)**0.5
    res[6] = ((y0u - coorr_guess[1]) ** 2 + (x0u - coorr_guess[0]) ** 2)**0.5

    return res

def get_alpha_at_cell_Center_reconstructed_front(elt_ribbon, elt_tip, alpha_tip, l_tip, zero_vertex, mesh, mat_prop):

    alpha = np.zeros((len(elt_ribbon), ), dtype=np.float64)
    slope = np.empty((len(elt_tip), ), dtype=np.float64)
    pnt_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    for i in range(len(elt_tip)):

        if zero_vertex[i] == 0:
            slope[i] = np.tan(-(np.pi/2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 0]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex[i] == 1:
            slope[i] = np.tan(np.pi/2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 1]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex[i] == 2:
            slope[i] = np.tan(-(np.pi/2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 2]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex[i] == 3:
            slope[i] = np.tan(np.pi/2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 3]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])

    # for i in range(mesh.nx+1):
    #     plt.plot([mesh.VertexCoor[i, 0], mesh.VertexCoor[i, 0]],
    #              [mesh.VertexCoor[0, 1],mesh.VertexCoor[-1, 1]],'k')
    # for i in range(mesh.ny+1):
    #     plt.plot([mesh.VertexCoor[0, 0], mesh.VertexCoor[-1, 0]],
    #              [mesh.VertexCoor[i*(mesh.nx+1), 1],mesh.VertexCoor[i*(mesh.nx+1), 1]],'k')
    #
    # for i in range(len(elt_tip)):
    #     plt.plot([pnt_on_line[i, 0], mesh.VertexCoor[mesh.Connectivity[elt_tip[i], zero_vertex[i]],0] ], [pnt_on_line[i, 1], mesh.VertexCoor[mesh.Connectivity[elt_tip[i], zero_vertex[i]],1]],'r')
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()
    # plt.close('all')
    b = np.ones((len(elt_tip),), dtype=np.float64)
    a = - slope
    c = -(pnt_on_line[:,1] - slope * pnt_on_line[:,0])

    points_lft = np.ones((len(elt_tip), 2), dtype=np.float64)
    points_rgt = np.ones((len(elt_tip), 2), dtype=np.float64)

    for i in range(len(elt_tip)):
        x = mesh.CenterCoor[elt_tip[i], 0] - mesh.hx / 2
        y = -c[i] - a[i] * x
        if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
            y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
            x = (-c[i] - y) / a[i]
        elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
            y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
            x = (-c[i] - y) / a[i]
        points_lft[i, 0] = x
        points_lft[i, 1] = y

        x = mesh.CenterCoor[elt_tip[i], 0] + mesh.hx / 2
        y = -c[i] - a[i] * x
        if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
            y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
            x = (-c[i] - y) / a[i]
        elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
            y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
            x = (-c[i] - y) / a[i]
        points_rgt[i, 0] = x
        points_rgt[i, 1] = y

    # neighbor_lft = np.empty((len(elt_tip),), dtype=np.int)
    # neighbor_rgt = np.empty((len(elt_tip),), dtype=np.int)
    # for i in range(len(elt_tip)):
    #     x = mesh.CenterCoor[elt_tip[i], 0] - mesh.hx / 2
    #     y = -c[i] - a[i] * x
    #     lft_in_tip = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 0])[0]
    #     if lft_in_tip.size > 0:
    #         neighbor_lft[i] = lft_in_tip
    #     if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neighbor_lft[i] = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 3])[0]
    #     elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neighbor_lft[i] = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 2])[0]
    #     points_lft[i, 0] = x
    #     points_lft[i, 1] = y
    #
    #     x = mesh.CenterCoor[elt_tip[i], 0] + mesh.hx / 2
    #     y = -c[i] - a[i] * x
    #     rgt_in_tip = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 1])[0]
    #     if rgt_in_tip.size > 0:
    #         neighbor_rgt[i] = rgt_in_tip
    #     if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neighbor_rgt[i] = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 3])[0]
    #     elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neighbor_rgt[i] = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 2])[0]
    #     points_rgt[i, 0] = x
    #     points_rgt[i, 1] = y

    closest_tip_cell = np.zeros((len(elt_ribbon),), dtype=np.int)
    dist_ribbon = np.zeros((len(elt_ribbon),), dtype=np.float64)

    for i in range(len(elt_ribbon)):
        dist_front_line = np.zeros((len(elt_tip),), dtype=np.float64)
        point_at_grid_line = np.zeros((len(elt_tip),), dtype=np.uint8)
        for j in range(len(elt_tip)):

            # if abs(alpha_tip[j]) < 1e-8:
            #     xx = mesh.CenterCoor[elt_ribbon[i], 0]
            #     yy = - c[j]
            # elif abs(alpha_tip[j]-np.pi/2)< 1e-8:
            #     yy = mesh.CenterCoor[elt_ribbon[i], 1]
            #     xx = points_lft[j, 0]
            # else:
            m = - 1/slope[j]
            intrcpt = mesh.CenterCoor[elt_ribbon[i], 1] - m * mesh.CenterCoor[elt_ribbon[i], 0]
            xx = -(intrcpt + c[j])/(a[j]+m)
            yy = m*xx + intrcpt

            # plt.plot([mesh.CenterCoor[elt_ribbon[i], 0], xx], [mesh.CenterCoor[elt_ribbon[i], 1], yy])
            # plt.plot([points_lft[j,0], points_rgt[j,0]], [points_lft[j,1], points_rgt[j,1]])
            # plt.axis("equal")
            # axes = plt.gca()
            # axes.set_xlim([-30., 30.])
            # axes.set_ylim([-30., 30.])


            if points_lft[j,0] > xx or points_rgt[j,0] < xx or min(points_lft[j,1],points_rgt[j,1]) > yy or max(points_lft[j,1],points_rgt[j,1]) < yy:
                dist_lft_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_lft[j,0])**2
                                 + (mesh.CenterCoor[elt_ribbon[i], 1] - points_lft[j,1])**2) ** 0.5
                dist_rgt_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_rgt[j, 0]) ** 2
                                 + (mesh.CenterCoor[elt_ribbon[i], 1] - points_rgt[j, 1]) ** 2) ** 0.5

                dist_front_line[j] = min(dist_lft_pnt, dist_rgt_pnt)
                if dist_lft_pnt < dist_rgt_pnt:
                    point_at_grid_line[j] = 1
                else:
                    point_at_grid_line[j] = 2
            else:
                dist_front_line[j] = abs(
                    mesh.CenterCoor[elt_ribbon[i], 0] * a[j] + mesh.CenterCoor[elt_ribbon[i], 1] * b[j] + c[j]) / (a[
                                                                                                             j] ** 2 +
                                                                                                      b[j] ** 2) ** 0.5
        # if not (np.logical_not(np.isinf(dist_front_line))).any():
        #     dist_lft_gridIntr = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_lft[:,0])**2
        #                          + (mesh.CenterCoor[elt_ribbon[i], 1] - points_lft[:,1])**2) ** 0.5
        #     dist_rgt_gridIntr = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_rgt[:, 0]) ** 2
        #                          + (mesh.CenterCoor[elt_ribbon[i], 1] - points_rgt[:, 1]) ** 2) ** 0.5
        #     dist_front_line = np.minimum(dist_lft_gridIntr, dist_rgt_gridIntr)
        #     print("found")

        closest_tip_cell[i] = np.argmin(dist_front_line)
        # chk_mult = np.where(dist_front_line == dist_front_line[closest_tip_cell[i]])[0]
        chk_mult = np.where(abs(1 - dist_front_line / dist_front_line[closest_tip_cell[i]]) < 1e-8)[0]
        if chk_mult.size <= 1:
            if point_at_grid_line[closest_tip_cell[i]] == 0:
                y = mesh.CenterCoor[elt_ribbon[i], 1]
                x = (-b[closest_tip_cell[i]] * y - c[closest_tip_cell[i]]) / a[closest_tip_cell[i]]
                alpha[i] = np.arccos(round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
                dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
            elif point_at_grid_line[closest_tip_cell[i]] == 1:
                y = mesh.CenterCoor[elt_ribbon[i], 1]
                x = (-y - c[closest_tip_cell[i]]) / a[closest_tip_cell[i]]
                alpha[i] = np.arccos(
                    round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
                # x = (-y - c[neig_lft[closest_tip_cell[i]]]) / a_tip[neig_lft[closest_tip_cell[i]]]
                # alpha_nei = np.arccos(
                #     round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
                # alpha[i] = (alpha_closest + alpha_nei) / 2
            elif point_at_grid_line[closest_tip_cell[i]] == 2:
                # slp = (points_rgt[closest_tip_cell[i],1]-mesh.CenterCoor[elt_ribbon[i], 1])/(points_rgt[closest_tip_cell[i],0]-mesh.CenterCoor[elt_ribbon[i], 0])
                # alpha[i] = abs(np.arctan(slp))
                y = mesh.CenterCoor[elt_ribbon[i], 1]
                x = (-y - c[closest_tip_cell[i]]) / a[closest_tip_cell[i]]
                alpha[i] = np.arccos(
                    round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))

        else:
            alpha_mult = np.array([])
            for k in chk_mult:
                y = mesh.CenterCoor[elt_ribbon[i], 1]
                x = (-b[k] * y - c[k]) / a[k]
                alpha_mult = np.append(alpha_mult, np.arccos(
                    round(dist_front_line[k] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5)))

                # if point_at_grid_line[k] == 0:
                #     y = mesh.CenterCoor[elt_ribbon[i], 1]
                #     x = (-b[k] * y - c[k]) / a[k]
                #     alpha_mult = np.append(alpha_mult, np.arccos(
                #         round(dist_front_line[k] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5)))
                # elif point_at_grid_line[k] == 1:
                #     slp = (points_lft[k, 1] - mesh.CenterCoor[elt_ribbon[i], 1]) / (
                #     points_lft[k, 0] - mesh.CenterCoor[elt_ribbon[i], 0])
                #     alpha_mult = np.append(alpha_mult, abs(np.arctan(slp)))
                # elif point_at_grid_line[k] == 2:
                #     slp = (points_rgt[k, 1] - mesh.CenterCoor[elt_ribbon[i], 1]) / (
                #     points_rgt[k, 0] - mesh.CenterCoor[elt_ribbon[i], 0])
                #     alpha_mult = np.append(alpha_mult, abs(np.arctan(slp)))
            Kprime_mult = mat_prop.KprimeFunc(alpha_mult)
            min_indx = np.argmin(Kprime_mult)
            alpha[i] = alpha_mult[min_indx]
            # print("found multiple potential fronts")

        dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
        # plt.show()


    zero_tip_alpha = np.where(abs(alpha_tip)<1e-6)[0]
    for i in range(len(zero_tip_alpha)):
        if zero_vertex[zero_tip_alpha[i]]== 0 or zero_vertex[zero_tip_alpha[i]] == 3:
            left_in_ribbon = np.where(elt_ribbon == mesh.NeiElements[elt_tip[zero_tip_alpha[i]],0])[0]
            alpha[left_in_ribbon] = 0.0
            dist_ribbon[left_in_ribbon] = l_tip[zero_tip_alpha[i]] + mesh.hx/2
        if zero_vertex[zero_tip_alpha[i]]== 1 or zero_vertex[zero_tip_alpha[i]] == 2:
            rgt_in_ribbon = np.where(elt_ribbon == mesh.NeiElements[elt_tip[zero_tip_alpha[i]],1])[0]
            alpha[rgt_in_ribbon] = 0.0
            dist_ribbon[rgt_in_ribbon] = l_tip[zero_tip_alpha[i]] + mesh.hx/2

    ninety_tip_alpha = np.where(abs(alpha_tip-np.pi/2) < 1e-6)[0]
    for i in range(len(ninety_tip_alpha)):
        if zero_vertex[ninety_tip_alpha[i]] == 0 or zero_vertex[ninety_tip_alpha[i]] == 1:
            bottom_in_ribbon = np.where(elt_ribbon == mesh.NeiElements[elt_tip[ninety_tip_alpha[i]], 2])[0]
            alpha[bottom_in_ribbon] = np.pi/2
        if zero_vertex[ninety_tip_alpha[i]] == 3 or zero_vertex[ninety_tip_alpha[i]] == 2:
            up_in_ribbon = np.where(elt_ribbon == mesh.NeiElements[elt_tip[ninety_tip_alpha[i]], 3])[0]
            alpha[up_in_ribbon] = np.pi/2

    # perp_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    # for i in range(len(elt_ribbon)):
    #
    #     if zero_vertex[closest_tip_cell[i]] == 0:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] + dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] + dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zero_vertex[closest_tip_cell[i]] == 1:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] - dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] + dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zero_vertex[closest_tip_cell[i]] == 2:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] - dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] - dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zero_vertex[closest_tip_cell[i]] == 3:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] + dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] - dist_ribbon[i] * np.sin(alpha[i])])
    #
    #
    # for i in range(len(elt_tip)):
    #     plt.plot([points_lft[i, 0], points_rgt[i, 0]], [points_lft[i, 1], points_rgt[i, 1]])
    #
    # for i in range(len(elt_ribbon)):
    #     plt.plot([perp_on_line[i, 0], mesh.CenterCoor[elt_ribbon[i], 0]], [perp_on_line[i, 1], mesh.CenterCoor[elt_ribbon[i], 1]])
    #
    #
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()
    if np.isnan(alpha).any():
        print("found nan")
    return alpha

def make_front_continous(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip):
    tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_c, smthed_tip_points_left, smthed_tip_points_rgt, tip_lft_neghb, tip_rgt_neghb = construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip)
    # slope = np.empty((len(elt_tip),), dtype=np.float64)
    # pnt_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    # for i in range(len(elt_tip)):
    #
    #     if zero_vertex_tip[i] == 0:
    #         slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
    #         zr_vrtx_global = mesh.Connectivity[elt_tip[i], 0]
    #         pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
    #                                    mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
    #     elif zero_vertex_tip[i] == 1:
    #         slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
    #         zr_vrtx_global = mesh.Connectivity[elt_tip[i], 1]
    #         pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
    #                                    mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
    #     elif zero_vertex_tip[i] == 2:
    #         slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
    #         zr_vrtx_global = mesh.Connectivity[elt_tip[i], 2]
    #         pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
    #                                    mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])
    #     elif zero_vertex_tip[i] == 3:
    #         slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
    #         zr_vrtx_global = mesh.Connectivity[elt_tip[i], 3]
    #         pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
    #                                    mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])
    #
    # for i in range(mesh.nx+1):
    #     plt.plot([mesh.VertexCoor[i, 0], mesh.VertexCoor[i, 0]],
    #              [mesh.VertexCoor[0, 1],mesh.VertexCoor[-1, 1]],'k')
    # for i in range(mesh.ny+1):
    #     plt.plot([mesh.VertexCoor[0, 0], mesh.VertexCoor[-1, 0]],
    #              [mesh.VertexCoor[i*(mesh.nx+1), 1],mesh.VertexCoor[i*(mesh.nx+1), 1]],'k')
    # for i in range(len(elt_tip)):
    #     plt.plot([pnt_on_line[i, 0], mesh.VertexCoor[mesh.Connectivity[elt_tip[i], zero_vertex_tip[i]],0] ], [pnt_on_line[i, 1], mesh.VertexCoor[mesh.Connectivity[elt_tip[i], zero_vertex_tip[i]],1]],'r')
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()
    #
    # b = np.ones((len(elt_tip),), dtype=np.float64)
    # a = - slope
    # c = -(pnt_on_line[:, 1] - slope * pnt_on_line[:, 0])
    #
    # points_lft = np.empty((len(elt_tip), 2), dtype=np.float64)
    # points_rgt = np.empty((len(elt_tip), 2), dtype=np.float64)
    # neighbor_lft = np.empty((len(elt_tip),), dtype=np.int)
    # neighbor_rgt = np.empty((len(elt_tip),), dtype=np.int)
    # left_not_found = np.array([], dtype=np.int)
    # rgt_not_found = np.array([], dtype=np.int)
    # for i in range(len(elt_tip)):
    #     x = mesh.CenterCoor[elt_tip[i], 0] - mesh.hx / 2
    #     y = -c[i] - a[i] * x
    #     neib = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 0])[0]
    #     if neib.size>0:
    #         neighbor_lft[i] = neib
    #     if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neib = np.where(elt_tip == mesh.NeiElements[elt_tip[i],3])[0]
    #         if neib.size<1:
    #             left_not_found = np.append(left_not_found, i)
    #         else:
    #             neighbor_lft[i] = neib
    #     elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neib = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 2])[0]
    #         if neib.size<1:
    #             left_not_found = np.append(left_not_found, i)
    #         else:
    #             neighbor_lft[i] = neib
    #     points_lft[i, 0] = x
    #     points_lft[i, 1] = y
    #
    #     x = mesh.CenterCoor[elt_tip[i], 0] + mesh.hx / 2
    #     y = -c[i] - a[i] * x
    #     rgt_in_tip = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 1])[0]
    #     if rgt_in_tip.size > 0:
    #         neighbor_rgt[i] = rgt_in_tip
    #     if y > mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] + mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neib = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 3])[0]
    #         if neib.size<1:
    #             rgt_not_found = np.append(rgt_not_found, i)
    #         else:
    #             neighbor_rgt[i] = neib
    #     elif y < mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2:
    #         y = mesh.CenterCoor[elt_tip[i], 1] - mesh.hy / 2
    #         x = (-c[i] - y) / a[i]
    #         neib = np.where(elt_tip == mesh.NeiElements[elt_tip[i], 2])[0]
    #         if neib.size<1:
    #             rgt_not_found = np.append(rgt_not_found, i)
    #         else:
    #             neighbor_rgt[i] = neib
    #     points_rgt[i, 0] = x
    #     points_rgt[i, 1] = y
    #
    # # for i in range(mesh.nx+1):
    # #     plt.plot([mesh.VertexCoor[i, 0], mesh.VertexCoor[i, 0]],
    # #              [mesh.VertexCoor[0, 1],mesh.VertexCoor[-1, 1]],'k')
    # # for i in range(mesh.ny+1):
    # #     plt.plot([mesh.VertexCoor[0, 0], mesh.VertexCoor[-1, 0]],
    # #              [mesh.VertexCoor[i*(mesh.nx+1), 1],mesh.VertexCoor[i*(mesh.nx+1), 1]],'k')
    #
    # # plt.plot([lft_x, rgt_x], [lft_y, rgt_y])
    # plt.plot(mesh.CenterCoor[elt_tip[left_not_found],0],mesh.CenterCoor[elt_tip[left_not_found],1],'o')
    # for i in range(len(elt_tip)):
    #     plt.plot([points_lft[i,0], points_rgt[i,0]], [points_lft[i,1], points_rgt[i,1]])
    #     # plt.plot([cont_pnts_lft_x[i],cont_pnts_rgt_x[i]],[cont_pnts_lft_y[i],cont_pnts_rgt_y[i]])
    #     # plt.plot([lft_x[i], rgt_x[i]], [lft_y[i], rgt_y[i]])
    #
    #     plt.axis("equal")
    #
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()
    # # plt.close('all')
    #
    # cont_pnts_lft_x = (points_lft[:, 0] + points_rgt[neighbor_lft, 0]) / 2
    # cont_pnts_lft_y = (points_lft[:, 1] + points_rgt[neighbor_lft, 1]) / 2
    # cont_pnts_rgt_x = (points_rgt[:, 0] + points_lft[neighbor_rgt, 0]) / 2
    # cont_pnts_rgt_y = (points_rgt[:, 1] + points_lft[neighbor_rgt, 1]) / 2
    #
    # slope_cont = (cont_pnts_rgt_y-cont_pnts_lft_y)/(cont_pnts_rgt_x-cont_pnts_lft_x)
    # # alpha_cont = np.arctan(1./abs(slope_cont))
    # a_cont = - slope_cont
    # coor_zr_vrtx = mesh.VertexCoor[mesh.Connectivity[elt_tip, zero_vertex_tip]]
    # c_cont = slope_cont * cont_pnts_lft_x - cont_pnts_lft_y
    # # l_cont =  abs((coor_zr_vrtx[:,0]* a_cont + coor_zr_vrtx[:,1] + c_cont) / (a_cont**2 + 1.)**0.5)
    #
    # lft_x = -30.*np.ones((len(elt_tip),))
    # rgt_x = 30.*np.ones((len(elt_tip),))
    # lft_y = -a_cont * lft_x - c_cont
    # rgt_y = -a_cont * rgt_x - c_cont


    # tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_c, smthed_tip_points_left, smthed_tip_points_rgt, tip_lft_neghb, tip_rgt_neghb
    return tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_c, smthed_tip_points_left[:,0],smthed_tip_points_left[:,1], smthed_tip_points_rgt[:,0], smthed_tip_points_rgt[:,1], tip_lft_neghb, tip_rgt_neghb

def find_projection(elt_ribbon, elt_tip, zr_vrtx_tip, a_tip, b_tip, c_tip, x_lft, y_lft, x_rgt, y_rgt, neig_lft, neig_rgt, mesh):

    closest_tip_cell = np.zeros((len(elt_ribbon),), dtype=np.int)
    dist_ribbon = np.zeros((len(elt_ribbon),), dtype=np.float64)
    alpha = np.zeros((len(elt_ribbon),), dtype=np.float64)
    for i in range(len(elt_ribbon)):
        dist_front_line = np.zeros((len(elt_tip),), dtype=np.float64)
        point_at_grid_line = np.zeros((len(elt_tip),), dtype=np.uint8)
        for j in range(len(elt_tip)):

            # if abs(alpha_tip[j]) < 1e-8:
            #     xx = mesh.CenterCoor[elt_ribbon[i], 0]
            #     yy = - c[j]
            # elif abs(alpha_tip[j]-np.pi/2)< 1e-8:
            #     yy = mesh.CenterCoor[elt_ribbon[i], 1]
            #     xx = points_lft[j, 0]
            # else:
            if x_rgt[j] - x_lft[j] == 0:
                xx = mesh.CenterCoor[elt_ribbon[i], 0]
                yy = - c_tip[j]
            else:
                slope_tip_line = (y_rgt[j] - y_lft[j]) / (x_rgt[j] - x_lft[j])
                m = -1. / slope_tip_line

                intrcpt = mesh.CenterCoor[elt_ribbon[i], 1] - m * mesh.CenterCoor[elt_ribbon[i], 0]
                xx = -(intrcpt + c_tip[j]) / (a_tip[j] + m)
                yy = m * xx + intrcpt


            if x_lft[j] > xx or x_rgt[j] < xx or min(y_lft[j], y_rgt[j]) > yy or max(
                    y_lft[j], y_rgt[j]) < yy:
                dist_lft_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_lft[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_lft[j]) ** 2) ** 0.5
                dist_rgt_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_rgt[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_rgt[j]) ** 2) ** 0.5

                dist_front_line[j] = min(dist_lft_pnt, dist_rgt_pnt)
                if dist_lft_pnt < dist_rgt_pnt:
                    point_at_grid_line[j] = 1
                else:
                    point_at_grid_line[j] = 2
            else:
                dist_front_line[j] = abs(
                    mesh.CenterCoor[elt_ribbon[i], 0] * a_tip[j] + mesh.CenterCoor[elt_ribbon[i], 1] + c_tip[j]) / (a_tip[
                                                                                                                       j] ** 2 +
                                                                                                                   1) ** 0.5
        # # if not (np.logical_not(np.isinf(dist_front_line))).any():
        # #     dist_lft_gridIntr = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_lft[:,0])**2
        # #                          + (mesh.CenterCoor[elt_ribbon[i], 1] - points_lft[:,1])**2) ** 0.5
        # #     dist_rgt_gridIntr = ((mesh.CenterCoor[elt_ribbon[i], 0] - points_rgt[:, 0]) ** 2
        # #                          + (mesh.CenterCoor[elt_ribbon[i], 1] - points_rgt[:, 1]) ** 2) ** 0.5
        # #     dist_front_line = np.minimum(dist_lft_gridIntr, dist_rgt_gridIntr)
        # #     print("found")
        #
        closest_tip_cell[i] = np.argmin(dist_front_line)
        # chk_mult = np.where(abs(1. - dist_front_line[closest_tip_cell[i]] / dist_front_line) < 1e-10)[0]
        # if chk_mult.size <= 1:
        if point_at_grid_line[closest_tip_cell[i]] == 0:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha[i] = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
        elif point_at_grid_line[closest_tip_cell[i]] == 1:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_lft[closest_tip_cell[i]]]) / a_tip[neig_lft[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2
        elif point_at_grid_line[closest_tip_cell[i]] == 2:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_rgt[closest_tip_cell[i]]]) / a_tip[neig_rgt[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2

        dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]

    # zero_angle = np.where(abs(1. - x_lft / x_rgt) < 1e-6)[0]
    zero_angle = np.where(x_lft == x_rgt)[0]
    for i in range(len(zero_angle)):
        if zr_vrtx_tip[zero_angle[i]] == 0 or zr_vrtx_tip[zero_angle[i]] == 3:
            for j in range(3):
                left_in_ribbon = np.where(elt_ribbon == elt_tip[zero_angle[i]]-(j+1))[0]
                if left_in_ribbon.size > 0:
                    break
            alpha[left_in_ribbon] = 0.0
            dist_ribbon[left_in_ribbon] = abs(abs(x_rgt[zero_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[left_in_ribbon],0]))
        if zr_vrtx_tip[zero_angle[i]]== 1 or zr_vrtx_tip[zero_angle[i]] == 2:
            for j in range(3):
                rgt_in_ribbon = np.where(elt_ribbon == elt_tip[zero_angle[i]]+(j+1))[0]
                if rgt_in_ribbon.size > 0:
                    break
            alpha[rgt_in_ribbon] = 0.0
            dist_ribbon[rgt_in_ribbon] = abs(abs(x_rgt[zero_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[rgt_in_ribbon],0]))

    ninety_angle = np.where(y_lft == y_rgt)[0]
    for i in range(len(ninety_angle)):
        if zr_vrtx_tip[ninety_angle[i]] == 0 or zr_vrtx_tip[ninety_angle[i]] == 1:
            for j in range(3):
                btm_in_ribbon = np.where(elt_ribbon == elt_tip[ninety_angle[i]] - (j+1)*mesh.nx)[0]
                if btm_in_ribbon.size > 0:
                    break
            alpha[btm_in_ribbon] = np.pi/2
            dist_ribbon[btm_in_ribbon] = abs(
                abs(y_rgt[ninety_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[btm_in_ribbon], 1]))
        if zr_vrtx_tip[ninety_angle[i]] == 2 or zr_vrtx_tip[ninety_angle[i]] == 3:
            for j in range(3):
                top_in_ribbon = np.where(elt_ribbon == elt_tip[ninety_angle[i]] + (j+1)*mesh.nx)[0]
                if top_in_ribbon.size > 0:
                    break
            alpha[top_in_ribbon] = np.pi/2
            dist_ribbon[top_in_ribbon] = abs(
                abs(y_rgt[ninety_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[top_in_ribbon], 1]))

    # perp_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    # for i in range(len(elt_ribbon)):
    #
    #     if zr_vrtx_tip[closest_tip_cell[i]] == 0:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] + dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] + dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zr_vrtx_tip[closest_tip_cell[i]] == 1:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] - dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] + dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zr_vrtx_tip[closest_tip_cell[i]] == 2:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] - dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] - dist_ribbon[i] * np.sin(alpha[i])])
    #     elif zr_vrtx_tip[closest_tip_cell[i]] == 3:
    #         perp_on_line[i] = np.array([mesh.CenterCoor[elt_ribbon[i], 0] + dist_ribbon[i] * np.cos(alpha[i]),
    #                                    mesh.CenterCoor[elt_ribbon[i], 1] - dist_ribbon[i] * np.sin(alpha[i])])
    #
    #
    #
    # plt.plot([x_lft, x_rgt], [y_lft, y_rgt])
    #
    # for i in range(len(elt_ribbon)):
    #     plt.plot([perp_on_line[i, 0], mesh.CenterCoor[elt_ribbon[i], 0]], [perp_on_line[i, 1], mesh.CenterCoor[elt_ribbon[i], 1]],'r')
    #
    #
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()
    if np.isnan(alpha).any():
        print("found nan")
    return alpha


def construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip):
    slope = np.empty((len(elt_tip),), dtype=np.float64)
    pnt_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    for i in range(len(elt_tip)):

        if zero_vertex_tip[i] == 0:
            slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 0]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 1:
            slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 1]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 2:
            slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 2]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 3:
            slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 3]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])

    zero_angle = np.where(alpha_tip == 0)[0]
    for i in zero_angle:
        if zero_vertex_tip[i] == 0 or zero_vertex_tip[i] == 1:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0])**2 + (pnt_on_line[:, 1] - pnt_on_line[i, 1] - mesh.hy)**2)**0.5
            closest = np.argmin(dist_from_added)
            if dist_from_added[closest] < (mesh.hx**2 + mesh.hy**2)**0.5 /10:
                np.delete(pnt_on_line, closest)
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0], pnt_on_line[i, 1] + mesh.hy])))

        if zero_vertex_tip[i] == 2 or zero_vertex_tip[i] == 3:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0]) ** 2 + (
                                pnt_on_line[:, 1] - pnt_on_line[i, 1] + mesh.hy) ** 2) ** 0.5
            closest = np.argmin(dist_from_added)
            if dist_from_added[closest] < (mesh.hx**2 + mesh.hy**2)**0.5 /10:
                np.delete(pnt_on_line, closest)
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0], pnt_on_line[i, 1] - mesh.hy])))
    ninety_angle = np.where(alpha_tip == np.pi / 2)[0]
    for i in ninety_angle:
        if zero_vertex_tip[i] == 0 or zero_vertex_tip[i] == 3:
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0] + mesh.hx, pnt_on_line[i, 1]])))
        if zero_vertex_tip[i] == 1 or zero_vertex_tip[i] == 2:
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0] - mesh.hx, pnt_on_line[i, 1]])))

    # rnd = 2
    # pnt_on_line = truncate(pnt_on_line, rnd)
    # pnt_on_line = np.vstack({tuple(row) for row in pnt_on_line})

    # for i in range(mesh.nx+1):
    #     plt.plot([mesh.VertexCoor[i, 0], mesh.VertexCoor[i, 0]],
    #              [mesh.VertexCoor[0, 1],mesh.VertexCoor[-1, 1]],'k')
    # for i in range(mesh.ny+1):
    #     plt.plot([mesh.VertexCoor[0, 0], mesh.VertexCoor[-1, 0]],
    #              [mesh.VertexCoor[i*(mesh.nx+1), 1],mesh.VertexCoor[i*(mesh.nx+1), 1]],'k')
    # for i in range(len(pnt_on_line)):
    #     plt.plot(pnt_on_line[i, 0], pnt_on_line[i, 1],'x')
    #
    # plt.axis("equal")
    # axes = plt.gca()
    # # axes.set_xlim([-30., 30.])
    # # axes.set_ylim([-30., 30.])
    # plt.show()



    grid_lines_x = np.unique(mesh.VertexCoor[:, 0])
    grid_lines_y = np.unique(mesh.VertexCoor[:, 1])
    polygon = np.empty((0,2), dtype=np.float64)

    # left_most = np.argmin(pnt_on_line[:,0])
    # rgt_most = np.argmax(pnt_on_line[:,0])
    # slope_lft_rgt = (pnt_on_line[rgt_most,1]-pnt_on_line[left_most,1])/(pnt_on_line[rgt_most,0]-pnt_on_line[left_most,0])
    # a_lft_rgt = -slope_lft_rgt
    # c_lft_rgt = -(pnt_on_line[left_most, 1] - slope_lft_rgt * pnt_on_line[left_most, 0])

    # inward_pnts = np.copy(pnt_on_line)
    # inward_pnts = np.delete(inward_pnts, np.array([left_most,rgt_most]),0)
    #
    #
    # dist_lft_rgt = inward_pnts[:,0] * a_lft_rgt + inward_pnts[:,1] + c_lft_rgt / (a_lft_rgt**2 + 1) ** 0.5
    # above_lftRgt_ln = np.where(dist_lft_rgt >= 0)[0]
    # below_lftRgt_ln = np.where(dist_lft_rgt < 0)[0]
    #
    # sorted_above = np.argsort(inward_pnts[above_lftRgt_ln,0])
    # sorted_below = np.argsort(inward_pnts[below_lftRgt_ln,0])
    # rvrse_sorted_above = sorted_above[::-1]
    #
    # pnt_in_order = pnt_on_line[left_most]
    # pnt_in_order = np.vstack((pnt_in_order,inward_pnts[below_lftRgt_ln[sorted_below]]))
    # pnt_in_order = np.vstack((pnt_in_order, pnt_on_line[rgt_most]))
    # pnt_in_order = np.vstack((pnt_in_order, inward_pnts[above_lftRgt_ln[rvrse_sorted_above]]))

#### closest point algorithm

    for i in range(pnt_on_line.size):
        remaining = np.copy(pnt_on_line)
        nxt = pnt_on_line[i]
        remaining = np.delete(remaining, i, 0)
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        direction = np.asarray([nxt[0]-remaining[nxt_indx,0], nxt[1]-remaining[nxt_indx,1]])
        nxt = remaining[nxt_indx]
        remaining = np.delete(remaining, nxt_indx, 0)
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        direction_sec = np.asarray([nxt[0] - remaining[nxt_indx, 0], nxt[1] - remaining[nxt_indx, 1]])
        if (np.sign(direction) == np.sign(direction_sec))[0] and (np.sign(direction) == np.sign(direction_sec))[1]:
            first = np.copy(pnt_on_line[0])
            pnt_on_line[0] = np.copy(pnt_on_line[i])
            pnt_on_line[i] = np.copy(first)
            break


    remaining = np.copy(pnt_on_line)
    pnt_in_order = np.array([remaining[0]])
    nxt = pnt_on_line[0]
    remaining = np.delete(remaining, 0,0)
    while remaining.size>0:
        dist_from_remnng = ((remaining[:,0]-nxt[0])**2 + (remaining[:,1]-nxt[1])**2)**0.5
        nxt_indx = np.argmin(dist_from_remnng)
        nxt = remaining[nxt_indx]
        remaining = np.delete(remaining, nxt_indx, 0)
        pnt_in_order = np.vstack((pnt_in_order,nxt))

    # for i in range(len(pnt_in_order)):
    #     plt.plot(pnt_in_order[i, 0], pnt_in_order[i, 1], 'o')
        # plt.show()


    i = 0
    while i <= pnt_in_order.shape[0]-1:
        i_next = (i+1)%pnt_in_order.shape[0]
        if pnt_in_order[i, 0] <= pnt_in_order[i_next, 0]:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] >= grid_lines_x, pnt_in_order[i, 0] < grid_lines_x))[0]
        else:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] <= grid_lines_x, pnt_in_order[i, 0] > grid_lines_x))[
                0]

        if grd_lns_btw_pnts_x.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
                pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_x:
                x_p = grid_lines_x[j]
                y_p = slope * (x_p - pnt_in_order[i_next, 0]) + pnt_in_order[i_next, 1]
                polygon = np.vstack((polygon, np.array([x_p, y_p])))

        if pnt_in_order[i, 1] <= pnt_in_order[i_next, 1]:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] >= grid_lines_y, pnt_in_order[i, 1] < grid_lines_y))[
                0]
        else:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] <= grid_lines_y, pnt_in_order[i, 1] > grid_lines_y))[
                0]

        if grd_lns_btw_pnts_y.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
            pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_y:
                y_p = grid_lines_y[j]

                x_p = (y_p - pnt_in_order[i_next, 1]) / slope + pnt_in_order[i_next, 0]
                polygon = np.vstack((polygon, np.array([x_p, y_p])))
        i+=1
    # polygon = np.round(polygon,6)
    polygon = np.vstack({tuple(row) for row in polygon})
    # for i in range(len(polygon)):
    #     plt.plot(polygon[i, 0], polygon[i, 1],'x')
        # plt.show()
    #     plt.plot(pnt_in_order2[i, 0], pnt_in_order2[i, 1], 'x')
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-30., 30.])
    # plt.show()

    # for i in range(pnt_on_line.shape[0]):
    #     remaining_points = np.copy(pnt_on_line)
    #     remaining_points = np.delete(remaining_points, i, 0)
    #     for k in range(2):
    #         dist_from_points = ((pnt_on_line[i,0]-remaining_points[:,0])**2 + (pnt_on_line[i,1]-remaining_points[:,1])**2)**0.5
    #         closest_point = np.argmin(dist_from_points)
    #         # if k>0:
    #         #     dist_from_first = ((remaining_points[closest_point,0]- closest_point_lst[0])**2-(remaining_points[closest_point,1]-closest_point_lst[1])**2)**0.5
    #         #     if dist_from_first < dist_from_points[closest_point]:
    #         #         tmp = np.delete(dist_from_points, closest_point)
    #         #         nxt_closest = np.argmin(tmp)
    #         #         closest_point = np.where(dist_from_points == tmp[nxt_closest])
    #
    #         if pnt_on_line[i,0] <= remaining_points[closest_point,0]:
    #             grd_lns_btw_pnts_x = np.where(np.logical_and(remaining_points[closest_point,0]>=grid_lines_x, pnt_on_line[i,0]<=grid_lines_x))[0]
    #         else:
    #             grd_lns_btw_pnts_x = np.where(np.logical_and(remaining_points[closest_point, 0] <= grid_lines_x, pnt_on_line[i, 0] >= grid_lines_x))[0]
    #
    #         if grd_lns_btw_pnts_x.size > 0:
    #             slope = (remaining_points[closest_point, 1]-pnt_on_line[i, 1])/(remaining_points[closest_point, 0]-pnt_on_line[i, 0])
    #             for j in grd_lns_btw_pnts_x:
    #                 x_p = grid_lines_x[j]
    #                 y_p = slope * (x_p - remaining_points[closest_point, 0]) + remaining_points[closest_point, 1]
    #                 polygon = np.vstack((polygon, np.array([x_p, y_p])))
    #
    #         if pnt_on_line[i,1] <= remaining_points[closest_point,1]:
    #             grd_lns_btw_pnts_y = np.where(np.logical_and(remaining_points[closest_point,1]>=grid_lines_y, pnt_on_line[i,1]<=grid_lines_y))[0]
    #         else:
    #             grd_lns_btw_pnts_y = np.where(np.logical_and(remaining_points[closest_point, 1] <= grid_lines_y, pnt_on_line[i, 1] >= grid_lines_y))[0]
    #
    #         if grd_lns_btw_pnts_y.size > 0:
    #             slope = (remaining_points[closest_point, 1]-pnt_on_line[i, 1])/(remaining_points[closest_point, 0]-pnt_on_line[i, 0])
    #             for j in grd_lns_btw_pnts_y:
    #                 y_p = grid_lines_y[j]
    #
    #                 x_p = (y_p - remaining_points[closest_point, 1]) / slope + remaining_points[closest_point, 0]
    #                 polygon = np.vstack((polygon, np.array([x_p, y_p])))
    #
    #         closest_point_lst = remaining_points[closest_point]
    #         dist_lst = dist_from_points[closest_point]
    #         remaining_points = np.delete(remaining_points, closest_point, 0)
    #
    # # rnd = 6
    # # polygon = truncate(polygon, rnd)
    # polygon = np.vstack({tuple(row) for row in polygon})
    tip_smoothed = np.array([],dtype=np.int)
    smthed_tip_points_left = np.empty((0, 2), dtype=np.float64)
    smthed_tip_points_rgt = np.empty((0, 2), dtype=np.float64)
    for i in range(mesh.NumberOfElts):
        in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0] #- mesh.hx*1e-6
        in_cell = np.logical_and(in_cell, polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0] )#+ 1e-6*mesh.hx)
        in_cell = np.logical_and(in_cell, polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1] )#- 1e-6*mesh.hy)
        in_cell = np.logical_and(in_cell, polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1] )#+ 1e-6*mesh.hy)
        # on_top_line = abs(1. - polygon[:, 1] / mesh.VertexCoor[mesh.Connectivity[i, 2], 1]) < 1e-3
        # on_left_line = abs(1. - polygon[:, 0] / mesh.VertexCoor[mesh.Connectivity[i, 0], 0]) < 1e-3
        # on_rgt_line = abs(1. - polygon[:, 0] / mesh.VertexCoor[mesh.Connectivity[i, 1], 0]) < 1e-3
        #
        # on_x_lines = np.intersect1d(np.where(polygon[:, 0] > mesh.VertexCoor[mesh.Connectivity[i, 0], 0] - 1e-14 )< 1e-3)[0],
        #                             np.where(abs(1. - mesh.VertexCoor[mesh.Connectivity[i, 1], 0]/polygon[:, 0])< 1e-3)[0])
        # on_y_lines = np.intersect1d(np.where(abs(1. - mesh.VertexCoor[mesh.Connectivity[i, 0], 1]/polygon[:, 1])< 1e-3)[0],
        #                             np.where(abs(1. - mesh.VertexCoor[mesh.Connectivity[i, 2], 1]/polygon[:, 1])< 1e-3)[0])
        cell_pnt = np.where(in_cell)[0]
        if cell_pnt.size > 2:
            # return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            dist = (polygon[cell_pnt[0],0]-polygon[cell_pnt,0])**2 + (polygon[cell_pnt[0],1]-polygon[cell_pnt,1])**2
            farthest = np.argmax(dist)
            to_delete = np.array([],dtype=np.int)
            for m in range(1,cell_pnt.size):
                if m != farthest:
                    to_delete = np.append(to_delete, cell_pnt[m])
            polygon = np.delete(polygon, to_delete, 0)

            in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0]  # - mesh.hx*1e-6
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0])  # + 1e-6*mesh.hx)
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1])  # - 1e-6*mesh.hy)
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1])  # + 1e-6*mesh.hy)
            cell_pnt = np.where(in_cell)[0]
            # print("found")
        # if i in elt_tip:
        #     plt.plot(mesh.CenterCoor[i,0],mesh.CenterCoor[i,1],'*')
        #     plt.axis("equal")
        #     axes = plt.gca()
        #     axes.set_xlim([-30., 30.])
        #     axes.set_ylim([-30., 30.])
            # plt.show()

        if cell_pnt.size>1:
            tip_smoothed = np.append(tip_smoothed, i)
            if polygon[cell_pnt[0],0] <= polygon[cell_pnt[1],0] :
                smthed_tip_points_left = np.vstack((smthed_tip_points_left,polygon[cell_pnt[0]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[1]]))
            else:
                smthed_tip_points_left = np.vstack((smthed_tip_points_left, polygon[cell_pnt[1]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[0]]))
            # plt.plot([smthed_tip_points_left[-1, 0], smthed_tip_points_rgt[-1, 0]],
            #          [smthed_tip_points_left[-1, 1], smthed_tip_points_rgt[-1, 1]])


    # zero_alpha = np.where(abs(alpha_tip) < 1e-6)[0]
    # for i in zero_alpha:
    #     zero_smthed_tip = np.where(tip_smoothed == elt_tip[i])[0]
    #     # smthed_tip_points_left
    #     smthed_tip_points_left[zero_smthed_tip,0] = (smthed_tip_points_left[zero_smthed_tip,0]+smthed_tip_points_rgt[zero_smthed_tip,0])/2
    #     smthed_tip_points_rgt[zero_smthed_tip, 0] = smthed_tip_points_left[zero_smthed_tip,0]
    #
    # ninety_alpha = np.where(abs(alpha_tip-np.pi/2) < 1e-6)[0]
    # for i in ninety_alpha:
    #     ninety_smthed_tip = np.where(tip_smoothed == elt_tip[i])[0]
    #     smthed_tip_points_left[ninety_smthed_tip, 1] = (
    #                                                  smthed_tip_points_left[ninety_smthed_tip, 1] + smthed_tip_points_rgt[
    #                                                      ninety_smthed_tip, 1]) / 2
    #     smthed_tip_points_rgt[ninety_smthed_tip, 1] = smthed_tip_points_left[ninety_smthed_tip, 1]

    smthed_tip_lines_slope = (smthed_tip_points_rgt[:, 1] - smthed_tip_points_left[:, 1]) / (
                                smthed_tip_points_rgt[:, 0] - smthed_tip_points_left[:, 0])
    smthed_tip_lines_a = -smthed_tip_lines_slope
    smthed_tip_lines_b = np.ones((len(tip_smoothed),),dtype=np.float64)
    smthed_tip_lines_c = -(smthed_tip_points_rgt[:, 1] - smthed_tip_lines_slope * smthed_tip_points_rgt[:, 0])

    zero_angle = np.where(smthed_tip_points_left[:,0]==smthed_tip_points_rgt[:,0])[0]
    smthed_tip_lines_b[zero_angle] = 0.
    smthed_tip_lines_a[zero_angle] = 1.
    smthed_tip_lines_c[zero_angle] = -smthed_tip_points_rgt[zero_angle,0]

    tip_lft_neghb = np.zeros((len(tip_smoothed),), dtype=np.int)
    tip_rgt_neghb = np.empty((len(tip_smoothed),), dtype=np.int)
    for i in range(len(tip_smoothed)):
        equal = smthed_tip_points_rgt == smthed_tip_points_left[i]
        # plt.plot(mesh.CenterCoor[tip_smoothed[i], 0],mesh.CenterCoor[tip_smoothed[i], 1], 'o')
        # plt.plot([smthed_tip_points_left[i, 0], smthed_tip_points_rgt[i, 0]],
        #          [smthed_tip_points_left[i, 1], smthed_tip_points_rgt[i, 1]], 'or-')
        # plt.axis("equal")
        # axes = plt.gca()
        # axes.set_xlim([-30., 30.])
        # axes.set_ylim([-30., 30.])
        # plt.show()
        left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
        if left_nei.size != 1:
            equal = smthed_tip_points_left == smthed_tip_points_left[i]
            left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
            if left_nei.size == 2:
                tip_lft_neghb[i] = left_nei[np.where(left_nei != i)[0]]
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                # tip_lft_neghb[i] = i
        else:
            tip_lft_neghb[i] = left_nei
        # plt.plot([mesh.CenterCoor[tip_smoothed[i],0],mesh.CenterCoor[tip_smoothed[tip_lft_neghb[i]],0]], [mesh.CenterCoor[tip_smoothed[i],1],mesh.CenterCoor[tip_smoothed[tip_lft_neghb[i]],1]], 'ob-')

        equal = smthed_tip_points_left == smthed_tip_points_rgt[i]
        # plt.plot(mesh.CenterCoor[tip_smoothed[i], 0],mesh.CenterCoor[tip_smoothed[i], 1], 'o')
        # plt.plot([smthed_tip_points_left[i, 0], smthed_tip_points_rgt[i, 0]],
        #          [smthed_tip_points_left[i, 1], smthed_tip_points_rgt[i, 1]], 'or-')
        # plt.axis("equal")
        # axes = plt.gca()
        # axes.set_xlim([-30., 30.])
        # axes.set_ylim([-30., 30.])
        # plt.show()
        rgt_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
        if rgt_nei.size != 1:
            equal = smthed_tip_points_rgt == smthed_tip_points_rgt[i]
            rgt_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
            if rgt_nei.size == 2:
                tip_rgt_neghb[i] = rgt_nei[np.where(rgt_nei != i)[0]]
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                # tip_rgt_neghb[i] = i
        else:
            tip_rgt_neghb[i] = rgt_nei

    # plt.plot(polygon[:, 0], polygon[:, 1], '*')
    # plt.plot(smthed_tip_points_left[:, 0], smthed_tip_points_left[:, 1], 'x')
    # plt.plot(smthed_tip_points_rgt[:, 0], smthed_tip_points_rgt[:, 1], 'o')
    # for i in range(len(tip_smoothed)):
    #     plt.plot([smthed_tip_points_left[i,0],smthed_tip_points_rgt[i,0]], [smthed_tip_points_left[i,1],smthed_tip_points_rgt[i,1]],'o-')


    # lft_x = -30.*np.ones((len(tip_smoothed),),)
    # rgt_x = 30.*np.ones((len(tip_smoothed),),)
    # lft_y = -smthed_tip_lines_a * lft_x - smthed_tip_lines_c
    # rgt_y = -smthed_tip_lines_a * rgt_x - smthed_tip_lines_c
    # plt.plot([smthed_tip_points_left[:,0],smthed_tip_points_rgt[:,0]],[smthed_tip_points_left[:,1],smthed_tip_points_rgt[:,1]])
    # plt.plot([lft_x,rgt_x], [lft_y, rgt_y])
    # plt.axis("equal")
    # axes = plt.gca()
    # axes.set_xlim([-30., 30.])
    # axes.set_ylim([-40., 40.])
    # plt.show()

    return tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_b, smthed_tip_lines_c, smthed_tip_points_left, smthed_tip_points_rgt, tip_lft_neghb, tip_rgt_neghb


def truncate(x, d):
    return (x*(10.0**d)).astype(int)/(10.0**d)