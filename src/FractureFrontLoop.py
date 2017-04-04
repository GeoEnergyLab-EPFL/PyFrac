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

import copy


def FractureFrontLoop(Frac, C,Material_properties,Fluid_properties,Simulation_Parameters,Injection_Parameters,TimeStep):
    """ Propagate fracture one time step
        Arguments:
            TimeStep (float):         time step
            C (ndarray-float):  Elasticity matrix
            tol_frntPos(float): tolerance for the front position iteration. The front position is assumed to be converged
                                if the norm of current iteration is below this tolerance.
            tol_Picard(float):  tolerance for Picard iteration.
            maxitr (int)        maximum iterations to find front position (default 25).
            turb (bool)         flag specifying if turbulence is taken into account.

        return:
            exitstatus (int):   possible values:
                                    0       -- not propagated
                                    1       -- iteration successfull
                                    2       -- evaluated level set is not valid
                                    3       -- front is not tracked correctly
                                    4       -- evaluated tip volume is not valied
                                    5       -- solution of elastohydrodynamic solver is not valid
                                    6       -- did not converge after max iterations
                                    7       -- tip inversion not successful
                                    8       -- Ribbon element not found in the enclosure of a tip cell
                                    9       -- Filling fraction not correct

    """

    CurrentRate = Injection_Parameters.injectionrate
    Qin = (np.zeros((Frac.mesh.NumberOfElts),float) )
    Qin[Injection_Parameters.source_location]=CurrentRate

    exitstatus = 0  # exit code returned

    f = open('log', 'a')

    C_EltTip = C[np.ix_(Frac.EltTip, Frac.EltTip)] # keep the tip element entries

    # filling fraction correction for element in the very tip region
    for e in range(0, len(Frac.EltTip)):
        r = Frac.FillF[e] - .25;
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r;
        C[Frac.EltTip[e], Frac.EltTip[e]] = C[Frac.EltTip[e], Frac.EltTip[e]] * (1. + ac * np.pi / 4.)


    # average injected fluid over footprint
    guess = TimeStep * CurrentRate / Frac.EltCrack.size * np.ones((Frac.EltCrack.size,),float)

    print('Solving the EHL system with the same fracture footprint')

    DLkOff = np.zeros((Frac.mesh.NumberOfElts,), float)

    wguess = np.copy(Frac.w)
    wguess[Frac.EltCrack] = wguess[Frac.EltCrack] + guess
    vk = velocity(wguess, Frac.EltCrack, Frac.mesh, Frac.InCrack, Frac.muPrime, C, Frac.SigmaO)

    argSameFP = (
        Frac.w, Frac.EltCrack, Qin, C, TimeStep, Frac.muPrime, Frac.mesh, Frac.InCrack, DLkOff, Frac.SigmaO, Frac.rho,
        Fluid_properties.turbulence)

    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_sameFP, MakeEquationSystemSameFP, guess, guess, vk,
                               1.0, Simulation_Parameters.ToleranceEHL, 100, *argSameFP)  # why 2 guess in input ?

    C[np.ix_(Frac.EltTip, Frac.EltTip)] = C_EltTip  # regain origional C (without fill fraction correction)

    w_k = np.copy(Frac.w)
    w_k[Frac.EltCrack] = w_k[Frac.EltCrack] + sol

    print('Starting Fracture Front loop')
    itrcount = 1

    FillFrac_km1 = []  # filling fraction last iteration; used to calculate norm
    DLkOffEltChannel = DLkOff[Frac.EltChannel]

    #   Fracture front loop
    norm=10.
    k=0
    maxitr=Simulation_Parameters.MaximumFrontIts

    # todo i don t like the way this loop is written - it is not a clean architecture to have and if break within a while
    # i have tried my best, far from readable code
    # it is not working - always do only 2 its.
#
    while (norm > Simulation_Parameters.ToleranceFractureFront) and ( k < maxitr ):
        k=k+1

        print('\n iteration ' + repr(k))
        # NEW FRAC FRONT ESTIMATION

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e10 * np.ones((Frac.mesh.NumberOfElts,), float);  # uncalculated cells get very large value , WTF?
        sgndDist_k[Frac.EltChannel] = 0
        # Tip asymptotic inversion
        # the order of fct arg is whacky here
        sgndDist_k[Frac.EltRibbon] = -TipAsymInversion(w_k, Frac.EltRibbon, Material_properties.Kprime, Material_properties.Eprime,
                                                       Simulation_Parameters.tip_asymptote,
                                                       Frac.muPrime, Material_properties.Cprime, Frac.sgndDist, TimeStep)

        if np.isnan(sgndDist_k[Frac.EltRibbon]).any():
            print('Tip inversion is not correct' + '\n time step failed .............')
            f.write('Tip inversion is not correct' + '\n time step failed .............\n\n')
            exitstatus = 7
            break

        # SOLVE EIKONAL eq via Fast Marching Method starting from the element close to the ribbon elt
        # (i.e. the tip element of the last time step)
        SolveFMM(sgndDist_k, Frac.EltRibbon, Frac.EltChannel, Frac.mesh)

        if max(sgndDist_k) == 1e10:  # why 1.e10 ???
            print(
                'FMM not worked properly = ' + repr(np.where(sgndDist_k == 1e10)) + '\ntime step failed .............')
            f.write(
                'FMM not worked properly = ' + repr(np.where(sgndDist_k == 1e10)) + '\ntime step failed .............')
            exitstatus = 2
            break

        print('Calculating the filling fraction of tip elements with the new fracture front location...')

        # gets the new tip elements & \ell_k & alpha_k (also containing the elements
        # which are fully filled after the front is moved outward)
        (EltsTipNew, l_k, alpha_k, CellStatus) = TrackFront(sgndDist_k, Frac.EltChannel, Frac.mesh) # should be named ReconstructFront not TrackFront

        tipNeighb = Frac.mesh.NeiElements[EltsTipNew, :]
        for i in range(0, len(EltsTipNew)):
            if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
                Frac.PlotFracture('complete', 'footPrint')
                f.write('Reached end of the grid. exiting....\n\n')
                raise SystemExit('Reached end of the grid. exiting....')

        InCrack_k = np.zeros((Frac.mesh.NumberOfElts,), dtype=np.int8)
        InCrack_k[Frac.EltChannel] = 1
        InCrack_k[EltsTipNew] = 1

        Vel_k = -(sgndDist_k[EltsTipNew] - Frac.sgndDist[EltsTipNew]) / TimeStep

        # Calculate filling fraction for current iteration
        FillFrac_k = VolumeIntegral(alpha_k, l_k, Frac.mesh.hx, Frac.mesh.hy, 'A', Material_properties.Kprime[EltsTipNew],
                                    Material_properties.Eprime, Frac.muPrime[EltsTipNew], Material_properties.Cprime[EltsTipNew],
                                    Vel_k) / Frac.mesh.EltArea
        print("fill frac "+ repr(FillFrac_k))

        FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-6)] = 1.0  # humm what is this fix for ?

        if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
            print('incorrect filling fraction ' + repr(FillFrac_k[np.where(FillFrac_k > 1.0 + 1e-6)]))
            print(repr(FillFrac_k[np.where(FillFrac_k < 0.0 - np.finfo(float).eps)]))
            print('Filling fraction not correct.\ntime step failed .............')
            f.write('Filling fraction not correct.\ntime step failed .............\n\n')
            exitstatus = 9
            break

        # some of the list are redundant to calculate on each iteration
        # Evaluate the element lists for current iteration
        (EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k, CellStatus_k) = UpdateLists(Frac.EltChannel,
                                                                                                 EltsTipNew,
                                                                                                 FillFrac_k,
                                                                                                 sgndDist_k,
                                                                                                 Frac.mesh)

        # EletsTipNew may contain fully filled elements also
        NewTipinTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]  # poor variable name !

        norm = abs((np.linalg.norm(FillFrac_k) - np.linalg.norm(FillFrac_km1)) / len(FillFrac_k))
        FillFrac_km1 = copy.deepcopy(FillFrac_k)
        print('Norm of subsequent filling fraction estimates = ' + repr(norm))  #

        # What is this for below ???
        nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
        if nan.any():
            #                problem = np.where(nan)[0]
            #                for i in range(0,len(problem)):
            #                    neighbors  = np.asarray(Neighbors(EltsTipNew[problem[i]],Frac.mesh.nx,Frac.mesh.ny))
            #                    inTip = np.asarray([],int)
            #                    for j in range(0,len(neighbors)):
            #                        inTip = np.append(inTip,np.where(EltsTipNew==neighbors[j]))
            #                    alpha_k[problem[i]]=np.mean(alpha_k[inTip])
            print('Front is not tracked correctly, ' + 'problem in cell(s) ' + repr(
                EltsTipNew[np.where(nan)]) + '\ntime step failed .............')
            f.write('Front is not tracked correctly, ' + 'problem in cell(s) ' + repr(
                EltsTipNew[np.where(nan)]) + '\ntime step failed .............\n\n')
            exitstatus = 3
            break


        print(' Solving the EHL system with the new trial footprint')
        # tip cells whose distance from front has not changed.
        stagnant = abs(1 - sgndDist_k[EltsTipNew] / Frac.sgndDist[EltsTipNew]) < 1e-8
        if stagnant.any():
            # calculate stress intensity factor for stagnant cells
            KIPrime = StressIntensityFactor(w_k, sgndDist_k, EltsTipNew, EltRibbon_k, stagnant, Frac.mesh,
                                                Frac.Eprime)

            if np.isnan(KIPrime).any():
                np.where(np.isnan(KIPrime))
                print('Ribbon element not found in the enclosure of tip cell. tip cell ' + repr(
                        EltsTipNew[np.where(np.isnan(KIPrime))]) + '\n time step failed .............')
                f.write('Ribbon element not found in the enclosure of tip cell. tip cell ' + repr(
                        EltsTipNew[np.where(np.isnan(KIPrime))]) + '\n time step failed .............\n\n')
                exitstatus = 8
                break

            wTip = VolumeIntegral(alpha_k, l_k, Frac.mesh.hx, Frac.mesh.hy, Simulation_Parameters.tip_asymptote,
                                      Frac.Kprime[EltsTipNew],
                                      Frac.Eprime, Frac.muPrime[EltsTipNew], Frac.Cprime[EltsTipNew], Vel_k,
                                      stagnant, KIPrime) / Frac.mesh.EltArea
        else:
            # directly calculate the tip volume from the propagation HF asymptote
            wTip = VolumeIntegral(alpha_k, l_k, Frac.mesh.hx, Frac.mesh.hy, Simulation_Parameters.tip_asymptote,
                                      Material_properties.Kprime[EltsTipNew],
                                      Material_properties.Eprime, Frac.muPrime[EltsTipNew], Material_properties.Cprime[EltsTipNew],
                                      Vel_k) / Frac.mesh.EltArea

        # check if the tip volume has gone into negative
        smallNgtvWTip = np.where(np.logical_and(wTip < 0, wTip > -10 ** -4 * np.mean(wTip))) #wtf with variable name
        if np.asarray(smallNgtvWTip).size > 0:
             #                    warnings.warn("Small negative volume integral(s) received, ignoring "+repr(wTip[smallngtvwTip])+' ...')
            wTip[smallNgtvWTip] = abs(wTip[smallNgtvWTip])

            # what is that for ? please comment
        if (wTip < -10 ** -4 * np.mean(wTip)).any():
            print('wTip not right' + '\n time step failed .............')
            f.write('wTip not right' + '\n time step failed .............\n\n')
            exitstatus = 4
            break

        DLkOff = np.zeros((Frac.mesh.NumberOfElts,), float) # what is this doing here ????

        guess = np.zeros((Frac.EltChannel.size + EltsTipNew.size,), float)
            # pguess = Frac.p[EltsTipNew]

        guess[np.arange(Frac.EltChannel.size)] = TimeStep * CurrentRate / Frac.EltCrack.size\
                                                     * np.ones( (Frac.EltCrack.size,), float)

        wguess = np.copy(Frac.w)
        wguess[Frac.EltChannel] = wguess[Frac.EltChannel] + guess[np.arange(Frac.EltChannel.size)]
        wguess[EltsTipNew] = wTip
        vk = velocity(wguess, EltCrack_k, Frac.mesh, InCrack_k, Frac.muPrime, C, Frac.SigmaO)

        # why is C not adjusted here for tip element correction HERE ?

        TypValue = np.copy(guess)
        TypValue[Frac.EltChannel.size + np.arange(EltsTipNew.size)] = 1e5
        arg = (
                Frac.EltChannel, EltsTipNew, Frac.w, wTip, EltCrack_k, Frac.mesh, TimeStep, Qin,
                C, Frac.muPrime,
                Fluid_properties.density,
                InCrack_k, DLkOff, Frac.SigmaO, Fluid_properties.turbulence)

        (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_ExtendedFP, MakeEquationSystemExtendedFP,
                                     guess, TypValue, vk, 1.0, Simulation_Parameters.ToleranceEHL, 100, *arg)

        w_k = np.copy(Frac.w)
        w_k[EltCrack_k] = w_k[EltCrack_k] + sol

        if np.isnan(w_k).any() or (w_k < 0).any():
            f.write('width solution not correct.\ntime step failed .............\n\n')
            exitstatus = 5
            break
    # ---- end of while loop

    if norm < Simulation_Parameters.ToleranceFractureFront :  # which means  convergence of the fracture front
        exitstatus = 1
        print('Fracture front has now converged, exiting loop...')
        w_k[EltsTipNew] = wTip
        Frac.w = w_k
        Frac.FillF = FillFrac_k[NewTipinTip]
        (Frac.EltChannel, Frac.EltTip, Frac.EltCrack, Frac.EltRibbon, Frac.ZeroVertex) = (
            EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k)  # lazy writing change
        Frac.p[Frac.EltCrack] = np.dot(C[np.ix_(Frac.EltCrack, Frac.EltCrack)], Frac.w[Frac.EltCrack])
        Frac.sgndDist = sgndDist_k
        (Frac.alpha, Frac.l, Frac.v) = (alpha_k[NewTipinTip], l_k[NewTipinTip], Vel_k[NewTipinTip])
        Frac.InCrack = InCrack_k
        Frac.time += TimeStep


    if k >= maxitr:
        print('did not converge after ' + repr(maxitr) + ' iterations' + '\n time step failed .............')
        f.write(
                'did not converge after ' + repr(maxitr) + ' iterations' + '\n time step failed .............\n\n')
        exitstatus = 6

    return exitstatus
