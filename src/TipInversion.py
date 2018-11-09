# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Nov  1 15:22:00 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.

Tip inversion for different flow regimes. These functions take width opening and gives distance from tip calculated
using the given propagation regime.
"""

# imports
import math
import numpy as np
from scipy.optimize import brentq
# from src.Utility import *
import matplotlib.pyplot as plt
import warnings

beta_m = 2**(1/3) * 3**(5/6)
beta_mtld = 4/(15**(1/4) * (2**0.5 - 1)**(1/4))
cnst_mc = 3 * beta_mtld**4 / (4 * beta_m**3)
cnst_m = beta_m**3 / 3
Ki_c = 3000

#-----------------------------------------------------------------------------------------------------------------------

def TipAsym_viscStor_Res(dist, *args):
    """Residual function for viscosity dominate regime, without leak off"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    return wEltRibbon - (18 * 3 ** 0.5 * (dist - DistLstTSEltRibbon) / dt * muPrime / Eprime) ** (1 / 3) * dist ** (
            2 / 3)


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_viscLeakOff_Res(dist, *args):
    """Residual function for viscosity dominated regime, with leak off"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    return wEltRibbon - 4 / (15 * np.tan(np.pi / 8)) ** 0.25 * (Cbar * muPrime / Eprime) ** 0.25 * ((dist -
            DistLstTSEltRibbon) / dt) ** 0.125 * dist ** (5 / 8)


# -----------------------------------------------------------------------------------------------------------------------

def TipAsym_MK_zrthOrder_Res(dist, *args):
    """Residual function for viscosity to toughness regime with transition, without leak off"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    if Kprime == 0:
        return TipAsym_viscStor_Res(dist, *args)
    if muPrime == 0:
        # return toughness dominated asymptote
        return wEltRibbon ** 2 * (Eprime / Kprime) ** 2

    w_tld = Eprime * wEltRibbon / (Kprime * dist**0.5)
    V = (dist - DistLstTSEltRibbon) / dt
    return w_tld - (1 + beta_m**3 * Eprime**2 * V * dist**0.5 * muPrime / Kprime**3)**(1/3)


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_MTildeK_zrthOrder_Res(dist, *args):
    """Residual function for zeroth-order solution for M~K edge tip asymptote"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    w_tld = Eprime * wEltRibbon / (Kprime * dist ** 0.5)
    V = (dist - DistLstTSEltRibbon) / dt
    return -w_tld + (1 + beta_mtld**4 * 2 * Cbar * Eprime**3 * dist**0.5 * V**0.5 * muPrime / Kprime**4)**(1/4)


# ----------------------------------------------------------------------------------------------------------------------

def f(K, Cb, C1):
    return 1 / (3 * C1) * (
        1 - K ** 3 - 3 * Cb * (1 - K ** 2) / 2 + 3 * Cb ** 2 * (1 - K) - 3 * Cb ** 3 * np.log((Cb + 1) / (Cb + K)))


# ----------------------------------------------------------------------------------------------------------------------
# todo: 1st order tip asymptote solutions
# def TipAsym_Universal_1stOrder_Res(dist, *args):
#     """More precise function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce)"""
#
#     (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args
#
#     Vel = (dist - DistLstTSEltRibbon) / dt
#     Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
#     Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
#     sh = muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)
#
#     g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
#     delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0
#
#     C1 = 4 * (1 - 2 * delt) / (delt * (1 - delt)) * np.tan(math.pi * delt)
#     C2 = 16 * (1 - 3 * delt) / (3 * delt * (2 - 3 * delt)) * np.tan(3 * math.pi * delt / 2)
#     b = C2 / C1
#
#     return sh - f(Kh, Ch * b, C1)


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Universal_zrthOrder_Res(dist, *args):
    """Function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce 2017)"""
    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, Dist_LstTS, dt) = args

    if Cbar == 0:
        return TipAsym_MK_zrthOrder_Res(dist, *args)

    Vel = (dist - Dist_LstTS) / dt
    Ki = 2 * Cbar * Eprime / (Vel**0.5 * Kprime)
    if Ki > Ki_c:
        return TipAsym_MTildeK_zrthOrder_Res(dist, *args)

    Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
    g0 = f(Kh, cnst_mc * Ch, cnst_m)
    sh = muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)

    return sh - g0


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_variable_Toughness_Res(dist, *args):

    (wEltRibbon, Eprime, Kprime_func, anisotropic_flag, alpha, zero_vertex, center_coord) = args

    if zero_vertex == 0:

        x = center_coord[0] + dist * np.cos(alpha)
        y = center_coord[1] + dist * np.sin(alpha)

    elif zero_vertex == 1:

        x = center_coord[0] - dist * np.cos(alpha)
        y = center_coord[1] + dist * np.sin(alpha)

    elif zero_vertex == 2:

        x = center_coord[0] - dist * np.cos(alpha)
        y = center_coord[1] - dist * np.sin(alpha)

    elif zero_vertex == 3:

        x = center_coord[0] + dist * np.cos(alpha)
        y = center_coord[1] - dist * np.sin(alpha)

    if anisotropic_flag:
        Kprime = Kprime_func(alpha)
    else:
        Kprime = Kprime_func(x,y)

    return dist - wEltRibbon ** 2 * (Eprime / Kprime) ** 2


#-----------------------------------------------------------------------------------------------------------------------

def FindBracket_dist(w, Kprime, Eprime, muPrime, Cprime, DistLstTS, dt, mesh, ResFunc):
    """ 
    Find the valid bracket for the root evaluation function.
    """

    a = -DistLstTS * (1 + 5e3 * np.finfo(float).eps)
    a[np.where(a<=np.finfo(float).eps * np.finfo(float).eps)[0]]= np.finfo(float).eps
    # b = 10 * (w / (Kprime / Eprime)) ** 2
    b = np.full((len(w),), 3 * (mesh.hx**2 + mesh.hy**2)**0.5, dtype=np.float64)
    # b[np.where(np.isinf(b))[0]] = 4 * (mesh.hx**2 + mesh.hy**2)**0.5

    for i in range(0, len(w)):

        TipAsmptargs = (w[i], Kprime[i], Eprime[i], muPrime[i], Cprime[i], -DistLstTS[i], dt)
        Res_a = ResFunc(a[i], *TipAsmptargs)
        Res_b = ResFunc(b[i], *TipAsmptargs)

        cnt = 0
        mid = b[i]
        while Res_a * Res_b > 0:
            mid = (a[i] + 2 * mid) / 3  # weighted
            Res_a = ResFunc(mid, *TipAsmptargs)
            a[i] = mid
            cnt += 1
            if cnt >= 30:  # Should assume not propagating. not set to check how frequently it happens.
                a[i] = np.nan
                b[i] = np.nan
                break

    return a, b


# ----------------------------------------------------------------------------------------------------------------------

def TipAsymInversion(w, frac, matProp, simParmtrs, dt=None, Kprime_k=None, Eprime_k=None):
    """ 
    Evaluate distance from the front using tip assymptotics according to the given regime, given the fracture width in
    the ribbon cells.

    Arguments:
        w (ndarray-float):                      fracture width
        frac (Fracture object):                 current fracture object
        matProp (MaterialProperties object):    Material properties
        simParmtrs (SimulationParameters object): Simulation parameters
        dt (float):                             time step
        Kprime_k (ndarray-float):               Kprime for current iteration of toughness loop. if not given, the Kprime
                                                from the given material properties object will be used.
    Returns:
        dist (ndarray-float):                   distance (unsigned) from the front to the ribbon cells.
    """

    if Kprime_k is None:
        Kprime = matProp.Kprime[frac.EltRibbon]
    else:
        Kprime = Kprime_k

    if Eprime_k is None:
        Eprime = np.full((frac.EltRibbon.size,), matProp.Eprime)
    else:
        Eprime = Eprime_k

    if simParmtrs.get_tipAsymptote() == 'U':
        ResFunc = TipAsym_Universal_zrthOrder_Res
    elif simParmtrs.get_tipAsymptote() == 'Kt':
        raise ValueError("Tip inversion with Kt regime is yet to be implemented")
    elif simParmtrs.get_tipAsymptote() == 'M':
        ResFunc = TipAsym_viscStor_Res
    elif simParmtrs.get_tipAsymptote() == 'Mt':
        ResFunc = TipAsym_viscLeakOff_Res
    elif simParmtrs.get_tipAsymptote() == 'MK':
        ResFunc = TipAsym_MK_zrthOrder_Res
    elif simParmtrs.get_tipAsymptote() == 'K':
        return w[frac.EltRibbon] ** 2 * (Eprime / Kprime) ** 2


    # checking propagation condition
    stagnant = np.where(Kprime * (-frac.sgndDist[frac.EltRibbon])**0.5 / (
                                        Eprime * w[frac.EltRibbon]) > 1)[0]
    moving = np.arange(frac.EltRibbon.shape[0])[~np.in1d(frac.EltRibbon, frac.EltRibbon[stagnant])]

    a, b = FindBracket_dist(w[frac.EltRibbon[moving]],
                            Kprime[moving],
                            Eprime[moving],
                            frac.muPrime[frac.EltRibbon[moving]],
                            matProp.Cprime[frac.EltRibbon[moving]],
                            frac.sgndDist[frac.EltRibbon[moving]],
                            dt,
                            frac.mesh,
                            ResFunc)

    dist = -frac.sgndDist[frac.EltRibbon]
    for i in range(0, len(moving)):

        TipAsmptargs = (w[frac.EltRibbon[moving[i]]],
                        Kprime[moving[i]],
                        Eprime[moving[i]],
                        frac.muPrime[frac.EltRibbon[moving[i]]],
                        matProp.Cprime[frac.EltRibbon[moving[i]]],
                        -frac.sgndDist[frac.EltRibbon[moving[i]]],
                        dt)
        try:
            dist[moving[i]] = brentq(ResFunc, a[i], b[i], TipAsmptargs)
        except RuntimeError:
            dist[moving[i]] = np.nan
        except ValueError:
            dist[moving[i]] = np.nan
    return dist

# -----------------------------------------------------------------------------------------------------------------------


def StressIntensityFactor(w, lvlSetData, EltTip, EltRibbon, stagnant, mesh, Eprime):
    """ 
    This function evaluate the stress intensity factor. See Donstov & Pierce Comput. Methods Appl. Mech. Engrn. 2017
    
    Arguments:
        w (ndarray-float):              fracture width
        lvlSetData (ndarray-float):     the level set values, i.e. distance from the fracture front
        EltTip (ndarray-int):           tip elements
        EltRibbon (ndarray-int):        ribbon elements
        stagnant (ndarray-boolean):     the stagnant tip cells
        mesh (CartesianMesh object):    mesh
        Eprime (ndarray):                 the plain strain modulus
        
    Returns:
        ndarray-float:                  the stress intensity factor of the stagnant cells. Zero is returned for the 
                                        tip cells that are moving.
    """
    KIPrime = np.zeros((EltTip.size,), float)
    for i in range(0, len(EltTip)):
        if stagnant[i]:
            neighbors = mesh.NeiElements[EltTip[i]]
            enclosing = np.append(neighbors, np.asarray(
                [neighbors[2] - 1, neighbors[2] + 1, neighbors[3] - 1, neighbors[3] + 1]))  # eight enclosing cells

            InRibbon = np.asarray([], int)  # find neighbors in Ribbon cells
            for e in range(8):
                found = np.where(EltRibbon == enclosing[e])[0]
                if found.size > 0:
                    InRibbon = np.append(InRibbon, EltRibbon[found[0]])

            if InRibbon.size == 1:
                KIPrime[i] = w[InRibbon[0]] * Eprime[i] / (-lvlSetData[InRibbon[0]]) ** 0.5
            elif InRibbon.size > 1:  # evaluate using least squares method
                KIPrime[i] = Eprime[i] * (w[InRibbon[0]] * (-lvlSetData[InRibbon[0]]) ** 0.5 + w[InRibbon[1]] * (
                    -lvlSetData[InRibbon[1]]) ** 0.5) / (-lvlSetData[InRibbon[0]] - lvlSetData[InRibbon[1]])
            else:  # ribbon cells not found in enclosure, evaluating with the closest ribbon cell
                RibbonCellsDist = ((mesh.CenterCoor[EltRibbon, 0] - mesh.CenterCoor[EltTip[i], 0]) ** 2 + (
                    mesh.CenterCoor[EltRibbon, 1] - mesh.CenterCoor[EltTip[i], 1]) ** 2) ** 0.5
                closest = EltRibbon[np.argmin(RibbonCellsDist)]
                KIPrime[i] = w[closest] * Eprime[i] / (-lvlSetData[closest]) ** 0.5

    return KIPrime

#-----------------------------------------------------------------------------------------------------------------------


def TipAsymInversion_hetrogenous_toughness(w, frac, mat_prop, level_set):
    """
    This function inverts the tip asymptote with the toughness value taken at the tip instead of taking at the ribbon
    cell.

    Argument:
        w (ndarray-float):                      fracture width
        frac (Fracture object):                 current fracture object
        matProp (MaterialProperties object):    material properties
        level_set (ndarray-float):              the level set values, i.e. signed distance from the fracture front

    Returns:
        ndarray-float:                          the inverted tip asymptote for the ribbon cells
    """

    zero_vrtx = find_zero_vertex(frac.EltRibbon, level_set, frac.mesh)
    dist = -level_set
    alpha = np.zeros((frac.EltRibbon.size,), dtype=np.float64)

    for i in range(0, len(frac.EltRibbon)):
        if zero_vrtx[i]==0:
            # north-east direction of propagation
            alpha[i] = np.arccos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]]) / frac.mesh.hx)

        elif zero_vrtx[i]==1:
            # north-west direction of propagation
            alpha[i] = np.arccos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 0]]) / frac.mesh.hx)

        elif zero_vrtx[i]==2:
            # south-west direction of propagation
            alpha[i] = np.arccos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 0]]) / frac.mesh.hx)

        elif zero_vrtx[i]==3:
            # south-east direction of propagation
            alpha[i] = np.arccos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]]) / frac.mesh.hx)

        warnings.filterwarnings("ignore")
        if abs(dist[frac.mesh.NeiElements[frac.EltRibbon[i], 0]] / dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]] - 1) < 1e-7:
            # if the angle is 90 degrees
            alpha[i] = np.pi / 2
        if abs(dist[frac.mesh.NeiElements[frac.EltRibbon[i], 2]] / dist[frac.mesh.NeiElements[frac.EltRibbon[i], 3]] - 1) < 1e-7:
            # if the angle is 0 degrees
            alpha[i] = 0

    sol = np.zeros((len(frac.EltRibbon),),dtype=np.float64)
    for i in range(0, len(frac.EltRibbon)):

        TipAsmptargs = (w[frac.EltRibbon[i]],
                        mat_prop.Eprime,
                        mat_prop.KprimeFunc,
                        mat_prop.anisotropic,
                        alpha[i],
                        zero_vrtx[i],
                        frac.mesh.CenterCoor[frac.EltRibbon[i]])

        # residual for zero distance; used as lower bracket
        residual_zero = TipAsym_variable_Toughness_Res(0, *TipAsmptargs)

        # the lower bracket (0) and the upper bracker (4x the maximum possible length in a cell) is divided into 16
        # equally distant points to sample the sign of the residual function. This is necessary to avoid missing a high
        # resolution variation in toughness. This also means that the toughness variations below the upper_bracket/16 is
        # not guaranteed to be caught.
        sample_lngths = np.linspace(4*(frac.mesh.hx**2 + frac.mesh.hy**2)**0.5 /
                                    16,4*(frac.mesh.hx**2 + frac.mesh.hy**2)**0.5,16)
        cnt = 0
        res_prdct = 0
        while res_prdct >= 0 and cnt < 16:
            res_prdct = residual_zero * TipAsym_variable_Toughness_Res(sample_lngths[cnt], *TipAsmptargs)
            cnt += 1

        if cnt == 16:
            sol[i] = np.nan
            return sol
        else:
            upper_bracket = sample_lngths[cnt-1]

        try:
            sol[i] = brentq(TipAsym_variable_Toughness_Res, 0, upper_bracket, TipAsmptargs)
        except RuntimeError:
            sol[i] = np.nan

    return sol-sol*1e-10

#-----------------------------------------------------------------------------------------------------------------------


def find_zero_vertex(Elts, level_set, mesh):
    """ find the vertex opposite to the propagation direction from which the perpendicular on the front is drawn"""

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

    return zero_vertex