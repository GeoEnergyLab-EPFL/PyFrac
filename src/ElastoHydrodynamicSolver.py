# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
# import numdifftools as nd
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from src.Utility import *


def finiteDiff_operator_laminar(w, EltCrack, muPrime, Mesh, InCrack):
    """
    The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
    Dontsov and Peirce 2008). THe matrix is evaluated with the laminar flow assumption.
    
    Arguments:
        w (ndarray-float):              the width of the trial fracture. 
        EltCrack (ndarray-int):         the list of elements inside the fracture
        muPrime (ndarray-float):        the scalled local viscosity of the injected fluid (12 * viscosity)
        Mesh (CartesianMesh object):    the mesh
        InCrack (ndarray-int):          An array specifying whether elements are inside the fracture or not with
                                        1 or 0 respectively
    
    Returns:
        ndarray-float:                  the finite difference matrix    
    """

    FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)
    dx = Mesh.hx
    dy = Mesh.hy

    # width at the cell edges evaluated by averaging
    wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 0]]
    wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 1]]
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 2]]
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 3]]

    # the finite difference operator matrix
    FinDiffOprtr[EltCrack, EltCrack] = -(wLftEdge ** 3 + wRgtEdge ** 3) / dx ** 2 / muPrime[EltCrack] - (
                                            wBtmEdge ** 3 + wTopEdge ** 3) / dy ** 2 / muPrime[EltCrack]
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 0]] = wLftEdge ** 3 / dx ** 2 / muPrime[EltCrack]
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 1]] = wRgtEdge ** 3 / dx ** 2 / muPrime[EltCrack]
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 2]] = wBtmEdge ** 3 / dy ** 2 / muPrime[EltCrack]
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 3]] = wTopEdge ** 3 / dy ** 2 / muPrime[EltCrack]

    return FinDiffOprtr

#-----------------------------------------------------------------------------------------------------------------------


def FiniteDiff_operator_turbulent_implicit(w, EltCrack, mu, Mesh, InCrack, rho, vkm1, C, sigma0, dgrain=1e-6):
    """
    The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
    Dontsov and Peirce 2008). THe matrix is evaluated by taking turbulence into account. The friction factor for 
    turbulent flow is evaluated using the Yang-Joseph approximation.

    Arguments:
        w (ndarray-float):              the width of the trial fracture. 
        EltCrack (ndarray-int):         the list of elements inside the fracture
        mu (ndarray-float):             the local viscosity of the injected fluid
        Mesh (CartesianMesh object):    the mesh
        InCrack (ndarray-int):          An array specifying whether elements are inside the fracture or not with
                                        1 or 0 respectively
        vkm1 (ndarray-float):           the velocity at cell edges from the previous iteration (if necessary). Here, it
                                        is used as the starting guess for the implicit solver.
        C (ndarray-float):              the elasticity matrix
        sigma0 (ndarrray-float):        the confining stress
        dgrain (float, default 1e-6)
                
    Returns:
        ndarray-float:                  the finite difference matrix
        ndarray-float:                  the velocity evaluated for current iteration
    """

    FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)

    dx = Mesh.hx
    dy = Mesh.hy

    # todo: can be evaluated at each cell edge
    rough = w[EltCrack]/dgrain
    rough[np.where(rough<5)[0]] = 10.

    # width on edges; evaluated by averaging the widths of adjacent cells
    wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2
    wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2

    # pressure gradient data structure. The rows store pressure gradient in the following order.
    # 0 - pressure gradient on the left edge in x-direction
    # 1 - pressure gradient on the right edge in x-direction
    # 2 - pressure gradient on the bottom edge in y-direction
    # 3 - pressure gradient on the top edge in y-direction
    # 4 - pressure gradient on the left edge in y-direction
    # 5 - pressure gradient on the right edge in y-direction
    # 6 - pressure gradient on the bottom edge in x-direction
    # 7 - pressure gradient on the top edge in x-direction

    dp = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)
    dp[0,EltCrack] = dpdxLft
    dp[1,EltCrack] = dpdxRgt
    dp[2,EltCrack] = dpdyBtm
    dp[3,EltCrack] = dpdyTop
    # linear interpolation for pressure gradient on the edges where central difference not available
    dp[4,EltCrack] = (dp[2,Mesh.NeiElements[EltCrack,0]]+dp[3,Mesh.NeiElements[EltCrack,0]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[5,EltCrack] = (dp[2,Mesh.NeiElements[EltCrack,1]]+dp[3,Mesh.NeiElements[EltCrack,1]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[6,EltCrack] = (dp[0,Mesh.NeiElements[EltCrack,2]]+dp[1,Mesh.NeiElements[EltCrack,2]]+dp[0,EltCrack]+dp[1,EltCrack])/4
    dp[7,EltCrack] = (dp[0,Mesh.NeiElements[EltCrack,3]]+dp[1,Mesh.NeiElements[EltCrack,3]]+dp[0,EltCrack]+dp[1,EltCrack])/4

    # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
    dpLft = (dp[0, EltCrack] ** 2 + dp[4, EltCrack] ** 2) ** 0.5
    dpRgt = (dp[1, EltCrack] ** 2 + dp[5, EltCrack] ** 2) ** 0.5
    dpBtm = (dp[2, EltCrack] ** 2 + dp[6, EltCrack] ** 2) ** 0.5
    dpTop = (dp[3, EltCrack] ** 2 + dp[7, EltCrack] ** 2) ** 0.5

    vk = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)

    # loop to calculate velocity on each cell edge implicitly
    for i in range(0,len(EltCrack)):
        # todo !!! Hack. zero velocity if the pressure gradient is zero or very small width
        if dpLft[i] < 1e-8 or wLftEdge[i]<1e-10:
            vk[0, EltCrack[i]] = 0.0
        else:
            arg = (wLftEdge[i], mu[EltCrack[i]], rho, dpLft[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[0, EltCrack[i]], *arg) * Velocity_Residual(
                            10 * vkm1[0, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[0, EltCrack[i]], *arg)
                vk[0, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[0, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[0, EltCrack[i]],
                                            10 * vkm1[0, EltCrack[i]], arg)

        if dpRgt[i] < 1e-8 or wRgtEdge[i] < 1e-10:
            vk[1, EltCrack[i]] = 0.0
        else:
            arg = (wRgtEdge[i], mu[EltCrack[i]], rho, dpRgt[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[1, EltCrack[i]], *arg) * Velocity_Residual(
                            10 * vkm1[1, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[1, EltCrack[i]], *arg)
                vk[1, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[1, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[1, EltCrack[i]],
                                            10 * vkm1[1, EltCrack[i]], arg)

        if dpBtm[i] < 1e-8 or wBtmEdge[i] < 1e-10:
            vk[2, EltCrack[i]] = 0.0
        else:
            arg = (wBtmEdge[i], mu[EltCrack[i]], rho, dpBtm[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[2, EltCrack[i]], *arg) * Velocity_Residual(
                            10 * vkm1[2, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[2, EltCrack[i]], *arg)
                vk[2, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[2, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[2, EltCrack[i]],
                                            10 * vkm1[2, EltCrack[i]], arg)

        if dpTop[i] < 1e-8 or wTopEdge[i] < 1e-10:
            vk[3, EltCrack[i]] = 0.0
        else:
            arg = (wTopEdge[i], mu[EltCrack[i]], rho, dpTop[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[3, EltCrack[i]],*arg)*Velocity_Residual(
                            10 * vkm1[3, EltCrack[i]],*arg)>0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[3, EltCrack[i]], *arg)
                vk[3, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[3, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[3, EltCrack[i]],
                                            10 * vkm1[3, EltCrack[i]], arg)

    # calculating Reynold's number with the velocity
    ReLftEdge = 4 / 3 * rho * wLftEdge * vk[0, EltCrack] / mu[EltCrack]
    ReRgtEdge = 4 / 3 * rho * wRgtEdge * vk[1, EltCrack] / mu[EltCrack]
    ReBtmEdge = 4 / 3 * rho * wBtmEdge * vk[2, EltCrack] / mu[EltCrack]
    ReTopEdge = 4 / 3 * rho * wTopEdge * vk[3, EltCrack] / mu[EltCrack]


    # non zeros Reynolds numbers
    ReLftEdge_nonZero = np.where(ReLftEdge > 0.)[0]
    ReRgtEdge_nonZero = np.where(ReRgtEdge > 0.)[0]
    ReBtmEdge_nonZero = np.where(ReBtmEdge > 0.)[0]
    ReTopEdge_nonZero = np.where(ReTopEdge > 0.)[0]



    # calculating friction factor with the Yang-Joseph explicit function
    ffLftEdge = np.zeros((EltCrack.size), dtype=np.float64)
    ffRgtEdge = np.zeros((EltCrack.size), dtype=np.float64)
    ffBtmEdge = np.zeros((EltCrack.size), dtype=np.float64)
    ffTopEdge = np.zeros((EltCrack.size), dtype=np.float64)
    ffLftEdge[ReLftEdge_nonZero] = FF_YangJoseph(ReLftEdge[ReLftEdge_nonZero], rough[ReLftEdge_nonZero])
    ffRgtEdge[ReRgtEdge_nonZero] = FF_YangJoseph(ReRgtEdge[ReRgtEdge_nonZero], rough[ReRgtEdge_nonZero])
    ffBtmEdge[ReBtmEdge_nonZero] = FF_YangJoseph(ReBtmEdge[ReBtmEdge_nonZero], rough[ReBtmEdge_nonZero])
    ffTopEdge[ReTopEdge_nonZero] = FF_YangJoseph(ReTopEdge[ReTopEdge_nonZero], rough[ReTopEdge_nonZero])

    # the conductivity matrix
    cond = np.zeros((4, EltCrack.size), dtype=np.float64)
    cond[0, ReLftEdge_nonZero] = wLftEdge[ReLftEdge_nonZero] ** 2 / (rho * ffLftEdge[ReLftEdge_nonZero]
                                                                     * vk[0, EltCrack[ReLftEdge_nonZero]])
    cond[1, ReRgtEdge_nonZero] = wRgtEdge[ReRgtEdge_nonZero] ** 2 / (rho * ffRgtEdge[ReRgtEdge_nonZero]
                                                                     * vk[1, EltCrack[ReRgtEdge_nonZero]])
    cond[2, ReBtmEdge_nonZero] = wBtmEdge[ReBtmEdge_nonZero] ** 2 / (rho * ffBtmEdge[ReBtmEdge_nonZero]
                                                                     * vk[2, EltCrack[ReBtmEdge_nonZero]])
    cond[3, ReTopEdge_nonZero] = wTopEdge[ReTopEdge_nonZero] ** 2 / (rho * ffTopEdge[ReTopEdge_nonZero]
                                                                     * vk[3, EltCrack[ReTopEdge_nonZero]])

    # cond[0, np.where(np.isinf(cond[0, :]))] = 0 # for cells with neighbors outside the fracture
    # cond[1, np.where(np.isinf(cond[1, :]))] = 0
    # cond[2, np.where(np.isinf(cond[2, :]))] = 0
    # cond[3, np.where(np.isinf(cond[3, :]))] = 0

    # assembling the finite difference matrix
    FinDiffOprtr[EltCrack, EltCrack] = -(cond[0, :] + cond[1, :]) / dx ** 2 - (cond[2, :] + cond[3, :]) / dy ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 0]] = cond[0, :] / dx ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 1]] = cond[1, :] / dx ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 2]] = cond[2, :] / dy ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 3]] = cond[3, :] / dy ** 2

    return FinDiffOprtr, vk

#-----------------------------------------------------------------------------------------------------------------------


def Velocity_Residual(v,*args):
    """
    This function gives the residual of the velocity equation. It is used by the root finder.
    Arguments:
        v (float):          current velocity guess
        args (
            w (float):          width at the given cell edge
            mu (float):         viscosity at the given cell edge
            rho (float):        density of the injected fluid
            dp (float):         pressure gradient at the given cell edge
            rough (float):      roughness (width / grain size) at the cell center
             )
             
    Returns:
         float:             residual of the velocity equation
    """
    (w, mu, rho, dp, rough) = args

    # Reynolds number
    Re = 4/3 * rho * w * v / mu

    # friction factor using Yang-Joseph approximation
    f = FF_YangJoseph_float(Re,rough)

    return v-w*dp/(v*rho*f)

#-----------------------------------------------------------------------------------------------------------------------


def findBracket(func,guess,*args):
    a = np.finfo(float).eps * guess
    b = max(1000*guess,1)
    Res_a = func(a,*args)
    Res_b = func(b,*args)

    cnt = 0
    while Res_a * Res_b > 0:
        b = 10*b
        Res_b = func(b, *args)
        cnt += 1
        if cnt >= 60:
            raise SystemExit('Velocity bracket cannot be found')

    return a, b



def FiniteDiff_operator_turbulent(w, EltCrack, mu, Mesh, InCrack, rho, vkm1, C, sigma0):
    FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)
    dx = Mesh.hx
    dy = Mesh.hy


    wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2
    wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2

    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)

    ReLftEdge = 4 / 3 * rho * wLftEdge * (vkm1[0, EltCrack] ** 2 + vkm1[4, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReRgtEdge = 4 / 3 * rho * wRgtEdge * (vkm1[1, EltCrack] ** 2 + vkm1[5, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReBtmEdge = 4 / 3 * rho * wBtmEdge * (vkm1[2, EltCrack] ** 2 + vkm1[6, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReTopEdge = 4 / 3 * rho * wTopEdge * (vkm1[3, EltCrack] ** 2 + vkm1[7, EltCrack] ** 2) ** 0.5 / mu[EltCrack]


    rough = 10000 * np.ones((EltCrack.size,), np.float64)
    ffLftEdge = FF_YangJoseph(ReLftEdge, rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge, rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge, rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge, rough)

    # velocity current iteration, arrangement row wise: left x, right x, bottom y, top y, left y, right y, bottom x, top x
    vk = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    vk[0, EltCrack] = -wLftEdge / (rho * ffLftEdge * (vkm1[0, EltCrack] ** 2 + vkm1[4, EltCrack] ** 2) ** 0.5) * dpdxLft
    vk[1, EltCrack] = -wRgtEdge / (rho * ffRgtEdge * (vkm1[1, EltCrack] ** 2 + vkm1[5, EltCrack] ** 2) ** 0.5) * dpdxRgt
    vk[2, EltCrack] = -wBtmEdge / (rho * ffBtmEdge * (vkm1[2, EltCrack] ** 2 + vkm1[6, EltCrack] ** 2) ** 0.5) * dpdyBtm
    vk[3, EltCrack] = -wTopEdge / (rho * ffTopEdge * (vkm1[3, EltCrack] ** 2 + vkm1[7, EltCrack] ** 2) ** 0.5) * dpdyTop
    vk[0, np.where(np.isnan(vk[0, :]))] = 0  # for edges adjacent to cells outside fracture
    vk[1, np.where(np.isnan(vk[1, :]))] = 0
    vk[2, np.where(np.isnan(vk[2, :]))] = 0
    vk[3, np.where(np.isnan(vk[3, :]))] = 0

    vk[4, EltCrack] = (vk[2, Mesh.NeiElements[EltCrack, 0]] + vk[3, Mesh.NeiElements[EltCrack, 0]] + vk[2, EltCrack] +
                       vk[3, EltCrack]) / 4
    vk[5, EltCrack] = (vk[2, Mesh.NeiElements[EltCrack, 1]] + vk[3, Mesh.NeiElements[EltCrack, 1]] + vk[2, EltCrack] +
                       vk[3, EltCrack]) / 4
    vk[6, EltCrack] = (vk[0, Mesh.NeiElements[EltCrack, 2]] + vk[1, Mesh.NeiElements[EltCrack, 2]] + vk[0, EltCrack] +
                       vk[1, EltCrack]) / 4
    vk[7, EltCrack] = (vk[0, Mesh.NeiElements[EltCrack, 3]] + vk[1, Mesh.NeiElements[EltCrack, 3]] + vk[0, EltCrack] +
                       vk[1, EltCrack]) / 4

    ReLftEdge = 4 / 3 * rho * wLftEdge * (vk[0, EltCrack] ** 2 + vk[4, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReRgtEdge = 4 / 3 * rho * wRgtEdge * (vk[1, EltCrack] ** 2 + vk[5, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReBtmEdge = 4 / 3 * rho * wBtmEdge * (vk[2, EltCrack] ** 2 + vk[6, EltCrack] ** 2) ** 0.5 / mu[EltCrack]
    ReTopEdge = 4 / 3 * rho * wTopEdge * (vk[3, EltCrack] ** 2 + vk[7, EltCrack] ** 2) ** 0.5 / mu[EltCrack]

    ffLftEdge = FF_YangJoseph(ReLftEdge, rough)
    ffRgtEdge = FF_YangJoseph(ReRgtEdge, rough)
    ffBtmEdge = FF_YangJoseph(ReBtmEdge, rough)
    ffTopEdge = FF_YangJoseph(ReTopEdge, rough)

    ffLftEdge[np.where(np.isinf(ffLftEdge))] = 0  # for edges adjacent to cells outside fracture
    ffRgtEdge[np.where(np.isinf(ffRgtEdge))] = 0
    ffBtmEdge[np.where(np.isinf(ffBtmEdge))] = 0
    ffTopEdge[np.where(np.isinf(ffTopEdge))] = 0

    cond = np.zeros((4, EltCrack.size), dtype=np.float64)
    cond[0, :] = wLftEdge ** 2 / (rho * ffLftEdge * (vk[0, EltCrack] ** 2 + vk[4, EltCrack] ** 2) ** 0.5)
    cond[1, :] = wRgtEdge ** 2 / (rho * ffRgtEdge * (vk[1, EltCrack] ** 2 + vk[5, EltCrack] ** 2) ** 0.5)
    cond[2, :] = wBtmEdge ** 2 / (rho * ffBtmEdge * (vk[2, EltCrack] ** 2 + vk[6, EltCrack] ** 2) ** 0.5)
    cond[3, :] = wTopEdge ** 2 / (rho * ffTopEdge * (vk[3, EltCrack] ** 2 + vk[7, EltCrack] ** 2) ** 0.5)

    cond[0, np.where(np.isinf(cond[0, :]))] = 0
    cond[1, np.where(np.isinf(cond[1, :]))] = 0
    cond[2, np.where(np.isinf(cond[2, :]))] = 0
    cond[3, np.where(np.isinf(cond[3, :]))] = 0

    FinDiffOprtr[EltCrack, EltCrack] = -(cond[0, :] + cond[1, :]) / dx ** 2 - (cond[2, :] + cond[3, :]) / dy ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 0]] = cond[0, :] / dx ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 1]] = cond[1, :] / dx ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 2]] = cond[2, :] / dy ** 2
    FinDiffOprtr[EltCrack, Mesh.NeiElements[EltCrack, 3]] = cond[3, :] / dy ** 2

    return (FinDiffOprtr, vk)


######################################

def MakeEquationSystemExtendedFP(solk, vkm1, *args):
    (EltChannel, EltsTipNew, wLastTS, wTip, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff, sigma0,
     turb) = args

    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltsTipNew)]

    A = np.zeros((EltChannel.size + EltsTipNew.size, EltChannel.size + EltsTipNew.size), dtype=np.float64)
    S = np.zeros((EltChannel.size + EltsTipNew.size,), dtype=np.float64)

    delwK = solk[np.arange(EltChannel.size)]
    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[EltChannel] = wcNplusOne[EltChannel] + delwK
    wcNplusOne[EltsTipNew] = wTip

    if turb:
        # (FinDiffOprtr, vk) = FiniteDiff_operator_turbulent(wcNplusOne, EltCrack, muPrime / 12, Mesh, InCrack, rho,
        #                                                      vkm1, C, sigma0)
        (FinDiffOprtr, vk) = FiniteDiff_operator_turbulent_implicit(wcNplusOne, EltCrack, muPrime/12, Mesh, InCrack,
                                                                    rho, vkm1, C, sigma0)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne, EltCrack, muPrime, Mesh, InCrack)
        vk = vkm1

    condCC = FinDiffOprtr[np.ix_(EltChannel, EltChannel)]
    condCT = FinDiffOprtr[np.ix_(EltChannel, EltsTipNew)]
    condTC = FinDiffOprtr[np.ix_(EltsTipNew, EltChannel)]
    condTT = FinDiffOprtr[np.ix_(EltsTipNew, EltsTipNew)]

    Channel = np.arange(EltChannel.size)
    Tip = Channel.size + np.arange(EltsTipNew.size)

    A[np.ix_(Channel, Channel)] = np.identity(Channel.size) - dt * np.dot(condCC, Ccc)
    A[np.ix_(Channel, Tip)] = -dt * condCT
    A[np.ix_(Tip, Channel)] = -dt * np.dot(condTC, Ccc)
    A[np.ix_(Tip, Tip)] = -dt * condTT

    S[Channel] = dt * np.dot(condCC, np.dot(Ccc, wLastTS[EltChannel]) + np.dot(Cct, wTip) + sigma0[
        EltChannel]) + dt / Mesh.hx / Mesh.hy * Q[EltChannel] \
                 - LeakOff[EltChannel] / Mesh.hx / Mesh.hy
    S[Tip] = -(wTip - wLastTS[EltsTipNew]) + dt * np.dot(condTC,
                                                         np.dot(Ccc, wLastTS[EltChannel]) + np.dot(Cct, wTip) + sigma0[
                                                             EltChannel]) \
             - LeakOff[EltsTipNew] / Mesh.hx / Mesh.hy

    return (A, S, vk)


######################################
#
def MakeEquationSystemSameFP(delwk, vkm1, *args):
    (w, EltCrack, Q, C, dt, muPrime, mesh, InCrack, LeakOff, sigma0, rho, turb) = args
    wnPlus1 = np.copy(w)
    wnPlus1[EltCrack] = wnPlus1[EltCrack] + delwk

    if turb:
        #(con, vk) = FiniteDiff_operator_turbulent(wnPlus1, EltCrack, muPrime / 12, mesh, InCrack, rho, vkm1, C, sigma0)
        (con, vk) = FiniteDiff_operator_turbulent_implicit(wnPlus1, EltCrack, muPrime / 12, mesh, InCrack, rho, vkm1,
                                                           C, sigma0)
    else:
        con = finiteDiff_operator_laminar(wnPlus1, EltCrack, muPrime, mesh, InCrack)
        vk = vkm1
    con = con[np.ix_(EltCrack, EltCrack)]
    CCrack = C[np.ix_(EltCrack, EltCrack)]

    A = np.identity(EltCrack.size) - dt * np.dot(con, CCrack)
    S = dt * np.dot(con, np.dot(CCrack, w[EltCrack]) + sigma0[EltCrack]) + dt / mesh.EltArea * Q[EltCrack] - LeakOff[
                                                                                                                 EltCrack] / mesh.EltArea
    return (A, S, vk)


#######################################

def Elastohydrodynamic_ResidualFun_sameFP(solk, interItr, *args):
    (A, S, vk) = MakeEquationSystemSameFP(solk, interItr, *args)
    return (np.dot(A, solk) - S, vk)


#######################################

def velocity(w, EltCrack, Mesh, InCrack, muPrime, C, sigma0):
    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)

    vel = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    vel[0, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2) ** 2 / muPrime[EltCrack] * dpdxLft
    vel[1, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2) ** 2 / muPrime[EltCrack] * dpdxRgt
    vel[2, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2) ** 2 / muPrime[EltCrack] * dpdyBtm
    vel[3, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2) ** 2 / muPrime[EltCrack] * dpdyTop

    vel[4, EltCrack] = (vel[2, Mesh.NeiElements[EltCrack, 0]] + vel[3, Mesh.NeiElements[EltCrack, 0]] + vel[
        2, EltCrack] + vel[3, EltCrack]) / 4
    vel[5, EltCrack] = (vel[2, Mesh.NeiElements[EltCrack, 1]] + vel[3, Mesh.NeiElements[EltCrack, 1]] + vel[
        2, EltCrack] + vel[3, EltCrack]) / 4
    vel[6, EltCrack] = (vel[0, Mesh.NeiElements[EltCrack, 2]] + vel[1, Mesh.NeiElements[EltCrack, 2]] + vel[
        0, EltCrack] + vel[1, EltCrack]) / 4
    vel[7, EltCrack] = (vel[0, Mesh.NeiElements[EltCrack, 3]] + vel[1, Mesh.NeiElements[EltCrack, 3]] + vel[
        0, EltCrack] + vel[1, EltCrack]) / 4

    return vel


#######################################

def pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack):
    pf = np.zeros((Mesh.NumberOfElts,), dtype=np.float64)
    pf[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], w[EltCrack]) + sigma0[EltCrack]

    dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) * InCrack[Mesh.NeiElements[EltCrack, 0]]
    dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 1]]
    dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) * InCrack[Mesh.NeiElements[EltCrack, 2]]
    dpdyTop = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 3]]

    return (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop)


#######################################
#  in the future the following should be move to a separate files containing the fluid models....
def FF_YangJoseph(ReNum, rough):
    ff = np.full((len(ReNum),), np.inf, dtype=np.float64)

    lam = np.where(abs(ReNum) < 2100)[0]
    ff[lam] = 16 / ReNum[lam]

    turb = np.where(abs(ReNum) >= 2100)[0]
    lamdaS = (-(
    (-64 / ReNum[turb] + 0.000083 * ReNum[turb] ** 0.75) / (1 + 2320 ** 50 / ReNum[turb] ** 50) ** 0.5) - 64 / ReNum[
                  turb] + 0.3164 / ReNum[turb] ** 0.25) / (1 + 3810 ** 15 / ReNum[turb] ** 15) ** 0.5 + (-((-(
    (-64 / ReNum[turb] + 0.000083 * ReNum[turb] ** 0.75) / (1 + 2320 ** 50 / ReNum[turb] ** 50) ** 0.5) - 64 / ReNum[
                                                                                                                turb] + 0.3164 /
                                                                                                            ReNum[
                                                                                                                turb] ** 0.25) / (
                                                                                                           1 + 3810 ** 15 /
                                                                                                           ReNum[
                                                                                                               turb] ** 15) ** 0.5) - (
                                                                                                         -64 / ReNum[
                                                                                                             turb] + 0.000083 *
                                                                                                         ReNum[
                                                                                                             turb] ** 0.75) / (
                                                                                                         1 + 2320 ** 50 /
                                                                                                         ReNum[
                                                                                                             turb] ** 50) ** 0.5 - 64 /
                                                                                                         ReNum[
                                                                                                             turb] + 0.1537 /
                                                                                                         ReNum[
                                                                                                             turb] ** 0.185) / (
                                                                                                                               1 + 1680700000000000000000000 /
                                                                                                                               ReNum[
                                                                                                                                   turb] ** 5) ** 0.5 + (
                                                                                                                                                        -(
                                                                                                                                                        (
                                                                                                                                                        -(
                                                                                                                                                        (
                                                                                                                                                        -64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.000083 *
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.75) / (
                                                                                                                                                        1 + 2320 ** 50 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 50) ** 0.5) - 64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.3164 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.25) / (
                                                                                                                                                        1 + 3810 ** 15 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 15) ** 0.5) - (
                                                                                                                                                        -(
                                                                                                                                                        (
                                                                                                                                                        -(
                                                                                                                                                        (
                                                                                                                                                        -64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.000083 *
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.75) / (
                                                                                                                                                        1 + 2320 ** 50 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 50) ** 0.5) - 64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.3164 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.25) / (
                                                                                                                                                        1 + 3810 ** 15 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 15) ** 0.5) - (
                                                                                                                                                        -64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.000083 *
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.75) / (
                                                                                                                                                        1 + 2320 ** 50 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 50) ** 0.5 - 64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.1537 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.185) / (
                                                                                                                                                        1 + 1680700000000000000000000 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 5) ** 0.5 - (
                                                                                                                                                        -64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.000083 *
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.75) / (
                                                                                                                                                        1 + 2320 ** 50 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 50) ** 0.5 - 64 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] + 0.0753 /
                                                                                                                                                        ReNum[
                                                                                                                                                            turb] ** 0.136) / (
                                                                                                                                                                              1 + 4000000000000 /
                                                                                                                                                                              ReNum[
                                                                                                                                                                                  turb] ** 2) ** 0.5 + (
                                                                                                                                                                                                       -64 /
                                                                                                                                                                                                       ReNum[
                                                                                                                                                                                                           turb] + 0.000083 *
                                                                                                                                                                                                       ReNum[
                                                                                                                                                                                                           turb] ** 0.75) / (
                                                                                                                                                                                                                            1 + 2320 ** 50 /
                                                                                                                                                                                                                            ReNum[
                                                                                                                                                                                                                                turb] ** 50) ** 0.5 + 64 / \
                                                                                                                                                                                                                                                      ReNum[
                                                                                                                                                                                                                                                          turb]
    lamdaR = ReNum[turb] ** (-0.2032 + 7.348278 / rough[turb] ** 0.96433953) * (
    -0.022 + (-0.978 + 0.92820419 * rough[turb] ** 0.03569244 - 0.00255391 * rough[turb] ** 0.8353877) / (
    1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249 / rough[
        turb] ** 50) ** 0.5 + 0.00255391 * rough[turb] ** 0.8353877) + (-(
    ReNum[turb] ** (-0.2032 + 7.348278 / rough[turb] ** 0.96433953) * (
    -0.022 + (-0.978 + 0.92820419 * rough[turb] ** 0.03569244 - 0.00255391 * rough[turb] ** 0.8353877) / (
    1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249 / rough[
        turb] ** 50) ** 0.5 + 0.00255391 * rough[turb] ** 0.8353877)) + 0.01105244 * ReNum[turb] ** (
                                                                        -0.191 + 0.62935712 / rough[
                                                                            turb] ** 0.28022284) * rough[
                                                                            turb] ** 0.23275646 + (ReNum[turb] ** (
    0.015 + 0.26827956 / rough[turb] ** 0.28852025) * (0.0053 + 0.02166401 / rough[turb] ** 0.30702955) - 0.01105244 *
                                                                                                   ReNum[turb] ** (
                                                                                                   -0.191 + 0.62935712 /
                                                                                                   rough[
                                                                                                       turb] ** 0.28022284) *
                                                                                                   rough[
                                                                                                       turb] ** 0.23275646 + (
                                                                                                   ReNum[
                                                                                                       turb] ** 0.002 * (
                                                                                                   0.011 + 0.18954211 /
                                                                                                   rough[
                                                                                                       turb] ** 0.510031) -
                                                                                                   ReNum[turb] ** (
                                                                                                   0.015 + 0.26827956 /
                                                                                                   rough[
                                                                                                       turb] ** 0.28852025) * (
                                                                                                   0.0053 + 0.02166401 /
                                                                                                   rough[
                                                                                                       turb] ** 0.30702955) + (
                                                                                                   0.0098 - ReNum[
                                                                                                       turb] ** 0.002 * (
                                                                                                   0.011 + 0.18954211 /
                                                                                                   rough[
                                                                                                       turb] ** 0.510031) + 0.17805185 /
                                                                                                   rough[
                                                                                                       turb] ** 0.46785053) / (
                                                                                                   1 + (
                                                                                                   8.733801045300249e10 *
                                                                                                   rough[
                                                                                                       turb] ** 0.90870686) /
                                                                                                   ReNum[
                                                                                                       turb] ** 2) ** 0.5) / (
                                                                                                   1 + (
                                                                                                   6.44205549308073e15 *
                                                                                                   rough[
                                                                                                       turb] ** 5.168887) /
                                                                                                   ReNum[
                                                                                                       turb] ** 5) ** 0.5) / (
                                                                        1 + (1.1077593467238922e13 * rough[
                                                                            turb] ** 4.9771653) / ReNum[
                                                                            turb] ** 5) ** 0.5) / (1 + (
    2.9505925619934144e14 * rough[turb] ** 3.7622822) / ReNum[turb] ** 5) ** 0.5
    ff[turb] = np.asarray(
        lamdaS + (lamdaR - lamdaS) / (1 + (ReNum[turb] / (45.196502 * rough[turb] ** 1.2369807 + 1891)) ** -5) ** 0.5,
        float) / 4
    return ff


#######################################

def FF_YangJoseph_float(ReNum, rough):

    if ReNum<2100:
        return 16/ReNum
    else:
        lamdaS = (-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5 + (-((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5) - (-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 - 64/ReNum + 0.1537/ReNum**0.185)/(1 + 1680700000000000000000000/ReNum**5)**0.5 + (-((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5) - (-((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5) - (-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 - 64/ReNum + 0.1537/ReNum**0.185)/(1 + 1680700000000000000000000/ReNum**5)**0.5 - (-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 - 64/ReNum + 0.0753/ReNum**0.136)/(1 + 4000000000000/ReNum**2)**0.5 + (-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 + 64/ReNum
        lamdaR = ReNum**(-0.2032 + 7.348278/rough**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough**0.03569244 - 0.00255391*rough**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough**50)**0.5 + 0.00255391*rough**0.8353877) + (-(ReNum**(-0.2032 + 7.348278/rough**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough**0.03569244 - 0.00255391*rough**0.8353877)/(1 + 265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough**50)**0.5 + 0.00255391*rough**0.8353877)) + 0.01105244*ReNum**(-0.191 + 0.62935712/rough**0.28022284)*rough**0.23275646 + (ReNum**(0.015 + 0.26827956/rough**0.28852025)*(0.0053 + 0.02166401/rough**0.30702955) - 0.01105244*ReNum**(-0.191 + 0.62935712/rough**0.28022284)*rough**0.23275646 + (ReNum**0.002*(0.011 + 0.18954211/rough**0.510031) - ReNum**(0.015 + 0.26827956/rough**0.28852025)*(0.0053 + 0.02166401/rough**0.30702955) + (0.0098 - ReNum**0.002*(0.011 + 0.18954211/rough**0.510031) + 0.17805185/rough**0.46785053)/(1 + (8.733801045300249e10*rough**0.90870686)/ReNum**2)**0.5)/(1 + (6.44205549308073e15*rough**5.168887)/ReNum**5)**0.5)/(1 + (1.1077593467238922e13*rough**4.9771653)/ReNum**5)**0.5)/(1 + (2.9505925619934144e14*rough**3.7622822)/ReNum**5)**0.5
        return (lamdaS + (lamdaR - lamdaS) / (1 + (ReNum / (45.196502 * rough ** 1.2369807 + 1891)) ** -5) ** 0.5) / 4


#######################################
def Elastohydrodynamic_ResidualFun_ExtendedFP(solk, interItr, *args):
    (A, S, vk) = MakeEquationSystemExtendedFP(solk, interItr, *args)
    return (np.dot(A, solk) - S, vk)


#######################################

def Picard_Newton(Res_fun, sys_fun, guess, TypValue, interItr, Tol, maxitr, *args, relax=1.0):
    """
    Mixed Picard Newton solver for nonlinear systems.
        
    :param Res_fun: The function calculating the residual
    :param sys_fun: The function giving the system A,b for the Picard solver to solve the linear system of the form Ax=b
    :param guess:   The initial guess
    :param TypValue:Typical value of the variable to estimate the Epsilon to calculate Jacobian
    :param interItr:Initial value of the variable(s) exchanged between the iterations, if any 
    :param relax:   The relaxation factor
    :param Tol:     Tolerance
    :param maxitr:  Maximum number of iterations
    :param args:    arguments given to the residual and systems functions
    :return:        solution
    """
    solk = guess
    k = 1
    norm = 1
    normlist = np.ones((maxitr,), float)

    tryNewton = False

    newton = 0

    while norm > Tol and k < maxitr:

        solkm1 = solk
        if k % 100 == 0 or tryNewton:
            (Fx, interItr) = Res_fun(solk, interItr, *args)
            if newton % 3 == 0:
                Jac = Jacobian(Res_fun, solk, interItr, TypValue, *args)
            dx = np.linalg.solve(Jac, -Fx)
            solk = solkm1 + dx
            newton += 1
        else:
            (A, b, interItr) = sys_fun(solk, interItr, *args)
            solk = (1 - relax) * solkm1 + relax * np.linalg.solve(A, b)

        norm = np.linalg.norm(abs(solk - solkm1)) / np.linalg.norm(abs(solkm1))

        normlist[k] = norm

        # todo !!! Hack: Consider coverged if norm grows and last norm is greater than 2e-4
        if norm > normlist[k - 1] and normlist[k - 1] < 1e-4:
            break
        k = k + 1

        if k == maxitr:  # returns nan as solution if does not converge
            print('Picard iteration not converged after ' + repr(maxitr) + ' iterations')
            solk = np.full((len(solk),), np.nan, dtype=np.float64)
            return solk, None

    print('Successful: iterations = ' + repr(k) + ', exiting norm = ' + repr(norm))
    return (solk, interItr)


#######################################
# ARHHHHHHH DON T USE pointers !!!!!!

def Jacobian(Residual_function, x, interItr, TypValue, *args):
    (Fx, interItr) = Residual_function(x, interItr, *args)
    Jac = np.zeros((len(x), len(x)), dtype=np.float64)
    for i in range(0, len(x)):
        Epsilon = np.finfo(float).eps ** 0.5 * max(x[i], TypValue[i])
        xip = np.copy(x)
        # xin = np.copy(x)
        xip[i] = xip[i] + Epsilon
        # xin[i] = xin[i]-Epsilon
        (Fxi, interItr) = Residual_function(xip, interItr, *args)
        Jac[:, i] = (Fxi - Fx) / Epsilon
        # Jac[:,i] = (Residual_function(xip,interItr,*args)[0] - Residual_function(xin,interItr,*args)[0])/(2*Epsilon)
    return Jac
