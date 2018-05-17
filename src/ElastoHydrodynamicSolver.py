# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.Utility import *
from src.FluidModel import *
from scipy import sparse


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

    # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
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


def FiniteDiff_operator_turbulent_implicit(w, EltCrack, mu, Mesh, InCrack, rho, vkm1, C, sigma0, dgrain):
    """
    The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
    Dontsov and Peirce 2008). The matrix is evaluated by taking turbulence into account. The full evolution of friction
    factor as a function of Reynold's number and relative roughness is incorporated.

    Arguments:
        w (ndarray-float):              the width of the trial fracture. 
        EltCrack (ndarray-int):         the list of elements inside the fracture
        mu (ndarray-float):             the local viscosity of the injected fluid
        Mesh (CartesianMesh object):    the mesh
        InCrack (ndarray-int):          an array specifying whether elements are inside the fracture or not with
                                        1 or 0 respectively
        vkm1 (ndarray-float):           the velocity at cell edges from the previous iteration (if necessary). Here, it
                                        is used as the starting guess for the implicit solver.
        C (ndarray-float):              the elasticity matrix
        sigma0 (ndarrray-float):        the confining stress
        dgrain (float, default 1e-6)    the grain size. Used to get the relative roughness.
                
    Returns:
        ndarray-float:                  the finite difference matrix
        ndarray-float:                  the velocity evaluated for current iteration
    """

    FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)

    dx = Mesh.hx
    dy = Mesh.hy

    # todo: can be evaluated at each cell edge
    rough = w[EltCrack]/dgrain
    rough[np.where(rough < 3)[0]] = 3.

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
    ffLftEdge[ReLftEdge_nonZero] = friction_factor_vector(ReLftEdge[ReLftEdge_nonZero], rough[ReLftEdge_nonZero])
    ffRgtEdge[ReRgtEdge_nonZero] = friction_factor_vector(ReRgtEdge[ReRgtEdge_nonZero], rough[ReRgtEdge_nonZero])
    ffBtmEdge[ReBtmEdge_nonZero] = friction_factor_vector(ReBtmEdge[ReBtmEdge_nonZero], rough[ReBtmEdge_nonZero])
    ffTopEdge[ReTopEdge_nonZero] = friction_factor_vector(ReTopEdge[ReTopEdge_nonZero], rough[ReTopEdge_nonZero])

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
    f = friction_factor(Re,rough)

    return v-w*dp/(v*rho*f)

#-----------------------------------------------------------------------------------------------------------------------


def findBracket(func,guess,*args):
    """
    This function can be used to find bracket for a root finding algorithm.

    Arguments:
        func (callable function): the function giving the residual for which zero is to be found
        guess (float): starting guess
        args (tupple): arguments passed to the function

    Returns:
         float : the lower bracket
         float : the higher bracket
    """
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


# -----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_viscousFluid_sameFP(delw_k, inter_iter, *args ):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the same footprint as of the last time step. The viscosity of the injected fluid is taken into account by
    calculating fluid fluxes using Poiseuille Law (see Peirce and Detounay 2008, CMAME for details) or turbulent rough,
    friction factor based calculation.

    Arguments:
        delw_k (ndarray-float): the trial change in width for the current iteration of fracture front

        args (tupple): arguments passed to the function
            w (ndarray-float): fracture width from the last time step
            EltCrack (ndarray-int): list of elements in the fracture
            Q (float) fluid injection rate at the current time step
            C (ndarray-float): the elasticity matrix
            dt (float): the current time step
            muPrime (ndarray-float): 12 time viscosity of the injected fluid
            mesh (CartesianMesh object): the mesh
            InCrack (ndarray-float): an array with one for all the elements in the fracture and zero for rest
            LeakOff (ndarray-float): the leaked off fluid volume for each cell
            sigma0 (ndarray-float): the confining stress
            rho (float): density of the injected fluid
            turb (boolean): turbulence will be taken into account if true
            dgrain (float): the grain size of the rock. it will be used to calculate the fracture roughness

        inter_iter (ndarray-float ): the data passed between iterations if any.

    Returns:
        ndarray-float : the A matrix (in the system Ax=b) to be solved by a linear system solver
        ndarray-float : the b matrix (in the system Ax=b) to be solved by a linear system slover
        ndarray-float : the velocity at cell edges
    """

    (w, EltCrack, Q, C, dt, muPrime, mesh, InCrack, LeakOff, sigma0, rho, turb, dgrain) = args
    wnPlus1 = np.copy(w)
    wnPlus1[EltCrack] = wnPlus1[EltCrack] + delw_k

    if turb:
        (con, vk) = FiniteDiff_operator_turbulent_implicit(wnPlus1,
                                                           EltCrack,
                                                           muPrime / 12,
                                                           mesh,
                                                           InCrack,
                                                           rho,
                                                           inter_iter,
                                                           C,
                                                           sigma0,
                                                           dgrain)
    else:
        con = finiteDiff_operator_laminar(wnPlus1,
                                          EltCrack,
                                          muPrime,
                                          mesh,
                                          InCrack)
        vk = inter_iter

    con = sparse.csr_matrix(con[np.ix_(EltCrack, EltCrack)])
    CCrack = C[np.ix_(EltCrack, EltCrack)]

    A = np.identity(EltCrack.size) - dt * con.dot(CCrack)
    S = dt * con.dot(np.dot(CCrack, w[EltCrack]) + sigma0[EltCrack]) + dt * Q[EltCrack] / mesh.EltArea - LeakOff[
                                                                                            EltCrack] / mesh.EltArea
    return (A, S, vk)


#-----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_viscousFluid_extendedFP(solk, vkm1, *args):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly; see description of the
    ILSA algorithm) as of the last time step.

    Arguments:
        sol_k (ndarray-float)               -- the trial change in width and pressure for the current iteration of
                                               fracture front
        vkm1 (ndarray-float)                -- the velosity from the last iteration.

        args (tupple): arguments passed to the function
            EltChannel (ndarray-int)        -- list of channel elements
            EltsTipNew (ndarray-int)        -- list of new tip elements. This list also contains the elements that has
                                               been fully traversed.
            wLastTS (ndarray-float)         -- fracture width from the last time step
            wTip (ndarray-float)            -- fracture width in the tip elements
            EltCrack (ndarray-int)          -- list of elements in the fracture
            Mesh (CartesianMesh object):    -- the mesh
            dt (float)                      -- the current time step
            Q (float)                       -- fluid injection rate at the current time step
            C (ndarray-float)               -- the elasticity matrix
            muPrime (ndarray-float)         -- 12 time viscosity of the injected fluid
            rho (float)                     -- density of the injected fluid
            InCrack (ndarray-float)         -- an array with one for all the elements in the fracture and zero for rest
            LeakOff (ndarray-float)         -- the leaked off fluid volume for each cell
            sigma0 (ndarray-float)          -- the confining stress
            turb (boolean)                  -- turbulence will be taken into account if true
            dgrain (float)                  -- the grain size of the rock. it will be used to calculate the fracture
                                               roughness.

    Returns:
        A (ndarray-float)       -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        S (ndarray-float)       -- the b matrix (in the system Ax=b) to be solved by a linear system slover.
        vk (ndarray-float)      -- the velocity at cell edges.
    """

    (EltChannel, EltsTipNew, wLastTS, wTip, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff, sigma0,
     turb, dgrain) = args

    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltsTipNew)]

    A = np.zeros((EltChannel.size + EltsTipNew.size, EltChannel.size + EltsTipNew.size), dtype=np.float64)
    S = np.zeros((EltChannel.size + EltsTipNew.size,), dtype=np.float64)

    delwK = solk[np.arange(EltChannel.size)]
    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[EltChannel] = wcNplusOne[EltChannel] + delwK
    wcNplusOne[EltsTipNew] = wTip

    if turb:
        (FinDiffOprtr, vk) = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                    EltCrack,
                                                                    muPrime/12,
                                                                    Mesh,
                                                                    InCrack,
                                                                    rho,
                                                                    vkm1,
                                                                    C,
                                                                    sigma0,
                                                                    dgrain)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack)
        vk = vkm1

    condCC = sparse.csr_matrix(FinDiffOprtr[np.ix_(EltChannel, EltChannel)])
    condCT = FinDiffOprtr[np.ix_(EltChannel, EltsTipNew)]
    condTC = sparse.csr_matrix(FinDiffOprtr[np.ix_(EltsTipNew, EltChannel)])
    condTT = FinDiffOprtr[np.ix_(EltsTipNew, EltsTipNew)]

    Channel = np.arange(EltChannel.size)
    Tip = Channel.size + np.arange(EltsTipNew.size)

    A[np.ix_(Channel, Channel)] = np.identity(Channel.size) - dt * condCC.dot(Ccc)
    A[np.ix_(Channel, Tip)] = -dt * condCT
    A[np.ix_(Tip, Channel)] = -dt * condTC.dot(Ccc)
    A[np.ix_(Tip, Tip)] = -dt * condTT

    S[Channel] = dt * condCC.dot(np.dot(Ccc, wLastTS[EltChannel]) + np.dot(Cct, wTip) + sigma0[
        EltChannel]) + dt * Q[EltChannel] / Mesh.EltArea - LeakOff[EltChannel] / Mesh.EltArea
    S[Tip] = -(wTip - wLastTS[EltsTipNew]) + dt * condTC.dot(np.dot(Ccc, wLastTS[EltChannel]) + np.dot(Cct, wTip)
                                                         + sigma0[EltChannel]) - LeakOff[EltsTipNew] / Mesh.EltArea

    return (A, S, vk)

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_mechLoading_sameFP(w_LoadedElts, EltCrack, EltLoaded, C):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    so that the given width is imposed on the loaded elements (see Zia and Lecampion 2018).
    """

    C_Crack = C[np.ix_(EltCrack, EltCrack)]

    A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
    A = np.vstack((A, np.zeros((1, EltCrack.size + 1), dtype=np.float64)))
    A[-1, np.where(EltCrack == EltLoaded)[0]] = 1

    S = np.zeros((EltCrack.size + 1), dtype=np.float64)
    S[-1] = w_LoadedElts

    return A, S


#-----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_mechLoading_extendedFP(wTip, EltChannel, EltTip, C, EltLoaded, w_loaded):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The given width is
    imposed on the given loaded elements (see Zia and Lecampion 2018)
    """

    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc, -np.ones((EltChannel.size, 1),dtype=np.float64)))
    A = np.vstack((A,np.zeros((1,EltChannel.size+1),dtype=np.float64)))
    A[-1, np.where(EltChannel == EltLoaded)[0]] = 1

    S = - np.dot(Cct, wTip)
    S = np.append(S, w_loaded)

    return A, S

#-----------------------------------------------------------------------------------------------------------------------
def MakeEquationSystem_volumeControl_sameFP(w, EltCrack, C, dt, Q, ElemArea):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    so that the volume of the fracture is equal to the fluid injected into the fracture (see Zia and Lecampion 2018)
    """
    C_Crack = C[np.ix_(EltCrack, EltCrack)]

    A = np.hstack((C_Crack,-np.ones((EltCrack.size,1),dtype=np.float64)))
    A = np.vstack((A,np.ones((1,EltCrack.size+1),dtype=np.float64)))
    A[-1,-1] = 0

    S = -np.dot(C_Crack,w[EltCrack])
    S = np.append(S,Q * dt / ElemArea)

    return A, S

#-----------------------------------------------------------------------------------------------------------------------
def MakeEquationSystem_volumeControl_extendedFP(w_lst_tmstp, wTip, EltChannel, EltTip, C, dt, Q, ElemArea):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture (see Zia and Lecampion 2018).
    """
    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc,-np.ones((EltChannel.size,1),dtype=np.float64)))
    A = np.vstack((A, np.ones((1, EltChannel.size + 1), dtype=np.float64)))
    A[-1,-1] = 0

    S = -np.dot(Ccc,w_lst_tmstp[EltChannel]) - np.dot(Cct,wTip)
    S = np.append(S,Q * dt / ElemArea - (sum(wTip)-sum(w_lst_tmstp[EltTip])))

    return A, S


#-----------------------------------------------------------------------------------------------------------------------

# def MakeEquationSystem_volumeControl_extendedFP_width(wTip, EltChannel, EltTip, C, time, Q, ElemArea):
#
#     Ccc = C[np.ix_(EltChannel, EltChannel)]
#     Cct = C[np.ix_(EltChannel, EltTip)]
#
#     A = np.hstack((Ccc,-np.ones((EltChannel.size,1),dtype=np.float64)))
#     A = np.vstack((A, np.ones((1, EltChannel.size + 1), dtype=np.float64)))
#     A[-1,-1] = 0
#
#     S = - np.dot(Cct,wTip)
#     S = np.append(S,Q * time / ElemArea - sum(wTip))
#
#     return A, S


#-----------------------------------------------------------------------------------------------------------------------

def Elastohydrodynamic_ResidualFun_sameFP(solk, interItr, *args):
    """
    This function gives the residual of the solution of the current iteration for the viscous fluid, same footprint
    case.
    """
    (A, S, vk) = MakeEquationSystem_viscousFluid_sameFP(solk, interItr, *args)
    return (np.dot(A, solk) - S, vk)


#-----------------------------------------------------------------------------------------------------------------------

def Elastohydrodynamic_ResidualFun_ExtendedFP(solk, *args, interItr=None):
    """
    This function gives the residual of the solution of the current iteration for the viscous fluid, extended footprint
    case.
    """
    (A, S, vk) = MakeEquationSystem_viscousFluid_extendedFP(solk, interItr, *args)
    return (np.dot(A, solk) - S, vk)


#-----------------------------------------------------------------------------------------------------------------------
def velocity(w, EltCrack, Mesh, InCrack, muPrime, C, sigma0):
    """
    This function gives the velocity at the cell edges evaluated using the Poiseuille flow assumption.
    """
    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)

    # velocity at the edges in the following order (x-left edge, x-right edge, y-bottom edge, y-top edge, y-left edge,
    #                                               y-right edge, x-bottom edge, x-top edge)
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

    vel_magnitude = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)
    vel_magnitude[0, :] = (vel[0, :] ** 2 + vel[4, :] ** 2) ** 0.5
    vel_magnitude[1, :] = (vel[1, :] ** 2 + vel[5, :] ** 2) ** 0.5
    vel_magnitude[2, :] = (vel[2, :] ** 2 + vel[6, :] ** 2) ** 0.5
    vel_magnitude[3, :] = (vel[3, :] ** 2 + vel[7, :] ** 2) ** 0.5

    return vel_magnitude


#-----------------------------------------------------------------------------------------------------------------------

def pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack):
    """
    This function gives the pressure gradient at the cell edges evaluated with the pressure calculated from the
    elasticity relation for the given fracture width.
    """
    pf = np.zeros((Mesh.NumberOfElts,), dtype=np.float64)
    # pf[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], w[EltCrack]) + sigma0[EltCrack]
    pf = np.dot(C, w) + sigma0

    dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) * InCrack[Mesh.NeiElements[EltCrack, 0]]
    dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 1]]
    dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) * InCrack[Mesh.NeiElements[EltCrack, 2]]
    dpdyTop = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 3]]

    return (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop)


#-----------------------------------------------------------------------------------------------------------------------


def Picard_Newton(Res_fun, sys_fun, guess, TypValue, interItr, Tol, maxitr, *args, relax=1.0, PicardPerNewton = 100,
                  perf_node=None):
    """
    Mixed Picard Newton solver for nonlinear systems.
    Arguments:
        Res_fun (function)      -- The function calculating the residual.
        sys_fun (function)      -- The function giving the system A,b for the Picard solver to solve the linear system
                                   of the form Ax=b.
        guess (ndarray)         -- The initial guess.
        TypValue (ndarray)      -- Typical value of the variable to estimate the Epsilon to calculate Jacobian.
        interItr (ndarray)      -- Initial value of the variable(s) exchanged between the iterations, if any.
        relax (float)           -- The relaxation factor.
        Tol (float)             -- Tolerance.
        maxitr (int):           -- Maximum number of iterations.
        args (tuple)            -- arguments given to the residual and systems functions.
        PicardPerNewton (int)   -- For hybrid Picard/Newton solution. Number of picard iterations for every Newton
                                   iteration.
    Returns:
            solk (ndarray)      -- solution at the end of iteration.
            interItr            -- any data passed between iterations.
    """
    solk = guess
    k = 1
    norm = 1
    normlist = []

    tryNewton = False

    newton = 0

    while norm > Tol and k < maxitr:

        solkm1 = solk
        if k % PicardPerNewton == 0 or tryNewton:
            (Fx, interItr) = Res_fun(solk, interItr, *args)
            if newton % 3 == 0:
                Jac = Jacobian(Res_fun, solk, TypValue, interItr, *args)
            dx = np.linalg.solve(Jac, -Fx)
            solk = solkm1 + dx
            newton += 1
        else:
            try:
                (A, b, interItr) = sys_fun(solk, interItr, *args)
                solk = (1 - relax) * solkm1 + relax * np.linalg.solve(A, b)
            except np.linalg.linalg.LinAlgError:
                print('singlular matrix!')
                solk = np.full((len(solk),), np.nan, dtype=np.float64)
                return solk, None


        norm = np.linalg.norm(abs(solk - solkm1)) / np.linalg.norm(abs(solkm1))

        normlist.append(norm)

        # todo !!! Hack: Consider coverged if norm grows and last norm is less than 1e-4
        # if norm > normlist[k - 1] and normlist[k - 1] < 1e-4:
        #     break
        k = k + 1

        if k == maxitr:  # returns nan as solution if does not converge
            print('Picard iteration not converged after ' + repr(maxitr) + ' iterations, norm:' + repr(norm))
            solk = np.full((len(solk),), np.nan, dtype=np.float64)
            return solk, None

        if perf_node is not None:
            perf_node.iterations = k - 1
            perf_node.normList = normlist

    return solk, interItr


#-----------------------------------------------------------------------------------------------------------------------


def Jacobian(Residual_function, x, TypValue, *args, central=False, interItr=None):
    """
    This function returns the Jacobian of the given function.
    """

    (Fx, interItr) = Residual_function(x, interItr, *args)
    Jac = np.zeros((len(x), len(x)), dtype=np.float64)
    for i in range(0, len(x)):
        Epsilon = np.finfo(float).eps ** 0.5 * max(x[i], TypValue[i])
        xip = np.copy(x)
        xip[i] = xip[i] + Epsilon
        if central:
            xin = np.copy(x)
            xin[i] = xin[i]-Epsilon
            Jac[:,i] = (Residual_function(xip,interItr,*args)[0] - Residual_function(xin,interItr,*args)[0])/(2*Epsilon)
        else:
            (Fxi, interItr) = Residual_function(xip, interItr, *args)
            Jac[:, i] = (Fxi - Fx) / Epsilon

    return Jac
