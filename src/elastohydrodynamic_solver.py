# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

from scipy import sparse
from scipy.optimize import brentq
import numpy as np

#local imports
from fluid_model import friction_factor_vector, friction_factor_MDR
from properties import instrument_start, instrument_close


def finiteDiff_operator_laminar(w, EltCrack, muPrime, Mesh, InCrack, neiInCrack, sparse_flag=False):
    """
    The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
    equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated with the laminar flow assumption.

    Args:
        w (ndarray):            -- the width of the trial fracture.
        EltCrack (ndarray):     -- the list of elements inside the fracture.
        muPrime (ndarray):      -- the scaled local viscosity of the injected fluid (12 * viscosity).
        Mesh (CartesianMesh):   -- the mesh.
        InCrack (ndarray):      -- an array specifying whether elements are inside the fracture or not with
                                   1 or 0 respectively.
        neiInCrack (ndarray):   -- an ndarray giving indices of the neighbours of all the cells in the crack, in the
                                   EltCrack list.
        sparse_flag (bool):     -- if true, the finite difference operator will be given as a sparse matrix.

    Returns:
        FinDiffOprtr (ndarray): -- the finite difference matrix.

    """

    if sparse_flag:
        FinDiffOprtr = sparse.lil_matrix((len(EltCrack), len(EltCrack)+1), dtype=np.float64)
    else:
        FinDiffOprtr = np.zeros((len(EltCrack), len(EltCrack)+1), dtype=np.float64)

    dx = Mesh.hx
    dy = Mesh.hy

    # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
    wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 0]]
    wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 1]]
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 2]]
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 3]]

    indx_elts = np.arange(len(EltCrack))
    FinDiffOprtr[indx_elts, indx_elts] = (-(wLftEdge ** 3 + wRgtEdge ** 3) / dx ** 2 - (
                                            wBtmEdge ** 3 + wTopEdge ** 3) / dy ** 2) / muPrime[EltCrack]
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 0]] = wLftEdge ** 3 / dx ** 2 / muPrime[EltCrack]
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = wRgtEdge ** 3 / dx ** 2 / muPrime[EltCrack]
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = wBtmEdge ** 3 / dy ** 2 / muPrime[EltCrack]
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = wTopEdge ** 3 / dy ** 2 / muPrime[EltCrack]

    return FinDiffOprtr


#-----------------------------------------------------------------------------------------------------------------------

def Gravity_term(w, EltCrack, muPrime, Mesh, InCrack, density):
    """
    This function returns the gravity term (G in Zia and Lecampion 2019).

    Args:
        w (ndarray):                -- the width of the trial fracture.
        EltCrack (ndarray):         -- the list of elements inside the fracture.
        muPrime (ndarray):          -- the scaled local viscosity of the injected fluid (12 * viscosity)
        Mesh (CartesianMesh):       -- the mesh.
        InCrack (ndarray):          -- An array specifying whether elements are inside the fracture or not with
                                       1 or 0 respectively.
        density (float):            -- the density of the fluid.

    Returns:
        G (ndarray):                -- the matrix with the gravity terms.
    """

    G = np.zeros((Mesh.NumberOfElts,), dtype=np.float64)

    # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 2]]
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 3]]

    G[EltCrack] = density * 9.81 * (wTopEdge ** 3 - wBtmEdge ** 3) / Mesh.hy / muPrime[EltCrack]

    return G

#-----------------------------------------------------------------------------------------------------------------------


def FiniteDiff_operator_turbulent_implicit(w, EltCrack, mu, Mesh, InCrack, rho, vkm1, C, sigma0, dgrain, to_solve,
                                           active, to_impose, sparse_flag=False):
    """
    The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
    Dontsov and Peirce 2008). The matrix is evaluated by taking turbulence into account.

    Args:
        w (ndarray):                -- the width of the trial fracture.
        EltCrack (ndarray):         -- the list of elements inside the fracture
        mu (ndarray):               -- the local viscosity of the injected fluid
        Mesh (CartesianMesh):       -- the mesh.
        InCrack (ndarray):          -- an array specifying whether elements are inside the fracture or not with
                                       1 or 0 respectively.
        rho (float):                -- density of the fluid.
        vkm1 (ndarray):             -- the velocity at cell edges from the previous iteration (if necessary). Here,
                                       it is used as the starting guess for the implicit solver.
        C (ndarray):                -- the elasticity matrix.
        sigma0 (ndarrray):          -- the confining stress.
        dgrain (float):             -- the grain size. Used to get the relative roughness.
        to_solve (ndarray):         -- the channel elements to be solved.
        active (ndarray):           -- the channel elements where width constraint is active.
        to_impose (ndarray):        -- the tip elements to be imposed.
        sparse_flag (bool):         -- if true, the finite difference operator will be given as a sparse matrix.
                
    Returns:
        - FinDiffOprtr (ndarray)    -- the finite difference matrix.
        - vk (ndarray)              -- the velocity evaluated for current iteration.
    """

    if sparse_flag:
        FinDiffOprtr = sparse.lil_matrix((w.size, w.size), dtype=np.float64)
    else:
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
    # 0 - left edge in x-direction    # 1 - right edge in x-direction
    # 2 - bottom edge in y-direction  # 3 - top edge in y-direction
    # 4 - left edge in y-direction    # 5 - right edge in y-direction
    # 6 - bottom edge in x-direction  # 7 - top edge in x-direction

    dp = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)
    dp[0, EltCrack] = dpdxLft
    dp[1, EltCrack] = dpdxRgt
    dp[2, EltCrack] = dpdyBtm
    dp[3, EltCrack] = dpdyTop
    # linear interpolation for pressure gradient on the edges where central difference not available
    dp[4, EltCrack] = (dp[2,Mesh.NeiElements[EltCrack,0]]+dp[3,Mesh.NeiElements[EltCrack,0]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[5, EltCrack] = (dp[2,Mesh.NeiElements[EltCrack,1]]+dp[3,Mesh.NeiElements[EltCrack,1]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[6, EltCrack] = (dp[0,Mesh.NeiElements[EltCrack,2]]+dp[1,Mesh.NeiElements[EltCrack,2]]+dp[0,EltCrack]+dp[1,EltCrack])/4
    dp[7, EltCrack] = (dp[0,Mesh.NeiElements[EltCrack,3]]+dp[1,Mesh.NeiElements[EltCrack,3]]+dp[0,EltCrack]+dp[1,EltCrack])/4

    # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
    dpLft = (dp[0, EltCrack] ** 2 + dp[4, EltCrack] ** 2) ** 0.5
    dpRgt = (dp[1, EltCrack] ** 2 + dp[5, EltCrack] ** 2) ** 0.5
    dpBtm = (dp[2, EltCrack] ** 2 + dp[6, EltCrack] ** 2) ** 0.5
    dpTop = (dp[3, EltCrack] ** 2 + dp[7, EltCrack] ** 2) ** 0.5

    vk = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    # the factor to be multiplied to the velocity from last iteration to get the upper bracket
    upBracket_factor = 10

    # loop to calculate velocity on each cell edge implicitly
    for i in range(0, len(EltCrack)):
        # todo !!! Hack. zero velocity if the pressure gradient is zero or very small width
        if dpLft[i] < 1e-8 or wLftEdge[i] < 1e-10:
            vk[0, EltCrack[i]] = 0.0
        else:
            arg = (wLftEdge[i], mu[EltCrack[i]], rho, dpLft[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[0, EltCrack[i]], *arg) * Velocity_Residual(
                            upBracket_factor * vkm1[0, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[0, EltCrack[i]], *arg)
                vk[0, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[0, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[0, EltCrack[i]],
                                            upBracket_factor * vkm1[0, EltCrack[i]], arg)

        if dpRgt[i] < 1e-8 or wRgtEdge[i] < 1e-10:
            vk[1, EltCrack[i]] = 0.0
        else:
            arg = (wRgtEdge[i], mu[EltCrack[i]], rho, dpRgt[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[1, EltCrack[i]], *arg) * Velocity_Residual(
                            upBracket_factor * vkm1[1, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[1, EltCrack[i]], *arg)
                vk[1, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[1, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[1, EltCrack[i]],
                                            upBracket_factor * vkm1[1, EltCrack[i]], arg)

        if dpBtm[i] < 1e-8 or wBtmEdge[i] < 1e-10:
            vk[2, EltCrack[i]] = 0.0
        else:
            arg = (wBtmEdge[i], mu[EltCrack[i]], rho, dpBtm[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[2, EltCrack[i]], *arg) * Velocity_Residual(
                            upBracket_factor * vkm1[2, EltCrack[i]], *arg) > 0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[2, EltCrack[i]], *arg)
                vk[2, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[2, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[2, EltCrack[i]],
                                            upBracket_factor * vkm1[2, EltCrack[i]], arg)

        if dpTop[i] < 1e-8 or wTopEdge[i] < 1e-10:
            vk[3, EltCrack[i]] = 0.0
        else:
            arg = (wTopEdge[i], mu[EltCrack[i]], rho, dpTop[i], rough[i])
            # check if bracket gives residuals with opposite signs
            if Velocity_Residual(np.finfo(float).eps * vkm1[3, EltCrack[i]],*arg)*Velocity_Residual(
                            upBracket_factor * vkm1[3, EltCrack[i]],*arg)>0.0:
                # bracket not valid. finding suitable bracket
                (a, b) = findBracket(Velocity_Residual, vkm1[3, EltCrack[i]], *arg)
                vk[3, EltCrack[i]] = brentq(Velocity_Residual, a, b, arg)
            else:
                # find the root with brentq method.
                vk[3, EltCrack[i]] = brentq(Velocity_Residual, np.finfo(float).eps * vkm1[3, EltCrack[i]],
                                            upBracket_factor * vkm1[3, EltCrack[i]], arg)

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
    ffLftEdge = np.zeros(EltCrack.size, dtype=np.float64)
    ffRgtEdge = np.zeros(EltCrack.size, dtype=np.float64)
    ffBtmEdge = np.zeros(EltCrack.size, dtype=np.float64)
    ffTopEdge = np.zeros(EltCrack.size, dtype=np.float64)
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

    ch_indxs = np.arange(len(to_solve))
    act_indxs = len(to_solve) + np.arange(len(active))
    tip_indxs = len(to_solve) + len(active) + np.arange(len(to_impose))

    indx_elts = np.arange(len(EltCrack))
    FD_compressed = np.zeros((len(EltCrack), len(EltCrack)), dtype=np.float64)
    FD_compressed[np.ix_(indx_elts, ch_indxs)] = FinDiffOprtr[np.ix_(EltCrack, to_solve)]
    FD_compressed[np.ix_(indx_elts, act_indxs)] = FinDiffOprtr[np.ix_(EltCrack, active)]
    FD_compressed[np.ix_(indx_elts, tip_indxs)] = FinDiffOprtr[np.ix_(EltCrack, to_impose)]

    return FD_compressed, vk

#-----------------------------------------------------------------------------------------------------------------------


def Velocity_Residual(v,*args):
    """
    This function gives the residual of the velocity equation. It is used by the root finder.

    Args:
        v (float):      -- current velocity guess
        args (tuple):   -- a tuple consisting of the following:

                            - w (float)          width at the given cell edge
                            - mu (float)         viscosity at the given cell edge
                            - rho (float)        density of the injected fluid
                            - dp (float)         pressure gradient at the given cell edge
                            - rough (float)      roughness (width / grain size) at the cell center

    Returns:
         float:       -- residual of the velocity equation
    """
    (w, mu, rho, dp, rough) = args

    # Reynolds number
    Re = 4/3 * rho * w * v / mu

    # friction factor using MDR approximation
    f = friction_factor_MDR(Re, rough)

    return v-w*dp/(v*rho*f)

#-----------------------------------------------------------------------------------------------------------------------


def findBracket(func, guess,*args):
    """
    This function can be used to find bracket for a root finding algorithm.

    Args:
        func (callable function):   -- the function giving the residual for which zero is to be found
        guess (float):              -- starting guess
        args (tupple):              -- arguments passed to the function

    Returns:
         - a (float)                -- the lower bracket
         - b (float)                -- the higher bracket
    """
    a = np.finfo(float).eps * guess
    b = max(1000*guess,1)
    Res_a = func(a, *args)
    Res_b = func(b, *args)

    cnt = 0
    while Res_a * Res_b > 0:
        b = 10 * b
        Res_b = func(b, *args)
        cnt += 1
        if cnt >= 60:
            raise SystemExit('Velocity bracket cannot be found')

    return a, b

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_ViscousFluid(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm) as of the last time step. The cells where width constraint is active are solved
    for traction and pressure separately.

    Arguments:
        solk (ndarray):                -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - EltsTipNew (ndarray)          -- list of new tip elements. This list also contains the elements that has\
                                                 been fully traversed.
            - wLastTS (ndarray)             -- fracture width from the last time step.
            - wTip (ndarray)                -- fracture width in the tip elements.
            - EltCrack (ndarray)            -- list of elements in the fracture.
            - Mesh (CartesianMesh object)   -- the mesh.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - muPrime (ndarray)             -- 12 time viscosity of the injected fluid.
            - rho (float)                   -- density of the injected fluid.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - sigma0 (ndarray)              -- the confining stress.
            - turb (boolean)                -- turbulence will be taken into account if true.
            - dgrain (float)                -- the grain size of the rock. it will be used to calculate the fracture\
                                               roughness.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - wc_to_impose (ndarray)        -- the critical minimum width to be imposed in the active width constraint \
                                               cells.
            - wc (float)                    -- the critical minimum width for the material.
            - cf (float)                    -- fluid compressibility.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (tuple)   -- the information transferred between iterations.
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for channel, tip and active width constraint cells.

    """

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[to_solve] += solk[:len(to_solve)]
    wcNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wcNplusOne[active] = wc_to_impose
    wcNplusOne[np.where(wcNplusOne < wc)[0]] = wc
    vkm1 = interItr

    if turb:
        FinDiffOprtr, interItr_kp1 = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                    EltCrack, muPrime / 12, Mesh,
                                                                    InCrack, rho, vkm1, C, sigma0,
                                                                    dgrain, to_solve, active, to_impose)

    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack,
                                                   neiInCrack)
        interItr_kp1 = vkm1

    if gravity:
        G = Gravity_term(wcNplusOne,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)

    else:
        G = np.zeros((Mesh.NumberOfElts,))

    LeakOff_cp = np.copy(LeakOff)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_w = n_ch + n_act
    n_p = n_ch + n_act + n_tip
    n_total = n_w + n_p

    ch_w_row_no = np.arange(n_ch)
    act_w_row_no = n_ch + np.arange(n_act)
    ch_p_row_no = n_w + np.arange(n_ch)
    act_p_row_no = n_w + n_ch + np.arange(n_act)
    tip_p_row_no = n_w + n_ch + n_act + np.arange(n_tip)

    ch_w_col_no = np.arange(n_ch)
    act_tr_col_no = n_ch + np.arange(n_act)
    ch_p_col_no = n_w + np.arange(n_ch)
    act_p_col_no = n_w + n_ch + np.arange(n_act)
    tip_p_col_no = n_w + n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    A[np.ix_(ch_w_row_no, ch_w_col_no)] = C[np.ix_(to_solve, to_solve)]
    A[ch_w_row_no, ch_p_col_no] = -1.

    A[np.ix_(act_w_row_no, ch_w_col_no)] = C[np.ix_(active, to_solve)]
    A[act_w_row_no, act_tr_col_no] = -1.

    A[ch_p_row_no, ch_w_col_no] = 1.
    A[np.ix_(ch_p_row_no, ch_p_col_no)] = -dt * FinDiffOprtr[to_solve, :][:, to_solve].toarray()
    A[np.ix_(ch_p_row_no, act_p_col_no)] = -dt * FinDiffOprtr[to_solve, :][:, active].toarray()
    A[np.ix_(ch_p_row_no, tip_p_col_no)] = -dt * FinDiffOprtr[to_solve, :][:, to_impose].toarray()

    A[np.ix_(act_p_row_no, ch_p_col_no)] = -dt * FinDiffOprtr[active, :][:, to_solve].toarray()
    A[np.ix_(act_p_row_no, act_p_col_no)] = -dt * FinDiffOprtr[active, :][:, active].toarray()
    A[np.ix_(act_p_row_no, tip_p_col_no)] = -dt * FinDiffOprtr[active, :][:, to_impose].toarray()

    A[np.ix_(tip_p_row_no, ch_p_col_no)] = -dt * FinDiffOprtr[to_impose, :][:, to_solve].toarray()
    A[np.ix_(tip_p_row_no, act_p_col_no)] = -dt * FinDiffOprtr[to_impose, :][:, active].toarray()
    A[np.ix_(tip_p_row_no, tip_p_col_no)] = -dt * FinDiffOprtr[to_impose, :][:, to_impose].toarray()

    S = np.zeros((n_total, ), dtype=np.float64)

    S[ch_w_row_no] = - sigma0[to_solve] - \
                        np.dot(C[np.ix_(to_solve, EltCrack)], wLastTS[EltCrack]) - \
                        np.dot(C[np.ix_(to_solve, to_impose)], imposed_val - wLastTS[to_impose]) - \
                        np.dot(C[np.ix_(to_solve, active)], wc - wLastTS[active])

    S[act_w_row_no] = - sigma0[active] - \
                         np.dot(C[np.ix_(active, EltCrack)], wLastTS[EltCrack]) - \
                         np.dot(C[np.ix_(active, to_impose)], imposed_val - wLastTS[to_impose]) - \
                         np.dot(C[np.ix_(active, active)], wc - wLastTS[active])
                         # + pfLastTS[to_solve]

    S[ch_p_row_no] = dt * Q[to_solve] / Mesh.EltArea - LeakOff_cp[to_solve] / Mesh.EltArea + \
                        dt * G[to_solve]# + dt * cond.dot(pfLastTS[EltCrack_R])

    S[act_p_row_no] = dt * Q[active] / Mesh.EltArea - LeakOff_cp[active] / Mesh.EltArea - \
                      (wc - wLastTS[active]) + dt * G[active]  # + dt * cond.dot(pfLastTS[EltCrack_R])

    S[tip_p_row_no] = dt * Q[to_impose] / Mesh.EltArea - LeakOff_cp[to_impose] / Mesh.EltArea - \
                        (imposed_val - wLastTS[to_impose]) + dt * G[to_impose]  # + dt * cond.dot(pfLastTS[EltCrack_R])

    # indices of solved width, pressure and traction in the solution
    indices = [ch_w_row_no,
               np.concatenate((ch_p_row_no, act_p_row_no, tip_p_row_no)),
               act_w_row_no]

    return A, S, interItr_kp1, indices


#-----------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted_sparse(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The finite difference
    difference opertator is saved as a sparse matrix. The system is assembled with the extended footprint (treating the
    channel and the extended tip elements distinctly; see description of the ILSA algorithm). The pressure in the tip
    cells and the cells where width constraint is active are solved separately. The pressure in the channel cells to be
    solved for change in width is substituted with width using the elasticity relation (see Zia and Lecamption 2019).

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - EltsTipNew (ndarray)          -- list of new tip elements. This list also contains the elements that has\
                                                 been fully traversed.
            - wLastTS (ndarray)             -- fracture width from the last time step.
            - wTip (ndarray)                -- fracture width in the tip elements.
            - EltCrack (ndarray)            -- list of elements in the fracture.
            - Mesh (CartesianMesh object)   -- the mesh.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - muPrime (ndarray)             -- 12 time viscosity of the injected fluid.
            - rho (float)                   -- density of the injected fluid.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - sigma0 (ndarray)              -- the confining stress.
            - turb (boolean)                -- turbulence will be taken into account if true.
            - dgrain (float)                -- the grain size of the rock. it will be used to calculate the fracture\
                                               roughness.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - wc_to_impose (ndarray)        -- the critical minimum width to be imposed in the active width constraint \
                                               cells.
            - wc (float)                    -- the critical minimum width for the material.
            - cf (float)                    -- fluid compressibility.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (tuple)   -- the information transferred between iterations.
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                   obtained for channel, tip and active width constraint cells.
    """

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[to_solve] += solk[:len(to_solve)]
    wcNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wcNplusOne[active] = wc_to_impose

    below_wc = np.where(wcNplusOne[to_solve] < wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wcNplusOne[to_solve[below_wc]] = wc
    vkm1 = interItr[0]

    wcNplusHalf = (wLastTS + wcNplusOne) / 2

    if turb:
        FinDiffOprtr, interItr_kp1 = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                            EltCrack, muPrime / 12, Mesh,
                                                                            InCrack, rho, vkm1, C, sigma0,
                                                                            dgrain, to_solve, active, to_impose)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack,
                                                   neiInCrack,
                                                   sparse_flag=True)
        vk = vkm1

    if gravity:
        G = Gravity_term(wcNplusOne,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)

    else:
        G = np.zeros((Mesh.NumberOfElts,))

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), cf * wcNplusHalf[to_solve])], [0], format='csr')

    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)
    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), cf * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), cf * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], wLastTS[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wcNplusOne[active]) + \
                  sigma0[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / Mesh.EltArea - \
                  LeakOff[to_solve] / Mesh.EltArea + \
                  cf * wcNplusHalf[to_solve] * pfLastTS[to_solve]
    S[tip_indxs] = -(imposed_val - wLastTS[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   cf * wcNplusHalf[to_impose] * pfLastTS[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / Mesh.EltArea - LeakOff[to_impose] / Mesh.EltArea
    S[act_indxs] = -(wc_to_impose - wLastTS[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   cf * wcNplusHalf[active] * pfLastTS[active] + \
                   dt * G[active] + \
                   dt * Q[active] / Mesh.EltArea - LeakOff[active] / Mesh.EltArea

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs]
    interItr_kp1 = (vk, below_wc)

    return A, S, interItr_kp1, indices


#-----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
    is saved as a sparse matrix.

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - EltsTipNew (ndarray)          -- list of new tip elements. This list also contains the elements that has\
                                                 been fully traversed.
            - wLastTS (ndarray)             -- fracture width from the last time step.
            - wTip (ndarray)                -- fracture width in the tip elements.
            - EltCrack (ndarray)            -- list of elements in the fracture.
            - Mesh (CartesianMesh object)   -- the mesh.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - muPrime (ndarray)             -- 12 time viscosity of the injected fluid.
            - rho (float)                   -- density of the injected fluid.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - sigma0 (ndarray)              -- the confining stress.
            - turb (boolean)                -- turbulence will be taken into account if true.
            - dgrain (float)                -- the grain size of the rock. it will be used to calculate the fracture\
                                               roughness.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - wc_to_impose (ndarray)        -- the critical minimum width to be imposed in the active width constraint \
                                               cells.
            - wc (float)                    -- the critical minimum width for the material.
            - cf (float)                    -- fluid compressibility.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (tuple)   -- the information transferred between iterations.
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for channel, tip and active width constraint cells.
    """

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[to_solve] += solk[:len(to_solve)]
    wcNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wcNplusOne[active] = wc_to_impose

    below_wc = np.where(wcNplusOne[to_solve] < wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wcNplusOne[to_solve[below_wc]] = wc
    vkm1 = interItr[0]

    wcNplusHalf = (wLastTS + wcNplusOne) / 2

    if turb:
        FinDiffOprtr, interItr_kp1 = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                            EltCrack, muPrime / 12, Mesh,
                                                                            InCrack, rho, vkm1, C, sigma0,
                                                                            dgrain, to_solve, active, to_impose)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack,
                                                   neiInCrack,
                                                   sparse_flag=True)
        vk = vkm1

    if gravity:
        G = Gravity_term(wcNplusOne,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)

    else:
        G = np.zeros((Mesh.NumberOfElts,))

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), cf * wcNplusHalf[to_solve])], [0], format='csr')

    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), cf * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), cf * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], wLastTS[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wcNplusOne[active]) + \
                  sigma0[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).dot(pfLastTS[to_impose]) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).dot(pfLastTS[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / Mesh.EltArea - LeakOff[to_solve] / Mesh.EltArea \
                  + cf * wcNplusHalf[to_solve] * pfLastTS[to_solve]

    S[tip_indxs] = -(imposed_val - wLastTS[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs]).dot(pfLastTS[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).dot(pfLastTS[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / Mesh.EltArea - LeakOff[to_impose] / Mesh.EltArea

    S[act_indxs] = -(wc_to_impose - wLastTS[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).dot(pfLastTS[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs]).dot(pfLastTS[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / Mesh.EltArea - LeakOff[active] / Mesh.EltArea

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs]
    interItr_kp1 = (vk, below_wc)

    return A, S, interItr_kp1, indices


# -----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The pressure in the tip cells and the cells where width constraint is active
    are solved separately. The pressure in the channel cells to be solved for change in width is substituted with width
    using the elasticity relation (see Zia and Lecampion 2019).

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - EltsTipNew (ndarray)          -- list of new tip elements. This list also contains the elements that has\
                                                 been fully traversed.
            - wLastTS (ndarray)             -- fracture width from the last time step.
            - wTip (ndarray)                -- fracture width in the tip elements.
            - EltCrack (ndarray)            -- list of elements in the fracture.
            - Mesh (CartesianMesh object)   -- the mesh.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - muPrime (ndarray)             -- 12 time viscosity of the injected fluid.
            - rho (float)                   -- density of the injected fluid.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - sigma0 (ndarray)              -- the confining stress.
            - turb (boolean)                -- turbulence will be taken into account if true.
            - dgrain (float)                -- the grain size of the rock. it will be used to calculate the fracture\
                                               roughness.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - wc_to_impose (ndarray)        -- the critical minimum width to be imposed in the active width constraint \
                                               cells.
            - wc (float)                    -- the critical minimum width for the material.
            - cf (float)                    -- fluid compressibility.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (tuple)   -- the information transferred between iterations.
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for channel, tip and active width constraint cells.
    """

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[to_solve] += solk[:len(to_solve)]
    wcNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wcNplusOne[active] = wc_to_impose

    below_wc = np.where(wcNplusOne[to_solve] < wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wcNplusOne[to_solve[below_wc]] = wc
    vkm1 = interItr[0]

    wcNplusHalf = (wLastTS + wcNplusOne) / 2

    if turb:
        FinDiffOprtr, interItr_kp1 = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                         EltCrack, muPrime / 12, Mesh,
                                                                         InCrack, rho, vkm1, C, sigma0,
                                                                         dgrain, to_solve, active, to_impose)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack,
                                                   neiInCrack)

        vk = vkm1

    if gravity:
        G = Gravity_term(wcNplusOne,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)

    else:
        G = np.zeros((Mesh.NumberOfElts,))

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= cf * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += cf * wcNplusHalf[to_impose]

    A[np.ix_(tip_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += cf * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], wLastTS[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wcNplusOne[active]) + \
                  sigma0[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / Mesh.EltArea - \
                  LeakOff[to_solve] / Mesh.EltArea + \
                  cf * wcNplusHalf[to_solve] * pfLastTS[to_solve]
    S[tip_indxs] = -(imposed_val - wLastTS[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   cf * wcNplusHalf[to_impose] * pfLastTS[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / Mesh.EltArea - LeakOff[to_impose] / Mesh.EltArea
    S[act_indxs] = -(wc_to_impose - wLastTS[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   cf * wcNplusHalf[active] * pfLastTS[active] + \
                   dt * G[active] + \
                   dt * Q[active] / Mesh.EltArea - LeakOff[active] / Mesh.EltArea

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs]
    interItr_kp1 = (vk, below_wc)

    return A, S, interItr_kp1, indices


#-----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019).

    Arguments:
        solk (ndarray):                -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - EltsTipNew (ndarray)          -- list of new tip elements. This list also contains the elements that has\
                                                 been fully traversed.
            - wLastTS (ndarray)             -- fracture width from the last time step.
            - wTip (ndarray)                -- fracture width in the tip elements.
            - EltCrack (ndarray)            -- list of elements in the fracture.
            - Mesh (CartesianMesh object)   -- the mesh.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - muPrime (ndarray)             -- 12 time viscosity of the injected fluid.
            - rho (float)                   -- density of the injected fluid.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - sigma0 (ndarray)              -- the confining stress.
            - turb (boolean)                -- turbulence will be taken into account if true.
            - dgrain (float)                -- the grain size of the rock. it will be used to calculate the fracture\
                                               roughness.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - wc_to_impose (ndarray)        -- the critical minimum width to be imposed in the active width constraint \
                                               cells.
            - wc (float)                    -- the critical minimum width for the material.
            - cf (float)                    -- fluid compressibility.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (tuple)   -- the information transferred between iterations.
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for channel, tip and active width constraint cells.
    """

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    wcNplusOne = np.copy(wLastTS)
    wcNplusOne[to_solve] += solk[:len(to_solve)]
    wcNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wcNplusOne[active] = wc_to_impose

    below_wc = np.where(wcNplusOne[to_solve] < wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wcNplusOne[to_solve[below_wc]] = wc
    vkm1 = interItr[0]

    wcNplusHalf = (wLastTS + wcNplusOne) / 2

    if turb:
        FinDiffOprtr, interItr_kp1 = FiniteDiff_operator_turbulent_implicit(wcNplusOne,
                                                                            EltCrack, muPrime / 12, Mesh,
                                                                            InCrack, rho, vkm1, C, sigma0,
                                                                            dgrain, to_solve, active, to_impose)
    else:
        FinDiffOprtr = finiteDiff_operator_laminar(wcNplusOne,
                                                   EltCrack,
                                                   muPrime,
                                                   Mesh,
                                                   InCrack,
                                                   neiInCrack)
        vk = vkm1

    if gravity:
        G = Gravity_term(wcNplusOne,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)

    else:
        G = np.zeros((Mesh.NumberOfElts,))

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= cf * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                    C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += cf * wcNplusHalf[to_impose]
    A[np.ix_(tip_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += cf * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], wLastTS[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wcNplusOne[active]) + \
                  sigma0[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)], pfLastTS[to_impose]) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, act_indxs)], pfLastTS[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / Mesh.EltArea - LeakOff[to_solve] / Mesh.EltArea \
                  + cf * wcNplusHalf[to_solve] * pfLastTS[to_solve]

    S[tip_indxs] = -(imposed_val - wLastTS[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)], pfLastTS[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, act_indxs)], pfLastTS[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / Mesh.EltArea - LeakOff[to_impose] / Mesh.EltArea

    S[act_indxs] = -(wc_to_impose - wLastTS[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, tip_indxs)], pfLastTS[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, act_indxs)], pfLastTS[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / Mesh.EltArea - LeakOff[active] / Mesh.EltArea

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs]
    interItr_kp1 = (vk, below_wc)

    return A, S, interItr_kp1, indices

# -----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_mechLoading(wTip, EltChannel, EltTip, C, EltLoaded, w_loaded):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The given width is
    imposed on the given loaded elements (see Zia and Lecampion 2019).
    """

    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc, -np.ones((EltChannel.size, 1), dtype=np.float64)))
    A = np.vstack((A,np.zeros((1,EltChannel.size+1), dtype=np.float64)))
    A[-1, np.where(EltChannel == EltLoaded)[0]] = 1

    S = - np.dot(Cct, wTip)
    S = np.append(S, w_loaded)

    return A, S

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_volumeControl(w_lst_tmstp, wTip, EltChannel, EltTip, sigma_o, C, dt, Q, ElemArea, lkOff):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture (see Zia and Lecampion 2019).
    """
    Ccc = C[np.ix_(EltChannel, EltChannel)]
    Cct = C[np.ix_(EltChannel, EltTip)]

    A = np.hstack((Ccc,-np.ones((EltChannel.size,1),dtype=np.float64)))
    A = np.vstack((A, np.ones((1, EltChannel.size + 1), dtype=np.float64)))
    A[-1,-1] = 0

    S = - sigma_o[EltChannel] - np.dot(Ccc,w_lst_tmstp[EltChannel]) - np.dot(Cct,wTip)
    S = np.append(S, sum(Q) * dt / ElemArea - (sum(wTip)-sum(w_lst_tmstp[EltTip])) - np.sum(lkOff))

    return A, S

#-----------------------------------------------------------------------------------------------------------------------


def Elastohydrodynamic_ResidualFun(solk, system_func, *args, interItr=None):
    """
    This function gives the residual of the solution for the system of equations formed using the given function.
    """
    A, S, interItr, indices = system_func(solk, interItr, *args)
    return np.dot(A, solk) - S, interItr, indices

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_volumeControl_symmetric(w_lst_tmstp, wTip_sym, EltChannel_sym, EltTip_sym, C_s, dt, Q, sigma_o,
                                                          ElemArea, LkOff, vol_weights, sym_elements, dwTip):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture (see Zia and Lecampion 2018).
    """

    Ccc = C_s[np.ix_(EltChannel_sym, EltChannel_sym)]
    Cct = C_s[np.ix_(EltChannel_sym, EltTip_sym)]

    A = np.hstack((Ccc, -np.ones((EltChannel_sym.size, 1),dtype=np.float64)))
    weights = vol_weights[EltChannel_sym]
    weights = np.concatenate((weights, np.array([0.0])))
    A = np.vstack((A, weights))

    S = - sigma_o[EltChannel_sym] - np.dot(Ccc, w_lst_tmstp[sym_elements[EltChannel_sym]]) - np.dot(Cct, wTip_sym)
    S = np.append(S, np.sum(Q) * dt / ElemArea - np.sum(dwTip) - np.sum(LkOff))

    return A, S

#-----------------------------------------------------------------------------------------------------------------------


def Picard_Newton(Res_fun, sys_fun, guess, TypValue, interItr_init, sim_prop, *args,
                  PicardPerNewton=1000, perf_node=None):
    """
    Mixed Picard Newton solver for nonlinear systems.

    Args:
        Res_fun (function):                 -- The function calculating the residual.
        sys_fun (function):                 -- The function giving the system A, b for the Picard solver to solve the
                                               linear system of the form Ax=b.
        guess (ndarray):                    -- The initial guess.
        TypValue (ndarray):                 -- Typical value of the variable to estimate the Epsilon to calculate
                                               Jacobian.
        interItr_init (ndarray):            -- Initial value of the variable(s) exchanged between the iterations (if
                                               any).
        sim_prop (SimulationProperties):    -- the SimulationProperties object giving simulation parameters.
        relax (float):                      -- The relaxation factor.
        args (tuple):                       -- arguments given to the residual and systems functions.
        PicardPerNewton (int):              -- For hybrid Picard/Newton solution. Number of picard iterations for every
                                               Newton iteration.
        perf_node (IterationProperties):    -- the IterationProperties object passed to be populated with data.

    Returns:
        - solk (ndarray)       -- solution at the end of iteration.
        - data (tuple)         -- any data to be returned
    """
    relax = sim_prop.relaxation_factor
    solk = guess
    k = 0
    normlist = []
    interItr = interItr_init
    newton = 0
    converged = False

    while not converged:

        solkm1 = solk
        if (k + 1) % PicardPerNewton == 0:
            Fx, interItr, indices = Res_fun(solk, sys_fun, *args, interItr,)
            if newton % 3 == 0:
                Jac = Jacobian(Res_fun, solk, TypValue, *args, interItr)
            dx = np.linalg.solve(Jac, -Fx)
            solk = solkm1 + dx
            newton += 1
        else:
            try:
                (A, b, interItr, indices) = sys_fun(solk, interItr, *args)
                perfNode_linSolve = instrument_start("linear system solve", perf_node)
                solk = (1 - relax) * solkm1 + relax * np.linalg.solve(A, b)
            except np.linalg.linalg.LinAlgError:
                print('singular matrix!')
                solk = np.full((len(solk),), np.nan, dtype=np.float64)
                if perf_node is not None:
                    instrument_close(perf_node, perfNode_linSolve, None,
                                     len(b), False, 'singular matrix', None)
                    perf_node.linearSolve_data.append(perfNode_linSolve)
                return solk, None

        converged, norm = check_covergance(solk, solkm1, indices, sim_prop.toleranceEHL)
        normlist.append(norm)
        k = k + 1

        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, norm, len(b), True, None, None)
            perf_node.linearSolve_data.append(perfNode_linSolve)

        if k == sim_prop.maxSolverItrs:  # returns nan as solution if does not converge
            print('Picard iteration not converged after ' + repr(sim_prop.maxSolverItrs) + \
                  ' iterations, norm:' + repr(norm))
            solk = np.full((len(solk),), np.nan, dtype=np.float64)
            if perf_node is not None:
                perfNode_linSolve.failure_cause = 'singular matrix'
                perfNode_linSolve.status = 'failed'
            return solk, None

    if sim_prop.verbosity > 1:
        print("Converged after " + repr(k) + " iterations")
    data = interItr
    return solk, data


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

#-----------------------------------------------------------------------------------------------------------------------


def check_covergance(solk, solkm1, indices, tol):
    """ This function checks for convergence of the solution

    Args:
        solk (ndarray)      -- the evaluated solution on this iteration
        solkm1 (ndarray)    -- the evaluated solution on last iteration
        indices (list)      -- the list containing 3 arrays giving indices of the cells where the solution is obtained
                               for channel, tip and active width constraint cells.
        tol (float)         -- tolerance

    Returns:
         - converged (bool) -- True if converged
         - norm (float)     -- the evaluated norm which is checked against tolerance
    """

    # if delta w is zero in some cells
    solkm1_is_0 = np.where(solkm1 == 0)[0]
    if len(solkm1_is_0) > 0:
        delw = solk[indices[0]]
        delw_km1 = solkm1[indices[0]]
        delw = np.delete(delw, solkm1_is_0)
        delw_km1 = np.delete(delw_km1, solkm1_is_0)
        norm_w = np.linalg.norm(abs(delw - delw_km1) / abs(delw_km1))
    else:
        norm_w = np.linalg.norm(abs(solk[indices[0]] - solkm1[indices[0]]) / abs(solkm1[indices[0]]))

    norm_p = np.linalg.norm(abs(solk[indices[1]] - solkm1[indices[1]]) / abs(solkm1[indices[1]]))
    if len(indices[2]) > 0:
        norm_tr = np.linalg.norm(abs(solk[indices[2]] - solkm1[indices[2]]) / abs(solkm1[indices[2]]))
    else:
        norm_tr = 0.
    norm = (norm_w + norm_p + norm_tr) / 3

    converged = (norm_w <= tol and norm_p <= tol and norm_tr <= tol)

    return converged, norm

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
    pf = np.zeros((Mesh.NumberOfElts, ), dtype=np.float64)
    pf[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], w[EltCrack]) + sigma0[EltCrack]

    dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) * InCrack[Mesh.NeiElements[EltCrack, 0]]
    dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 1]]
    dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) * InCrack[Mesh.NeiElements[EltCrack, 2]]
    dpdyTop = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 3]]

    return dpdxLft, dpdxRgt, dpdyBtm, dpdyTop


#-----------------------------------------------------------------------------------------------------------------------

def calculate_fluid_flow_characteristics_laminar(w, C, sigma0, Mesh, EltCrack, InCrack, muPrime, density):
    """
    This function calculate fluid flux and velocity at the cell edges evaluated with the pressure calculated from the
    elasticity relation for the given fracture width and the poisoille's Law.
    """
    dp = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
    (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)
    dp[0, EltCrack] = dpdxLft
    dp[1, EltCrack] = dpdxRgt
    dp[2, EltCrack] = dpdyBtm
    dp[3, EltCrack] = dpdyTop
    # linear interpolation for pressure gradient on the edges where central difference not available
    dp[4, EltCrack] = (dp[2, Mesh.NeiElements[EltCrack, 0]] + dp[3, Mesh.NeiElements[EltCrack, 0]] + dp[2, EltCrack] +
                       dp[3, EltCrack]) / 4
    dp[5, EltCrack] = (dp[2, Mesh.NeiElements[EltCrack, 1]] + dp[3, Mesh.NeiElements[EltCrack, 1]] + dp[2, EltCrack] +
                       dp[3, EltCrack]) / 4
    dp[6, EltCrack] = (dp[0, Mesh.NeiElements[EltCrack, 2]] + dp[1, Mesh.NeiElements[EltCrack, 2]] + dp[0, EltCrack] +
                       dp[1, EltCrack]) / 4
    dp[7, EltCrack] = (dp[0, Mesh.NeiElements[EltCrack, 3]] + dp[1, Mesh.NeiElements[EltCrack, 3]] + dp[0, EltCrack] +
                       dp[1, EltCrack]) / 4

    # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
    dpLft = (dp[0, EltCrack] ** 2 + dp[4, EltCrack] ** 2) ** 0.5
    dpRgt = (dp[1, EltCrack] ** 2 + dp[5, EltCrack] ** 2) ** 0.5
    dpBtm = (dp[2, EltCrack] ** 2 + dp[6, EltCrack] ** 2) ** 0.5
    dpTop = (dp[3, EltCrack] ** 2 + dp[7, EltCrack] ** 2) ** 0.5

    # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
    wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 0]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 0]]
    wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 1]]
    wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 2]]
    wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 * InCrack[Mesh.NeiElements[EltCrack, 3]]

    fluid_flux = np.vstack((-wLftEdge ** 3 * dpLft / muPrime, -wRgtEdge ** 3 * dpRgt / muPrime))
    fluid_flux = np.vstack((fluid_flux, -wBtmEdge ** 3 * dpBtm / muPrime))
    fluid_flux = np.vstack((fluid_flux, -wTopEdge ** 3 * dpTop / muPrime))

    fluid_vel = np.copy(fluid_flux)
    fluid_vel[0] /= wLftEdge
    fluid_vel[1] /= wRgtEdge
    fluid_vel[2] /= wBtmEdge
    fluid_vel[3] /= wTopEdge

    Rey_number = abs(4 / 3 * density * fluid_flux / muPrime * 12)

    return abs(fluid_flux), abs(fluid_vel), Rey_number

#-----------------------------------------------------------------------------------------------------------------------


def Anderson(sys_fun, guess, interItr_init, sim_prop, *args, perf_node=None):
    """
    Anderson solver for non linear system.

    Args:
        sys_fun (function):                 -- The function giving the system A, b for the Anderson solver to solve the
                                               linear system of the form Ax=b.
        guess (ndarray):                    -- The initial guess.
        interItr_init (ndarray):            -- Initial value of the variable(s) exchanged between the iterations (if
                                               any).
        sim_prop (SimulationProperties):    -- the SimulationProperties object giving simulation parameters.
        relax (float):                      -- The relaxation factor.
        args (tuple):                       -- arguments given to the residual and systems functions.
        perf_node (IterationProperties):    -- the IterationProperties object passed to be populated with data.
        m_Anderson                          -- value of the recursive time steps to consider for the anderson iteration

    Returns:
        - Xks[mk+1] (ndarray)  -- final solution at the end of the iterations.
        - data (tuple)         -- any data to be returned
    """
    m_Anderson = sim_prop.Anderson_parameter
    relax = sim_prop.relaxation_factor

    ## Initialization of solution vectors
    xks = np.full((m_Anderson+2,guess.size),0.)
    Fks = np.full((m_Anderson+1,guess.size),0.)
    Gks = np.full((m_Anderson+1,guess.size),0.)

    ## Initialization of iteration parameters
    k = 0
    normlist = []
    interItr = interItr_init
    converged = False
    try:
        perfNode_linSolve = instrument_start("linear system solve", perf_node)
        # First iteration
        xks[0,::] = np.array([guess])                                       # xo
        (A, b, interItr, indices) = sys_fun(xks[0,::], interItr, *args)     # assembling A and b
        Gks[0,::] = np.linalg.solve(A, b)                                   # solve the linear system
        Fks[0,::] = Gks[0,::] - xks[0,::]
        xks[1,::] = Gks[0,::]                                               # x1
    except np.linalg.linalg.LinAlgError:
        print('singular matrix!')
        solk = np.full((len(xks[0]),), np.nan, dtype=np.float64)
        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, None,
                             len(b), False, 'singular matrix', None)
            perf_node.linearSolve_data.append(perfNode_linSolve)
        return solk, None

    while not converged:

        try:
            mk = np.min([k, m_Anderson-1])  # Asses the amount of solutions available for the least square problem
            if k >= m_Anderson:# + 1:
                (A, b, interItr, indices) = sys_fun(xks[mk + 2,::], interItr, *args)
                Gks = np.roll(Gks, -1, axis=0)
                Fks = np.roll(Fks, -1, axis=0)
            else:
                (A, b, interItr, indices) = sys_fun(xks[mk + 1, ::], interItr, *args)
            perfNode_linSolve = instrument_start("linear system solve", perf_node)

            Gks[mk + 1, ::] = np.linalg.solve(A, b)
            Fks[mk + 1, ::] = Gks[mk + 1, ::] - xks[mk + 1, ::]

            ## Setting up the Least square problem of Anderson
            A_Anderson = np.transpose(Fks[:mk+1:1,::] - Fks[mk+1,::])
            b_Anderson = - Fks[mk+1,::]

            ## Solving the least square problem for the coefficients
            omega_s = np.linalg.lstsq(A_Anderson,b_Anderson)[0]
            omega_s = np.append(omega_s,1.0 - np.sum(omega_s))

            ## Updating xk in a relaxed version
            if k >= m_Anderson:# + 1:
                xks = np.roll(xks, -1, axis=0)

            xks[mk + 2, ::] = (1-relax) * np.sum(np.transpose(np.multiply(np.transpose(xks[:mk+2,::]), omega_s)),axis=0)\
                 + relax * np.sum(np.transpose(np.multiply(np.transpose(Gks[:mk+2,::]), omega_s)),axis=0)

        except np.linalg.linalg.LinAlgError:
            print('singular matrix!')
            solk = np.full((len(xks[mk]),), np.nan, dtype=np.float64)
            if perf_node is not None:
                instrument_close(perf_node, perfNode_linSolve, None,
                                 len(b), False, 'singular matrix', None)
                perf_node.linearSolve_data.append(perfNode_linSolve)
            return solk, None
        ## Check for convergency of the solution
        #converged, norm = check_covergance(xks[0,::], xks[1,::], indices, sim_prop.toleranceEHL)
        converged, norm = check_covergance(xks[mk + 1, ::], xks[mk + 2, ::], indices, sim_prop.toleranceEHL)
        normlist.append(norm)
        k = k + 1

        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, norm, len(b), True, None, None)
            perf_node.linearSolve_data.append(perfNode_linSolve)

        if k == sim_prop.maxSolverItrs:  # returns nan as solution if does not converge
            print('Anderson iteration not converged after ' + repr(sim_prop.maxSolverItrs) + \
                  ' iterations, norm:' + repr(norm))
            solk = np.full((np.size(xks[0,::]),), np.nan, dtype=np.float64)
            if perf_node is not None:
                perfNode_linSolve.failure_cause = 'singular matrix'
                perfNode_linSolve.status = 'failed'
            return solk, None

    if sim_prop.verbosity > 1:
        print("Converged after " + repr(k) + " iterations")
    data = interItr
    return xks[mk + 2, ::], data


#-----------------------------------------------------------------------------------------------------------------------
