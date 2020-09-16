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
#import numdifftools as nd
import copy
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

#local imports
from fluid_model import friction_factor_vector, friction_factor_MDR
from properties import instrument_start, instrument_close


def finiteDiff_operator_laminar(w, EltCrack, muPrime, Mesh, InCrack, neiInCrack, simProp):
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
        simProp (object):       -- An object of the SimulationProperties class.

    Returns:
        FinDiffOprtr (ndarray): -- the finite difference matrix.

    """

    if simProp.solveSparse:
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
                                            wBtmEdge ** 3 + wTopEdge ** 3) / dy ** 2) / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 0]] = wLftEdge ** 3 / dx ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = wRgtEdge ** 3 / dx ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = wBtmEdge ** 3 / dy ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = wTopEdge ** 3 / dy ** 2 / muPrime

    return FinDiffOprtr


#-----------------------------------------------------------------------------------------------------------------------

def Gravity_term(w, EltCrack, fluidProp, mesh, InCrack, simProp):
    """
    This function returns the gravity term (G in Zia and Lecampion 2019).

    Args:
        w (ndarray):                -- the width of the trial fracture.
        EltCrack (ndarray):         -- the list of elements inside the fracture.
        fluidProp (object):         -- FluidProperties class object giving the fluid properties.
        Mesh (CartesianMesh):       -- the mesh.
        InCrack (ndarray):          -- An array specifying whether elements are inside the fracture or not with
                                       1 or 0 respectively.
        simProp (object):           -- An object of the SimulationProperties class.

    Returns:
        G (ndarray):                -- the matrix with the gravity terms.
    """

    G = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    
    if simProp.gravity:
        if fluidProp.rheology == "Newtonian" and not fluidProp.turbulence:
            # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
            wBtmEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 2]]) / 2 * InCrack[mesh.NeiElements[EltCrack, 2]]
            wTopEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 3]]) / 2 * InCrack[mesh.NeiElements[EltCrack, 3]]
        
            G[EltCrack] = fluidProp.density * 9.81 * (wTopEdge ** 3 - wBtmEdge ** 3) / mesh.hy / fluidProp.muPrime
        else:
            raise SystemExit("Effect of gravity is only supported for Newtonian fluid in laminar flow regime yet!")

    return G

#-----------------------------------------------------------------------------------------------------------------------


def FiniteDiff_operator_turbulent_implicit(w, pf, EltCrack, fluidProp, matProp, simProp, mesh, InCrack, vkm1, to_solve,
                                           active, to_impose):
    """
    The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
    Dontsov and Peirce 2008). The matrix is evaluated by taking turbulence into account.

    Args:
        w (ndarray):                -- the width of the trial fracture.
        EltCrack (ndarray):         -- the list of elements inside the fracture
        fluidProp (object):         -- FluidProperties class object giving the fluid properties.
        matProp (object):           -- An instance of the MaterialProperties class giving the material properties.
        simProp (object):           -- An object of the SimulationProperties class.
        mesh (CartesianMesh):       -- the mesh.
        InCrack (ndarray):          -- an array specifying whether elements are inside the fracture or not with
                                       1 or 0 respectively.
        vkm1 (ndarray):             -- the velocity at cell edges from the previous iteration (if necessary). Here,
                                       it is used as the starting guess for the implicit solver.
        to_solve (ndarray):         -- the channel elements to be solved.
        active (ndarray):           -- the channel elements where width constraint is active.
        to_impose (ndarray):        -- the tip elements to be imposed.
                
    Returns:
        - FinDiffOprtr (ndarray)    -- the finite difference matrix.
        - vk (ndarray)              -- the velocity evaluated for current iteration.
    """

    if simProp.solveSparse:
        FinDiffOprtr = sparse.lil_matrix((w.size, w.size), dtype=np.float64)
    else:
        FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)

    dx = mesh.hx
    dy = mesh.hy

    # todo: can be evaluated at each cell edge
    rough = w[EltCrack]/matProp.grainSize
    rough[np.where(rough < 3)[0]] = 3.

    # width on edges; evaluated by averaging the widths of adjacent cells
    wLftEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 0]]) / 2
    wRgtEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 1]]) / 2
    wBtmEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 2]]) / 2
    wTopEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 3]]) / 2

    # pressure gradient data structure. The rows store pressure gradient in the following order.
    # 0 - left edge in x-direction    # 1 - right edge in x-direction
    # 2 - bottom edge in y-direction  # 3 - top edge in y-direction
    # 4 - left edge in y-direction    # 5 - right edge in y-direction
    # 6 - bottom edge in x-direction  # 7 - top edge in x-direction

    dp = np.zeros((8, mesh.NumberOfElts), dtype=np.float64)
    dp[0, EltCrack] = (pf[EltCrack] - pf[mesh.NeiElements[EltCrack, 0]]) / dx
    dp[1, EltCrack] = (pf[mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) / dx
    dp[2, EltCrack] = (pf[EltCrack] - pf[mesh.NeiElements[EltCrack, 2]]) / dy
    dp[3, EltCrack] = (pf[mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) / dy
    # linear interpolation for pressure gradient on the edges where central difference not available
    dp[4, EltCrack] = (dp[2,mesh.NeiElements[EltCrack,0]]+dp[3,mesh.NeiElements[EltCrack,0]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[5, EltCrack] = (dp[2,mesh.NeiElements[EltCrack,1]]+dp[3,mesh.NeiElements[EltCrack,1]]+dp[2,EltCrack]+dp[3,EltCrack])/4
    dp[6, EltCrack] = (dp[0,mesh.NeiElements[EltCrack,2]]+dp[1,mesh.NeiElements[EltCrack,2]]+dp[0,EltCrack]+dp[1,EltCrack])/4
    dp[7, EltCrack] = (dp[0,mesh.NeiElements[EltCrack,3]]+dp[1,mesh.NeiElements[EltCrack,3]]+dp[0,EltCrack]+dp[1,EltCrack])/4

    # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
    dpLft = (dp[0, EltCrack] ** 2 + dp[4, EltCrack] ** 2) ** 0.5
    dpRgt = (dp[1, EltCrack] ** 2 + dp[5, EltCrack] ** 2) ** 0.5
    dpBtm = (dp[2, EltCrack] ** 2 + dp[6, EltCrack] ** 2) ** 0.5
    dpTop = (dp[3, EltCrack] ** 2 + dp[7, EltCrack] ** 2) ** 0.5

    vk = np.zeros((8, mesh.NumberOfElts), dtype=np.float64)
    # the factor to be multiplied to the velocity from last iteration to get the upper bracket
    upBracket_factor = 10

    # loop to calculate velocity on each cell edge implicitly
    for i in range(0, len(EltCrack)):
        # todo !!! Hack. zero velocity if the pressure gradient is zero or very small width
        if dpLft[i] < 1e-8 or wLftEdge[i] < 1e-10:
            vk[0, EltCrack[i]] = 0.0
        else:
            arg = (wLftEdge[i], fluidProp.viscosity, fluidProp.density, dpLft[i], rough[i])
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
            arg = (wRgtEdge[i], fluidProp.viscosity, fluidProp.density, dpRgt[i], rough[i])
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
            arg = (wBtmEdge[i], fluidProp.viscosity, fluidProp.density, dpBtm[i], rough[i])
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
            arg = (wTopEdge[i], fluidProp.viscosity, fluidProp.density, dpTop[i], rough[i])
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
    ReLftEdge = 4 / 3 * fluidProp.density * wLftEdge * vk[0, EltCrack] / fluidProp.viscosity
    ReRgtEdge = 4 / 3 * fluidProp.density * wRgtEdge * vk[1, EltCrack] / fluidProp.viscosity
    ReBtmEdge = 4 / 3 * fluidProp.density * wBtmEdge * vk[2, EltCrack] / fluidProp.viscosity
    ReTopEdge = 4 / 3 * fluidProp.density * wTopEdge * vk[3, EltCrack] / fluidProp.viscosity


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
    cond[0, ReLftEdge_nonZero] = wLftEdge[ReLftEdge_nonZero] ** 2 / (fluidProp.density * ffLftEdge[ReLftEdge_nonZero]
                                                                     * vk[0, EltCrack[ReLftEdge_nonZero]])
    cond[1, ReRgtEdge_nonZero] = wRgtEdge[ReRgtEdge_nonZero] ** 2 / (fluidProp.density * ffRgtEdge[ReRgtEdge_nonZero]
                                                                     * vk[1, EltCrack[ReRgtEdge_nonZero]])
    cond[2, ReBtmEdge_nonZero] = wBtmEdge[ReBtmEdge_nonZero] ** 2 / (fluidProp.density * ffBtmEdge[ReBtmEdge_nonZero]
                                                                     * vk[2, EltCrack[ReBtmEdge_nonZero]])
    cond[3, ReTopEdge_nonZero] = wTopEdge[ReTopEdge_nonZero] ** 2 / (fluidProp.density * ffTopEdge[ReTopEdge_nonZero]
                                                                     * vk[3, EltCrack[ReTopEdge_nonZero]])

    # assembling the finite difference matrix
    FinDiffOprtr[EltCrack, EltCrack] = -(cond[0, :] + cond[1, :]) / dx ** 2 - (cond[2, :] + cond[3, :]) / dy ** 2
    FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 0]] = cond[0, :] / dx ** 2
    FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 1]] = cond[1, :] / dx ** 2
    FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 2]] = cond[2, :] / dy ** 2
    FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 3]] = cond[3, :] / dy ** 2

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

def finiteDiff_operator_power_law(w, pf, EltCrack, fluidProp, Mesh, InCrack, neiInCrack, edgeInCrk_lst, simProp):
    """
    The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
    equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated for Herschel-Bulkley fluid rheology.

    Args:
        w (ndarray):            -- the width of the trial fracture.
        pf (ndarray):           -- the fluid pressure.
        EltCrack (ndarray):     -- the list of elements inside the fracture.
        fluidProp (object):     -- FluidProperties class object giving the fluid properties.
        Mesh (CartesianMesh):   -- the mesh.
        InCrack (ndarray):      -- an array specifying whether elements are inside the fracture or not with
                                   1 or 0 respectively.
        neiInCrack (ndarray):   -- an ndarray giving indices of the neighbours of all the cells in the crack, in the
                                   EltCrack list.
        edgeInCrk_lst (ndarray):-- this list provides the indices of those cells in the EltCrack list whose neighbors are not 
                                   outside the crack. It is used to evaluate the conductivity on edges of only these cells who
                                   are inside. It consists of four lists, one for each edge.
        simProp (object):       -- An object of the SimulationProperties class.

    Returns:
        FinDiffOprtr (ndarray): -- the finite difference matrix.

    """

    if simProp.solveSparse:
        FinDiffOprtr = sparse.lil_matrix((w.size, w.size), dtype=np.float64)
    else:
        FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)

    dx = Mesh.hx
    dy = Mesh.hy

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
    dp[0, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) / dx
    dp[1, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) / dx
    dp[2, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) / dy
    dp[3, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) / dy
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
    dpLft = np.sqrt(dp[0, EltCrack] ** 2 + dp[4, EltCrack] ** 2)
    dpRgt = np.sqrt(dp[1, EltCrack] ** 2 + dp[5, EltCrack] ** 2)
    dpBtm = np.sqrt(dp[2, EltCrack] ** 2 + dp[6, EltCrack] ** 2)
    dpTop = np.sqrt(dp[3, EltCrack] ** 2 + dp[7, EltCrack] ** 2)

    cond = np.zeros((4, EltCrack.size), dtype=np.float64)
    cond[0, edgeInCrk_lst[0]] = (wLftEdge[edgeInCrk_lst[0]] ** (2 * fluidProp.n + 1) * dpLft[edgeInCrk_lst[0]] / \
                         fluidProp.Mprime) ** (1 / fluidProp.n) / dpLft[edgeInCrk_lst[0]]
    cond[1, edgeInCrk_lst[1]] = (wRgtEdge[edgeInCrk_lst[1]] ** (2 * fluidProp.n + 1) * dpRgt[edgeInCrk_lst[1]] / \
                         fluidProp.Mprime) ** (1 / fluidProp.n) / dpRgt[edgeInCrk_lst[1]]
    cond[2, edgeInCrk_lst[2]] = (wBtmEdge[edgeInCrk_lst[2]] ** (2 * fluidProp.n + 1) * dpBtm[edgeInCrk_lst[2]] / \
                         fluidProp.Mprime) ** (1 / fluidProp.n) / dpBtm[edgeInCrk_lst[2]]
    cond[3, edgeInCrk_lst[3]] = (wTopEdge[edgeInCrk_lst[3]] ** (2 * fluidProp.n + 1) * dpTop[edgeInCrk_lst[3]] / \
                         fluidProp.Mprime) ** (1 / fluidProp.n) / dpTop[edgeInCrk_lst[3]]

    indx_elts = np.arange(len(EltCrack))
    FinDiffOprtr[indx_elts, indx_elts] = -(cond[0, :] + cond[1, :]) / dx ** 2 - (cond[2, :] + cond[3, :]) / dy ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 0]] = cond[0, :] / dx ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = cond[1, :] / dx ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = cond[2, :] / dy ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = cond[3, :] / dy ** 2
    
    eff_mu = None
    if simProp.saveEffVisc:
        eff_mu = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)
        eff_mu[0, EltCrack] =  wLftEdge ** 3 / (12 * cond[0, :])
        eff_mu[1, EltCrack] =  wRgtEdge ** 3 / (12 * cond[1, :])
        eff_mu[2, EltCrack] =  wBtmEdge ** 3 / (12 * cond[2, :])
        eff_mu[3, EltCrack] =  wTopEdge ** 3 / (12 * cond[3, :])
    
    
    return FinDiffOprtr, eff_mu


#-----------------------------------------------------------------------------------------------------------------------

def finiteDiff_operator_Herschel_Bulkley(w, pf, EltCrack, fluidProp, Mesh, InCrack, neiInCrack, edgeInCrk_lst, simProp):
    """
    The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
    equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated for Herschel-Bulkley fluid rheology.

    Args:
        w (ndarray):            -- the width of the trial fracture.
        pf (ndarray):           -- the fluid pressure.
        EltCrack (ndarray):     -- the list of elements inside the fracture.
        fluidProp (object):     -- FluidProperties class object giving the fluid properties.
        Mesh (CartesianMesh):   -- the mesh.
        InCrack (ndarray):      -- an array specifying whether elements are inside the fracture or not with
                                   1 or 0 respectively.
        neiInCrack (ndarray):   -- an ndarray giving indices of the neighbours of all the cells in the crack, in the
                                   EltCrack list.
        edgeInCrk_lst (ndarray):-- this list provides the indices of those cells in the EltCrack list whose neighbors are not 
                                   outside the crack. It is used to evaluate the conductivity on edges of only these cells who
                                   are inside. It consists of four lists, one for each edge.
        simProp (object):       -- An object of the SimulationProperties class.

    Returns:
        FinDiffOprtr (ndarray): -- the finite difference matrix.

    """

    if simProp.solveSparse:
        FinDiffOprtr = sparse.lil_matrix((w.size, w.size), dtype=np.float64)
    else:
        FinDiffOprtr = np.zeros((w.size, w.size), dtype=np.float64)

    dx = Mesh.hx
    dy = Mesh.hy

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
    dp[0, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) / dx
    dp[1, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) / dx
    dp[2, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) / dy
    dp[3, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) / dy
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

    cond = np.zeros((4, EltCrack.size), dtype=np.float64)
    

    x0 = np.maximum(1 - 2 * fluidProp.T0 / wLftEdge / dpLft, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[0, edgeInCrk_lst[0]] = fluidProp.var1 * dpLft[edgeInCrk_lst[0]] ** fluidProp.var2 * wLftEdge[edgeInCrk_lst[0]] ** \
                                fluidProp.var3 * x0[edgeInCrk_lst[0]]**fluidProp.var4 * \
                                (1 + 2*fluidProp.T0 / wLftEdge[edgeInCrk_lst[0]] / dpLft[edgeInCrk_lst[0]] * fluidProp.var5)
    x1 = np.maximum(1 - 2*fluidProp.T0 / wRgtEdge / dpRgt, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[1, edgeInCrk_lst[1]] = fluidProp.var1 * dpRgt[edgeInCrk_lst[1]] ** fluidProp.var2 * wRgtEdge[edgeInCrk_lst[1]] ** \
                                fluidProp.var3 * x1[edgeInCrk_lst[1]]**fluidProp.var4 * \
                                (1 + 2*fluidProp.T0 / wRgtEdge[edgeInCrk_lst[1]] / dpRgt[edgeInCrk_lst[1]] * fluidProp.var5)
    x2 = np.maximum(1 - 2*fluidProp.T0 / wBtmEdge / dpBtm, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[2, edgeInCrk_lst[2]] = fluidProp.var1 * dpBtm[edgeInCrk_lst[2]] ** fluidProp.var2 * wBtmEdge[edgeInCrk_lst[2]] ** \
                                fluidProp.var3 * x2[edgeInCrk_lst[2]]**fluidProp.var4 * \
                                (1 + 2*fluidProp.T0 / wBtmEdge[edgeInCrk_lst[2]] / dpBtm[edgeInCrk_lst[2]] * fluidProp.var5)
    x3 = np.maximum(1 - 2*fluidProp.T0 / wTopEdge / dpTop, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[3, edgeInCrk_lst[3]] = fluidProp.var1 * dpTop[edgeInCrk_lst[3]] ** fluidProp.var2 * wTopEdge[edgeInCrk_lst[3]] ** \
                                fluidProp.var3 * x3[edgeInCrk_lst[3]]**fluidProp.var4 * \
                                (1 + 2*fluidProp.T0 / wTopEdge[edgeInCrk_lst[3]] / dpTop[edgeInCrk_lst[3]] * fluidProp.var5)
    
        
    indx_elts = np.arange(len(EltCrack))
    FinDiffOprtr[indx_elts, indx_elts] = -(cond[0, :] + cond[1, :]) / dx ** 2 - (cond[2, :] + cond[3, :]) / dy ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 0]] = cond[0, :] / dx ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = cond[1, :] / dx ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = cond[2, :] / dy ** 2
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = cond[3, :] / dy ** 2

    eff_mu = None
    if simProp.saveEffVisc:
        with np.errstate(divide='ignore'):
            eff_mu = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)
            eff_mu[0, EltCrack[edgeInCrk_lst[0]]] =  wLftEdge[edgeInCrk_lst[0]] ** 3 / (12 * cond[0, edgeInCrk_lst[0]])
            eff_mu[1, EltCrack[edgeInCrk_lst[1]]] =  wRgtEdge[edgeInCrk_lst[1]] ** 3 / (12 * cond[1, edgeInCrk_lst[1]])
            eff_mu[2, EltCrack[edgeInCrk_lst[2]]] =  wBtmEdge[edgeInCrk_lst[2]] ** 3 / (12 * cond[2, edgeInCrk_lst[2]])
            eff_mu[3, EltCrack[edgeInCrk_lst[3]]] =  wTopEdge[edgeInCrk_lst[3]] ** 3 / (12 * cond[3, edgeInCrk_lst[3]])
    
    if simProp.saveYieldRatio:
        yielded = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)
        yielded[0, EltCrack] = x0
        yielded[1, EltCrack] = x1
        yielded[2, EltCrack] = x2
        yielded[3, EltCrack] = x3
        
    return FinDiffOprtr, eff_mu, yielded


#----------------------------------------------------------------------------------------------------------------------------------------

def get_finite_difference_matrix(wNplusOne, sol, frac_n, EltCrack, neiInCrack, fluid_prop, mat_prop, sim_prop, mesh,
                                 InCrack, C, interItr, to_solve, to_impose, active, interItr_kp1, list_edgeInCrack):
    
   

    if fluid_prop.rheology == 'Newtonian' and not fluid_prop.turbulence:
        FinDiffOprtr =  finiteDiff_operator_laminar(wNplusOne,
                                                    EltCrack,
                                                    fluid_prop.muPrime,
                                                    mesh,
                                                    InCrack,
                                                    neiInCrack,
                                                    sim_prop)
       
    else:       
        pf = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        # pressure evaluated by dot product of width and elasticity matrix
        pf[to_solve] = np.dot(C[np.ix_(to_solve, EltCrack)], wNplusOne[EltCrack]) +  mat_prop.SigmaO[to_solve]
        if sim_prop.solveDeltaP:
            pf[active] = frac_n.pFluid[active] + sol[len(to_solve):len(to_solve) + len(active)]
            pf[to_impose] = frac_n.pFluid[to_impose] + sol[len(to_solve) + len(active):]
        else:
            pf[active] = sol[len(to_solve):len(to_solve) + len(active)]
            pf[to_impose] = sol[len(to_solve) + len(active):]
            
        
        if fluid_prop.turbulence:
            FinDiffOprtr, interItr_kp1[0] = FiniteDiff_operator_turbulent_implicit(wNplusOne,
                                                        pf,
                                                        EltCrack, 
                                                        fluid_prop,
                                                        mat_prop,
                                                        sim_prop,
                                                        mesh,
                                                        InCrack,
                                                        interItr[0],
                                                        to_solve,
                                                        active,
                                                        to_impose)
        elif fluid_prop.rheology in ["Herschel-Bulkley", "HBF"]:
            FinDiffOprtr, interItr_kp1[2], interItr_kp1[3] = finiteDiff_operator_Herschel_Bulkley(wNplusOne,
                                                        pf,
                                                        EltCrack,
                                                        fluid_prop,
                                                        mesh,
                                                        InCrack,
                                                        neiInCrack,
                                                        list_edgeInCrack,
                                                        sim_prop)
            
        elif fluid_prop.rheology in ['power law', 'PLF']:
            FinDiffOprtr, interItr_kp1[2] = finiteDiff_operator_power_law(wNplusOne,
                                                        pf,          
                                                        EltCrack,
                                                        fluid_prop,
                                                        mesh,
                                                        InCrack,
                                                        neiInCrack,
                                                        list_edgeInCrack,
                                                        sim_prop)
        
    return FinDiffOprtr
            
            
#--------------------------------------------------------------------------------------------------------------------------------

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
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.    
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.
                                               
    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args


    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2
    
    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)
    

    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)


    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), fluid_prop.compressibility * wcNplusHalf[to_solve])], [0], format='csr')

    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)
    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - \
                  LeakOff[to_solve] / frac.mesh.EltArea + \
                  fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making 
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)
    
        if len(to_del) > 0:            
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]
    
    interItr_kp1[1] = below_wc

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
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.    
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.
                                               
    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)
    

    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), fluid_prop.compressibility * wcNplusHalf[to_solve])], [0], format='csr')

    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                  + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]

    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea

    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making 
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)
    
        if len(to_del) > 0:            
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]
    
    interItr_kp1[1] = below_wc

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
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.    
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.
                                               
    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)
    


    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]

    A[np.ix_(tip_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - \
                  LeakOff[to_solve] / frac.mesh.EltArea + \
                  fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making 
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)
    
        if len(to_del) > 0:            
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]
    
    interItr_kp1[1] = below_wc

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
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.    
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.
                                               
    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)
    

    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                    C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]
    A[np.ix_(tip_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, act_indxs)], frac.pFluid[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                  + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]

    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, act_indxs)], frac.pFluid[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea

    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, act_indxs)], frac.pFluid[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea


    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making 
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)
        if len(to_del) > 0:
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]
    
    interItr_kp1[1] = below_wc
    return A, S, interItr_kp1, indices

# -----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_mechLoading(wTip, EltChannel, EltTip, C, EltLoaded, w_loaded):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The given width is
    imposed on the given loaded elements.
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


def MakeEquationSystem_volumeControl_double_fracture(w_lst_tmstp, wTipFR0, wTipFR1, EltChannel0,EltChannel1, EltTip0, EltTip1, sigma_o, C, dt, QFR0, QFR1, ElemArea, lkOff):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture.
    """
    """
    Scheme of the system of equations that we are going to make
    
       CC0  CC01| 1 1   Dw0   sigma00
                | 1 1   Dw0   sigma00
       CC10 CC1 | 0 1 * Dw1 = sigma01
       --------------   ---   --------
       -1 -1 -1 0 0 0   Dp0   Q00*Dt/A0
       0  0  0 -1 0 0   Dp1   Q01*Dt/A1
    """
    wTip = np.concatenate((wTipFR0, wTipFR1))
    EltChannel = np.concatenate((EltChannel0,EltChannel1))
    EltTip = np.concatenate((EltTip0, EltTip1))
    Ccc = C[np.ix_(EltChannel, EltChannel)] # elasticity Channel Channel
    Cct = C[np.ix_(EltChannel, EltTip)]

    varray0 = np.zeros((EltChannel.size,1),dtype=np.float64)
    varray0[0:EltChannel0.size] = 1.
    varray1 = np.zeros((EltChannel.size,1),dtype=np.float64)
    varray1[EltChannel0.size:EltChannel.size] = 1.

    A = np.hstack((Ccc,-varray0,-varray1))

    harray0 = np.zeros((1,EltChannel.size+2),dtype=np.float64)
    harray0[0,0:EltChannel0.size] = 1.
    harray1 = np.zeros((1,EltChannel.size+2),dtype=np.float64)
    harray1[0,EltChannel0.size:EltChannel.size] = 1.

    A = np.vstack((A,harray0,harray1))

    S = - sigma_o[EltChannel] - np.dot(Ccc,w_lst_tmstp[EltChannel]) - np.dot(Cct,wTip)
    S = np.append(S, sum(QFR0) * dt / ElemArea - (sum(wTipFR0) - sum(w_lst_tmstp[EltTip0])) - np.sum(lkOff[np.concatenate((EltChannel0,EltTip0))]))
    S = np.append(S, sum(QFR1) * dt / ElemArea - (sum(wTipFR1) - sum(w_lst_tmstp[EltTip1])) - np.sum(lkOff[np.concatenate((EltChannel1,EltTip1))]))

    return A, S

#-----------------------------------------------------------------------------------------------------------------------


def MakeEquationSystem_volumeControl(w_lst_tmstp, wTip, EltChannel, EltTip, sigma_o, C, dt, Q, ElemArea, lkOff):
    """
    This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
    with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
    fracture is imposed to be equal to the fluid injected into the fracture.
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


def Elastohydrodynamic_ResidualFun(solk, system_func, interItr, *args):
    """
    This function gives the residual of the solution for the system of equations formed using the given function.
    """
    A, S, interItr, indices = system_func(solk, interItr, *args)
    return np.dot(A, solk) - S, interItr, indices


#-----------------------------------------------------------------------------------------------------------------------

def Elastohydrodynamic_ResidualFun_nd(solk, system_func, interItr, InterItr_o, indices_o,*args):
    """
    This function gives the residual of the solution for the system of equations formed using the given function.
    """
    A, S, interItr, indices = system_func(solk, interItr, *args)
    if len(indices[3]) == 0:
        Fx = np.dot(A, solk) - S
    else:
        Fx_red = np.dot(A, np.delete(solk, len(indices[0]) + np.asarray(indices[3]))) - S
        Fx = populate_full(indices, Fx_red)
    InterItr_o = interItr
    indices_o = indices
    return Fx
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
            Fx, interItr, indices = Elastohydrodynamic_ResidualFun(solk, sys_fun, interItr, *args)
            Jac = Jacobian(Elastohydrodynamic_ResidualFun, sys_fun, solk, TypValue, interItr, *args)
            # Jac = nd.Jacobian(Elastohydrodynamic_ResidualFun)(solk, sys_fun, interItr, interItr_o, indices, *args)
            dx = np.linalg.solve(Jac, -Fx)
            solk = solkm1 + dx
            newton += 1
        else:
            try:
                A, b, interItr, indices = sys_fun(solk, interItr, *args)
                perfNode_linSolve = instrument_start("linear system solve", perf_node)
                sol = np.linalg.solve(A, b)
                if len(indices[3]) > 0:             # if the size of system is varying between iterations (in case of HB fluid)
                    solk = relax * solkm1 + (1 - relax) * get_complete_solution(sol, indices, *args)
                else:
                    solk = relax * solkm1 + (1 - relax) * sol 
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
    data = [interItr[0], interItr[2], interItr[3]]
    return solk, data


#-----------------------------------------------------------------------------------------------------------------------

def Jacobian(Residual_function, sys_func, x, TypValue, interItr, *args):
    """
    This function returns the Jacobian of the given function.
    """

    central = False
    Fx, interItr, indices = Residual_function(x, sys_func, interItr, *args)
    Jac = np.zeros((len(x), len(x)), dtype=np.float64)
    for i in range(0, len(x)):
        Epsilon = np.finfo(float).eps ** 0.5 * abs(max(x[i], TypValue[i]))
        if Epsilon == 0:
            Epsilon = np.finfo(float).eps ** 0.5
        xip = np.copy(x)
        xip[i] = xip[i] + Epsilon
        if central:
            xin = np.copy(x)
            xin[i] = xin[i]-Epsilon
            Jac[:,i] = (Residual_function(xip, sys_func, interItr, *args)[0] - Residual_function(
                xin, sys_func, interItr, *args)[0])/(2*Epsilon)
            if np.isnan(Jac[:, i]).any():
                Jac[:,:] = np.nan
                return Jac
        else:
            Fxi, interItr, indices = Residual_function(xip, sys_func, interItr, *args)
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

    w_normalization = np.linalg.norm(solkm1[indices[0]])
    if w_normalization > 0.:
        norm_w = np.linalg.norm(abs(solk[indices[0]] - solkm1[indices[0]]) / w_normalization)
    else:
        norm_w = np.linalg.norm(abs(solk[indices[0]] - solkm1[indices[0]]))

    p_normalization = np.linalg.norm(solkm1[indices[1]])
    if p_normalization > 0.:
        norm_p = np.linalg.norm(abs(solk[indices[1]] - solkm1[indices[1]]) / p_normalization)
    else:
        norm_p = np.linalg.norm(abs(solk[indices[1]] - solkm1[indices[1]]) )

    if len(indices[2]) > 0: #these are the cells with the active width constraints
        tr_normalization = np.linalg.norm(solkm1[indices[2]])
        if tr_normalization > 0.:
            norm_tr = np.linalg.norm(abs(solk[indices[2]] - solkm1[indices[2]]) / tr_normalization)
        else:
            norm_tr = np.linalg.norm(abs(solk[indices[2]] - solkm1[indices[2]]))
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

def pressure_gradient_form_pressure( pf, Mesh, EltCrack, InCrack):
    """
    This function gives the pressure gradient at the cell edges evaluated with the pressure
    """

    dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 0]]) * InCrack[Mesh.NeiElements[EltCrack, 0]] /Mesh.hx
    dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 1]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 1]] /Mesh.hx
    dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 2]]) * InCrack[Mesh.NeiElements[EltCrack, 2]] /Mesh.hy
    dpdyTop = (pf[Mesh.NeiElements[EltCrack, 3]] - pf[EltCrack]) * InCrack[Mesh.NeiElements[EltCrack, 3]] /Mesh.hy

    return dpdxLft, dpdxRgt, dpdyBtm, dpdyTop

#-----------------------------------------------------------------------------------------------------------------------

def calculate_fluid_flow_characteristics_laminar(w, pf, sigma0, Mesh, EltCrack, InCrack, muPrime, density):
    """
    This function calculate fluid flux and velocity at the cell edges evaluated with the pressure calculated from the
    elasticity relation for the given fracture width and the poisoille's Law.
    """
    """
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                             0     1      2      3
    """
    if muPrime != 0:
        dp = np.zeros((8, Mesh.NumberOfElts), dtype=np.float64)
        (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient_form_pressure( pf, Mesh, EltCrack, InCrack)
        # dp = [dpdxLft , dpdxRgt, dpdyBtm, dpdyTop, dpdyLft, dpdyRgt, dpdxBtm, dpdxTop]
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

        #          0    ,    1   ,     2  ,    3   ,    4   ,    5   ,    6   ,    7
        # dp = [dpdxLft , dpdxRgt, dpdyBtm, dpdyTop, dpdyLft, dpdyRgt, dpdxBtm, dpdxTop]

        # fluid_flux_components = [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge, fy bottom edge, fx top edge, fy top edge]
        #                                                      fx left edge          ,              fy left edge
        fluid_flux_components = np.vstack((-wLftEdge ** 3 * dp[0, EltCrack] / muPrime, -wLftEdge ** 3 * dp[4, EltCrack] / muPrime))
        #                                                      fx right edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wRgtEdge ** 3 * dp[1, EltCrack] / muPrime))
        #                                                      fy right edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wRgtEdge ** 3 * dp[5, EltCrack] / muPrime))
        #                                                      fx bottom edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wBtmEdge ** 3 * dp[6, EltCrack] / muPrime))
        #                                                      fy bottom edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wBtmEdge ** 3 * dp[2, EltCrack] / muPrime))
        #                                                      fx top edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wTopEdge ** 3 * dp[7, EltCrack] / muPrime))
        #                                                      fy top edge
        fluid_flux_components = np.vstack((fluid_flux_components, -wTopEdge ** 3 * dp[3, EltCrack] / muPrime))



        fluid_vel = np.copy(fluid_flux)
        wEdges = [wLftEdge,wRgtEdge,wBtmEdge,wTopEdge]
        for i in range(4):
            local_nonzero_indexes=fluid_vel[i].nonzero()
            fluid_vel[i][local_nonzero_indexes] /= wEdges[i][local_nonzero_indexes]

        fluid_vel_components = np.copy(fluid_flux_components)
        for i in range(8):
            local_nonzero_indexes=fluid_vel_components[i].nonzero()
            fluid_vel_components[i][local_nonzero_indexes] /= wEdges[int(np.trunc(i/2))][local_nonzero_indexes]

        Rey_number = abs(4 / 3 * density * fluid_flux / muPrime * 12)

        return abs(fluid_flux), abs(fluid_vel), Rey_number, fluid_flux_components, fluid_vel_components
    else:
        raise SystemExit('ERROR: if the fluid viscosity is equal to 0 does not make sense to compute the fluid velocity or the fluid flux')


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
    xks = np.full((m_Anderson+2, guess.size), 0.)
    Fks = np.full((m_Anderson+1, guess.size), 0.)
    Gks = np.full((m_Anderson+1, guess.size), 0.)

    ## Initialization of iteration parameters
    k = 0
    normlist = []
    interItr = interItr_init
    converged = False
    try:
        perfNode_linSolve = instrument_start("linear system solve", perf_node)
        # First iteration
        xks[0, ::] = np.array([guess])                                       # xo
        (A, b, interItr, indices) = sys_fun(xks[0, ::], interItr, *args)     # assembling A and b
        # solk = np.linalg.solve(A, b)                                        # solve the linear system
        # if len(indices[3]) > 0:                                             # if the size of system is varying between \
        #     Gks[0, ::] = get_complete_solution(solk, indices, *args)        # iterations (in case of HB fluid)
        # else:
        #     Gks[0, ::] = solk
        Gks[0, ::] = np.linalg.solve(A, b)
        Fks[0, ::] = Gks[0, ::] - xks[0, ::]
        xks[1, ::] = Gks[0, ::]                                               # x1
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
                (A, b, interItr, indices) = sys_fun(xks[mk + 2, ::], interItr, *args)
                Gks = np.roll(Gks, -1, axis=0)
                Fks = np.roll(Fks, -1, axis=0)
            else:
                (A, b, interItr, indices) = sys_fun(xks[mk + 1, ::], interItr, *args)
            perfNode_linSolve = instrument_start("linear system solve", perf_node)
            
            # solk = np.linalg.solve(A, b)
            # if len(indices[3]) > 0:                                             # if the size of system is varying between \
            #     Gks[mk + 1, ::] = get_complete_solution(solk, indices, *args)        # iterations (in case of HB fluid)
            # else:
            #     Gks[mk + 1, ::] = solk
            Gks[mk + 1, ::] = np.linalg.solve(A, b)
            Fks[mk + 1, ::] = Gks[mk + 1, ::] - xks[mk + 1, ::]

            ## Setting up the Least square problem of Anderson
            A_Anderson = np.transpose(Fks[:mk+1, ::] - Fks[mk+1, ::])
            b_Anderson = -Fks[mk+1, ::]

            # Solving the least square problem for the coefficients
            omega_s = np.linalg.lstsq(A_Anderson, b_Anderson, rcond=None)[0]
            omega_s = np.append(omega_s, 1.0 - sum(omega_s))

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

    data = [interItr[0], interItr[2], interItr[3]]
    return xks[mk + 2, ::], data


#-----------------------------------------------------------------------------------------------------------------------

def get_complete_solution(sol, indices, *args):

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    tip_act = np.concatenate((to_impose, active))

    w = np.copy(frac.w)
    w[to_solve] += sol[:len(to_solve)]
    w[to_impose] = imposed_val
    w[active] = wc_to_impose

    [ch_indxs, tip_indxs, act_indxs, deleted] = indices
    
    if sim_prop.solveDeltaP:
        values = np.dot(C[np.ix_(tip_act[deleted], EltCrack)], w[EltCrack]) + \
                    mat_prop.SigmaO[tip_act[deleted]]- frac.pFluid[tip_act[deleted]]
    else:
        values = np.dot(C[np.ix_(tip_act[deleted], EltCrack)], w[EltCrack]) + \
                    mat_prop.SigmaO[tip_act[deleted]]
    sol_full = populate_full(indices, sol, values)

    return sol_full

def populate_full(indices, sol, values=None):

    [ch_indxs, tip_indxs, act_indxs, deleted] = indices
    sol_full = np.empty(len(ch_indxs) + len(tip_indxs) + len(act_indxs))
    sol_full[:len(ch_indxs)] = sol[:len(ch_indxs)]

    if values is None:
        values = np.zeros(len(deleted))
    sol_full[len(ch_indxs) + np.asarray(deleted, dtype=int)] = values
    sol_full[len(ch_indxs) + np.setdiff1d(np.arange(len(tip_indxs) + len(act_indxs)), deleted)] = sol[len(ch_indxs):]

    return sol_full
