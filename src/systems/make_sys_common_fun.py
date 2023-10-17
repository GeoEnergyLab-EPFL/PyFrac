# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Dec 28 14:43:38 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from numba import jit, prange
import copy

# Internal imports
from fluid.fluid_model import friction_factor_vector, friction_factor_MDR

#@jit()
def finiteDiff_operator_laminar(w, EltCrack, muPrime, NeiElements, dx, dy, InCrack, neiInCrack, sparse_flag=True):
    """
    The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
    equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated with the laminar flow assumption.

    Args:
        w (ndarray):            -- the width of the trial fracture.
        EltCrack (ndarray):     -- the list of elements inside the fracture.
        muPrime (ndarray):      -- the scaled local viscosity of the injected fluid (12 * viscosity).
        Mesh.NeiElements (CartesianMesh):   -- see mesh class
        dx (CartesianMesh):     -- see Mesh.hx in the mesh class
        dy (CartesianMesh):     -- see Mesh.hy in the mesh class
        InCrack (ndarray):      -- an array specifying whether elements are inside the fracture or not with
                                   1 or 0 respectively.
        neiInCrack (ndarray):   -- an ndarray giving indices of the neighbours of all the cells in the crack, in the
                                   EltCrack list.
        sparse_flag (bool):     -- A bool to decide if returning a sparse matrix or a full matrix.

    Returns:
        FinDiffOprtr (ndarray): -- the finite difference matrix.

    """

    if sparse_flag:
        FinDiffOprtr = sparse.lil_matrix((len(EltCrack), len(EltCrack) + 1), dtype=np.float64)
    else:
        FinDiffOprtr = np.zeros((len(EltCrack), len(EltCrack) + 1), dtype=np.float64)

    # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
    wLftEdge = (w[EltCrack] + w[NeiElements[EltCrack, 0]]) / 2 * InCrack[NeiElements[EltCrack, 0]]
    wRgtEdge = (w[EltCrack] + w[NeiElements[EltCrack, 1]]) / 2 * InCrack[NeiElements[EltCrack, 1]]
    wBtmEdge = (w[EltCrack] + w[NeiElements[EltCrack, 2]]) / 2 * InCrack[NeiElements[EltCrack, 2]]
    wTopEdge = (w[EltCrack] + w[NeiElements[EltCrack, 3]]) / 2 * InCrack[NeiElements[EltCrack, 3]]

    indx_elts = np.arange(len(EltCrack))
    FinDiffOprtr[indx_elts, indx_elts] = (-(wLftEdge ** 3 + wRgtEdge ** 3) / dx ** 2 - (
            wBtmEdge ** 3 + wTopEdge ** 3) / dy ** 2) / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 0]] = wLftEdge ** 3 / dx ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = wRgtEdge ** 3 / dx ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = wBtmEdge ** 3 / dy ** 2 / muPrime
    FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = wTopEdge ** 3 / dy ** 2 / muPrime

    return FinDiffOprtr


# -----------------------------------------------------------------------------------------------------------------------

def Gravity_term(w, EltCrack, fluidProp, mesh, InCrack, solidProp, simProp):
    """
    This function returns the gravity term (G in Zia and Lecampion 2019).

    Args:
        w (ndarray):                -- the width of the trial fracture.
        EltCrack (ndarray):         -- the list of elements inside the fracture.
        fluidProp (object):         -- FluidProperties class object giving the fluid properties.
        Mesh (CartesianMesh):       -- the mesh.
        InCrack (ndarray):          -- An array specifying whether elements are inside the fracture or not with
                                       1 or 0 respectively.
        solidProp (object):         -- An object of the materialProperties class.
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
            wLeftEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 0]]) / 2 * InCrack[mesh.NeiElements[EltCrack, 0]]
            wRightEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 1]]) / 2 * InCrack[mesh.NeiElements[EltCrack, 1]]

            G[EltCrack] = -fluidProp.density * solidProp.gravityValue[2 * EltCrack] * \
                          (wLeftEdge ** 3 - wRightEdge ** 3) / mesh.hy / fluidProp.muPrime - \
                          fluidProp.density * solidProp.gravityValue[2 * EltCrack - 1] * \
                          (wTopEdge ** 3 - wBtmEdge ** 3) / mesh.hy / fluidProp.muPrime
        else:
            raise SystemExit("Effect of gravity is only supported for Newtonian fluid in laminar flow regime yet!")

    return G


# -----------------------------------------------------------------------------------------------------------------------


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
    rough = w[EltCrack] / matProp.grainSize
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
    dp[4, EltCrack] = (dp[2, mesh.NeiElements[EltCrack, 0]] + dp[3, mesh.NeiElements[EltCrack, 0]] + dp[2, EltCrack] +
                       dp[3, EltCrack]) / 4
    dp[5, EltCrack] = (dp[2, mesh.NeiElements[EltCrack, 1]] + dp[3, mesh.NeiElements[EltCrack, 1]] + dp[2, EltCrack] +
                       dp[3, EltCrack]) / 4
    dp[6, EltCrack] = (dp[0, mesh.NeiElements[EltCrack, 2]] + dp[1, mesh.NeiElements[EltCrack, 2]] + dp[0, EltCrack] +
                       dp[1, EltCrack]) / 4
    dp[7, EltCrack] = (dp[0, mesh.NeiElements[EltCrack, 3]] + dp[1, mesh.NeiElements[EltCrack, 3]] + dp[0, EltCrack] +
                       dp[1, EltCrack]) / 4

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
            if Velocity_Residual(np.finfo(float).eps * vkm1[3, EltCrack[i]], *arg) * Velocity_Residual(
                    upBracket_factor * vkm1[3, EltCrack[i]], *arg) > 0.0:
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


# -----------------------------------------------------------------------------------------------------------------------


def Velocity_Residual(v, *args):
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
    Re = 4 / 3 * rho * w * v / mu

    # friction factor using MDR approximation
    f = friction_factor_MDR(Re, rough)

    return v - w * dp / (v * rho * f)


# -----------------------------------------------------------------------------------------------------------------------


def findBracket(func, guess, *args):
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
    b = max(1000 * guess, 1)
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


# -----------------------------------------------------------------------------------------------------------------------

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
        eff_mu[0, EltCrack] = wLftEdge ** 3 / (12 * cond[0, :])
        eff_mu[1, EltCrack] = wRgtEdge ** 3 / (12 * cond[1, :])
        eff_mu[2, EltCrack] = wBtmEdge ** 3 / (12 * cond[2, :])
        eff_mu[3, EltCrack] = wTopEdge ** 3 / (12 * cond[3, :])

    return FinDiffOprtr, eff_mu


# -----------------------------------------------------------------------------------------------------------------------

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
    cond[0, edgeInCrk_lst[0]] = fluidProp.var1 * dpLft[edgeInCrk_lst[0]] ** fluidProp.var2 * wLftEdge[
        edgeInCrk_lst[0]] ** \
                                fluidProp.var3 * x0[edgeInCrk_lst[0]] ** fluidProp.var4 * \
                                (1 + 2 * fluidProp.T0 / wLftEdge[edgeInCrk_lst[0]] / dpLft[
                                    edgeInCrk_lst[0]] * fluidProp.var5)
    x1 = np.maximum(1 - 2 * fluidProp.T0 / wRgtEdge / dpRgt, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[1, edgeInCrk_lst[1]] = fluidProp.var1 * dpRgt[edgeInCrk_lst[1]] ** fluidProp.var2 * wRgtEdge[
        edgeInCrk_lst[1]] ** \
                                fluidProp.var3 * x1[edgeInCrk_lst[1]] ** fluidProp.var4 * \
                                (1 + 2 * fluidProp.T0 / wRgtEdge[edgeInCrk_lst[1]] / dpRgt[
                                    edgeInCrk_lst[1]] * fluidProp.var5)
    x2 = np.maximum(1 - 2 * fluidProp.T0 / wBtmEdge / dpBtm, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[2, edgeInCrk_lst[2]] = fluidProp.var1 * dpBtm[edgeInCrk_lst[2]] ** fluidProp.var2 * wBtmEdge[
        edgeInCrk_lst[2]] ** \
                                fluidProp.var3 * x2[edgeInCrk_lst[2]] ** fluidProp.var4 * \
                                (1 + 2 * fluidProp.T0 / wBtmEdge[edgeInCrk_lst[2]] / dpBtm[
                                    edgeInCrk_lst[2]] * fluidProp.var5)
    x3 = np.maximum(1 - 2 * fluidProp.T0 / wTopEdge / dpTop, np.zeros(len(wLftEdge), dtype=np.float64))
    cond[3, edgeInCrk_lst[3]] = fluidProp.var1 * dpTop[edgeInCrk_lst[3]] ** fluidProp.var2 * wTopEdge[
        edgeInCrk_lst[3]] ** \
                                fluidProp.var3 * x3[edgeInCrk_lst[3]] ** fluidProp.var4 * \
                                (1 + 2 * fluidProp.T0 / wTopEdge[edgeInCrk_lst[3]] / dpTop[
                                    edgeInCrk_lst[3]] * fluidProp.var5)

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
            eff_mu[0, EltCrack[edgeInCrk_lst[0]]] = wLftEdge[edgeInCrk_lst[0]] ** 3 / (12 * cond[0, edgeInCrk_lst[0]])
            eff_mu[1, EltCrack[edgeInCrk_lst[1]]] = wRgtEdge[edgeInCrk_lst[1]] ** 3 / (12 * cond[1, edgeInCrk_lst[1]])
            eff_mu[2, EltCrack[edgeInCrk_lst[2]]] = wBtmEdge[edgeInCrk_lst[2]] ** 3 / (12 * cond[2, edgeInCrk_lst[2]])
            eff_mu[3, EltCrack[edgeInCrk_lst[3]]] = wTopEdge[edgeInCrk_lst[3]] ** 3 / (12 * cond[3, edgeInCrk_lst[3]])

    if simProp.saveYieldRatio:
        yielded = np.zeros((4, Mesh.NumberOfElts), dtype=np.float64)
        yielded[0, EltCrack] = x0
        yielded[1, EltCrack] = x1
        yielded[2, EltCrack] = x2
        yielded[3, EltCrack] = x3

    return FinDiffOprtr, eff_mu, yielded


# ----------------------------------------------------------------------------------------------------------------------------------------

def get_finite_difference_matrix(wNplusOne, sol, frac_n, EltCrack, neiInCrack, fluid_prop, mat_prop, sim_prop, mesh,
                                 InCrack, C, interItr, to_solve, to_impose, active, interItr_kp1, list_edgeInCrack):
    if fluid_prop.rheology == 'Newtonian' and not fluid_prop.turbulence:
        FinDiffOprtr = finiteDiff_operator_laminar(wNplusOne,
                                                   EltCrack,
                                                   fluid_prop.muPrime,
                                                   mesh.NeiElements,
                                                   mesh.hx,
                                                   mesh.hy,
                                                   InCrack,
                                                   neiInCrack,
                                                   sim_prop.solveSparse)

    else:
        pf = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        # pressure evaluated by dot product of width and elasticity matrix
        pf[to_solve] = np.dot(C[np.ix_(to_solve, EltCrack)], wNplusOne[EltCrack]) + mat_prop.SigmaO[to_solve]
        if sim_prop.solveDeltaP:
            pf[active] = frac_n.pFluid[active] + sol[len(to_solve):len(to_solve) + len(active)]
            #to implement injection line
            #pf[to_impose] = frac_n.pFluid[to_impose] + sol[len(to_solve) + len(active):]
            pf[to_impose] = frac_n.pFluid[to_impose] + sol[len(to_solve) + len(active):
                                                            len(to_solve) + len(active) + len(to_impose)]

        else:
            pf[active] = sol[len(to_solve):len(to_solve) + len(active)]
            #to implement injection line
            #pf[to_impose] = sol[len(to_solve) + len(active):]
            pf[to_impose] = sol[len(to_solve) + len(active):
                                len(to_solve) + len(active) + len(to_impose)]


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

def calculate_fluid_flow_characteristics_laminar(w, pf, sigma0, Mesh, EltCrack, InCrack, muPrime, density, simulProp,
                                                 solid):
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
        (dpdxLft, dpdxRgt, dpdyBtm, dpdyTop) = pressure_gradient_form_pressure(pf, Mesh, EltCrack, InCrack)
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

        if simulProp.gravity:
            # we average the density between neighboring cells
            rhoLftEdge = (solid.density[EltCrack] + solid.density[Mesh.NeiElements[EltCrack, 0]]) / 2
            rhoRgtEdge = (solid.density[EltCrack] + solid.density[Mesh.NeiElements[EltCrack, 1]]) / 2
            rhoBtmEdge = (solid.density[EltCrack] + solid.density[Mesh.NeiElements[EltCrack, 2]]) / 2
            rhoTopEdge = (solid.density[EltCrack] + solid.density[Mesh.NeiElements[EltCrack, 3]]) / 2

            # we add the gravity term for fluid flux
            fluid_flux[0, :] = fluid_flux[-1, :] +\
                                (-wLftEdge ** 3 / muPrime * (rhoLftEdge - density) *
                                 simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux[1, :] = fluid_flux[-2, :] +\
                                (-wRgtEdge ** 3 / muPrime * (rhoRgtEdge - density) *
                                 simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux[2, :] = fluid_flux[-2, :] +\
                                (-wBtmEdge ** 3 / muPrime * (rhoBtmEdge - density) *
                                 simulProp.gravityValue[2 * EltCrack])
            fluid_flux[3, :] = fluid_flux[-1, :] +\
                                (-wTopEdge ** 3 / muPrime * (rhoTopEdge - density) *
                                 simulProp.gravityValue[2 * EltCrack])

            # we add the gravity term for the fluid flux components.
            fluid_flux_components[0, :] = fluid_flux_components[0, :] +\
                                          (-wLftEdge ** 3 / muPrime * (rhoLftEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux_components[1, :] = fluid_flux_components[1, :] +\
                                          (-wLftEdge ** 3 / muPrime * (rhoLftEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack])
            fluid_flux_components[2, :] = fluid_flux_components[2, :] +\
                                          (-wRgtEdge ** 3 / muPrime * (rhoRgtEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux_components[3, :] = fluid_flux_components[3, :] +\
                                          (-wRgtEdge ** 3 / muPrime * (rhoRgtEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack])
            fluid_flux_components[4, :] = fluid_flux_components[4, :] +\
                                          (-wBtmEdge ** 3 / muPrime * (rhoBtmEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux_components[5, :] = fluid_flux_components[5, :] +\
                                          (-wBtmEdge ** 3 / muPrime * (rhoBtmEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack])
            fluid_flux_components[6, :] = fluid_flux_components[6, :] +\
                                          (-wTopEdge ** 3 / muPrime * (rhoTopEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack - 1])
            fluid_flux_components[7, :] = fluid_flux_components[7, :] +\
                                          (-wTopEdge ** 3 / muPrime * (rhoTopEdge - density) *
                                           simulProp.gravityValue[2 * EltCrack])

        fluid_vel = copy.deepcopy(fluid_flux)
        wEdges = [wLftEdge, wRgtEdge, wBtmEdge, wTopEdge]
        for i in range(4):
            local_nonzero_indexes=fluid_vel[i].nonzero()
            fluid_vel[i][local_nonzero_indexes] /= wEdges[i][local_nonzero_indexes]

        fluid_vel_components = copy.deepcopy(fluid_flux_components)
        for i in range(8):
            local_nonzero_indexes=fluid_vel_components[i].nonzero()
            fluid_vel_components[i][local_nonzero_indexes] /= wEdges[int(np.trunc(i/2))][local_nonzero_indexes]

        Rey_number = abs(4 / 3 * density * fluid_flux / muPrime * 12)

        return abs(fluid_flux), abs(fluid_vel), Rey_number, fluid_flux_components, fluid_vel_components
    else:
        raise SystemExit('ERROR: if the fluid viscosity is equal to 0 does not make sense to compute the fluid velocity or the fluid flux')



