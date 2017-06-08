#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Nov 16 18:33:56 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.

Hydraulic fracture propagation Analytical solutions
notably for
 - planar radial fracture with constant injection rate
 - height contained fracture
"""

# imports
import numpy as np
from scipy import interpolate


# ----------------------------------------------------
def M_vertex_solution_R_given(Eprime, Q0, muPrime, Mesh, R):
    """
    Analytical solution for Viscosity dominated (M vertex) fracture propagation, given fracture radius. The solution
    does not take leak off into account.
    
    Arguments:
        Eprime (float):         plain strain elastic modulus
        Q0 (float):             injection rate
        muPrime (float):        12*viscosity
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        R (float):              the given radius for which the solution is evaluated
        
    Returns:
        float:                  time at which the fracture reached the given radius
        ndarray-float:          pressure at each cell when the fracture has propagated to the given radius
        ndarray-float:          width at each cell when the fracture has propagated to the given radius
        float:                  fracture propagation velocity
    """

    t = (2.24846 * R ** (9 / 4) * muPrime ** (1 / 4)) / (Eprime ** (1 / 4) * Q0 ** (3 / 4))

    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actv = np.where(rho <= 1)[0] # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = -2 + 2 * rho[actv]
    var2 = 1 - rho[actv]

    # todo: cite where the solution is taken from
    w[actv] = (1 / (Eprime ** (2 / 9))) * 0.6976 * Q0 ** (1 / 3) * t ** (1 / 9) * muPrime ** (2 / 9) * (
    1.89201 * var2 ** (2 / 3) + 0.000663163 * var2 ** (2 / 3) * (
    35 / 9 + 80 / 9 * var1 + 38 / 9 * var1 ** 2) + 0.00314291 * var2 ** (2 / 3) * (
    455 / 81 + 1235 / 54 * var1 + 2717 / 108 * var1 ** 2 + 5225 / 648 * var1 ** 3) + 0.000843517 * var2 ** (2 / 3) * (
    1820 / 243 + 11440 / 243 * var1 + 7150 / 81 * var1 ** 2 + 15400 / 243 * var1 ** 3 + (
    59675 * var1 ** 4) / 3888) + 0.102366 * var2 ** (2 / 3) * (1 / 3 + 13 / 3 * (-1 + 2 * rho[actv])) + 0.237267 * (
    (1 - rho[actv] ** 2) ** 0.5 - rho[actv] * np.arccos(rho[actv])))

    p[actv] = (0.0931746 * Eprime ** (2 / 3) * muPrime ** (1 / 3) * (
    -2.20161 + 8.81828 * (1 - rho[actv]) ** (1 / 3) - 0.0195787 * rho[actv] - 0.171565 * rho[actv] ** 2 - 0.103558 *
    rho[actv] ** 3 + (1 - rho[actv]) ** (1 / 3) * np.log(1 / rho[actv]))) / (t ** (1 / 3) * (1 - rho[actv]) ** (1 / 3))
    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure)
    t1 = (2.24846 * (1.01 * R) ** (9 / 4) * muPrime ** (1 / 4)) / (Eprime ** (1 / 4) * Q0 ** (3 / 4))
    v = 0.01 * R / (t1 - t)

    return (t, p, w, v)

# ----------------------------------------------------------------------------------------------------------------------

def M_vertex_solution_t_given(Eprime, Q0, muPrime, Mesh, t):
    """
    Analytical solution for Viscosity dominated (M vertex, see e.g. Detournay 2016, Annu. Rev. Fluid Mech.) fracture
    propagation, given current time. The solution does not take leak off into account.
    
    Arguments:
        Eprime (float):         plain strain elastic modulus
        Q0 (float):             injection rate
        muPrime (float):        12*viscosity
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        t (float):              the given time for which the solution is evaluated    
    
    Returns:
        float:                  radius of the fracture attained at the given time
        ndarray-float:          pressure at each cell at the given time
        ndarray-float:          width at each cell at the given time
        float:                  fracture propagation velocity
    """

    R = (0.6976 * Eprime ** (1 / 9) * Q0 ** (1 / 3) * t ** (4 / 9)) / muPrime ** (1 / 9)
    w = np.zeros((Mesh.NumberOfElts,), float)
    p = np.zeros((Mesh.NumberOfElts,), float)
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actv = np.where(rho <= 1) # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = -2 + 2 * rho[actv]
    var2 = 1 - rho[actv]

    w[actv] = (1 / (Eprime ** (2 / 9))) * 0.6976 * Q0 ** (1 / 3) * t ** (1 / 9) * muPrime ** (2 / 9) * (
    1.89201 * var2 ** (2 / 3) + 0.000663163 * var2 ** (2 / 3) * (
    35 / 9 + 80 / 9 * var1 + 38 / 9 * var1 ** 2) + 0.00314291 * var2 ** (2 / 3) * (
    455 / 81 + 1235 / 54 * var1 + 2717 / 108 * var1 ** 2 + 5225 / 648 * var1 ** 3) + 0.000843517 * var2 ** (2 / 3) * (
    1820 / 243 + 11440 / 243 * var1 + 7150 / 81 * var1 ** 2 + 15400 / 243 * var1 ** 3 + (
    59675 * var1 ** 4) / 3888) + 0.102366 * var2 ** (2 / 3) * (1 / 3 + 13 / 3 * (-1 + 2 * rho[actv])) + 0.237267 * (
    (1 - rho[actv] ** 2) ** 0.5 - rho[actv] * np.arccos(rho[actv])))

    p[actv] = (0.0931746 * Eprime ** (2 / 3) * muPrime ** (1 / 3) * (
    -2.20161 + 8.81828 * (1 - rho[actv]) ** (1 / 3) - 0.0195787 * rho[actv] - 0.171565 * rho[actv] ** 2 - 0.103558 *
    rho[actv] ** 3 + (1 - rho[actv]) ** (1 / 3) * np.log(1 / rho[actv]))) / (t ** (1 / 3) * (1 - rho[actv]) ** (1 / 3))

    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to acvance by one percent (not sure)
    t1 = (2.24846 * (1.01 * R) ** (9 / 4) * muPrime ** (1 / 4)) / (Eprime ** (1 / 4) * Q0 ** (3 / 4))
    v = 0.01 * R / (t1 - t)

    return (R, p, w, v)


def K_vertex_solution_r_given(Kprime, Eprime, Q0, mesh, R):
    """
    Analytical solution for toughness dominated (K vertex) fracture propagation, given current radius. The solution
    does not take leak off into account.
    
    Arguments:
        Kprime (float):         4*(2/pi)**0.5 * K1c, where K1c is the linear-elastic plane-strain fracture toughness
        Eprime (float):         plain strain elastic modulus
        Q0 (float):             injection rate
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        r (float):              the given radius for which the solution is evaluated    
    
    Returns:
        float:                  time at which the fracture attained the given radius
        ndarray-float:          pressure at each cell at the given radius
        ndarray-float:          width at each cell at the given radius
        float:                  propagation velocity
    """

    t = 2 ** 0.5 * Kprime[mesh.CenterElts][0] * np.pi * R ** (5 / 2) / (3 * Eprime * Q0)
    p = np.pi / 8 * (np.pi / 12) ** (1 / 5) * (Kprime[mesh.CenterElts][0] ** 6 / (Eprime * Q0 * t)) ** (1 / 5) * np.ones(
        (mesh.NumberOfElts,), float)

    w = np.zeros((mesh.NumberOfElts,))
    rad = (mesh.CenterCoor[:, 0] ** 2 + mesh.CenterCoor[:, 1] ** 2) ** 0.5 # distance from center
    actv = np.where(rad < R) # active cells (inside fracture)
    w[actv] = (3 / 8 / np.pi) ** 0.2 * (Q0 * Kprime[mesh.CenterElts][0] ** 4 * t / Eprime ** 4) ** 0.2 * (
                                                                                        1 - (rad[actv] / R) ** 2) ** 0.5

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure about that)
    t1 = 2 ** 0.5 * Kprime[mesh.CenterElts][0] * np.pi * (1.01 * R) ** (5 / 2) / (3 * Eprime * Q0)
    v = 0.01 * R / (t1 - t)
    return (t, p, w, v)


def K_vertex_solution_t_given(Kprime, Eprime, Q0, mesh, t):
    """
    Analytical solution for toughness dominated (K vertex) fracture propagation, given current time. The solution
    does not take leak off into account.
    
    Arguments:
        Kprime (float):         4*(2/pi)**0.5 * K1c, where K1c is the linear-elastic plane-strain fracture toughness
        Eprime (float):         plain strain elastic modulus
        Q0 (float):             injection rate
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        t (float):              the given time for which the solution is evaluated    
    
    Returns:
        float:                  radius attained by the fracture at the given time
        ndarray-float:          pressure at each cell at the given time
        ndarray-float:          width at each cell at the given time
        float:                  propagation velocity
    """

    R = (3 / 2 ** 0.5 / np.pi * Q0 * Eprime * t / Kprime[mesh.CenterElts][0]) ** 0.4
    p = np.pi / 8 * (np.pi / 12) ** (1 / 5) * (Kprime[mesh.CenterElts][0] ** 6 / (Eprime * Q0 * t)) ** (1 / 5)* np.ones(
        (mesh.NumberOfElts,), float)

    w = np.zeros((mesh.NumberOfElts,))
    rad = (mesh.CenterCoor[:, 0] ** 2 + mesh.CenterCoor[:, 1] ** 2) ** 0.5 # distance from center
    actv = np.where(rad < R) # active cells (inside fracture)
    w[actv] = (3 / 8 / np.pi) ** 0.2 * (Q0 * Kprime[mesh.CenterElts][0] ** 4 * t / Eprime ** 4) ** 0.2 * (
                                                                                        1 - (rad[actv] / R) ** 2) ** 0.5

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure about that)
    t1 = 2 ** 0.5 * Kprime[mesh.CenterElts][0] * np.pi * (1.01 * R) ** (5 / 2) / (3 * Eprime * Q0)
    v = 0.01 * R / (t1 - t)
    return (R, p, w, v)


def Mt_vertex_solution_r_given(Eprime, Cprime, Q0, muPrime, Mesh, R):
    """
    Analytical solution for viscosity dominated (M tilde vertex) fracture propagation, given current time. The solution
    takes leak off into account.
    
    Arguments:
        Eprime (float):         plain strain elastic modulus
        Cprime (float):         2*C, where C is the Carter's leak off coefficient 
        Q0 (float):             injection rate
        muPrime (float):        12*viscosity
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        R (float):              the given radius for which the solution is evaluated    
    
    Returns:
        float:                  time at which the given radius is attained by the fracture
        ndarray-float:          pressure at each cell at the given radius
        ndarray-float:          width at each cell at the given radius
        float:                  propagation velocity
    """

    t = Cprime ** 2 * R ** 4 * np.pi ** 4 / (4 * Q0 ** 2)

    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actv = np.where(rho <= 1) # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = (1 - rho[actv]) ** 0.375
    var2 = (1 - rho[actv]) ** 0.625

    # todo: cite where the solution is taken from
    w[actv] = (0.07627790025007182 * Q0 ** 0.375 * t ** 0.0625 * muPrime ** 0.25 * (
    11.40566553791626 * var2 + 7.049001601162521 * var2 * rho[actv] - 0.6802327798216378 * var2 * rho[
        actv] ** 2 - 0.828297356390819 * var2 * rho[actv] ** 3 + var2 * rho[actv] ** 4 + 2.350633434009811 * (
    1 - rho[actv] ** 2) ** 0.5 - 2.350633434009811 * rho[actv] * np.arccos(rho[actv]))) / (
              Cprime ** 0.125 * Eprime ** 0.25)

    p[actv] = (0.156415 * Cprime ** 0.375 * Eprime ** 0.75 * muPrime ** 0.25 * (
    -1.0882178530759854 + 6.3385626500863985 * var1 - 0.07314343477396379 * rho[actv] - 0.21802875891750756 * rho[
        actv] ** 2 - 0.04996007983993901 * rho[actv] ** 3 + 1. * var1 * np.log(1 / rho[actv]))) / (
              Q0 ** 0.125 * var1 * t ** 0.1875)

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure about that)
    t1 = Cprime ** 2 * (1.01 * R) ** 4 * np.pi ** 4 / 4 / Q0 ** 2
    v = 0.01 * R / (t1 - t)

    return (t, p, w, v)


def MT_vertex_solution_t_given(Eprime, Cprime, Q0, muPrime, Mesh, t):
    """
    Analytical solution for viscosity dominated (M tilde vertex) fracture propagation, given current time. The solution
    takes leak off into account.
    
    Arguments:
        Eprime (float):         plain strain elastic modulus
        Cprime (float):         2*C, where C is the Carter's leak off coefficient 
        Q0 (float):             injection rate
        muPrime (float):        12*viscosity
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        t (float):              the given time for which the solution is evaluated    
    
    Returns:
        float:                  radius attained by the fracture at the given time 
        ndarray-float:          pressure at each cell at the given time
        ndarray-float:          width at each cell at the given time
        float:                  propagation velocity
    """

    R = (2 * Q0 / Cprime) ** 0.5 * t ** 0.25 / np.pi

    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actv = np.where(rho <= 1) # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = (1 - rho[actv]) ** 0.375
    var2 = (1 - rho[actv]) ** 0.625

    w[actv] = (0.07627790025007182 * Q0 ** 0.375 * t ** 0.0625 * muPrime ** 0.25 * (
    11.40566553791626 * var2 + 7.049001601162521 * var2 * rho[actv] - 0.6802327798216378 * var2 * rho[
        actv] ** 2 - 0.828297356390819 * var2 * rho[actv] ** 3 + 1. * var2 * rho[actv] ** 4 + 2.350633434009811 * (
    1 - rho[actv] ** 2) ** 0.5 - 2.350633434009811 * rho[actv] * np.arccos(rho[actv]))) / (
              Cprime ** 0.125 * Eprime ** 0.25)

    p[actv] = (0.156415 * Cprime ** 0.375 * Eprime ** 0.75 * muPrime ** 0.25 * (
    -1.0882178530759854 + 6.3385626500863985 * var1 - 0.07314343477396379 * rho[actv] - 0.21802875891750756 * rho[
        actv] ** 2 - 0.04996007983993901 * rho[actv] ** 3 + 1. * var1 * np.log(1 / rho[actv]))) / (
              Q0 ** 0.125 * var1 * t ** 0.1875)

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure about that)
    t1 = Cprime ** 2 * (1.01 * R) ** 4 * np.pi ** 4 / 4 / Q0 ** 2
    v = 0.01 * R / (t1 - t)

    return (R, p, w, v)



def PKN_solution(Eprime, Q0, muPrime, Mesh, t, h):
    """
    Analytical solution for heigth contained hydraulic fracture (PKN geometry), given current time. The solution
    does not take leak off into account.

    Arguments:
        Eprime (float):         plain strain elastic modulus
        Q0 (float):             injection rate
        muPrime (float):        12*viscosity
        Mesh (CartesianMesh):   a CartesianMesh class object describing the grid.
        t (float):              the given time for which the solution is evaluated
        h (float):              the height of the PKN fracture

    Returns:
        float:                  length of the fracture at the given time 
        ndarray-float:          pressure at each cell at the given time
        ndarray-float:          width at each cell at the given time
        float:                  propagation velocity
        ndarray-int:            list of cells inside the PKN fracture at the given time
    """

    # length of the fracture at the given time
    sol_l = (2 * (Q0 / 2) ** 3 * Eprime / np.pi ** 3 / muPrime * 12 / h ** 4) ** (1 / 5) * (t) ** (4 / 5)

    x = np.linspace(-sol_l, sol_l, int(Mesh.nx))

    # one dimensional solution for average width along the width of the PKN fracture. The solution is approximated with
    # the power of 1/3 and not evaluate with the series.
    sol_w = (np.pi ** 3 * muPrime / 12 * (Q0 / 2) ** 2 * (t) / Eprime / h / 2) ** (1 / 5) * 1.32 * (1 - abs(
        x) / sol_l) ** (1 / 3)
    # interpolation function to calculate width at any length.
    anltcl_w = interpolate.interp1d(x, sol_w)

    # cells inside the PKN fracture
    PKN_v = np.where(abs(Mesh.CenterCoor[:, 1]) <= h / 2)
    PKN_h = np.where(abs(Mesh.CenterCoor[:, 0]) <= sol_l)
    PKN = np.intersect1d(PKN_v, PKN_h)

    w = np.zeros((Mesh.NumberOfElts,), float)
    # calculating width across the cross section of the fracture from the average width.
    # The average width is given by the interpolation function.
    w[PKN] = 4 / np.pi * anltcl_w(Mesh.CenterCoor[PKN, 0]) * (1 - 4 * Mesh.CenterCoor[PKN, 1] ** 2 / h ** 2) ** 0.5

    # calculating pressure from width
    p = np.zeros((Mesh.NumberOfElts,), float)
    p[PKN] = 2 * Eprime * anltcl_w(Mesh.CenterCoor[PKN, 0]) / (np.pi * h)

    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to acvance by one percent (not sure)
    t1 = (1.01 * sol_l / (2 * (Q0 / 2) ** 3 * Eprime / np.pi ** 3 / muPrime * 12 / h ** 4) ** (1 / 5)) ** (5 / 4)
    v = 0.01 * sol_l / (t1 - t)

    return (sol_l, p, w, v, PKN)
