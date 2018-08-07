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
import warnings
from scipy import special
import scipy.integrate as integrate


#  radial zero toughness turbulent MDR approximation
def MDR_M_vertex_solution(Eprime,Q0,density,visc,Mesh,R=None,t=None):
    """
       Analytical solution for a radial hydraulic fracture (zero toughness case) in the Max drag reduction flow regime
       propagation, given fracture radius or the time since the start of the injection. The solution does not take leak off into account.

       Arguments:
           Eprime (float)       -- plain strain elastic modulus.
           Q0 (float)           -- injection rate.
           density (float)      -- fluid density.
           visc (float)         -- fluid viscosity.
           Mesh (CartesianMesh) -- a CartesianMesh class object describing the grid.
           R (float)            -- the given radius for which the solution is evaluated.
           t (float)            -- time since the start of the injection.

       Returns:
           t (float)            -- time at which the fracture reaches the given radius.
           R (float)            -- radius of the fracture at the given time.
           p (ndarray)          -- pressure at each cell when the fracture has propagated to the given radius
           w (ndarray)          -- width opening at each cell when the fracture has propagated to the given radius or time
           v (float)            -- fracture propagation velocity
           actvElts (ndarray)   -- list of cells inside the fracture at the given time
       """
    fo = 1.78
    gammam= 0.758244

    if R is None and t is None:
        raise ValueError("Either radius or time must be provided!")
    elif t is None:
        t = ((3**0.175)*(fo**0.25)*(R**2.175)*(visc**0.175)*(density**0.075))/((2**0.35)*(Eprime**0.25)*(Q0**0.675)*(gammam**2.175))
    elif R is None:
        R = gammam*((2**(14./87))*(Eprime**(10./87))*(Q0**(9./29))*(t**(40/87))
                    )/((3**(7./87))*(fo**(10./87))*(visc**(7./87))*density**(1./29))

    wscale =((3**0.16091954022988506)*(fo**0.22988505747126436)*(Q0**0.3793103448275862)*(t**0.08045977011494253)*
             (visc**0.16091954022988506)*(density**0.06896551724137931))/((2**0.3218390804597701)*(Eprime**0.22988505747126436))

    pscale = ((3**0.2413793103448276)*
              (Eprime**0.6551724137931034)*(fo**0.3448275862068966)*(Q0**0.06896551724137931)*(visc**0.2413793103448276)*(density**0.10344827586206896))/((2**0.4827586206896552)*(t**0.3793103448275862))

    v = gammam * 0.45977011494252873* (t ** (0.45977011494252873-1.))  * (((2 ** 0.16091954022988506) * (Eprime ** 0.11494252873563218) * (Q0 ** 0.3103448275862069) ) /
    ((3 ** 0.08045977011494253) * (fo ** 0.11494252873563218) * (visc ** 0.08045977011494253) * (density** 0.034482758620689655)) )

    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actvElts = np.where(rho <= 1)[0] # active cells (inside fracture)

    var1 = 1 - rho[actvElts]

    w[actvElts]= wscale * (0.916848 * (var1)**0.740741 - 0.283356 * (rho[actvElts])**(7/10) + 0.683013 * (var1)**0.740741 * rho[actvElts]
    - 0.500775 * (var1)**0.740741 * (rho[actvElts]**2) +  1.07431* (var1)**0.740741 * (rho[actvElts]**3)
    - 1.2138 *(var1)**0.740741 * (rho[actvElts]**4)  + 0.577514 * (var1)**0.740741  *(rho[actvElts]**5)
    - 0.502666  * (1 - rho[actvElts]**2)**0.5 + 0.718095 * special.hyp2f1(-(7/20), 1/2, 13/20, rho[actvElts]**2.))  # Hypergeometric2F1[-(7/20), 1/2, 13/20, r^2]

    p[actvElts]= pscale * (
    0.5948935210154036 - 0.2717984030270058 / (var1 ** 0.25925925925925924) + 0.23531180369007717855 / (rho[actvElts] ** 0.3) -
    (0.16828914936530234 * rho[actvElts]) / (var1 ** 0.25925925925925924) + (0.2225733568270942 * (rho[actvElts] ** 2)) /
    (var1 ** 0.25925925925925924) - (0.2158763695654084 * (rho[actvElts] ** 3)) / (var1 ** 0.25925925925925924)
    + (0.07471989686220308 * (rho[actvElts] ** 4)) /(var1 ** 0.25925925925925924)
    )

    return t, R, p, w, v, actvElts


# ----------------------------------------------------
def M_vertex_solution(Eprime, Q0, muPrime, Mesh, R=None, t=None):
    """
    Analytical solution for Viscosity dominated (M vertex) fracture propagation, given fracture radius or the time
    since the start of the injection. The solution does not take leak off into account.
    
    Arguments:
        Eprime (float)       -- plain strain elastic modulus.
        Q0 (float)           -- injection rate.
        muPrime (float)      -- 12*viscosity.
        Mesh (CartesianMesh) -- a CartesianMesh class object describing the grid.
        R (float)            -- the given radius for which the solution is evaluated.
        t (float)            -- time since the start of the injection.
        
    Returns:
        t (float)            -- time at which the fracture reaches the given radius.
        R (float)            -- radius of the fracture at the given time.
        p (ndarray)          -- pressure at each cell when the fracture has propagated to the given radius
        w (ndarray)          -- width opening at each cell when the fracture has propagated to the given radius or time
        v (float)            -- fracture propagation velocity
        actvElts (ndarray)   -- list of cells inside the fracture at the given time
    """

    if R is None and t is None:
        raise ValueError("Either radius or time must be provided!")
    elif t is None:
        t = (2.24846 * R ** (9 / 4) * muPrime ** (1 / 4)) / (Eprime ** (1 / 4) * Q0 ** (3 / 4))
    elif R is None:
        R = (0.6976 * Eprime ** (1 / 9) * Q0 ** (1 / 3) * t ** (4 / 9)) / muPrime ** (1 / 9)

    v = (4 / 9) *  (t ** (4 / 9 -1.))  * ( (0.6976 * Eprime ** (1 / 9) * Q0 ** (1 / 3) ) / muPrime ** (1 / 9) )

    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actvElts = np.where(rho <= 1)[0] # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = -2 + 2 * rho[actvElts]
    var2 = 1 - rho[actvElts]

    # todo: cite where the solution is taken from
    w[actvElts] = (1 / (Eprime ** (2 / 9))) * 0.6976 * Q0 ** (1 / 3) * t ** (1 / 9) * muPrime ** (2 / 9) * (
    1.89201 * var2 ** (2 / 3) + 0.000663163 * var2 ** (2 / 3) * (
    35 / 9 + 80 / 9 * var1 + 38 / 9 * var1 ** 2) + 0.00314291 * var2 ** (2 / 3) * (
    455 / 81 + 1235 / 54 * var1 + 2717 / 108 * var1 ** 2 + 5225 / 648 * var1 ** 3) + 0.000843517 * var2 ** (2 / 3) * (
    1820 / 243 + 11440 / 243 * var1 + 7150 / 81 * var1 ** 2 + 15400 / 243 * var1 ** 3 + (
    59675 * var1 ** 4) / 3888) + 0.102366 * var2 ** (2 / 3) * (1 / 3 + 13 / 3 * (-1 + 2 * rho[actvElts])) + 0.237267 * (
    (1 - rho[actvElts] ** 2) ** 0.5 - rho[actvElts] * np.arccos(rho[actvElts])))

    warnings.filterwarnings("ignore")

    p[actvElts] = (0.0931746 * Eprime ** (2 / 3) * muPrime ** (1 / 3) * (
    -2.20161 + 8.81828 * (1 - rho[actvElts]) ** (1 / 3) - 0.0195787 * rho[actvElts] - 0.171565 * rho[actvElts] ** 2 -
    0.103558 * rho[actvElts] ** 3 + (1 - rho[actvElts]) ** (1 / 3) * np.log(1 / rho[actvElts]))) / (t ** (1 / 3) * (1 -
                                                                                            rho[actvElts]) ** (1 / 3))

    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to advance by one percent (not sure)
    #t1 = (2.24846 * (1.01 * R) ** (9 / 4) * muPrime ** (1 / 4)) / (Eprime ** (1 / 4) * Q0 ** (3 / 4))
    #v = 0.01 * R / (t1 - t)

    return t, R, p, w, v, actvElts

# ----------------------------------------------------------------------------------------------------------------------

def K_vertex_solution(Kprime, Eprime, Q0, mesh, R=None, t=None):
    """
    Analytical solution for toughness dominated (K vertex) fracture propagation, given current radius or time. The
    solution does not take leak off into account.
    
    Arguments:
        Kprime (float)          -- 4*(2/pi)**0.5 * K1c, where K1c is the linear-elastic plane-strain fracture toughness
        Eprime (float)          -- plain strain elastic modulus
        Q0 (float)              -- injection rate
        Mesh (CartesianMesh)    -- a CartesianMesh class object describing the grid.
        R (float)               -- the given radius for which the solution is evaluated.
        t (float)               -- time since the start of the injection.
    
    Returns:
        t (float)               -- time at which the fracture reaches the given radius.
        R (float)               -- radius of the fracture at the given time.
        p (ndarray)             -- pressure at each cell when the fracture has propagated to the given radius
        w (ndarray)             -- width opening at each cell when the fracture has propagated to the given radius or time
        v (float)               -- fracture propagation velocity
        actvElts (ndarray)      -- list of cells inside the fracture at the given time
    """

    if R is None and t is None:
        raise ValueError("Either radius or time must be provided!")
    elif t is None:
        t = 2 ** 0.5 * Kprime * np.pi * R ** (5 / 2) / (3 * Eprime * Q0)
    elif R is None:
        R = (3 / 2 ** 0.5 / np.pi * Q0 * Eprime * t / Kprime) ** 0.4

    p = np.pi / 8 * (np.pi / 12) ** (1 / 5) * (Kprime ** 6 / (Eprime * Q0 * t)) ** (1 / 5) * np.ones(
        (mesh.NumberOfElts,), float)

    w = np.zeros((mesh.NumberOfElts,))
    rad = (mesh.CenterCoor[:, 0] ** 2 + mesh.CenterCoor[:, 1] ** 2) ** 0.5 # distance from center
    actvElts = np.where(rad < R) # active cells (inside fracture)
    w[actvElts] = (3 / 8 / np.pi) ** 0.2 * (Q0 * Kprime ** 4 * t / Eprime ** 4) ** 0.2 * (
                                                                                1 - (rad[actvElts] / R) ** 2) ** 0.5

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent
    t1 = 2 ** 0.5 * Kprime * np.pi * (1.01 * R) ** (5 / 2) / (3 * Eprime * Q0)
    v = 0.01 * R / (t1 - t)

    return t, R, p, w, v, actvElts

#-----------------------------------------------------------------------------------------------------------------------


def Mt_vertex_solution(Eprime, Cprime, Q0, muPrime, Mesh, R=None, t=None):
    """
    Analytical solution for viscosity dominated (M tilde vertex) fracture propagation, given current time. The solution
    takes leak off into account.
    
    Arguments:
        Eprime (float)         -- plain strain elastic modulus
        Cprime (float)         -- 2*C, where C is the Carter's leak off coefficient
        Q0 (float)             -- injection rate
        muPrime (float)        -- 12*viscosity
        Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        R (float)              -- the given radius for which the solution is evaluated
    
    Returns:
        t (float)               -- time at which the fracture reaches the given radius.
        R (float)               -- radius of the fracture at the given time.
        p (ndarray)             -- pressure at each cell when the fracture has propagated to the given radius
        w (ndarray)             -- width opening at each cell when the fracture has propagated to the given radius or time
        v (float)               -- fracture propagation velocity
        actvElts (ndarray)      -- list of cells inside the fracture at the given time
    """

    if R is None and t is None:
        raise ValueError("Either radius or time must be provided!")
    elif t is None:
        if Cprime == 0:
            raise ValueError("leak off cannot be zero for Mt regime!")
        t = Cprime ** 2 * R ** 4 * np.pi ** 4 / (4 * Q0 ** 2)
    elif R is None:
        if Cprime == 0:
            raise ValueError("leak off cannot be zero for Mt regime!")
        R = (2 * Q0 / Cprime) ** 0.5 * t ** 0.25 / np.pi
    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R # normalized distance from center
    actvElts = np.where(rho <= 1) # active cells (inside fracture)

    # temporary variables to avoid recomputation
    var1 = (1 - rho[actvElts]) ** 0.375
    var2 = (1 - rho[actvElts]) ** 0.625

    # todo: cite where the solution is taken from
    w[actvElts] = (0.07627790025007182 * Q0 ** 0.375 * t ** 0.0625 * muPrime ** 0.25 * (
    11.40566553791626 * var2 + 7.049001601162521 * var2 * rho[actvElts] - 0.6802327798216378 * var2 * rho[
        actvElts] ** 2 - 0.828297356390819 * var2 * rho[actvElts] ** 3 + var2 * rho[actvElts] ** 4 + 2.350633434009811
    * (1 - rho[actvElts] ** 2) ** 0.5 - 2.350633434009811 * rho[actvElts] * np.arccos(rho[actvElts]))) / (
              Cprime ** 0.125 * Eprime ** 0.25)

    p[actvElts] = (0.156415 * Cprime ** 0.375 * Eprime ** 0.75 * muPrime ** 0.25 * (
    -1.0882178530759854 + 6.3385626500863985 * var1 - 0.07314343477396379 * rho[actvElts] - 0.21802875891750756 * rho[
        actvElts] ** 2 - 0.04996007983993901 * rho[actvElts] ** 3 + 1. * var1 * np.log(1 / rho[actvElts]))) / (
              Q0 ** 0.125 * var1 * t ** 0.1875)

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent
    t1 = Cprime ** 2 * (1.01 * R) ** 4 * np.pi ** 4 / 4 / Q0 ** 2
    v = 0.01 * R / (t1 - t)

    return t, R, p, w, v, actvElts

#-----------------------------------------------------------------------------------------------------------------------


def KT_vertex_solution(Eprime, Cprime, Q0, Kprime, Mesh, R=None, t=None):
    """
    Analytical solution for viscosity dominated (M tilde vertex) fracture propagation, given the current time or the
    radius.

    Arguments:
        Eprime (float)      -- plain strain elastic modulus
        Cprime (float)      -- 2*C, where C is the Carter's leak off coefficient
        Q0 (float)          -- injection rate
        Kprime (float)      -- 4*(2/pi)**0.5 * K1c, where K1c is the linear-elastic plane-strain fracture toughness
        Mesh (CartesianMesh)-- a CartesianMesh class object describing the grid.
        t (float)           -- the given time for which the solution is evaluated. Either of the time or radius can
                               be provided
        R (float)           -- the given radius for which the solution is evaluated. Either of the time or radius can
                               be provided

    Returns:
        t (float)           -- time at which the fracture reaches the given radius.
        R (float)           -- radius of the fracture at the given time.
        p (ndarray)         -- pressure at each cell when the fracture has propagated to the given radius
        w (ndarray)         -- width opening at each cell when the fracture has propagated to the given radius or time
        v (float)           -- fracture propagation velocity
        actvElts (ndarray)  -- list of cells inside the fracture at the given time
    """


    if R is None and t is None:
        raise ValueError("Either the time or the radius is required to evaluate the solution.")
    elif R is None:
        if Cprime == 0:
            raise ValueError("leak off cannot be zero for Kt regime!")
        R = 2**0.5 * Q0**0.5 * t**(1/4) / Cprime**0.5 / np.pi
    elif t is None:
        if Cprime == 0:
            raise ValueError("leak off cannot be zero for Kt regime!")
        t = (R * Cprime**0.5 * np.pi / (2 * Q0)**0.5)**4


    w = np.zeros((Mesh.NumberOfElts,))
    p = np.zeros((Mesh.NumberOfElts,))
    rho = (Mesh.CenterCoor[:, 0] ** 2 + Mesh.CenterCoor[:, 1] ** 2) ** 0.5 / R  # normalized distance from center
    actvElts = np.where(rho <= 1)  # active cells (inside fracture)

    w[actvElts] = Kprime * Q0**0.25 * (1-rho[actvElts])**0.5 * t**0.125 / (2**0.25 * Cprime**0.25 * Eprime * np.pi**0.5)

    p[actvElts] = Cprime**0.25 * Kprime * np.pi**(3/2) / (8 * 2**(3/4) * Q0**0.25 * t**0.125)

    # todo Hack: The velocity is evaluated with time taken by the fracture to advance by one percent
    t1 = (1.01 * R * Cprime**0.5 * np.pi / (2 * Q0)**0.5)**4
    v = 0.01 * R / (t1 - t)

    return t, R, p, w, v, actvElts

#-----------------------------------------------------------------------------------------------------------------------


def PKN_solution(Eprime, Q0, muPrime, Mesh, h, ell=None, t=None):
    """
    Analytical solution for heigth contained hydraulic fracture (PKN geometry), given current time. The solution
    does not take leak off into account.

    Arguments:
        Eprime (float)         -- plain strain elastic modulus
        Q0 (float)             -- injection rate
        muPrime (float)        -- 12*viscosity
        Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        t (float)              -- the given time for which the solution is evaluated
        h (float)              -- the height of the PKN fracture

    Returns:
        t (float)              -- time at which the fracture reaches the given length.
        ell (float)            -- length of the fracture at the given time
        p (ndarray-float)      -- pressure at each cell at the given time
        w (ndarray-float)      -- width at each cell at the given time
        v (float)              -- propagation velocity
        actvElts (ndarray)     -- list of cells inside the PKN fracture at the given time
    """

    if ell is None and t is None:
        raise ValueError("Either the length or the time is required to evaluate the solution.")
    elif ell is None:
        # length of the fracture at the given time
        ell = (2 * (Q0 / 2) ** 3 * Eprime / np.pi ** 3 / muPrime * 12 / h ** 4) ** (1 / 5) * t ** (4 / 5)
    elif t is None:
        t = 2**0.5 * h * ell**(5/4) * np.pi**(3/4) * muPrime**3 / (Eprime**(1/4) * Q0**(3/4))

    x = np.linspace(-ell, ell, int(Mesh.nx))

    # one dimensional solution for average width along the width of the PKN fracture. The solution is approximated with
    # the power of 1/3 and not evaluate with the series.
    sol_w = (np.pi ** 3 * muPrime / 12 * (Q0 / 2) ** 2 * (t) / Eprime / h / 2) ** (1 / 5) * 1.32 * (1 - abs(
                                                                                        x) / ell) ** (1 / 3)
    # interpolation function to calculate width at any length.
    anltcl_w = interpolate.interp1d(x, sol_w)

    # cells inside the PKN fracture
    actvElts_v = np.where(abs(Mesh.CenterCoor[:, 1]) <= h / 2)
    actvElts_h = np.where(abs(Mesh.CenterCoor[:, 0]) <= ell)
    actvElts = np.intersect1d(actvElts_v, actvElts_h)

    w = np.zeros((Mesh.NumberOfElts,), float)
    # calculating width across the cross section of the fracture from the average width.
    # The average width is given by the interpolation function.
    w[actvElts] = 4 / np.pi * anltcl_w(Mesh.CenterCoor[actvElts, 0]) * (1 - 4 * Mesh.CenterCoor[
                                                                actvElts, 1] ** 2 / h ** 2) ** 0.5

    # calculating pressure from width
    p = np.zeros((Mesh.NumberOfElts,), float)
    p[actvElts] = 2 * Eprime * anltcl_w(Mesh.CenterCoor[actvElts, 0]) / (np.pi * h)

    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to acvance by one percent
    t1 = (1.01 * ell / (2 * (Q0 / 2) ** 3 * Eprime / np.pi ** 3 / muPrime * 12 / h ** 4) ** (1 / 5)) ** (5 / 4)
    v = 0.01 * ell / (t1 - t)

    return t, ell, p, w, v, actvElts


# -----------------------------------------------------------------------------------------------------------------------

def KGD_solution_K(Eprime, Q0, Kprime, Mesh, height, ell=None, t=None):
    """
    Analytical solution plain strain hydraulic fracture (KGB geometry) in the toughness dominated regime, given current
    time or length. The solution does not take leak off into account.

    Arguments:
        Eprime (float)         -- plain strain elastic modulus
        Q0 (float)             -- injection rate
        KPrime (float)         -- 4*(2/pi)**0.5 * K1c, where K1c is the linear-elastic plane-strain fracture toughness
        Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        height (float)         -- the height of the KGD fracture (it should be much longer then length)
        ell (float)            -- length of fracture
        t (float)              -- the given time for which the solution is evaluated


    Returns:
        t (float)              -- time at which the fracture reaches the given length.
        ell (float)            -- length of the fracture at the given time
        p (ndarray-float)      -- pressure at each cell at the given time
        w (ndarray-float)      -- width at each cell at the given time
        v (float)              -- propagation velocity
        actvElts (ndarray)     -- list of cells inside the KGD fracture at the given time
    """
    # injection rate per unit height in one wing
    Q = Q0 / height

    if ell is None and t is None:
        raise ValueError("Either the length or the time is required to evaluate the solution.")
    elif ell is None:
        # length of the fracture at the given time
        ell = 0.932388 * (Eprime * Q * t / Kprime) ** (2 / 3)
    elif t is None:
        t = 1.11072 * Kprime / Eprime / Q * ell ** (3 / 2)

    x = np.linspace(-ell, ell, int(Mesh.nx))

    # one dimensional solution for average width along the width of the PKN fracture. The solution is approximated with
    # the power of 1/3 and not evaluate with the series.
    sol_w = 0.682784 * (Kprime ** 2 * Q * t / Eprime ** 2) ** (1 / 3) * (1 - (abs(x) / ell) ** 2) ** (1 / 2)

    # interpolation function to calculate width at any length.
    anltcl_w = interpolate.interp1d(x, sol_w)

    # cells inside the PKN fracture
    actvElts_v = np.where(abs(Mesh.CenterCoor[:, 1]) <= height / 2)
    actvElts_h = np.where(abs(Mesh.CenterCoor[:, 0]) <= ell)
    actvElts = np.intersect1d(actvElts_v, actvElts_h)

    w = np.zeros((Mesh.NumberOfElts,), float)
    # The average width is given by the interpolation function.
    w[actvElts] = anltcl_w(Mesh.CenterCoor[actvElts, 0])

    # calculating pressure from width
    p = np.zeros((Mesh.NumberOfElts,), float)
    p[actvElts] = 0.183074 * (Kprime ** 4 / (Eprime * Q0 * t)) ** (1 / 3)

    # todo !!! Hack: The velocity is evaluated with time taken by the fracture to acvance by one percent
    t1 = 1.11072 * Kprime / Eprime / Q * (1.01 * ell) ** (3 / 2)
    v = 0.01 * ell / (t1 - t)

    return t, ell, p, w, v, actvElts


#-----------------------------------------------------------------------------------------------------------------------

def anisotropic_toughness_elliptical_solution(KIc_max, KIc_min, Eprime, Q0, mesh, b=None, t=None):
    """
    Analytical solution for an elliptical fracture propagating in toughness dominated regime (see Zia and Lecampion,
    IJF, 2018).

    Arguments:
        KIc_max (float)        -- the fracture toughness along the minor axis
        KIc_min (float)        -- the fracture toughness along the major axis
        Eprime (float)         -- plain strain modulus
        Q0 (float)             -- injection rate
        Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        b (float)              -- the given minor axis length
        t (float)              -- the given time for which the solution is evaluated

    Returns:
        t (float)              -- time at which the fracture reaches the given length.
        b (float)              -- length of the fracture at the given time
        p (ndarray-float)      -- pressure at each cell at the given time
        w (ndarray-float)      -- width at each cell at the given time
        v (float)              -- propagation velocity
        actvElts (ndarray)     -- list of cells inside the fracture at the given time
    """

    if KIc_min is None:
        raise ValueError("Fracture toughness along both major and minor axis is to be provided! See MaterialProperties"
                         " class.")
    c = (KIc_min / KIc_max)**2

    if b is None and t is None:
        raise ValueError("Either the minor axis length or the time is required to evaluate the solution!")
    if b is None:
        b = (Q0 * t * 3 * c * Eprime / (8 * KIc_max * np.pi**0.5 ))**(2/5)
    if t is None:
        t = 8 * KIc_max * np.pi**0.5 * b**(5/2) / (3 * c * Eprime * Q0)

    a = (KIc_max / KIc_min)**2 * b
    eccentricity = (1 - b ** 2 / a ** 2) ** 0.5

    rho = 1 - (mesh.CenterCoor[:, 0] / a) ** 2 - (mesh.CenterCoor[:, 1] / b) ** 2
    actvElts = np.where(rho > 0)[0]  # active cells (inside fracture)

    p = np.zeros((mesh.NumberOfElts,), float)
    p[actvElts] = KIc_max * special.ellipe(eccentricity**2) / (np.pi * b)**0.5

    w = np.zeros((mesh.NumberOfElts,))
    w[actvElts] = 4 * b * p[mesh.CenterElts] / (Eprime * special.ellipe(eccentricity**2)) * (1 -
                                (mesh.CenterCoor[actvElts, 0] / a) ** 2 - (mesh.CenterCoor[actvElts, 1] / b) ** 2)**0.5

    t1 = 8 * KIc_max * np.pi**0.5 * (1.01 * b)**(5/2) / (3 * c * Eprime * Q0)
    v = 0.01 * b / (t1 - t)

    return t, b, p, w, v, actvElts

#-----------------------------------------------------------------------------------------------------------------------


def TI_Elasticity_elliptical_solution(mesh, gamma, Cij, Kc3, Ep3, Q0, t=None, b=None):
    """
    Analytical solution for an elliptical fracture propagating in toughness dominated regime (see Zia and Lecampion,
    IJF, 2018).

    Arguments:
        Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        b (float)              -- the given minor axis length
        a (float)              -- the given major axis length
        p (float)              -- pressure

    Returns:
        w (ndarray-float)      -- width at each cell at the given time

        actvElts (ndarray)     -- list of cells inside the fracture at the given time
    """

    if b is None and t is None:
        raise ValueError("Either the minor axis length or time is to be provided!")
    if b is None:
        b = (Q0 * t * 3 * Ep3 / (8 * gamma * Kc3 * np.pi**0.5))**(2/5)
    else:
        t = b**(5 / 2) * 8 * gamma * Kc3 * np.pi**0.5 / (3 * Q0 * Ep3)

    a = gamma * b

    C11 = Cij[0, 0]
    C12 = Cij[0, 1]
    C13 = Cij[0, 2]
    C33 = Cij[2, 2]
    C44 = Cij[3, 3]
    C66 = Cij[5, 5]

    m1 = (-C13 ** 2 + C11 * C33 - 2 * C13 * C44 - 2 * C44 ** 2 + ((C13 ** 2 - C11 * C33) * (C13 ** 2
                - C11 * C33 + 4 * C13 * C44 + 4 * C44 ** 2)) ** 0.5) / (
                     2 * (C13 * C44 + C44 ** 2))
    m2 = (-C13 ** 2 + C11 * C33 - 2 * C13 * C44 - 2 * C44 ** 2 - ((C13 ** 2 - C11 * C33) * (C13 ** 2
                - C11 * C33 + 4 * C13 * C44 + 4 * C44 ** 2)) ** 0.5) / (
                     2 * (C13 * C44 + C44 ** 2))

    gamma3 = (C44 / C66) ** 0.5

    args = (Cij, gamma, m1, m2, gamma3)

    sigma_intgrl = integrate.quad(TI_elasticity_sigma,
                                  0,
                                  2 * np.pi,
                                  args=args,
                                  points=[np.pi / 2, 3 * np.pi / 2])[0]

    w0 = 2 * (4 * (m2 - m1) * gamma3 ** 2) / (C66 * sigma_intgrl)
    rho = 1 - (mesh.CenterCoor[:, 0] / a) ** 2 - (mesh.CenterCoor[:, 1] / b) ** 2
    actvElts = np.where(rho > 0)[0]  # active cells (inside fracture)
    p = np.zeros((mesh.NumberOfElts, ), float)
    p[actvElts] = (4 * gamma * Kc3) / ((np.pi * b) ** 0.5 * w0 * Ep3)

    u0 = w0 * p * b**2 / a

    w = np.zeros((mesh.NumberOfElts,))
    w[actvElts] = u0[actvElts] * (1 - (mesh.CenterCoor[actvElts, 0] / a) ** 2 - (
                                    mesh.CenterCoor[actvElts, 1] / b) ** 2)**0.5

    return t, b, p, w, None, actvElts


#-----------------------------------------------------------------------------------------------------------------------

def TI_elasticity_sigma(theta, *args):

    (stiff_tensor, gamma, m1, m2, gamma3) = args

    C11 = stiff_tensor[0, 0]
    C13 = stiff_tensor[0, 2]
    C44 = stiff_tensor[3, 3]

    gamma1 = ((C44 + m1 * (C13 + C44)) / C11) ** 0.5
    gamma2 = ((C44 + m2 * (C13 + C44)) / C11) ** 0.5

    h1 = (m1 + 1) * (gamma3 * np.cos(theta)) ** 2 + 2 * np.sin(theta) ** 2
    h2 = (m2 + 1) * (gamma3 * np.cos(theta)) ** 2 + 2 * np.sin(theta) ** 2

    xi1 = ((gamma1 * np.cos(theta)) ** 2 + np.sin(theta) ** 2) ** 0.5
    xi2 = ((gamma2 * np.cos(theta)) ** 2 + np.sin(theta) ** 2) ** 0.5
    xi3 = ((gamma3 * np.cos(theta)) ** 2 + np.sin(theta) ** 2) ** 0.5

    rgam = (gamma ** 2 * np.cos(theta) ** 2 + np.sin(theta) ** 2) ** 0.5

    dd = m2 * h1 ** 2 * xi2 - m1 * h2 ** 2 * xi1 + 4 * (m1 - m2) * xi2 * xi1 * xi3 * np.sin(theta) ** 2

    sigma = dd / (xi1 * xi2 * np.cos(theta) ** 2 * rgam ** 3)

    return sigma

#-----------------------------------------------------------------------------------------------------------------------


def HF_analytical_sol(regime, mesh, Eprime, Q0, muPrime=None, Kprime=None, Cprime=None, length=None, t=None,
                      Kc_1=None, h=None, density=None, Cij=None, gamma=None):
    """
    This function provides the analytical solution for the given parameters according to the given propagation regime

    Arguments:
        regime (string)        -- the propagation regime. Possible options:
                                    - K  (toughness dominated regime, without leak off)
                                    - M  (viscosity dominated regime, without leak off)
                                    - Kt (viscosity dominated regime , with leak off)
                                    - Mt (viscosity dominated regime , with leak off)
                                    - PKN (height contained hydraulic fracture with PKN geometry)
                                    - E (elliptical fracture propagating in toughness dominated regime).
        mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
        Eprime (float)         -- plain strain modulus.
        Q0 (float)             -- injection rate.
        muPrime (float)        -- 12*viscosity.
        Kprime (float)         -- the fracture toughness (K') along the minor axis.
        Cprime (float)         -- 2*C, where C is the Carter's leak off coefficient.
        length (float)         -- the given length dimension (fracture length in the case of PKN, length of the minor
                                  axis in the case of elliptical fracture and the fracture radius in all of the rest).
        t (float)              -- the given time for which the solution is evaluated.
        Kp_perp (float)        -- the fracture toughness along the major axis.
        h (float)              -- the height of the PKN fracture.

    Returns:
        t (float)              -- time at which the fracture reaches the given length.
        r (float)              -- length of the fracture at the given time (fracture length in the case of PKN, length
                                  of the minor axis in the case of elliptical fracture and the fracture radius in all
                                  of the rest).
        p (ndarray-float)      -- pressure at each cell at the given time.
        w (ndarray-float)      -- width at each cell at the given time.
        v (float)              -- propagation velocity.
        actvElts (ndarray)     -- list of cells inside the fracture at the given time.

    """

    if regime is 'M':
        t, r, p, w, v, actvElts = M_vertex_solution(Eprime, Q0, muPrime, mesh, length, t)
    elif regime is 'K':
        t, r, p, w, v, actvElts = K_vertex_solution(Kprime, Eprime, Q0, mesh, length, t)
    elif regime is 'Mt':
        t, r, p, w, v, actvElts = Mt_vertex_solution(Eprime, Cprime, Q0, muPrime, mesh, length, t)
    elif regime is 'Kt':
        t, r, p, w, v, actvElts = KT_vertex_solution(Eprime, Cprime, Q0, Kprime, mesh, length, t)
    elif regime is 'PKN':
        t, r, p, w, v, actvElts = PKN_solution(Eprime, Q0, muPrime, mesh, h, length, t)
    elif regime is 'KGD_K':
        t, r, p, w, v, actvElts = KGD_solution_K(Eprime, Q0, Kprime, mesh, h, length, t)
    elif regime is 'MDR':
        t, r, p, w, v, actvElts = MDR_M_vertex_solution(Eprime, Q0, density, muPrime/12., mesh, length, t)
    elif regime is 'E_K':
        Kc_3 = Kprime / (32 / np.pi) ** 0.5
        t, r, p, w, v, actvElts = anisotropic_toughness_elliptical_solution( Kc_3, Kc_1, Eprime, Q0, mesh, length, t)
    elif regime is 'E_E':
        Kc_3 = Kprime / (32 / np.pi) ** 0.5
        t, r, p, w, v, actvElts = TI_Elasticity_elliptical_solution(mesh, gamma, Cij, Kc_3, Eprime, Q0, t, length)

    return t, r, p, w, v, actvElts

#-----------------------------------------------------------------------------------------------------------------------


def get_fracture_dimensions_analytical(regime, t, Eprime, Q0, muPrime=None, Kprime=None, Cprime=None,
                      Kc_1=None, h=None, density=None, gamma=None):
    if regime is 'M':
        x_len = y_len = (0.6976 * Eprime ** (1 / 9) * Q0 ** (1 / 3) * t ** (4 / 9)) / muPrime ** (1 / 9)
    elif regime is 'K':
        x_len = y_len = (3 / 2 ** 0.5 / np.pi * Q0 * Eprime * t / Kprime) ** 0.4
    elif regime is 'Mt':
        x_len = y_len = (2 * Q0 / Cprime) ** 0.5 * t ** 0.25 / np.pi
    elif regime is 'Kt':
        x_len = y_len = 2 ** 0.5 * Q0 ** 0.5 * t ** (1 / 4) / Cprime ** 0.5 / np.pi
    elif regime is 'PKN':
        x_len = (2 * (Q0 / 2) ** 3 * Eprime / np.pi ** 3 / muPrime * 12 / h ** 4) ** (1 / 5) * t ** (4 / 5)
        y_len = h
    elif regime is 'KGD_K':
        x_len = 0.932388 * (Eprime * Q0 * t / Kprime) ** (2 / 3)
        y_len = h
    elif regime is 'MDR':
        fo = 1.78
        gammam = 0.758244
        x_len = y_len = gammam * ((2 ** (14. / 87)) * (Eprime ** (10. / 87)) * (Q0 ** (9. / 29)) * (t ** (40 / 87))
                    ) / ((3 ** (7. / 87)) * (fo ** (10. / 87)) * ((muPrime / 12) ** (7. / 87)) * density ** (1. / 29))
    elif regime is 'E_K':
        Kc_3 = Kprime / (32 / np.pi) ** 0.5
        c = (Kc_1 / Kc_3) ** 2
        x_len = (Q0 * t * 3 * c * Eprime / (8 * Kc_3 * np.pi ** 0.5)) ** (2 / 5)
        y_len = x_len / c
    elif regime is 'E_E':
        Kc_3 = Kprime / (32 / np.pi) ** 0.5
        y_len = (Q0 * t * 3 * Eprime / (8 * gamma * Kc_3 * np.pi ** 0.5)) ** (2 / 5)
        x_len = y_len * gamma

    return x_len, y_len
