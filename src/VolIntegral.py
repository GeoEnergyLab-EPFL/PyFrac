# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Oct 14 18:27:39 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.


Functions to calculate tip volumes, given the propagation regime

    regime -- A  gives the area (fill fraction)
    regime -- K  gives tip volume according to the square root assymptote
    regime -- M  gives tip volume according to the viscocity dominated assymptote 
    regime -- Lk is used to calculate the leak off given the distance of the front l (note, its not tip volume) 
    regime -- Mt gives tip volume according to the viscocity, Leak-off assymptote 
    regime -- U  gives tip volume according to the Universal assymptote (Donstov and Pierce, 2017)
    regime -- MK gives tip volume according to the M-K transition assymptote
    
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.TipInversion import f


# ----------

def TipAsym_UniversalW_zero_Res(w, *args):
    """Function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce 2017)"""
    (dist, Kprime, Eprime, muPrime, Cbar, Vel) = args

    Kh = Kprime * dist ** 0.5 / (Eprime * w)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * w)
    g0 = f(Kh, 0.9911799823 * Ch, 6 * 3 ** 0.5)
    sh = muPrime * Vel * dist ** 2 / (Eprime * w ** 3)

    return sh - g0


def TipAsym_UniversalW_delt_Res(w, *args):
    """The residual function zero of which will give the General asymptote """

    (dist, Kprime, Eprime, muPrime, Cbar, Vel) = args

    Kh = Kprime * dist ** 0.5 / (Eprime * w)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * w)
    sh = muPrime * Vel * dist ** 2 / (Eprime * w ** 3)

    g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
    delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0

    C1 = 4 * (1 - 2 * delt) / (delt * (1 - delt)) * np.tan(np.pi * delt)
    C2 = 16 * (1 - 3 * delt) / (3 * delt * (2 - 3 * delt)) * np.tan(3 * np.pi * delt / 2)
    b = C2 / C1

    return sh - f(Kh, Ch * b, C1)


def MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime):
    """Moments of the General tip asymptote to calculate the volume integral (see Donstov and Pierce, 2017)"""

    TipAsmptargs = (dist, Kprime, Eprime, muPrime, Cbar, Vel)

    if stagnant:
        w = KIPrime * dist ** 0.5 / Eprime
    else:
        a, b = FindBracket_w(dist, Kprime, Eprime, muPrime, Cbar, Vel)
        try:
            w = brentq(TipAsym_UniversalW_zero_Res, a, b, TipAsmptargs)  # root finding
        except RuntimeError:
            M0, M1 = np.nan, np.nan
            return (M0, M1)
        except ValueError:
            M0, M1 = np.nan, np.nan
            return (M0, M1)

        if w < -1e-15:
            w = abs(w)

    if Vel < 1e-6:
        delt = 1 / 6
    else:
        Kh = Kprime * dist ** 0.5 / (Eprime * w)
        Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * w)
        g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
        delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0

    M0 = 2 * w * dist / (3 + delt)
    M1 = 2 * w * dist ** 2 / (5 + delt)

    if np.isnan(M0) or np.isnan(M1):
       M0, M1 = np.nan, np.nan

    return (M0, M1)


def Pdistance(x, y, slope, intercpt):
    """distance of a point from a line"""

    return (slope * x - y + intercpt) / (slope ** 2 + 1) ** 0.5


def VolumeTriangle(dist, *param):
    """
    Volume  of the triangle defined by perpendicular distance (dist) and em (em=1/sin(alpha)cos(alpha), where alpha
    is the angle of the perpendicular). The regime variable identifies the propagation regime.
    """

    regime, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime, arrival_t, em, t_lstTS, dt = param

    if regime == 'A':
        return dist ** 2 * em / 2

    elif regime == 'K':
        return 4 / 15 * Kprime / Eprime * dist ** 2.5 * em

    elif regime == 'M':
        return 0.7081526678 * (Vel * muPrime / Eprime) ** (1 / 3) * em * dist ** (8 / 3)

    elif regime == 'Lk':
        t = t_lstTS + dt
        if Vel <= 0:
            t_e = arrival_t
        else:
            t_e = t - dist / Vel

        intgrl_0_t = 4 / 15 * em * (t - t_e) ** (5 / 2) * Vel ** 2
        if (t - t_e - dt) < 0:
            intgrl_0_tm1 = 0.
        else:
            intgrl_0_tm1 = 4 / 15 * em * (t - t_e - dt) ** (5 / 2) * Vel ** 2

        return intgrl_0_t - intgrl_0_tm1

    elif regime == 'Mt':
        return 256 / 273 / (15 * np.tan(np.pi / 8)) ** 0.25 * (
                                    Cbar * muPrime / Eprime) ** 0.25 * em * Vel ** 0.125 * dist ** (21 / 8)

    elif regime == 'U':
        if Cbar == 0 and Kprime == 0: # if fully viscosity dominated
            return 0.7081526678 * (Vel * muPrime / Eprime) ** (1 / 3) * em * dist ** (8 / 3)

        (M0, M1) = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime)
        return em * (dist * M0 - M1)

    elif regime == 'MK':
        return (3.925544049000839e-9 * em * Kprime * (
        1.7320508075688772 * Kprime ** 9 * (Kprime ** 6 - 1872. * dist * Eprime ** 4 * muPrime ** 2 * Vel ** 2) + (
        1. + (31.17691453623979 * (dist) ** 0.5 * Eprime ** 2 * muPrime * Vel) / Kprime ** 3) ** 0.3333333333333333 * (
        -1.7320508075688772 * Kprime ** 15 + 18. * (
        dist) ** 0.5 * Eprime ** 2 * Kprime ** 12 * muPrime * Vel + 2868.2761373340604 * dist * Eprime ** 4 *
        Kprime ** 9 * muPrime ** 2 * Vel ** 2 - 24624. * dist ** 1.5 * Eprime ** 6 * Kprime ** 6 * muPrime ** 3 *
        Vel ** 3 + 464660.73424811783 * dist ** 2 * Eprime ** 8 * Kprime ** 3 * muPrime ** 4 * Vel ** 4 + 5.7316896e7
        * dist ** 2.5 * Eprime ** 10 * muPrime ** 5 * Vel ** 5))) / (Eprime ** 11 * muPrime ** 5 * Vel ** 5)


def Area(dist, *param):
    """Gives Area under the tip depending on the regime identifier ;  
    used in case of 0 or 90 degree angle; can be used for 1d case"""

    regime, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime, arrival_t, em, t_lstTS, dt = param

    if regime == 'A':
        return dist

    elif regime == 'K':
        return 2 / 3 * Kprime / Eprime * dist ** 1.5

    elif regime == 'M':
        return 1.8884071141 * (Vel * muPrime / Eprime) ** (1 / 3) * dist ** (5 / 3)

    elif regime == 'Lk':
        t = t_lstTS + dt
        if Vel <= 0:
            t_e = arrival_t
        else:
            t_e = t - dist / Vel

        intgrl_0_t = 2 / 3 * (t - t_e) ** (3 / 2) * Vel
        if (t - t_e - dt) < 0:
            intgrl_0_tm1 = 0.
        else:
            intgrl_0_tm1 = 2 / 3 * (t - t_e - dt) ** (3 / 2) * Vel

        return intgrl_0_t - intgrl_0_tm1

    elif regime == 'Mt':
        return 32 / 13 / (15 * np.tan(np.pi / 8)) ** 0.25 * (Cbar * muPrime / Eprime) ** 0.25 * Vel ** 0.125 * dist ** (
        13 / 8)

    elif regime == 'U':
        if Cbar == 0 and Kprime == 0:  # if fully viscosity dominated
            return 1.8884071141 * (Vel * muPrime / Eprime) ** (1 / 3) * dist ** (5 / 3)
        (M0, M1) = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime)
        return M0

    elif regime == 'MK':
        return (7.348618459729571e-6 * Kprime * (-1.7320508075688772 * Kprime ** 9 +
                (1. + (31.17691453623979 * (dist) ** 0.5 * Eprime ** 2 * muPrime * Vel) / Kprime ** 3) ** 0.3333333333333333 * (
                1.7320508075688772 * Kprime ** 9 - 18. * (dist) ** 0.5 * Eprime ** 2 * Kprime ** 6 * muPrime * Vel + (
                374.12297443487745 * dist * Eprime ** 4 * Kprime ** 3 * muPrime ** 2 * Vel ** 2) + (
                81648. * dist ** 1.5 * Eprime ** 6 * muPrime ** 3 * Vel ** 3)))) / (
                Eprime ** 7 * muPrime ** 3 * Vel ** 3)


def Integral_over_cell(EltTip, alpha, l, mesh, function, frac=None, mat_prop=None, fluid_prop=None, Vel=None,
                       Kprime=None, Eprime=None, stagnant=None, KIPrime=None, dt=None, arrival_t=None):
    """
    Calculate integral of the function specified by the argument function over the cell.

    Arguments:
        EltTip (ndarray)                -- the tip cells over which the integral is to be evaluated
        alpha (ndarray)                 -- the angle alpha of the perpendicular drawn on the front from the zero vertex.
        l (ndarray)                     -- the length of the perpendicular drawn on the front from the zero vertex.
        mesh (CartesianMesh)            -- the mesh object.
        function (string)               -- the string specifying the type of function that is to be integreated.
                                           Possible options are:
                                                'A'  gives the area (fill fraction)
                                                'K'  gives tip volume according to the square root assymptote
                                                'M'  gives tip volume according to the viscocity dominated assymptote
                                                'Lk' Lk is used to calculate the leak off given the distance of the
                                                     front l (note, its not tip volume)
                                                'Mt' gives tip volume according to the viscocity, Leak-off assymptote
                                                'U'  gives tip volume according to the Universal assymptote (Donstov
                                                     and Pierce, 2017)
                                                'MK' gives tip volume according to the M-K transition assymptote
        frac (Fracture)                 -- the fracture object.
        mat_prop (MaterialProperties)   -- the material properties object.
        fluid_prop (FluidProperties)    -- the fluid properties object
        Vel (ndarray)                   -- the velocity of the front in the given tip cells.
        Kprime (ndarray)                -- if provided, the toughness will be taken from the given array instead of
                                           taking it from the mat_prop object
        Eprime(ndarray-float):          -- plain strain TI modulus for current iteration. if not given, the Eprime
                                                from the given material properties object will be used.
        stagnant (ndarray)              -- list of tip cells where the front is not moving.
        KIPrime (ndarray)               -- the stress intensity factor of the cells where the fracture front is not
                                           moving
        dt (float)                      -- the time step, only used to calculate leak off.

    Returns:
        integral (ndarray)              -- the integral of the specifiend function over the given tip cells.
    """

    # Pass None as dummy if parameter is not required
    dummy = np.full((alpha.size,),None)

    if stagnant is None:
        stagnant = dummy
    if KIPrime is None:
        KIPrime = dummy

    if Kprime is None and not mat_prop is None:
        Kprime = mat_prop.Kprime[EltTip]
    if Kprime is None and mat_prop is None:
        Kprime = dummy

    if Eprime is None and mat_prop is not None:
        Eprime = np.full((alpha.size,), mat_prop.Eprime)
    if Eprime is None and mat_prop is None:
        Eprime = dummy

    if Vel is None:
        Vel = dummy

    if mat_prop is None:
        Cprime = dummy
    else:
        Cprime = mat_prop.Cprime[EltTip]

    if not fluid_prop is None:
        muPrimeTip = np.full((alpha.size,),fluid_prop.muPrime,dtype=np.float64)
    else:
        muPrimeTip = dummy

    if not frac is None:
        t_lstTS = frac.time
    else:
        t_lstTS = None

    if arrival_t is None:
        arrival_t = dummy

    integral = np.zeros((len(l),), float)
    for i in range(0, len(l)):

        m = 1 / (np.sin(alpha[i]) * np.cos(alpha[i]))  # the m parameter (see e.g. A. Pierce 2015)
        # packing parameters to pass
        param_pack = (function, Kprime[i], Eprime[i], muPrimeTip[i], Cprime[i], Vel[i], stagnant[i], KIPrime[i],
                      arrival_t[i], m, t_lstTS, dt)

        if abs(alpha[i]) < 1e-8:
            # the angle inscribed by the perpendicular is zero
            if l[i] <= mesh.hx:
                # the front is within the cell.
                integral[i] = Area(l[i], *param_pack) * mesh.hy
            else:
                # the front has surpassed this cell.
                integral[i] = (Area(l[i], *param_pack) - Area(l[i] - mesh.hx, *param_pack)) * mesh.hy

        elif abs(alpha[i] - np.pi / 2) < 1e-8:
            # the angle inscribed by the perpendicular is 90 degrees
            if l[i] <= mesh.hy:
                # the front is within the cell.
                integral[i] = Area(l[i], *param_pack) * mesh.hx
            else:
                # the front has surpassed this cell.
                integral[i] = (Area(l[i], *param_pack) - Area(l[i] - mesh.hy, *param_pack)) * mesh.hx
        else:
            yIntrcpt = l[i] / np.cos(np.pi / 2 - alpha[i]) # Y intercept of the front line
            grad = -1 / np.tan(alpha[i]) # gradient of the front line

            # integral of the triangle made by the front by intersecting the x and y directional lines of the cell
            TriVol = VolumeTriangle(l[i], *param_pack)

            # distance of the front from the upper left vertex of the grid cell
            lUp = Pdistance(0, mesh.hy, grad, yIntrcpt)

            if lUp > 0:  # upper vertex of the triangle is higher than the grid cell height
                UpTriVol = VolumeTriangle(lUp, *param_pack)
            else:
                UpTriVol = 0

            # distance of the front from the lower right vertex of the grid cell
            lRt = Pdistance(mesh.hx, 0, grad, yIntrcpt)

            if lRt > 0:  # right vertex of the triangle is wider than the grid cell width
                RtTriVol = VolumeTriangle(lRt, *param_pack)
            else:
                RtTriVol = 0

            # distance of the front from the upper right vertex of the grid cell
            IntrsctTriDist = Pdistance(mesh.hx, mesh.hy, grad, yIntrcpt)

            if IntrsctTriDist > 0:  # front has passed the grid cell
                IntrsctTri = VolumeTriangle(IntrsctTriDist, *param_pack)
            else:
                IntrsctTri = 0

            integral[i] = TriVol - UpTriVol - RtTriVol + IntrsctTri

    return integral


def FindBracket_w(dist, Kprime, Eprime, muPrime, Cprime, Vel):
    """
    This function finds the bracket to be used by the Universal tip asymptote root finder.
    """

    a = 0.01 * dist ** 0.5 * Kprime / Eprime  # lower bound on width
    b = 100 * dist ** 0.5 * Kprime / Eprime

    TipAsmptargs = (dist, Kprime, Eprime, muPrime, Cprime, Vel)
    Res_a = TipAsym_UniversalW_zero_Res(a, *TipAsmptargs)
    Res_b = TipAsym_UniversalW_zero_Res(b, *TipAsmptargs)


    # res_U = np.zeros((100,),)
    # x=np.linspace(a, b, 100)
    # for j in range(0,len(x)):
    #     res_U[j] = TipAsym_UniversalW_zero_Res(x[j], *TipAsmptargs)
    # plt.plot(x, res_U, 'b.-')
    # plt.plot(x, np.zeros((100,),),'k')
    # plt.show()


    if a == 0:
        a = 10 * np.finfo(float).eps
    if b == 0:
        b = 10.


    cnt = 0
    mid = b
    while Res_a * Res_b > 0:
        mid = (a + 2 * mid) / 3  # weighted
        Res_a = TipAsym_UniversalW_zero_Res(mid, *TipAsmptargs)
        cnt += 1
        if cnt >= 50:
            a = np.nan
            b = np.nan

    return a, b

#-----------------------------------------------------------------------------------------------------------------------

def toughness_at_tip(elts, mesh, mat_prop, alpha, l):
    if mat_prop.anisotropic:
        return mat_prop.KprimeFunc(alpha)
    else:

        return