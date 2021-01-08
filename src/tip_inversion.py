# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Nov 01 15:22:00 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import logging
from properties import instrument_start, instrument_close
import numpy as np
from scipy.optimize import brentq
import warnings
from scipy.optimize import fsolve


beta_m = 2**(1/3) * 3**(5/6)
beta_mtld = 4/(15**(1/4) * (2**0.5 - 1)**(1/4))
cnst_mc = 3 * beta_mtld**4 / (4 * beta_m**3)
cnst_m = beta_m**3 / 3
Ki_c = 3000

# ----------------------------------------------------------------------------------------------------------------------
def C1(delta):
    if (delta >= 1 or delta <= 0):
        return cnst_m
    else:
        return 4 * (1 - 2 * delta) / (delta * (1 - delta)) * np.tan(np.pi * delta)

# ----------------------------------------------------------------------------------------------------------------------
def C2(delta):
    if delta == 1/3:
        return beta_mtld ** 4 / 4
    else:
        return 16 * (1 - 3 * delta) / (3 * delta * (2 - 3 * delta)) * np.tan(3 * np.pi / 2 * delta)

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_k_exp(dist, *args):
    """Residual function for the near-field k expansion (Garagash & Detournay, 2011)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    V = (dist - DistLstTSEltRibbon) / dt
    l_mk = (Kprime ** 3 / (Eprime ** 2 * fluidProp.muPrime * V)) ** 2
    l_mtk = Kprime ** 8 / (Eprime ** 6 * fluidProp.muPrime ** 2 * (2 * Cbar) ** 2 * V)
    l1 = (l_mk ** (-1/2) + l_mtk ** (-1/2)) ** (-2)
    l2 = (2 / 3 * l_mk ** (-1/2) + l_mtk ** (-1/2)) ** (-2)

    return -wEltRibbon + ( Kprime / Eprime ) ** 2 * dist ** (1/2) * ( 1 + 4 * np.pi * (dist/l1) ** (1/2) + 64 *
                                                                      (dist * np.log(dist) / (l1 * l2) ** (1/2)))

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_m_exp(dist, *args):
    """Residual function for the far-field m expansion (Garagash & Detournay, 2011)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    V = (dist - DistLstTSEltRibbon) / dt
    l_mmt = (2 * Cbar) ** 6 * Eprime ** 2 / ( V ** 5 * fluidProp.muPrime ** 2)

    return -wEltRibbon + ( V * fluidProp.muPrime / Eprime ) ** (1/3) * dist ** (2/3) * ( beta_m + 1 / 2 * (l_mmt / dist) ** (1/6)
                                                                               - 3 ** (1/6) / 2 ** (7/3) * (l_mmt / dist) ** (1/3)
                                                                               + 2 ** (7/3) / 3 ** (5/3) * (l_mmt / dist) ** (1/2)
                                                                               - 0.7406 * (l_mmt / dist) ** (2/3 - 0.1387))

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_mt_exp(dist, *args):
    """Residual function for the intermediate-field m expansion (Garagash & Detournay, 2011)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    V = (dist - DistLstTSEltRibbon) / dt
    l_mtk = Kprime ** 8 / (Eprime ** 6 * fluidProp.muPrime ** 2 * (2 * Cbar) ** 2 * V)
    l_mmt = (2 * Cbar) ** 6 * Eprime ** 2 / (V ** 5 * fluidProp.muPrime ** 2)

    return -wEltRibbon + (2 * Cbar * V ** (1/2) * fluidProp.muPrime / Eprime ) ** (1/4) * dist ** (5/8) * (0.0161 * (l_mtk / dist) ** (5/8 - 0.06999)
                                                                                                 + 2.53356 + 1.30165 * (dist/l_mmt) ** (1/8)
                                                                                                 - 0.451609 * (dist/l_mmt) ** (1/4)
                                                                                                 + 0.183355 * (dist/l_mmt) ** (3/8))

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_viscStor_Res(dist, *args):
    """Residual function for viscosity dominate regime, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    return wEltRibbon - (18 * 3 ** 0.5 * (dist - DistLstTSEltRibbon) / dt * fluidProp.muPrime / Eprime) ** (1 / 3) * dist ** (
            2 / 3)

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_MDR_Res(dist, *args):
    """Residual function for viscosity dominate regime, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    density = 1000

    return wEltRibbon - (1.89812 * dist ** 0.740741 * ((dist - DistLstTSEltRibbon) / dt) ** 0.481481 * (
                fluidProp.muPrime ** 0.7 * density ** 0.3) ** 0.37037) / Eprime ** 0.37037

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_M_MDR_Res(dist, *args):
    """Residual function for viscosity dominate regime, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    density = 1000
    Vel = (dist - DistLstTSEltRibbon) / dt

    return wEltRibbon - 3.14735 * dist ** (2/3) * ((dist - DistLstTSEltRibbon) / dt) ** (1/3) * fluidProp.muPrime ** (1/3) * (1 +
    0.255286 * dist ** 0.2 * Vel ** 0.4 * density ** 0.3 / (Eprime ** 0.1 * fluidProp.muPrime ** 0.2)) ** 0.37037 / Eprime**(1/3)

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_viscLeakOff_Res(dist, *args):
    """Residual function for viscosity dominated regime, with leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    return wEltRibbon - 4 / (15 * np.tan(np.pi / 8)) ** 0.25 * (2 * Cbar * fluidProp.muPrime / Eprime) ** 0.25 * ((dist -
            DistLstTSEltRibbon) / dt) ** 0.125 * dist ** (5 / 8)

# -----------------------------------------------------------------------------------------------------------------------

def TipAsym_MK_zrthOrder_Res(dist, *args):
    """Residual function for viscosity to toughness regime with transition, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    if Kprime == 0:
        return TipAsym_viscStor_Res(dist, *args)
    if fluidProp.muPrime == 0:
        # return toughness dominated asymptote
        return dist - wEltRibbon ** 2 * (Eprime / Kprime) ** 2

    w_tld = Eprime * wEltRibbon / (Kprime * dist**0.5)
    V = (dist - DistLstTSEltRibbon) / dt
    return w_tld - (1 + beta_m**3 * Eprime**2 * V * dist**0.5 * fluidProp.muPrime / Kprime**3)**(1/3)

# -----------------------------------------------------------------------------------------------------------------------

def TipAsym_MK_deltaC_Res(dist, *args):
    """Residual function for viscosity to toughness regime with transition, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    if Kprime == 0:
        return TipAsym_viscStor_Res(dist, *args)
    if fluidProp.muPrime == 0:
        # return toughness dominated asymptote
        return dist - wEltRibbon ** 2 * (Eprime / Kprime) ** 2

    w_tld = Eprime * wEltRibbon / (Kprime * dist ** 0.5)

    V = (dist - DistLstTSEltRibbon) / dt
    l_mk = (Kprime ** 3 / (Eprime ** 2 * fluidProp.muPrime * V)) ** 2
    x_tld = (dist / l_mk) ** (1/2)
    delta = 1 / 3 * beta_m ** 3 * x_tld / (1 + beta_m ** 3 * x_tld)
    return w_tld - (1 + 3 * C1(delta) * x_tld) ** (1/3)

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_MTildeK_zrthOrder_Res(dist, *args):
    """Residual function for zeroth-order solution for M~K edge tip asymptote"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    w_tld = Eprime * wEltRibbon / (Kprime * dist ** 0.5)
    V = (dist - DistLstTSEltRibbon) / dt
    return -w_tld + (1 + beta_mtld**4 * 2 * Cbar * Eprime**3 * dist**0.5 * V**0.5 * fluidProp.muPrime / Kprime**4)**(1/4)

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_MTildeK_deltaC_Res(dist, *args):
    """Residual function for viscosity to toughness regime with transition, without leak off"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    w_tld = Eprime * wEltRibbon / (Kprime * dist ** 0.5)

    V = (dist - DistLstTSEltRibbon) / dt
    l_mk = (Kprime ** 3 / (Eprime ** 2 * fluidProp.muPrime * V)) ** 2
    chi = 2 * Cbar * Eprime / (V**0.5 * Kprime)
    x_tld = (dist / l_mk) ** (1/2)
    delta = 1 / 4 * beta_mtld ** 4 * chi * x_tld / (1 + beta_mtld ** 4 * chi * x_tld)
    return w_tld - (1 + 4 * C2(delta) * x_tld * chi) ** (1/3)

# ----------------------------------------------------------------------------------------------------------------------

def f(K, Cb, Con):
    if K >= 1:
        return 0
    elif Cb > 100:
        return (1 - K ** 4) / (4 * cnst_m * Cb)
    elif Cb == 0 and K == 0:
        return 1 / (3 * Con)
    elif Cb == 0:
        return 1 / (3 * Con) * ( 1 - K ** 3)
    else:
        return 1 / (3 * Con) * (
            1 - K ** 3 - 3 * Cb * (1 - K ** 2) / 2 + 3 * Cb ** 2 * (1 - K) - 3 * Cb ** 3 * np.log((Cb + 1) / (Cb + K)))

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Universal_1stOrder_Res(dist, *args):
    """More precise function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    if Cbar == 0:
        return TipAsym_MK_deltaC_Res(dist, *args)

    Vel = (dist - DistLstTSEltRibbon) / dt
    Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
    sh = fluidProp.muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)

    g0 = f(Kh, cnst_mc * Ch, cnst_m)
    delt = cnst_m * (1 + cnst_mc * Ch) * g0
    gdelt = f(Kh, Ch * C2(delt) / C1(delt), C1(delt))

    return sh - gdelt

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Universal_zrthOrder_Res(dist, *args):
    """Function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce 2017)"""
    
    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    if Cbar == 0:
        return TipAsym_MK_zrthOrder_Res(dist, *args)

    Vel = (dist - DistLstTSEltRibbon) / dt

    Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
    g0 = f(Kh, cnst_mc * Ch, cnst_m)
    sh = fluidProp.muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)

    return sh - g0


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Hershcel_Burkley_Res(dist, *args):
    """Function to be minimized to find root for Herschel Bulkley (see Bessmertnykh and Donstov 2019)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args
    
    if Cbar == 0:
        return TipAsym_power_law_MK_Res(dist, *args)
    
    Vel = (dist - DistLstTSEltRibbon) / dt
    n = fluidProp.n
    alpha = -0.3107 * n + 1.9924
    X = 2 * Cbar * Eprime / np.sqrt(Vel) / Kprime
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    ell = (Kprime**(n + 2) / Mprime / Vel**n / Eprime**(n + 1))**(2 / (2 - n))
    xt = np.sqrt(dist / ell)
    T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
    wtTau = 2 * np.sqrt(np.pi * T0t) * xt
    wt = ((wEltRibbon * Eprime / Kprime / np.sqrt(dist))**alpha - wtTau**alpha)**(1 / alpha)

    theta = 0.0452 * n**2 - 0.178 * n + 0.1753
    Vm = 1 - wt ** -((2 + n) / (1 + theta))
    Vmt = 1 - wt ** -((2 + 2 * n) / (1 + theta))
    dm = (2 - n) / (2 + n)
    dmt = (2 - n) / (2 + 2 * n)
    Bm = (2 * (2 + n)**2 / n * np.tan(np.pi * n / (2 + n)))**(1 / (2 + n))
    Bmt = (64 * (1 + n) ** 2 / (3 * n *(4 + n)) * np.tan(3 * np.pi * n / (4 * (1 + n))))**(1 / (2 + 2 * n))
    
    dt1 = dmt * dm * Vmt * Vm * \
          (Bm**((2 + n) / n) * Vmt**((1 + theta) / n) + X / wt * Bmt**(2 * (1 + n) / n) * Vm**((1 + theta) / n)) / \
          (dmt * Vmt * Bm**((2 + n) / n) * Vmt**((1 + theta) / n) +
           dm * Vm * X / wt * Bmt**(2 * (1 + n) / n) * Vm**((1 + theta) / n))

    return xt**((2 - n) / (1 + theta)) - dt1 * wt**((2 + n) / (1 + theta)) * (dm**(1 + theta) * Bm**(2 + n) +
                            dmt**(1 + theta) * Bmt**(2 * (1 + n)) * ((1 + X / wt)**n - 1))**(-1 / (1 + theta))


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_power_law_Res(dist, *args):
    """Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args
    
    if Cbar == 0:
        return TipAsym_power_law_MK_Res(dist, *args)
    
    Vel = (dist - DistLstTSEltRibbon) / dt
    n = fluidProp.n
    X = 2 * Cbar * Eprime / np.sqrt(Vel) / Kprime
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    ell = (Kprime**(n + 2) / Mprime / Vel**n / Eprime**(n + 1))**(2 / (2 - n))
    xt = np.sqrt(dist / ell)
    wt = wEltRibbon * Eprime / Kprime / np.sqrt(dist)

    theta = 0.0452 * n**2 - 0.178 * n + 0.1753
    Vm = 1 - wt ** -((2 + n) / (1 + theta))
    Vmt = 1 - wt ** -((2 + 2 * n) / (1 + theta))
    dm = (2 - n) / (2 + n)
    dmt = (2 - n) / (2 + 2 * n)
    Bm = (2 * (2 + n)**2 / n * np.tan(np.pi * n / (2 + n)))**(1 / (2 + n))
    Bmt = (64 * (1 + n) ** 2 / (3 * n *(4 + n)) * np.tan(3 * np.pi * n / (4 * (1 + n))))**(1 / (2 + 2 * n))

    dt1 = dmt * dm * Vmt * Vm * \
          (Bm**((2 + n) / n) * Vmt**((1 + theta) / n) + X / wt * Bmt**(2 * (1 + n) / n) * Vm**((1 + theta) / n)) / \
          (dmt * Vmt * Bm**((2 + n) / n) * Vmt**((1 + theta) / n) +
           dm * Vm * X / wt * Bmt**(2 * (1 + n) / n) * Vm**((1 + theta) / n))

    return xt**((2 - n) / (1 + theta)) - dt1 * wt**((2 + n) / (1 + theta)) * (dm**(1 + theta) * Bm**(2 + n) +
                            dmt**(1 + theta) * Bmt**(2 * (1 + n)) * ((1 + X / wt)**n - 1))**(-1 / (1 + theta))


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Hershcel_Burkley_MK_Res(dist, *args):
    """Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    Vel = (dist - DistLstTSEltRibbon) / dt
    n = fluidProp.n
    alpha = -0.3107 * n + 1.9924
    X = 2 * Cbar * Eprime / np.sqrt(Vel) / Kprime
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    ell = (Kprime**(n + 2) / Mprime / Vel**n / Eprime**(n + 1))**(2 / (2 - n))
    xt = np.sqrt(dist / ell)
    T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
    wtTau = 2 * np.sqrt(np.pi * T0t) * xt
    wt = ((wEltRibbon * Eprime / Kprime / np.sqrt(dist))**alpha - wtTau**alpha)**(1 / alpha)

    theta = 0.0452 * n**2 - 0.178 * n + 0.1753
    dm = (2 - n) / (2 + n)
    Bm = (2 * (2 + n)**2 / n * np.tan(np.pi * n / (2 + n)))**(1 / (2 + n))

    return wt - (1 + (Bm**(2 + n) * xt**(2 - n))**(1 / (1 + theta)))**((1 + theta) / (2 + n)) 

# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_power_law_MK_Res(dist, *args):
    """Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)"""

    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

    Vel = (dist - DistLstTSEltRibbon) / dt
    n = fluidProp.n
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    ell = (Kprime**(n + 2) / Mprime / Vel**n / Eprime**(n + 1))**(2 / (2 - n))
    xt = np.sqrt(dist / ell)
    wt = wEltRibbon * Eprime / Kprime / np.sqrt(dist)

    theta = 0.0452 * n**2 - 0.178 * n + 0.1753
    dm = (2 - n) / (2 + n)
    Bm = (2 * (2 + n)**2 / n * np.tan(np.pi * n / (2 + n)))**(1 / (2 + n))

    return wt - (1 + (Bm**(2 + n) * xt**(2 - n))**(1 / (1 + theta)))**((1 + theta) / (2 + n)) 


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_PowerLaw_M_vertex_Res(dist, *args):
    
    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args
    n = fluidProp.n    
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    Vel = (dist - DistLstTSEltRibbon) / dt
    Bm = (2 * (2 + n)**2 / n * np.tan(np.pi * n / (2 + n)))**(1 / (2 + n))
    
    return wEltRibbon - Bm * (Mprime * Vel**n / Eprime) ** (1 / (2 + n)) * dist ** (2 / (2 + n))

    
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

def Vm_residual(dist, *args):
    
    (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args
    
    Vel = (dist - DistLstTSEltRibbon) / dt
    n = fluidProp.n
    alpha = -0.3107 * n + 1.9924
    X = 2 * Cbar * Eprime / np.sqrt(Vel) / Kprime
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * fluidProp.k
    ell = (Kprime**(n + 2) / Mprime / Vel**n / Eprime**(n + 1))**(2 / (2 - n))
    xt = np.sqrt(dist / ell)
    T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
    wtTau = 2 * np.sqrt(np.pi * T0t) * xt
    wt = ((wEltRibbon * Eprime / Kprime / np.sqrt(dist))**alpha - wtTau**alpha)**(1 / alpha)
    theta = 0.0452 * n**2 - 0.178 * n + 0.1753
    
    return 100 * np.finfo(float).eps - 1 + wt ** -((2 + 2 * n) / (1 + theta))


#-----------------------------------------------------------------------------------------------------------------------

def FindBracket_dist(w, Kprime, Eprime, fluidProp, Cprime, DistLstTS, dt, mesh, ResFunc, simProp):
    """ 
    Find the valid bracket for the root evaluation function.
    """

    a = -DistLstTS * (1 + np.finfo(float).eps)
    if fluidProp.rheology == "Newtonian" or sum(Cprime) == 0:
        b = np.full((len(w),), 6 * (mesh.hx**2 + mesh.hy**2)**0.5, dtype=np.float64)
    elif simProp.get_tipAsymptote()  in ["HBF", "HBF_aprox", "HBF_num_quad", "PLF", "PLF_aprox", "PLF_num_quad"]:
        b = np.zeros(len(w), dtype=np.float64)
        for i in range(0, len(w)):
            TipAsmptargs = (w[i], Kprime[i], Eprime[i], fluidProp, Cprime[i], -DistLstTS[i], dt)
            b[i] = fsolve(Vm_residual, (w[i] * Eprime[i] / Kprime[i])**2, args=TipAsmptargs)
        
        
    for i in range(0, len(w)):

        TipAsmptargs = (w[i], Kprime[i], Eprime[i], fluidProp, Cprime[i], -DistLstTS[i], dt)
        Res_a = ResFunc(a[i], *TipAsmptargs)
        Res_b = ResFunc(b[i], *TipAsmptargs)

        cnt = 0
        mid = b[i]
        while Res_a * Res_b > 0:
            mid = (a[i] + 2 * mid) / 3  # weighted
            Res_a = ResFunc(mid, *TipAsmptargs)
            cnt += 1
            if Res_a * Res_b < 0:
                a[i] = mid
                break
            elif Res_a > 0.0 and Res_b > 0.0:
                mid_b = b[i] * 2 ** cnt
                Res_b = ResFunc(mid_b, *TipAsmptargs)
                if Res_a * Res_b < 0:
                    a[i] = mid
                    b[i] = mid_b
                    break
            if cnt >= 100:  # Should assume not propagating. not set to check how frequently it happens.
                a[i] = np.nan
                b[i] = np.nan
                break

    return a, b


# ----------------------------------------------------------------------------------------------------------------------

def TipAsymInversion(w, frac, matProp, fluidProp, simParmtrs, dt=None, Kprime_k=None, Eprime_k=None, perfNode=None):
    """ 
    Evaluate distance from the front using tip assymptotics according to the given regime, given the fracture width in
    the ribbon cells.

    Arguments:
        w (ndarray):                        -- fracture width.
        frac (Fracture):                    -- current fracture object.
        matProp (MaterialProperties):       -- material properties.
        fluidProp (FluidProperties):        -- fluid properties.
        simParmtrs (SimulationParameters):  -- Simulation parameters.
        dt (float):                         -- time step.
        Kprime_k (ndarray-float):           -- Kprime for current iteration of toughness loop. if not given, the Kprime
                                               from the given material properties object will be used.
        Eprime_k (float):                   -- the plain strain modulus.
    Returns:
        dist (ndarray):                     -- distance (unsigned) from the front to the ribbon cells.
    """
    log = logging.getLogger('PyFrac.TipAsymInversion')
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
    elif simParmtrs.get_tipAsymptote() == 'U1':
        ResFunc = TipAsym_Universal_1stOrder_Res
    elif simParmtrs.get_tipAsymptote() == 'K':
        return w[frac.EltRibbon] ** 2 * (Eprime / Kprime) ** 2
    elif simParmtrs.get_tipAsymptote() == 'Kt':
        return w[frac.EltRibbon] ** 2 * (Eprime / Kprime) ** 2
    elif simParmtrs.get_tipAsymptote() == 'M':
        ResFunc = TipAsym_viscStor_Res
    elif simParmtrs.get_tipAsymptote() == 'Mt':
        ResFunc = TipAsym_viscLeakOff_Res
    elif simParmtrs.get_tipAsymptote() == 'MK':
        ResFunc = TipAsym_MK_zrthOrder_Res
    elif simParmtrs.get_tipAsymptote() == 'MDR':
        ResFunc = TipAsym_MDR_Res
    elif simParmtrs.get_tipAsymptote() == 'M_MDR':
        ResFunc = TipAsym_M_MDR_Res
    elif simParmtrs.get_tipAsymptote() in ["HBF", "HBF_aprox", "HBF_num_quad"]:
        ResFunc = TipAsym_Hershcel_Burkley_Res
    elif simParmtrs.get_tipAsymptote() in ["PLF", "PLF_aprox", "PLF_num_quad"]:
        ResFunc = TipAsym_power_law_Res
    elif simParmtrs.get_tipAsymptote() == 'PLF_M':
        ResFunc = TipAsym_PowerLaw_M_vertex_Res
    else:
        raise SystemExit("Tip asymptote type not supported!")

    # checking propagation condition
    stagnant = np.where(Kprime * (abs(frac.sgndDist[frac.EltRibbon]))**0.5 / (
                                        Eprime * w[frac.EltRibbon]) > 1)[0]
    moving = np.arange(frac.EltRibbon.shape[0])[~np.in1d(frac.EltRibbon, frac.EltRibbon[stagnant])]

    a, b = FindBracket_dist(w[frac.EltRibbon[moving]],
                            Kprime[moving],
                            Eprime[moving],
                            fluidProp,
                            matProp.Cprime[frac.EltRibbon[moving]],
                            frac.sgndDist[frac.EltRibbon[moving]],
                            dt,
                            frac.mesh,
                            ResFunc,
                            simParmtrs)
    ## AM: part added to take care of nan's in the bracketing if bracketing is no longer possible.
    if any(np.isnan(a)):
        stagnant_from_bracketing = np.argwhere(np.isnan(a))[::,0]
        a = np.delete(a, stagnant_from_bracketing)
        b = np.delete(b, stagnant_from_bracketing)
        if not stagnant.size == 0:
            stagnant = np.sort(np.unique(np.hstack((stagnant, moving[stagnant_from_bracketing]))))
        else:
            stagnant = stagnant_from_bracketing
        moving = np.arange(frac.EltRibbon.shape[0])[~np.in1d(frac.EltRibbon, frac.EltRibbon[stagnant])]
    ## End of adaption

    dist = -frac.sgndDist[frac.EltRibbon]
    for i in range(0, len(moving)):
        TipAsmptargs = (w[frac.EltRibbon[moving[i]]],
                        Kprime[moving[i]],
                        Eprime[moving[i]],
                        fluidProp,
                        matProp.Cprime[frac.EltRibbon[moving[i]]],
                        -frac.sgndDist[frac.EltRibbon[moving[i]]],
                        dt)
        try:
            if perfNode is None:
                dist[moving[i]] = brentq(ResFunc, a[i], b[i], TipAsmptargs)
            else:
                brentq_itr = instrument_start('Brent method', perfNode)
                dist[moving[i]], data = brentq(ResFunc, a[i], b[i], TipAsmptargs, full_output=True)
                instrument_close(perfNode, brentq_itr, None, None, data.converged, None, None)
                brentq_itr.iterations = data.iterations
                perfNode.brentMethod_data.append(brentq_itr)
        except RuntimeError:
            dist[moving[i]] = np.nan
        except ValueError:
            if simParmtrs.get_tipAsymptote() == 'U1':
                log.warning("First order did not converged: try with zero order.")
                try:
                    if perfNode is None:
                        dist[moving[i]] = brentq(TipAsym_Universal_zrthOrder_Res, a[i], b[i], TipAsmptargs)
                    else:
                        brentq_itr = instrument_start('Brent method', perfNode)
                        dist[moving[i]], data = brentq(ResFunc, a[i], b[i], TipAsmptargs, full_output=True)
                        instrument_close(perfNode, brentq_itr, None, None, data.converged, None, None)
                        brentq_itr.iterations = data.iterations
                        perfNode.brentMethod_data.append(brentq_itr)
                except RuntimeError:
                    dist[moving[i]] = np.nan
                except ValueError:
                    dist[moving[i]] = np.nan
            else:
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

            if KIPrime[i] < 0.:
                KIPrime[i] = 0.

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