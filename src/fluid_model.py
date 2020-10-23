# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 11:51:00 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np
from scipy.special import factorial
from scipy.optimize import fsolve


def FF_YangJoseph_vector(ReNum, rough):
    """
    This function approximate the friction factor for the given Reynold's number and the relative roughness arrays with
    the Yang Joseph approximation (see Virtual Nikuradse Yang & Joseph 2009).
    
    Arguments:
        ReNum (ndarray):      -- Reynold's number
        rough (ndarray):      -- 1/relative roughness (w/roughness length scale)
    
    Returns:
         ff (ndarray):        -- frction factor
    """
    ff = np.full((len(ReNum),), np.inf, dtype=np.float64)

    lam = np.where(abs(ReNum) < 2100)[0]
    ff[lam] = 16 / ReNum[lam]

    turb = np.where(abs(ReNum) >= 2100)[0]
    lamdaS = (-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(
        1 + 3810**15/ReNum[turb]**15)**0.5 + (-((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) -
        64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5) - (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1
        + 2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.1537/ReNum[turb]**0.185)/(1 + 1680700000000000000000000/ReNum[turb]**5)**0.5 + (-
        ((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb] + 0.3164/ReNum[turb]**0.25)/(1 +
        3810**15/ReNum[turb]**15)**0.5) - (-((-((-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5) - 64/ReNum[turb]
        + 0.3164/ReNum[turb]**0.25)/(1 + 3810**15/ReNum[turb]**15)**0.5) - (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 +
        2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.1537/ReNum[turb]**0.185)/(1 + 1680700000000000000000000/ReNum[turb]**5)**0.5 - (
        -64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 - 64/ReNum[turb] + 0.0753/ReNum[turb]**0.136)/(1 +
        4000000000000/ReNum[turb]**2)**0.5 + (-64/ReNum[turb] + 0.000083*ReNum[turb]**0.75)/(1 + 2320**50/ReNum[turb]**50)**0.5 + 64/ReNum[turb]

    lamdaR = ReNum[turb]**(-0.2032 + 7.348278/rough[turb]**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough[turb]**0.03569244 -
        0.00255391*rough[turb]**0.8353877)/(1 +
        265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[turb]**50
        )**0.5 + 0.00255391*rough[turb]**0.8353877) + (-(ReNum[turb]**(-0.2032 + 7.348278/rough[turb]**0.96433953)*(-0.022 + (-0.978 +
        0.92820419*rough[turb]**0.03569244 - 0.00255391*rough[turb]**0.8353877)/(1 +
        265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough[turb]**50
        )**0.5 + 0.00255391*rough[turb]**0.8353877)) + 0.01105244*ReNum[turb]**(-0.191 + 0.62935712/rough[turb]**0.28022284)*
        rough[turb]**0.23275646 + (ReNum[turb]**(0.015 + 0.26827956/rough[turb]**0.28852025)*(0.0053 + 0.02166401/rough[turb]**0.30702955) -
        0.01105244*ReNum[turb]**(-0.191 + 0.62935712/rough[turb]**0.28022284)*rough[turb]**0.23275646 + (ReNum[turb]**0.002*(0.011 +
        0.18954211/rough[turb]**0.510031) - ReNum[turb]**(0.015 + 0.26827956/rough[turb]**0.28852025)*(0.0053 + 0.02166401/rough[turb]**
        0.30702955) + (0.0098 - ReNum[turb]**0.002*(0.011 + 0.18954211/rough[turb]**0.510031) + 0.17805185/rough[turb]**0.46785053)/(1 +
        (8.733801045300249e10*rough[turb]**0.90870686)/ReNum[turb]**2)**0.5)/(1 + (6.44205549308073e15*rough[turb]**5.168887)/ReNum[turb]**5)
        **0.5)/(1 + (1.1077593467238922e13*rough[turb]**4.9771653)/ReNum[turb]**5)**0.5)/(1 + (2.9505925619934144e14*rough[turb]**
        3.7622822)/ReNum[turb]**5)**0.5

    ff[turb] = np.asarray(
        lamdaS + (lamdaR - lamdaS) / (1 + (ReNum[turb] / (45.196502 * rough[turb] ** 1.2369807 + 1891)) ** -5) ** 0.5, float) / 4

    return ff


#######################################
def FF_YangJoseph(ReNum, rough):
    """
    This function approximate the friction factor for the given Reynold's number and the relative roughness float 
    with the Yang Joseph approximation (see Virtual Nikuradse Yang & Joseph 2009).

    Arguments:
        ReNum (float):      -- Reynold's number
        rough (float):      -- 1/relative roughness (w/roughness length scale)

    Returns:
         float :            -- friction factor
    """
    if ReNum < 1e-8:
        return 0
    elif ReNum < 2100:
        return 16/ReNum
    else:
        lamdaS = (-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(
        1 + 3810**15/ReNum**15)**0.5 + (-((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) -
        64/ReNum + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5) - (-64/ReNum + 0.000083*ReNum**0.75)/(1
        + 2320**50/ReNum**50)**0.5 - 64/ReNum + 0.1537/ReNum**0.185)/(1 + 1680700000000000000000000/ReNum**5)**0.5 + (-
        ((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum + 0.3164/ReNum**0.25)/(1 +
        3810**15/ReNum**15)**0.5) - (-((-((-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5) - 64/ReNum
        + 0.3164/ReNum**0.25)/(1 + 3810**15/ReNum**15)**0.5) - (-64/ReNum + 0.000083*ReNum**0.75)/(1 +
        2320**50/ReNum**50)**0.5 - 64/ReNum + 0.1537/ReNum**0.185)/(1 + 1680700000000000000000000/ReNum**5)**0.5 - (
        -64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 - 64/ReNum + 0.0753/ReNum**0.136)/(1 +
        4000000000000/ReNum**2)**0.5 + (-64/ReNum + 0.000083*ReNum**0.75)/(1 + 2320**50/ReNum**50)**0.5 + 64/ReNum

        lamdaR = ReNum**(-0.2032 + 7.348278/rough**0.96433953)*(-0.022 + (-0.978 + 0.92820419*rough**0.03569244 -
        0.00255391*rough**0.8353877)/(1 +
        265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough**50
        )**0.5 + 0.00255391*rough**0.8353877) + (-(ReNum**(-0.2032 + 7.348278/rough**0.96433953)*(-0.022 + (-0.978 +
        0.92820419*rough**0.03569244 - 0.00255391*rough**0.8353877)/(1 +
        265550686013728218770454203489441165109061383639474724663955742569518708077419167245843482753466249/rough**50
        )**0.5 + 0.00255391*rough**0.8353877)) + 0.01105244*ReNum**(-0.191 + 0.62935712/rough**0.28022284)*
        rough**0.23275646 + (ReNum**(0.015 + 0.26827956/rough**0.28852025)*(0.0053 + 0.02166401/rough**0.30702955) -
        0.01105244*ReNum**(-0.191 + 0.62935712/rough**0.28022284)*rough**0.23275646 + (ReNum**0.002*(0.011 +
        0.18954211/rough**0.510031) - ReNum**(0.015 + 0.26827956/rough**0.28852025)*(0.0053 + 0.02166401/rough**
        0.30702955) + (0.0098 - ReNum**0.002*(0.011 + 0.18954211/rough**0.510031) + 0.17805185/rough**0.46785053)/(1 +
        (8.733801045300249e10*rough**0.90870686)/ReNum**2)**0.5)/(1 + (6.44205549308073e15*rough**5.168887)/ReNum**5)
        **0.5)/(1 + (1.1077593467238922e13*rough**4.9771653)/ReNum**5)**0.5)/(1 + (2.9505925619934144e14*rough**
        3.7622822)/ReNum**5)**0.5

        return (lamdaS + (lamdaR - lamdaS) / (1 + (ReNum / (45.196502 * rough ** 1.2369807 + 1891)) ** -5) ** 0.5) / 4

def FF_Yang_Dou_residual(vbyu, *args):
    """
    The Yang_Dou residual function; to be used by numerical root finder
    """
    (Re, rough) = args

    Rstar = Re / (2 * vbyu * rough)
    theta = np.pi * np.log( Rstar / 1.25) / np.log(100 / 1.25)
    alpha = (1 - np.cos(theta)) / 2
    beta = 1 - (1 - 0.107) * (alpha + theta/np.pi) / 2
    R = Re / (2 * vbyu)

    rt = 1.
    for i in range(1,5):
        rt = rt - 1. / np.e * ( i / factorial(i) * (67.8 / R) ** (2 * i))

    return vbyu - (1 - rt) * R / 4. - rt * (2.5 * np.log(R) - 66.69 * R**-0.72 + 1.8 - (2.5 * np.log(
        (1 + alpha * Rstar / 5) / (1 + alpha * beta * Rstar / 5)) + (5.8 + 1.25) * (alpha * Rstar / (
        5 + alpha * Rstar)) ** 2 + 2.5 * (alpha * Rstar / (5 + alpha * Rstar)) - (5.8 + 1.25)
        * (alpha * beta * Rstar / (5 + alpha * beta * Rstar)) ** 2 - 2.5 * (alpha * beta * Rstar / (
        5 + alpha * beta * Rstar))))


def FF_Yang_Dou(Re, rough):
    """
    This function approximate the friction factor for the given Reynold's number and the relative roughness arrays with
    the Yang Dou approximation (see Yang, S. Dou, G. (2010). Turbulent drag reduction with polymer additive in rough 
    pipes. Journal of Fluid Mechanics, 642 279-294). The function is implicit and utilize a numerical root finder

    Arguments:
        Re (ndarray):       -- Reynold's number.
        rough (ndarray):    -- 1/relative roughness (w/roughness length scale).

    Returns:
         ff (ndarray) :     -- friction factor.
    """

    ff_args = (Re, rough)
    sol_vbyu = fsolve(FF_Yang_Dou_residual, 15., ff_args)
    # sol_vbyu = brentq(FF_Yang_Dou_residual, 18., 100., ff_args)

    ff_Yang_Dou = 2 / sol_vbyu ** 2
    Rplus = Re / (2 * sol_vbyu)
    ff_Man_Strkl = 0.143 / 4 / rough ** (1 / 3)

    if Rplus >= 100 * rough:
        ff = ff_Man_Strkl
    else:
        ff = ff_Yang_Dou
    if rough < 32 and ff > ff_Man_Strkl:
        ff = ff_Man_Strkl

    return ff

#-----------------------------------------------------------------------------------------------------------------------

def friction_factor_lam_turb_rough(Re, roughness):
    """
    This function approximate the friction factor for the given Reynold's number and the relative roughness arrays. The
    analytical friction factor of 16/Re is returned in case of laminar flow, Yang Joseph (see Virtual Nikuradse Yang & 
    Joseph 2009) approximation is used for the turbulent flow cases where the 1/relative roughness is greater than 15, 
    and Yang Dou approximation (see Yang, S. Dou, G. (2010), Turbulent drag reduction with polymer additive in rough 
    pipes. Journal of Fluid Mechanics, 642 279-294) is used in the case of turbulent flow with 1/relative roughness 
    less than 15.

    Arguments:
        Re(ndarray):            -- Reynold's number
        roughness (ndarray):    -- 1/relative roughness (w/roughness length scale)

    Returns:
         ndarray:               -- friction factor
    """
    if Re < 1e-8:
        return 0
    elif Re < 2300:
        return 16./Re
    # elif roughness >= 15.:
    #     return FF_YangJoseph_float(Re, roughness)
    else:
        return FF_Yang_Dou(Re, roughness)

#-----------------------------------------------------------------------------------------------------------------------

def friction_factor_MDR(ReNum, roughness):
    """
    This function approximate the friction factor for the given Reynold's number and the relative roughness arrays. The
    analytical friction factor of 16/Re is returned in case of laminar flow, and an explicit approximation of the
    maximum drag reduction asymptote is returned in case the Reynold's number is larger than 1760.

    Arguments:
        ReNum (ndarray):        -- Reynold's number
        roughness (ndarray):    -- relative roughness
    Returns:
         ndarray :              -- friction factor
    """
    if ReNum < 1e-8:
        return 0
    elif ReNum < 1510: #18112100:
       return 16. / ReNum
    else:
        return 1.78 / ReNum**0.7 #1.1 / ReNum**0.65 #


#-----------------------------------------------------------------------------------------------------------------------

def friction_factor_vector(Re, roughness):
    """
    Vector version of the friction_factor function (see the documentation of the friction_factor function)  
    """
    ff = np.zeros((Re.size,),dtype=np.float64)
    for i in range(0,Re.size):
        ff[i] = friction_factor_MDR(Re[i], roughness[i])
        # ff[i] = friction_factor_lam_turb_rough(Re[i], roughness[i])
    return ff