# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# External imports
import logging
import time

# Internal imports
from systems.sol_sys_EHL import sol_sys_EHL
from systems.sol_sys_volume_and_load_control import sol_sys_volume_and_load_control


def solve_width_pressure(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,Boundary,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon,
                         doublefracturedictionary = None, inj_same_footprint = False):
    """
    This function evaluates the width and pressure by constructing and solving the coupled elasticity and fluid flow
    equations. The system of equations are formed according to the type of solver given in the simulation properties.
    """
    #log = logging.getLogger('PyFrac.solve_width_pressure')
    if sim_properties.get_volumeControl():
        return sol_sys_volume_and_load_control(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,Boundary,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon,
                         doublefracturedictionary = doublefracturedictionary, inj_same_footprint = inj_same_footprint)

    if sim_properties.get_viscousInjection():
        return sol_sys_EHL(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,Boundary,
                                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon,
                                         doublefracturedictionary = doublefracturedictionary, inj_same_footprint = inj_same_footprint)