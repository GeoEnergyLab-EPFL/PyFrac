# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

#external imports
import numpy as np
import logging

# internal imports
from systems.sys_back_subst_EHL import check_covergance, Elastohydrodynamic_ResidualFun
from properties import instrument_start, instrument_close


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
    log = logging.getLogger('PyFrac.Picard_Newton')
    relax = sim_prop.relaxation_factor
    solk = guess
    k = 0
    normlist = []
    interItr = interItr_init
    newton = 0
    converged = False

    while not converged: #todo:check system change (AM)

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
                # if len(indices[3]) > 0:             # if the size of system is varying between iterations (in case of HB fluid)
                #     solk = relax * solkm1 + (1 - relax) * get_complete_solution(sol, indices, *args)
                # else:
                solk = relax * solkm1 + (1 - relax) * sol
            except np.linalg.linalg.LinAlgError:
                log.error('singular matrix!')
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
            log.warning('Picard iteration not converged after ' + repr(sim_prop.maxSolverItrs) + \
                  ' iterations, norm:' + repr(norm))
            solk = np.full((len(solk),), np.nan, dtype=np.float64)
            if perf_node is not None:
                perfNode_linSolve.failure_cause = 'singular matrix'
                perfNode_linSolve.status = 'failed'
            return solk, None


    log.debug("Converged after " + repr(k) + " iterations")
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