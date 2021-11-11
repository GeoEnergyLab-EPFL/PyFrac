# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external import
import logging
import numpy as np

# internal import
from systems.sys_back_subst_EHL import check_covergance
from properties import instrument_start, instrument_close

#@profile
def Anderson(linear_solver, guess, interItr_init, sim_prop, *args, perf_node=None):
    """
    Anderson solver for non linear system.

    Args:
        linear_solver (Linear_solver):      -- An object creating and solving the linear system A(x) * x = b(x).
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
    log=logging.getLogger('PyFrac.Anderson')
    m_Anderson = sim_prop.Anderson_parameter
    relax = sim_prop.relaxation_factor

    ## Initialization of solution vectors
    xks = np.full((m_Anderson+2, guess.size), 0.)
    Fks = np.full((m_Anderson+1, guess.size), 0.)
    Gks = np.full((m_Anderson+1, guess.size), 0.)

    ## Initialization of iteration parameters
    k = 0
    normlist = []
    #cond_num = [] #this is expensive to compute! do it only while debugging
    interItr = interItr_init
    converged = False
    try:
        perfNode_linSolve = instrument_start("linear system solve", perf_node)
        # First iteration
        xks[0, ::] = np.array([guess])                                       # xo
        #cond_num.append(np.linalg.cond(A)) #this is expensive to compute! do it only while debugging
        sol = linear_solver.solve(xks[0, ::], interItr, *args)
        if sim_prop.solve_monolithic:
            Gks[0, ::] = linear_solver.sys_func._matvec_PastSolution(sol)
        else:
            Gks[0, ::] = sol
        interItr = linear_solver.interItr
        Fks[0, ::] = Gks[0, ::] - xks[0, ::]
        xks[1, ::] = Gks[0, ::]                                               # x1

    except np.linalg.linalg.LinAlgError:
        log.error('singular matrix!')
        solk = np.full((len(xks[0]),), np.nan, dtype=np.float64)
        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, None,len(linear_solver.b), False, 'singular matrix', None)
            perf_node.linearSolve_data.append(perfNode_linSolve)
        return solk, None

    while not converged:

        try:
            mk = np.min([k, m_Anderson-1])  # Asses the amount of solutions available for the least square problem
            if k >= m_Anderson:
                xks_current = xks[mk + 2, ::]
                Gks = np.roll(Gks, -1, axis=0)
                Fks = np.roll(Fks, -1, axis=0)
            else:
                xks_current = xks[mk + 1, ::]

            perfNode_linSolve = instrument_start("linear system solve", perf_node)

            sol = linear_solver.solve(xks_current, interItr, *args)
            if sim_prop.solve_monolithic:
                Gks[mk + 1, ::] = linear_solver.sys_func._matvec_PastSolution(sol)
            else:
                Gks[mk + 1, ::] = sol
            interItr = linear_solver.interItr
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
            log.error('singular matrix!')
            solk = np.full((len(xks[mk]),), np.nan, dtype=np.float64)
            if perf_node is not None:
                instrument_close(perf_node, perfNode_linSolve, None,
                                 len(linear_solver.b), False, 'singular matrix', None)
                perf_node.linearSolve_data.append(perfNode_linSolve)
            return solk, None

        ## Check for convergency of the solution
        converged, norm = check_covergance(xks[mk + 1, ::], xks[mk + 2, ::], linear_solver.indices, sim_prop.toleranceEHL)
        normlist.append(norm)
        k = k + 1

        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, norm, len(linear_solver.b), True, None, None)
            perf_node.linearSolve_data.append(perfNode_linSolve)

        if k == sim_prop.maxSolverItrs:  # returns nan as solution if does not converge
            log.warning('Anderson iteration not converged after ' + repr(sim_prop.maxSolverItrs) + \
                  ' iterations, norm:' + repr(norm))
            solk = np.full((np.size(xks[0,::]),), np.nan, dtype=np.float64)
            if perf_node is not None:
                perfNode_linSolve.failure_cause = 'singular matrix'
                perfNode_linSolve.status = 'failed'
            return solk, None

    log.debug("Converged after " + repr(k) + " iterations")

    data = [interItr[0], interItr[2], interItr[3]]
    return xks[mk + 2, ::], data
