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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spilu

# internal import
from systems.sys_back_subst_EHL import check_covergance
from properties import instrument_start, instrument_close
from systems.preconditioners.prec_back_subst_EHL import APrec
from utilities.utility import gmres_counter

#@profile
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
        if not sim_prop.EHL_GMRES: (A, b, interItr, indices) = sys_fun(xks[0, ::], interItr, *args)     # assembling A and b
        #else: (A, b, interItr, indices) = sys_fun._getsys(xks[0, ::], interItr, *args)
        else:
            (A, b, interItr, indices) = sys_fun._getsys_simplif(xks[0, ::], interItr, *args, decay_tshold = 0.68, probability = 0.25)

        #cond_num.append(np.linalg.cond(A)) #this is expensive to compute! do it only while debugging

        if sim_prop.EHL_GMRES:
            # (A1, b1, interItr1, indices1) = sys_fun._getsys(xks[0, ::], interItr, *args)
            # ss, sss = np.linalg.eig(A1)
            # (A1, b1, interItr1, indices1) = sys_fun._getsys_simplif(xks[0, ::], interItr, *args)
            # EHL_iLU = spilu(csc_matrix(A1), drop_tol=0., fill_factor=1)
            # ss, sss = np.linalg.eig((np.identity(A1.shape[0]) / A1[0, 0]).dot(A1))
            # ss
            # (A1, b1, interItr1, indices1) = sys_fun._getsys(xks[0, ::], interItr, *args)
            # EHL_iLU = spilu(csc_matrix(A1), drop_tol=0., fill_factor=1)

            # ---> to check the sparsity pattern
            # import matplotlib
            # matplotlib.pyplot.spy(EHL_iLU.L)

            # prepare preconditioner
            EHL_iLU = spilu(csc_matrix(A), drop_tol=0., fill_factor=1)
            Aprec = APrec(EHL_iLU)
            counter = gmres_counter()  # to obtain the number of iteration and residual
            sol_GMRES = gmres(sys_fun,
                              b,
                              M=Aprec,
                              atol=sim_prop.gmres_tol,
                              tol=1.e-9,
                              maxiter=sim_prop.gmres_maxiter,
                              callback=counter,
                              restart=1000)
            # sol_GMRES = bicgstab(sys_fun,
            #                   b,
            #                   M=Aprec,
            #                   atol=sim_prop.gmres_tol,
            #                   tol=1.e-9,
            #                   maxiter=sim_prop.gmres_maxiter,
            #                   callback=counter)
            if sol_GMRES[1] > 0:
                log.warning("EHL system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
            elif sol_GMRES[1] == 0:
                log.debug(" --> GMRES EHL converged after " + str(counter.niter) + " iter. ")
                #file_name = "/Users/carloperuzzo/Desktop/Pyfrac_formulation/_gmres_dev/_preconditioner/_data&performances/Gmres_with_parallel_Hdot/_data_Elast_stencil/gmres_iter_vs_size.txt"
                #append_new_line(file_name, str(len(b)) + ' ' + str(counter.niter))
            Gks[0, ::] = sol_GMRES[0]
        else:
            # DIRECT SOLVER #
            Gks[0, ::] = np.linalg.solve(A, b)
        Fks[0, ::] = Gks[0, ::] - xks[0, ::]
        xks[1, ::] = Gks[0, ::]                                               # x1



    except np.linalg.linalg.LinAlgError:
        log.error('singular matrix!')
        solk = np.full((len(xks[0]),), np.nan, dtype=np.float64)
        if perf_node is not None:
            instrument_close(perf_node, perfNode_linSolve, None,
                             len(b), False, 'singular matrix', None)
            perf_node.linearSolve_data.append(perfNode_linSolve)
        return solk, None

    while not converged:

        try:
            mk = np.min([k, m_Anderson-1])  # Asses the amount of solutions available for the least square problem
            if k >= m_Anderson:
                if not sim_prop.EHL_GMRES: (A, b, interItr, indices) = sys_fun(xks[mk + 2, ::], interItr, *args)
                else : (b, interItr) = sys_fun._update_sys(xks[mk + 2, ::], interItr); x0 = xks[mk + 2, ::]
                Gks = np.roll(Gks, -1, axis=0)
                Fks = np.roll(Fks, -1, axis=0)
            else:
                if not sim_prop.EHL_GMRES: (A, b, interItr, indices) = sys_fun(xks[mk + 1, ::], interItr, *args)
                else: (b, interItr) = sys_fun._update_sys(xks[mk + 1, ::], interItr); x0 = xks[mk + 1, ::]

            perfNode_linSolve = instrument_start("linear system solve", perf_node)

            if sim_prop.EHL_GMRES:
                counter = gmres_counter()  # to obtain the number of iteration and residual
                sol_GMRES = gmres(sys_fun,
                                  b,
                                  M=Aprec,
                                  x0=x0,
                                  atol=sim_prop.gmres_tol,
                                  tol=1.e-9,
                                  maxiter=sim_prop.gmres_maxiter,
                                  callback=counter,
                                  restart=1000)
                if sol_GMRES[1] > 0:
                    log.warning( "EHL system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
                elif sol_GMRES[1] == 0:
                    log.debug(" --> GMRES EHL converged after " + str(counter.niter) + " iter. ")
                    #file_name = "/Users/carloperuzzo/Desktop/Pyfrac_formulation/_gmres_dev/_preconditioner/_data&performances/Gmres_with_parallel_Hdot/_data_Elast_stencil/gmres_iter_vs_size.txt"
                    #append_new_line(file_name, str(len(b)) + ' ' + str(counter.niter))
                Gks[mk + 1, ::] = sol_GMRES[0]
            else: # DIRECT SOLVER #
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
            log.error('singular matrix!')
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
