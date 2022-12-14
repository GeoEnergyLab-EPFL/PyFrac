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
from scipy.sparse.linalg import LinearOperator
import time

# Internal imports
from systems.make_sys_common_fun import get_finite_difference_matrix
from systems.make_sys_common_fun import Gravity_term

def MakeEquationSystem_ViscousFluid_pressure_substituted_sparse(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The finite difference
    difference opertator is saved as a sparse matrix. The system is assembled with the extended footprint (treating the
    channel and the extended tip elements distinctly; see description of the ILSA algorithm). The pressure in the tip
    cells and the cells where width constraint is active are solved separately. The pressure in the channel cells to be
    solved for change in width is substituted with width using the elasticity relation (see Zia and Lecamption 2019).

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args


    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)


    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)


    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), fluid_prop.compressibility * wcNplusHalf[to_solve])], [0], format='csr')

    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)
    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - \
                  LeakOff[to_solve] / frac.mesh.EltArea + \
                  fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)

        if len(to_del) > 0:
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]

    interItr_kp1[1] = below_wc

    return A, S, interItr_kp1, indices

#-----------------------------------------------------------------------------------------------------------------------
#@profile
def MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse(solk, interItr, *args, return_w=False, dtype = np.float64):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
    is saved as a sparse matrix.

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """
    # see https://matteding.github.io/2019/04/25/sparse-matrices/ for more info about CSC matrix

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4

    # Account for the presence of boundaries
    if Boundary is not None:
        tb_np1 = Boundary.getTraction(wNplusOne, EltCrack)
        # from utility import plot_as_matrix
        # K = tb_np1
        # plot_as_matrix(K, frac.mesh)
        tb_n = frac.boundEffTraction
        delta_tb = tb_np1 - tb_n

    else:
        tb_n = np.zeros((len(wNplusOne),), dtype=dtype)
        delta_tb = np.zeros((len(wNplusOne),), dtype=dtype)

    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)


    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=dtype)

    ch_AplusCf = dt * FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), fluid_prop.compressibility * wcNplusHalf[to_solve])], [0], format='csr')

    # 1
    A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])

    # 2
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=dtype)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).toarray()

    # 3
    A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])

    # 4
    A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs] +
                                       sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
                                                    [0], format='csr')).toarray()
    A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).toarray()

    # 5
    A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
                                        ).dot(C[np.ix_(to_solve, to_solve)])

    # 6
    A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs] +
                                       sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
                                                    [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                  dt * (FinDiffOprtr.tocsr()[ch_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                  + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]+ \
                  + (ch_AplusCf.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea + \
                   - dt * (FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea+ \
                   - dt * (FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)

        if len(to_del) > 0:
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]

    interItr_kp1[1] = below_wc

    if not return_w:
        return A, S, interItr_kp1, indices
    else:
        return A, S, interItr_kp1, indices, wcNplusHalf, FinDiffOprtr.tocsr()

# -----------------------------------------------------------------------------------------------------------------------
#@profile
def make_local_elast_sys(solk, interItr, *args, return_w=False, dtype = np.float64, decay_tshold = 0.9, probability = 0.15):
    """
    ATTENTION: derived from

    MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse


    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
    is saved as a sparse matrix.

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """
    # see https://matteding.github.io/2019/04/25/sparse-matrices/ for more info about CSC matrix

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4

    # Account for the presence of boundaries
    if Boundary is not None:
        tb_np1 = Boundary.getTraction(wNplusOne, EltCrack)
        # from utility import plot_as_matrix
        # K = tb_np1
        # plot_as_matrix(K, frac.mesh)
        tb_n = frac.boundEffTraction
        delta_tb = tb_np1 - tb_n

    else:
        tb_n = np.zeros((len(wNplusOne),), dtype=dtype)
        delta_tb = np.zeros((len(wNplusOne),), dtype=dtype)
    #a = -time.time()
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)
    #a = a + time.time()
    #print(f'1 {a}')


    #a = -time.time()
    FinDiffOprtr = FinDiffOprtr.tocsr()

    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)
    #a = a + time.time()
    #print(f'2 {a}')

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    #A = np.zeros((n_total, n_total), dtype=dtype)
    #a = -time.time()
    ch_AplusCf = dt * FinDiffOprtr[ch_indxs, :].tocsc()[:, ch_indxs] \
                 - sparse.diags([np.full((n_ch,), fluid_prop.compressibility * wcNplusHalf[to_solve])], [0], format='csr')
    #a = a + time.time()
    #print(f'3 {a}')

    #a = -time.time()
    C_loc = C._get9stencilC(to_solve, decay_tshold = decay_tshold, probability = probability)
    #a = a + time.time()
    #print(f'4 {a}')
    """
    (1)
    *ch_ch*  ch_act    ch_tip
    act_ch   act_act   act_tip
    tip_ch   tip_act   tip_tip
    """
    # A[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    # A[np.ix_(ch_indxs, ch_indxs)] = - ((ch_AplusCf.tocsc()).dot(C_loc)).toarray()
    # A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=dtype)
    A_ch_ch = - ((ch_AplusCf.tocsc()).dot(C_loc)) + sparse.identity(n_ch, dtype=dtype, format='csc')

    if n_act == 0 and n_tip == 0 :
        A = A_ch_ch
    else:
        A_ch_ch = A_ch_ch.tocoo(copy=False)


        """
        (2)
        (ch_ch)    ch_act    *ch_tip*
        act_ch    act_act    act_tip
        tip_ch    tip_act    tip_tip
        """
        # A[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
        A_temp = (-dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs])).tocoo(copy=False)
        row = [*A_ch_ch.row, *A_temp.row]
        col = [*A_ch_ch.col, *A_temp.col + n_ch + n_act]
        data = [*A_ch_ch.data, *A_temp.data]


        """
        (3)
        (ch_ch)  ch_act    (ch_tip)
        act_ch   act_act   act_tip
        *tip_ch*   tip_act   tip_tip
        """
        # with full C
        # A[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[tip_indxs, :].tocsc()[:, ch_indxs]
        #                                     ).dot(C[np.ix_(to_solve, to_solve)])
        # with approx C but slow
        # A[np.ix_(tip_indxs, ch_indxs)] = - (((dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]
        #                                     ).tocsc()).dot(C_loc)).toarray()
        A_temp = (- (((dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).tocsc()).dot(C_loc))).tocoo(copy=False)
        row.extend(A_temp.row + n_ch + n_act)
        col.extend(A_temp.col)
        data.extend(A_temp.data)


        """
        (4)
        (ch_ch)  ch_act    (ch_tip)
        act_ch   act_act   act_tip
        (tip_ch)   tip_act   *tip_tip*
        """
        # A[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs] +
        #                                    sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
        #                                                 [0], format='csc')).toarray()
        A_temp = (- dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs] + sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],[0],format='csc')).tocoo(copy=False)
        row.extend(A_temp.row + n_ch + n_act)
        col.extend(A_temp.col + n_ch + n_act)
        data.extend(A_temp.data)


        """
        (5)
        (ch_ch)      ch_act    (ch_tip)
        act_ch      act_act     act_tip
        (tip_ch)   *tip_act*   (tip_tip)
        """
        # A[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs]).toarray()
        A_temp = (-dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs])).tocoo(copy=False)
        row.extend(A_temp.row + n_ch + n_act)
        col.extend(A_temp.col + n_ch)
        data.extend(A_temp.data)


        """
        (6)
        (ch_ch)    *ch_act*     (ch_tip)
        act_ch     act_act     act_tip
        (tip_ch)  (tip_act)   (tip_tip)
        """
        # A[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs]).toarray()
        A_temp = (-dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs])).tocoo(copy=False)
        row.extend(A_temp.row)
        col.extend(A_temp.col + n_ch)
        data.extend(A_temp.data)


        """
        (7)
        (ch_ch)   (ch_act)    (ch_tip)
        *act_ch*   act_act     act_tip
        (tip_ch)  (tip_act)   (tip_tip)
        """
        # with full C
        # A[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr.tocsr()[act_indxs, :].tocsc()[:, ch_indxs]
        #                                     ).dot(C[np.ix_(to_solve, to_solve)])
        # with approx C but slow
        # A[np.ix_(act_indxs, ch_indxs)] = - (((dt * FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]
        #                                 ).tocsc()).dot(C_loc)).toarray()
        A_temp = (- (((dt * FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).tocsc()).dot(C_loc))).tocoo(copy=False)
        row.extend(A_temp.row + n_ch)
        col.extend(A_temp.col)
        data.extend(A_temp.data)

        """
        (8)
        (ch_ch)   (ch_act)    (ch_tip)
        (act_ch)   act_act    *act_tip*
        (tip_ch)  (tip_act)   (tip_tip)
        """
        # A[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
        A_temp = (-dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs])).tocoo(copy=False)
        row.extend(A_temp.row + n_ch)
        col.extend(A_temp.col + n_ch + n_act)
        data.extend(A_temp.data)


        """
        (9)
        (ch_ch)   (ch_act)    (ch_tip)
        (act_ch)  *act_act*    (act_tip)
        (tip_ch)  (tip_act)   (tip_tip)
        """
        # A[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs] +
        #                                    sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
        #                                                 [0], format='csc')).toarray()
        A_temp = (- dt * FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs] + sparse.diags(
            [np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])], [0], format='csc')).tocoo(copy=False)
        row.extend(A_temp.row + n_ch)
        col.extend(A_temp.col + n_ch)
        data.extend(A_temp.data)


        A = sparse.csc_matrix((data, (row, col)), shape=(n_total, n_total))

    # from scipy.sparse.linalg import spilu
    # from scipy.sparse import csc_matrix
    # import numpy as np
    # import matplotlib.pyplot as plt
    # EHL_iLU = spilu(csc_matrix(A), drop_tol=1.e-10, fill_factor=5)
    #
    # ei = np.linalg.eig(EHL_iLU.solve(A))
    # x = np.arange(len(ei[0]))
    # y = ei[0]
    # plt.scatter(x, y, s=10, marker='o', cmap='hue')
    #
    # EHL_iLU = spilu(csc_matrix(A), drop_tol=1.e-10, fill_factor=10)
    # ei = np.linalg.eig(EHL_iLU.solve(A))
    # x = np.arange(len(ei[0]))
    # y = ei[0]
    # plt.scatter(x, y, s=1, marker='o', cmap='hue')
    #
    # plt.show()
    # import matplotlib.pyplot as plt
    # plt.spy(C_loc, marker='o',markersize=0.5)
    # aaaa = np.count_nonzero(C_loc.toarray())

    # from numpy.linalg import inv
    # import numpy as np
    # import matplotlib.pyplot as plt
    # Iapp = np.dot(inv(A), Afull)
    # fig, ax = plt.subplots()
    # ax.matshow(Iapp, cmap=plt.cm.Blues)

    # from scipy.sparse.linalg import spilu
    # from scipy.sparse import csc_matrix
    # EHL_iLU = spilu(csc_matrix(A), drop_tol=0., fill_factor=1.)
    #
    # Afull = np.zeros((n_total, n_total), dtype=dtype)
    #
    # Afull[np.ix_(ch_indxs, ch_indxs)] = - ch_AplusCf.dot(C[np.ix_(to_solve, to_solve)])
    # #Afull[np.ix_(ch_indxs, ch_indxs)] = - ((ch_AplusCf.tocsc()).dot(C_loc)).toarray()
    #
    # # 2
    # Afull[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=dtype)
    #
    # Afull[np.ix_(ch_indxs, tip_indxs)] = -dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs]).toarray()
    # Afull[np.ix_(ch_indxs, act_indxs)] = -dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs]).toarray()
    #
    # # 3
    # Afull[np.ix_(tip_indxs, ch_indxs)] = - (dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]
    #                                     ).dot(C[np.ix_(to_solve, to_solve)])
    # # Afull[np.ix_(tip_indxs, ch_indxs)] = - (((dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]
    # #                                     ).tocsc()).dot(C_loc)).toarray()
    #
    # # 4
    # Afull[np.ix_(tip_indxs, tip_indxs)] = (- dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs] +
    #                                    sparse.diags([np.full((n_tip,), fluid_prop.compressibility * wcNplusHalf[to_impose])],
    #                                                 [0], format='csr')).toarray()
    # Afull[np.ix_(tip_indxs, act_indxs)] = -dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs]).toarray()
    #
    # # 5
    # Afull[np.ix_(act_indxs, ch_indxs)] = - (dt * FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]
    #                                     ).dot(C[np.ix_(to_solve, to_solve)])
    # # Afull[np.ix_(act_indxs, ch_indxs)] = - (((dt * FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]
    # #                                 ).tocsc()).dot(C_loc)).toarray()
    #
    # # 6
    # Afull[np.ix_(act_indxs, tip_indxs)] = -dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs]).toarray()
    # Afull[np.ix_(act_indxs, act_indxs)] = (- dt * FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs] +
    #                                    sparse.diags([np.full((n_act,), fluid_prop.compressibility * wcNplusHalf[active])],
    #                                                 [0], format='csr')).toarray()

    S = np.zeros((n_total,), dtype=np.float64)

    # pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
    #               np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
    #               np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
    #               mat_prop.SigmaO[to_solve]

    C._set_domain_and_codomain_IDX(to_solve, to_solve, )
    pf_ch_prime = C._matvec(frac.w[to_solve])

    if n_tip > 0:
        C._set_domain_and_codomain_IDX(to_impose, to_solve)
        pf_ch_prime = pf_ch_prime + C._matvec(imposed_val)

    if n_act > 0:
        C._set_domain_and_codomain_IDX(active, to_solve)
        pf_ch_prime = pf_ch_prime + C._matvec(wNplusOne[active])

    pf_ch_prime = pf_ch_prime + mat_prop.SigmaO[to_solve]


    S[ch_indxs] = ch_AplusCf.dot(pf_ch_prime) + \
                  dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                  dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                  + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]+ \
                  + (ch_AplusCf.tocsr()[ch_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea + \
                   - dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                   dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                   dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea+ \
                   - dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

    # --- OMITTED AT THE MOMENT ---
    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    # if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
    #     for i in range(n_tip + n_act):
    #             if not A[n_ch + i, :].any():
    #                 to_del.append(i)
    #
    #     if len(to_del) > 0:
    #         deleted = n_ch + np.asarray(to_del)
    #         A = np.delete(A, deleted, 0)
    #         A = np.delete(A, deleted, 1)
    #         S = np.delete(S, deleted)
    #  --- OMITTED AT THE MOMENT ---

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]

    interItr_kp1[1] = below_wc

    if not return_w:
        return A, S, interItr_kp1, indices
    else:
        # A is a csc matrix
        return A, S, interItr_kp1, indices, wcNplusHalf, FinDiffOprtr

# -----------------------------------------------------------------------------------------------------------------------
class EHL_sys_obj(LinearOperator):
  # TESTED FOR NEWTONIAN FLUIDS ONLY!
  def __init__(self, system_dim, dtype=np.float64):
    self.dtype_ = dtype
    self.shape_ = (system_dim,system_dim)
    super().__init__(self.dtype_, self.shape_)
    self.args = None
    self.wcNplusHalf = None
    self.FinDiffOprtr = None

  def _matvec(self, xks):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    return EHL_dot(xks, self.args, self.wcNplusHalf, self.FinDiffOprtr )

  def _getsys(self, solk, interItr, *args):
    """
    This function gets the system of equations to be solved
    """
    self.args = args # args never change within the solution of 1 non-linear system

    (A, b, interItr, indices, wcNplusHalf, FinDiffOprtr) = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse(solk, interItr, *args,  return_w=True)

    self.wcNplusHalf = wcNplusHalf
    self.FinDiffOprtr = FinDiffOprtr
    return (A, b, interItr, indices)

  def _getsys_simplif(self, solk, interItr, *args, decay_tshold = 0.9, probability = 0.15):
    """
    This function gets a simplified version of the system of equations to be decomposed (see ILU) and used as preconditioner
    """
    self.args = args # args never change within the solution of 1 non-linear system

    (A, b, interItr, indices, wcNplusHalf, FinDiffOprtr) = make_local_elast_sys(solk, interItr, *args,  return_w=True, decay_tshold = decay_tshold , probability = probability)

    self.wcNplusHalf = wcNplusHalf
    self.FinDiffOprtr = FinDiffOprtr
    return (A, b, interItr, indices)

#  @profile
  def _update_sys(self, solk, interItr):
      """
      Consider the system of equations: A(x) * x = b(x)
      This function updates x (for A(x)) and returns b(x)
      """
      (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
       sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = self.args

      wNplusOne = np.copy(frac.w)
      wNplusOne[to_solve] += solk[:len(to_solve)]
      wNplusOne[to_impose] = imposed_val
      if len(wc_to_impose) > 0:
          wNplusOne[active] = wc_to_impose

      below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
      below_wc_km1 = interItr[1]
      below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
      wNplusOne[to_solve[below_wc]] = mat_prop.wc

      wcNplusHalf = (frac.w + wNplusOne) / 2.

      # store data
      self.wcNplusHalf = wcNplusHalf

      # return updated list of cells below k1c
      interItr_kp1 = [None] * 4
      interItr_kp1[1] = below_wc

      # get b (rhs of the system)

      n_ch = len(to_solve)
      n_act = len(active)
      n_tip = len(imposed_val)
      n_total = n_ch + n_act + n_tip

      ch_indxs = np.arange(n_ch)
      act_indxs = n_ch + np.arange(n_act)
      tip_indxs = n_ch + n_act + np.arange(n_tip)

      # Account for the presence of boundaries
      if Boundary is not None:
          tb_np1 = Boundary.getTraction(wNplusOne, EltCrack)
          # from utility import plot_as_matrix
          # K = tb_np1
          # plot_as_matrix(K, frac.mesh)
          tb_n = frac.boundEffTraction
          delta_tb = tb_np1 - tb_n

      else:
          tb_n = np.zeros((len(wNplusOne),), dtype=self.dtype)
          delta_tb = np.zeros((len(wNplusOne),), dtype=self.dtype)

      FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk, frac,
                                                  EltCrack, neiInCrack, fluid_prop,
                                                  mat_prop, sim_prop, frac.mesh,
                                                  InCrack, C, interItr, to_solve,
                                                  to_impose, active, interItr_kp1,
                                                  lst_edgeInCrk)
      FinDiffOprtr = FinDiffOprtr.tocsr()
      self.FinDiffOprtr = FinDiffOprtr

      G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                       frac.mesh, InCrack, sim_prop)

      S = np.zeros((n_total,), dtype=self.dtype)

      # compute pf_ch_prime:
      C._set_domain_and_codomain_IDX(to_solve, to_solve)
      pf_ch_prime = C._matvec(frac.w[to_solve])

      if len(to_impose) > 0:
          C._set_domain_and_codomain_IDX(to_impose, to_solve)
          pf_ch_prime = pf_ch_prime + C._matvec(imposed_val)

      if len(active) > 0:
          C._set_domain_and_codomain_IDX(active, to_solve)
          pf_ch_prime = pf_ch_prime + C._matvec(wNplusOne[active]) + mat_prop.SigmaO[to_solve]
      else:
          pf_ch_prime = pf_ch_prime + mat_prop.SigmaO[to_solve]

      S[ch_indxs] = dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) - \
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * pf_ch_prime+ \
                    dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                    dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                    dt * G[to_solve] + \
                    dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                    + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve] + \
                    + dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve]) - \
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * pf_ch_prime * delta_tb[to_solve]

      S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                     dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                     dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                     dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                     dt * G[to_impose] + \
                     dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea + \
                     - dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])

      S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                     dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).dot(pf_ch_prime) + \
                     dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs]).dot(frac.pFluid[to_impose]) + \
                     dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs]).dot(frac.pFluid[active]) + \
                     dt * G[active] + \
                     dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea + \
                     - dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).dot(delta_tb[to_solve])


      # indices of solved width, pressure and active width constraint in the solution
      to_del = []
      indices = [ch_indxs, tip_indxs, act_indxs, to_del]
      # S is the right hand side vector
      return S, interItr_kp1, indices

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

# -----------------------------------------------------------------------------------------------------------------------
#@profile
def EHL_dot(solk, args, wcNplusHalf, FinDiffOprtr, dtype=np.float64):
    """
    This function has been coded from:
    MakeEquationSystem_DOT_ViscousFluid_pressure_substituted_deltaP_sparse

    It implements the dot product of the linearized system of equations to be solved by a linear system solver. The system is
    never assembled but accounts for the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
    is saved as a sparse matrix.

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.
        wcNplusHalf (ndarray)         -- [wN + w(N+1)] / 2. where wN is the opening at time step N
        FinDiffOprtr (matrix)         -- the finite difference operator

    Returns:
        - res (ndarray)           -- the res vector obtained by the multiplication A*x (in the system Ax=b).

    """
    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    res = np.zeros(n_total, dtype=dtype)

    """
    We can divide the whole matrix in blocks:
        ch   act   tip
      [1,1] [1,2] [1,3] ch
      [2,1] [2,2] [2,3] act
      [3,1] [3,2] [3,3] tip
    """
    # [1,1] INDEXES: ch ch (generally)
    C._set_domain_and_codomain_IDX(to_solve, to_solve)
    res[ch_indxs] = C._matvec(solk[ch_indxs])
    c_dot_solk = np.copy(res[ch_indxs])

    res[ch_indxs] = - (dt * FinDiffOprtr[ch_indxs, :].tocsc()[:, ch_indxs]).dot(res[ch_indxs]) \
                    + fluid_prop.compressibility * wcNplusHalf[to_solve] * res[ch_indxs] + solk[ch_indxs]

    # [2,1] INDEXES: act ch (generally)
    res[act_indxs] = - (dt * FinDiffOprtr[act_indxs, :].tocsc()[:, ch_indxs]).dot(c_dot_solk)

    # [3,1] INDEXES: tip ch (generally)
    res[tip_indxs] = - (dt * FinDiffOprtr[tip_indxs, :].tocsc()[:, ch_indxs]).dot(c_dot_solk)

    # [1,2] INDEXES: ch act (generally)
    res[ch_indxs] = res[ch_indxs] - dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, act_indxs]).dot(solk[act_indxs])

    # [2,2] INDEXES: act act (generally)
    res[act_indxs] = res[act_indxs] - dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, act_indxs]).dot(solk[act_indxs]) \
                     + fluid_prop.compressibility * wcNplusHalf[active] * solk[act_indxs]

    # [3,2] INDEXES: tip act
    res[tip_indxs] = res[tip_indxs] -dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, act_indxs]).dot(solk[act_indxs])


    # [1,3] INDEXES: ch tip (generally)
    res[ch_indxs] = res[ch_indxs] -dt * (FinDiffOprtr[ch_indxs, :].tocsc()[:, tip_indxs]).dot(solk[tip_indxs])

    # [2,3] INDEXES: act tip
    res[act_indxs] = res[act_indxs] -dt * (FinDiffOprtr[act_indxs, :].tocsc()[:, tip_indxs]).dot(solk[tip_indxs])

    # [3,3] INDEXES: tip tip
    res[tip_indxs] = res[tip_indxs] - dt * (FinDiffOprtr[tip_indxs, :].tocsc()[:, tip_indxs]).dot(solk[tip_indxs]) \
                     + fluid_prop.compressibility * wcNplusHalf[to_impose]* solk[tip_indxs]

    return res

# -----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The pressure in the tip cells and the cells where width constraint is active
    are solved separately. The pressure in the channel cells to be solved for change in width is substituted with width
    using the elasticity relation (see Zia and Lecampion 2019).

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4
    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)



    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]

    A[np.ix_(tip_indxs, act_indxs)] = -dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = -dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - \
                  LeakOff[to_solve] / frac.mesh.EltArea + \
                  fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)

        if len(to_del) > 0:
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]

    interItr_kp1[1] = below_wc

    return A, S, interItr_kp1, indices


#-----------------------------------------------------------------------------------------------------------------------

def MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP(solk, interItr, *args):
    """
    This function makes the linearized system of equations to be solved by a linear system solver. The system is
    assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
    description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
    active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
    with width using the elasticity relation (see Zia and Lecamption 2019).

    Arguments:
        solk (ndarray):               -- the trial change in width and pressure for the current iteration of
                                          fracture front.
        interItr (ndarray):            -- the information from the last iteration.
        args (tupple):                 -- arguments passed to the function. A tuple containing the following in order:

            - EltChannel (ndarray)          -- list of channel elements
            - to_solve (ndarray)            -- the cells where width is to be solved (channel cells).
            - to_impose (ndarray)           -- the cells where width is to be imposed (tip cells).
            - imposed_vel (ndarray)         -- the values to be imposed in the above list (tip volumes)
            - wc_to_impose (ndarray)        -- the values to be imposed in the cells where the width constraint is active. \
                                               These can be different then the minimum width if the overall fracture width is \
                                               small and it has not reached the minimum width yet.
            - frac (Fracture)               -- fracture from last time step to get the width and pressure.
            - fluidProp (object):           -- FluidProperties class object giving the fluid properties.
            - matProp (object):             -- an instance of the MaterialProperties class giving the material properties.
            - sim_prop (object):            -- An object of the SimulationProperties class.
            - dt (float)                    -- the current time step.
            - Q (float)                     -- fluid injection rate at the current time step.
            - C (ndarray)                   -- the elasticity matrix.
            - InCrack (ndarray)             -- an array with one for all the elements in the fracture and zero for rest.
            - LeakOff (ndarray)             -- the leaked off fluid volume for each cell.
            - active (ndarray)              -- index of cells where the width constraint is active.
            - neiInCrack (ndarray)          -- an ndarray giving indices(in the EltCrack list) of the neighbours of all\
                                               the cells in the crack.
            - edgeInCrk_lst (ndarray)       -- this list provides the indices of those cells in the EltCrack list whose neighbors are not\
                                               outside the crack. It is used to evaluate the conductivity on edges of only these cells who\
                                               are inside. It consists of four lists, one for each edge.

    Returns:
        - A (ndarray)            -- the A matrix (in the system Ax=b) to be solved by a linear system solver.
        - S (ndarray)            -- the b vector (in the system Ax=b) to be solved by a linear system solver.
        - interItr_kp1 (list)    -- the information transferred between iterations. It has three ndarrays
                                        - fluid velocity at edges
                                        - cells where width is closed
                                        - effective newtonian viscosity
        - indices (list)         -- the list containing 3 arrays giving indices of the cells where the solution is\
                                    obtained for width, pressure and active width constraint cells.
    """

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, Boundary, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    wNplusOne = np.copy(frac.w)
    wNplusOne[to_solve] += solk[:len(to_solve)]
    wNplusOne[to_impose] = imposed_val
    if len(wc_to_impose) > 0:
        wNplusOne[active] = wc_to_impose

    below_wc = np.where(wNplusOne[to_solve] < mat_prop.wc)[0]
    below_wc_km1 = interItr[1]
    below_wc = np.append(below_wc_km1, np.setdiff1d(below_wc, below_wc_km1))
    wNplusOne[to_solve[below_wc]] = mat_prop.wc

    wcNplusHalf = (frac.w + wNplusOne) / 2

    interItr_kp1 = [None] * 4

    # Account for the presence of boundaries
    if Boundary is not None:
        tb_np1 = Boundary.getTraction(wNplusOne, EltCrack)
        # from utility import plot_as_matrix
        # K = tb_np1
        # plot_as_matrix(K, frac.mesh)
        tb_n = frac.boundEffTraction
        delta_tb = tb_np1 - tb_n

    else:
        tb_n = np.zeros((len(wNplusOne),), dtype=np.float64)
        delta_tb = np.zeros((len(wNplusOne),), dtype=np.float64)


    FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk,   frac,
                                 EltCrack,  neiInCrack, fluid_prop,
                                 mat_prop,  sim_prop,   frac.mesh,
                                 InCrack,   C,  interItr,   to_solve,
                                 to_impose, active, interItr_kp1,
                                 lst_edgeInCrk)


    G = Gravity_term(wNplusOne, EltCrack,   fluid_prop,
                    frac.mesh,  InCrack,    sim_prop)

    n_ch = len(to_solve)
    n_act = len(active)
    n_tip = len(imposed_val)
    n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)
    act_indxs = n_ch + np.arange(n_act)
    tip_indxs = n_ch + n_act + np.arange(n_tip)

    A = np.zeros((n_total, n_total), dtype=np.float64)

    ch_AplusCf = dt * FinDiffOprtr[np.ix_(ch_indxs, ch_indxs)]
    ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

    A[np.ix_(ch_indxs, ch_indxs)] = - np.dot(ch_AplusCf, C[np.ix_(to_solve, to_solve)])
    A[ch_indxs, ch_indxs] += np.ones(len(ch_indxs), dtype=np.float64)

    A[np.ix_(ch_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)]
    A[np.ix_(ch_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(ch_indxs, act_indxs)]

    A[np.ix_(tip_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)],
                                                    C[np.ix_(to_solve, to_solve)])
    A[np.ix_(tip_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)]
    A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]
    A[np.ix_(tip_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(tip_indxs, act_indxs)]

    A[np.ix_(act_indxs, ch_indxs)] = - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)],
                                                   C[np.ix_(to_solve, to_solve)])
    A[np.ix_(act_indxs, tip_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, tip_indxs)]
    A[np.ix_(act_indxs, act_indxs)] = - dt * FinDiffOprtr[np.ix_(act_indxs, act_indxs)]
    A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

    S = np.zeros((n_total,), dtype=np.float64)
    pf_ch_prime = np.dot(C[np.ix_(to_solve, to_solve)], frac.w[to_solve]) + \
                  np.dot(C[np.ix_(to_solve, to_impose)], imposed_val) + \
                  np.dot(C[np.ix_(to_solve, active)], wNplusOne[active]) + \
                  mat_prop.SigmaO[to_solve] + tb_n[to_solve]

    S[ch_indxs] = np.dot(ch_AplusCf, pf_ch_prime) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                  dt * np.dot(FinDiffOprtr[np.ix_(ch_indxs, act_indxs)], frac.pFluid[active]) + \
                  dt * G[to_solve] + \
                  dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea \
                  + fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]+ \
                  + np.dot(ch_AplusCf[ch_indxs, ch_indxs], delta_tb[to_solve])

    S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, act_indxs)], frac.pFluid[active]) + \
                   dt * G[to_impose] + \
                   dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea + \
                   - dt * np.dot(FinDiffOprtr[np.ix_(tip_indxs, ch_indxs)], delta_tb[to_solve])

    S[act_indxs] = -(wc_to_impose - frac.w[active]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], pf_ch_prime) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, tip_indxs)], frac.pFluid[to_impose]) + \
                   dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, act_indxs)], frac.pFluid[active]) + \
                   dt * G[active] + \
                   dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea + \
                   - dt * np.dot(FinDiffOprtr[np.ix_(act_indxs, ch_indxs)], delta_tb[to_solve])


    # In the case of HB fluid, there can be tip or active constraint cells with no flux going in and out, making
    # the matrix singular. These pressure in these cells is not solved but is obtained from elasticity relaton.
    to_del = []
    if fluid_prop.rheology  in ["Herschel-Bulkley", "HBF"]:
        for i in range(n_tip + n_act):
                if not A[n_ch + i, :].any():
                    to_del.append(i)
        if len(to_del) > 0:
            deleted = n_ch + np.asarray(to_del)
            A = np.delete(A, deleted, 0)
            A = np.delete(A, deleted, 1)
            S = np.delete(S, deleted)

    # indices of solved width, pressure and active width constraint in the solution
    indices = [ch_indxs, tip_indxs, act_indxs, to_del]

    interItr_kp1[1] = below_wc
    return A, S, interItr_kp1, indices

# -----------------------------------------------------------------------------------------------------------------------


def Elastohydrodynamic_ResidualFun(solk, system_func, interItr, *args):
    """
    This function gives the residual of the solution for the system of equations formed using the given function.
    """
    A, S, interItr, indices = system_func(solk, interItr, *args)
    return np.dot(A, solk) - S, interItr, indices


#-----------------------------------------------------------------------------------------------------------------------

def Elastohydrodynamic_ResidualFun_nd(solk, system_func, interItr, InterItr_o, indices_o,*args):
    """
    This function gives the residual of the solution for the system of equations formed using the given function.
    """
    A, S, interItr, indices = system_func(solk, interItr, *args)
    if len(indices[3]) == 0:
        Fx = np.dot(A, solk) - S
    else:
        Fx_red = np.dot(A, np.delete(solk, len(indices[0]) + np.asarray(indices[3]))) - S
        Fx = populate_full(indices, Fx_red)
    InterItr_o = interItr
    indices_o = indices
    return Fx
#-----------------------------------------------------------------------------------------------------------------------


def check_covergance(solk, solkm1, indices, tol):
    """ This function checks for convergence of the solution

    Args:
        solk (ndarray)      -- the evaluated solution on this iteration
        solkm1 (ndarray)    -- the evaluated solution on last iteration
        indices (list)      -- the list containing 3 arrays giving indices of the cells where the solution is obtained
                               for channel, tip and active width constraint cells.
        tol (float)         -- tolerance

    Returns:
         - converged (bool) -- True if converged
         - norm (float)     -- the evaluated norm which is checked against tolerance
    """

    w_normalization = np.linalg.norm(solkm1[indices[0]])
    if w_normalization > 0.:
        norm_w = np.linalg.norm(abs(solk[indices[0]] - solkm1[indices[0]]) / w_normalization)
    else:
        norm_w = np.linalg.norm(abs(solk[indices[0]] - solkm1[indices[0]]))

    p_normalization = np.linalg.norm(solkm1[indices[1]])
    if p_normalization > 0.:
        norm_p = np.linalg.norm(abs(solk[indices[1]] - solkm1[indices[1]]) / p_normalization)
    else:
        norm_p = np.linalg.norm(abs(solk[indices[1]] - solkm1[indices[1]]) )

    if len(indices[2]) > 0: #these are the cells with the active width constraints
        tr_normalization = np.linalg.norm(solkm1[indices[2]])
        if tr_normalization > 0.:
            norm_tr = np.linalg.norm(abs(solk[indices[2]] - solkm1[indices[2]]) / tr_normalization)
        else:
            norm_tr = np.linalg.norm(abs(solk[indices[2]] - solkm1[indices[2]]))
        cnt = 3.
    else:
        norm_tr = 0.
        cnt = 2.

    norm = (norm_w + norm_p + norm_tr) / cnt
    # todo is not better to show the max of the 3 norms rather than the mean value
    converged = (norm_w <= tol and norm_p <= tol and norm_tr <= tol)

    return converged, norm

#-----------------------------------------------------------------------------------------------------------------------

def get_complete_solution(sol, indices, *args):

    (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
    sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

    tip_act = np.concatenate((to_impose, active))

    w = np.copy(frac.w)
    w[to_solve] += sol[:len(to_solve)]
    w[to_impose] = imposed_val
    w[active] = wc_to_impose

    [ch_indxs, tip_indxs, act_indxs, deleted] = indices

    if sim_prop.solveDeltaP:
        values = np.dot(C[np.ix_(tip_act[deleted], EltCrack)], w[EltCrack]) + \
                    mat_prop.SigmaO[tip_act[deleted]]- frac.pFluid[tip_act[deleted]]
    else:
        values = np.dot(C[np.ix_(tip_act[deleted], EltCrack)], w[EltCrack]) + \
                    mat_prop.SigmaO[tip_act[deleted]]
    sol_full = populate_full(indices, sol, values)

    return sol_full

def populate_full(indices, sol, values=None):

    [ch_indxs, tip_indxs, act_indxs, deleted] = indices
    sol_full = np.empty(len(ch_indxs) + len(tip_indxs) + len(act_indxs))
    sol_full[:len(ch_indxs)] = sol[:len(ch_indxs)]

    if values is None:
        values = np.zeros(len(deleted))
    sol_full[len(ch_indxs) + np.asarray(deleted, dtype=int)] = values
    sol_full[len(ch_indxs) + np.setdiff1d(np.arange(len(tip_indxs) + len(act_indxs)), deleted)] = sol[len(ch_indxs):]

    return sol_full
