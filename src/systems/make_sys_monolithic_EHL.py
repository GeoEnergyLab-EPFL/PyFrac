# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

# Internal imports
from systems.make_sys_common_fun import Gravity_term, get_finite_difference_matrix

class Monolithic_EHL_sys_obj(LinearOperator):
  # TESTED FOR NEWTONIAN FLUIDS ONLY!
  # NO BOUNDARY EFFECT YET!
  def __init__(self, system_dim, *args, dtype=np.float64):
    self.dtype_ = dtype
    self.shape_ = (system_dim,system_dim)
    super().__init__(self.dtype_, self.shape_)
    self.args = args
    self.wcNplusHalf = None
    self.FinDiffOprtr = None

  def _matvec(self, xks):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    return monolitic_EHL_dot(xks, self.args, self.wcNplusHalf, self.FinDiffOprtr )

  def _matvec_PastSolution(self, xks):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    return monolitic_EHL_dot_PastSolution(xks, self.args, self.wcNplusHalf, self.FinDiffOprtr )

#  @profile
  def _update_sys(self, solk, interItr):
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
      n_total = n_ch + n_ch + n_act + n_tip

      all_indxs_dp = n_ch + np.arange(n_ch + n_act + n_tip)
      act_tip_indxs_dp = 2 * n_ch + np.arange(n_act + n_tip)
      # act_indxs = n_ch + np.arange(n_act)
      # tip_indxs = n_ch + n_act + np.arange(n_tip)

      # Account for the presence of boundaries
      if Boundary is not None:
          SystemExit('Boundary effect not yet implemented with solve_monolithic')
          tb_np1 = Boundary.getTraction(wNplusOne, EltCrack)
          # from utility import plot_as_matrix
          # K = tb_np1
          # plot_as_matrix(K, frac.mesh)
          tb_n = frac.boundEffTraction
          delta_tb = tb_np1 - tb_n
      # else:
      #     tb_n = np.zeros((len(wNplusOne),), dtype=self.dtype)
      #     delta_tb = np.zeros((len(wNplusOne),), dtype=self.dtype)

      FinDiffOprtr = get_finite_difference_matrix(wNplusOne, solk, frac,
                                                  EltCrack, neiInCrack, fluid_prop,
                                                  mat_prop, sim_prop, frac.mesh,
                                                  InCrack, C, interItr, to_solve,
                                                  to_impose, active, interItr_kp1,
                                                  lst_edgeInCrk)
      # FinDiffOprtr = FinDiffOprtr.tocsr()
      self.FinDiffOprtr = FinDiffOprtr

      G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                       frac.mesh, InCrack, sim_prop)

      S = np.zeros((n_total,), dtype=self.dtype)

      # compute RHS dw:
      active_and_toimpose = np.concatenate((active,to_impose)) #can be saved between iters
      C._set_domain_and_codomain_IDX(active_and_toimpose, to_solve)
      dw_active_and_tip = np.concatenate(((wc_to_impose - frac.w[active]),(imposed_val - frac.w[to_impose])))
      S[:n_ch] = - (C._matvec(dw_active_and_tip))

      # compute RHS dp:
      ch_act_toimpose = np.concatenate((to_solve, active, to_impose))  # can be saved between iters

      # p = pf - sig0


      V_pn = np.dot((FinDiffOprtr.tocsr()[np.arange(n_ch + n_act + n_tip), :].tocsc()[:, np.arange(n_ch + n_act + n_tip)]).toarray(), frac.pFluid[ch_act_toimpose] - mat_prop.SigmaO[ch_act_toimpose])
      #V_pn = np.dot(FinDiffOprtr.toarray()[:,:-1], frac.pFluid[ch_act_toimpose] - mat_prop.SigmaO[ch_act_toimpose])
      S[all_indxs_dp] = dt * ( - V_pn
                               + G[ch_act_toimpose]
                               + Q[ch_act_toimpose] / frac.mesh.EltArea ) \
                        + LeakOff[ch_act_toimpose] / frac.mesh.EltArea

      # S[all_indxs_dp] = dt * ( FinDiffOprtr[:,0:-1].dot(frac.pFluid[ch_act_toimpose]) -
      #                          G[ch_act_toimpose] -
      #                          Q[ch_act_toimpose] / frac.mesh.EltArea ) \
      #                   + LeakOff[ch_act_toimpose] / frac.mesh.EltArea
      S[act_tip_indxs_dp] = S[act_tip_indxs_dp] + dw_active_and_tip



      # indices of solved width, pressure and active width constraint in the solution
      to_del = []
      indices = [range(n_ch), n_ch + np.arange(n_ch), act_tip_indxs_dp, to_del]
      # S is the right hand side vector
      return S, interItr_kp1, indices

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

# -----------------------------------------------------------------------------------------------------------------------
#@profile
def monolitic_EHL_dot(solk, args, wcNplusHalf, FinDiffOprtr, dtype=np.float64):
    #todo: write the description of the arguments
    """
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
    n_total = n_ch + n_ch + n_act + n_tip

    w_ch_indxs = np.arange(n_ch)
    p_ch_indxs = n_ch + np.arange(n_ch)
    p_all_indxs = n_ch + np.arange(n_ch + n_act + n_tip)
    #p_act_indxs = n_ch + n_ch + np.arange(n_act)
    #p_tip_indxs = n_ch + n_act + np.arange(n_tip)

    res = np.zeros(n_total, dtype=dtype)

    # Consider the part dw(cc) of the solution
    C._set_domain_and_codomain_IDX(to_solve, to_solve)
    res[w_ch_indxs] = C._matvec(solk[w_ch_indxs]) / C.diag_val
    res[w_ch_indxs] = res[w_ch_indxs] + C._matvec(solk[p_ch_indxs]) / C.diag_val - solk[p_ch_indxs]

    # Consider the part dp(cc) of the solution
    res[p_ch_indxs] = (solk[w_ch_indxs] - solk[p_ch_indxs]) / C.diag_val

    # Consider the part dp(cc,tt,act) of the solution

    V_dp = dt * np.dot((FinDiffOprtr.tocsr()[range(n_ch + n_act + n_tip), :].tocsc()[:, range(n_ch + n_act + n_tip)]).toarray(), solk[p_all_indxs])
    #V_dp = dt * np.dot(FinDiffOprtr.toarray()[:,:-1], solk[p_all_indxs])
    res[p_all_indxs] = res[p_all_indxs] - V_dp - fluid_prop.compressibility * wcNplusHalf[
                           np.concatenate((to_solve, active, to_impose))] * solk[p_all_indxs]
    #res[p_all_indxs] = res[p_all_indxs] - dt * (FinDiffOprtr[:,0:-1].dot(solk[p_all_indxs])) - fluid_prop.compressibility * wcNplusHalf[np.concatenate((to_solve, to_impose, active))] * solk[p_all_indxs]

    #A, S, interItr_kp1, indices = MakeEquationSystem_Monolithic_precond(solk, self.interItr, *args, return_w=False, dtype=np.float64)
    return res

# -----------------------------------------------------------------------------------------------------------------------


def monolitic_EHL_dot_PastSolution(solk, args, wcNplusHalf, FinDiffOprtr, dtype=np.float64):
    #todo: write the description of the arguments
    """
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
    n_total = n_ch + n_ch + n_act + n_tip

    w_ch_indxs = np.arange(n_ch)
    p_ch_indxs = n_ch + np.arange(n_ch)
    p_all_indxs = n_ch + np.arange(n_ch +  n_act + n_tip)
    #p_act_indxs = n_ch + n_ch + np.arange(n_act)
    #p_tip_indxs = n_ch + n_act + np.arange(n_tip)

    res = np.zeros(n_total, dtype=dtype)

    # Consider the part dw(cc) of the solution

    BT_S_z2 = dt * np.dot((FinDiffOprtr.tocsr()[range(n_ch), :].tocsc()[:, range(n_ch + n_act + n_tip)]).toarray(), solk[p_all_indxs]) \
              + fluid_prop.compressibility * wcNplusHalf[to_solve] * solk[p_ch_indxs]\
              + solk[p_ch_indxs] / C.diag_val

    # BT_S_z2 = dt * np.dot(FinDiffOprtr.toarray()[w_ch_indxs,:-1], solk[p_all_indxs]) \
    #           + fluid_prop.compressibility * wcNplusHalf[to_solve] * solk[p_ch_indxs]\
    #           + solk[p_ch_indxs] / C.diag_val
    res[w_ch_indxs] = solk[w_ch_indxs] * C.diag_val + BT_S_z2

    S_dot_solk = np.zeros(len(p_all_indxs))
    all_cells_ids = np.concatenate((to_solve, active, to_impose))

    # -- 1st application of S --
    # Consider the part dp(cc) of the solution
    S_dot_solk[:n_ch] = - solk[p_ch_indxs] / C.diag_val

    # Consider the part dp(cc,tt,act) of the solution
    L_dot_solk = dt * np.dot((FinDiffOprtr.tocsr()[ range(n_ch + n_act + n_tip), :].tocsc()[:, range(n_ch + n_act + n_tip)]).toarray(), solk[p_all_indxs]) \
                 + fluid_prop.compressibility * wcNplusHalf[all_cells_ids] * solk[p_all_indxs]
    # L_dot_solk = dt * np.dot(FinDiffOprtr.toarray()[:,:-1], solk[p_all_indxs]) \
    #              + fluid_prop.compressibility * wcNplusHalf[all_cells_ids] * solk[p_all_indxs]
    S_dot_solk = S_dot_solk - L_dot_solk



    # -- 2nd application of S --
    # Consider the part dp(cc) of the solution
    res[p_ch_indxs] = - S_dot_solk[:n_ch] / C.diag_val

    # Consider the part dp(cc,tt,act) of the solution
    L_dot_Ssolk = dt * np.dot((FinDiffOprtr.tocsr()[ range(n_ch + n_act + n_tip), :].tocsc()[:, range(n_ch + n_act + n_tip)]).toarray(), S_dot_solk) \
                  + fluid_prop.compressibility * wcNplusHalf[all_cells_ids] * S_dot_solk
    # L_dot_Ssolk = dt * np.dot(FinDiffOprtr.toarray()[:,:-1], S_dot_solk) \
    #               + fluid_prop.compressibility * wcNplusHalf[all_cells_ids] * S_dot_solk
    res[p_all_indxs] = res[p_all_indxs] - L_dot_Ssolk

    return res

# -----------------------------------------------------------------------------------------------------------------------
#
def MakeEquationSystem_Monolithic_precond(solk, interItr, *args, return_w=False, dtype = np.float64):
    """

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
    n_total = n_ch + n_ch + n_act + n_tip

    ch_indxs_dw = np.arange(n_ch)
    ch_indxs_dp = n_ch + np.arange(n_ch)
    all_indxs_dp = n_ch + np.arange(n_ch + n_act + n_tip)

    A = makeA()
    #np.linalg.eig(A)
    S = np.zeros((n_total,), dtype=np.float64)


    S[ch_indxs_dw] = -np.dot(C[np.ix_(to_solve, active)],solk[2*n_ch + np.arange(n_act)])+ \
                     np.dot(C[np.ix_(to_solve, to_impose)], solk[2 * n_ch + n_act + np.arange(n_tip)])

    all = np.concatenate((to_solve, active, to_impose))

    S[n_ch:] = -(- dt * (FinDiffOprtr.tocsr()[np.arange(n_ch + n_act + n_tip), :].tocsc()[:, np.arange(n_ch + n_act + n_tip)]).dot(frac.pNet[all])
               + dt * G[all]
               + dt * Q[all] / frac.mesh.EltArea
               - LeakOff[all] / frac.mesh.EltArea)

    S[2 * n_ch + np.arange(n_act)] = -(wc_to_impose - frac.w[active])

    S[2 * n_ch + n_act + np.arange(n_tip)] = -(imposed_val - frac.w[to_impose])

    to_del = []

    # indices of solved width, pressure and active width constraint in the solution
    act_tip_indxs_dp = 2 * n_ch + np.arange(n_act + n_tip)
    indices = [range(n_ch), n_ch + np.arange(n_ch), act_tip_indxs_dp, to_del]

    interItr_kp1[1] = below_wc

    if not return_w:
        return A, S, interItr_kp1, indices
    else:
        return A, S, interItr_kp1, indices, wcNplusHalf, FinDiffOprtr.tocsr()

# -----------------------------------------------------------------------------------------------------------------------

def makeA(C, wcNplusHalf, dt, FinDiffOprtr, fluid_prop, n_act, n_tip, n_total, n_ch, ch_indxs_dw, ch_indxs_dp, all_indxs_dp, to_solve, active, to_impose, dtype):
    A = np.zeros((n_total, n_total), dtype=dtype)

    # 11
    A[np.ix_(ch_indxs_dw, ch_indxs_dw)] = C[np.ix_(to_solve, to_solve)] / C.diag_val
    # 12
    A[np.ix_(ch_indxs_dw, ch_indxs_dp)] = C[np.ix_(to_solve, to_solve)] / C.diag_val - np.identity(n_ch)
    # 21
    A[np.ix_(ch_indxs_dp, ch_indxs_dw)] = (1./C.diag_val) * np.identity(n_ch)
    # 22 (part)
    A[np.ix_(ch_indxs_dp, ch_indxs_dp)] = (-1./C.diag_val) * np.identity(n_ch)
    # (2-4)(2-4)
    A[np.ix_(all_indxs_dp, all_indxs_dp)] = A[np.ix_(all_indxs_dp, all_indxs_dp)] - \
                                          dt * (FinDiffOprtr.tocsr()[np.arange(n_ch + n_act + n_tip), :].tocsc()[:, np.arange(n_ch + n_act + n_tip)]).toarray() - \
                                          np.identity((n_ch + n_act + n_tip)) * fluid_prop.compressibility * wcNplusHalf[np.concatenate((to_solve,active,to_impose))]