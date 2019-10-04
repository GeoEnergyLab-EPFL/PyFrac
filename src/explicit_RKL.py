# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Sep 6 16:53:19 2019.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

from elastohydrodynamic_solver import finiteDiff_operator_laminar
import numpy as np
from math import ceil
# from elasticity import calculate_fluid_flow_characteristics_laminar
import copy

s_max = 1000
a = np.zeros(s_max)
b = np.zeros(s_max)
mu = np.zeros(s_max)
nu = np.zeros(s_max)

b[:2] = 1 / 3
a[:2] = 1 - b[:2]
for j in range(2, s_max):
    b[j] = (j * j + j - 2) / (2 * j * (j + 1))
    a[j] = 1 - b[j]
    mu[j] = (2 * j - 1) * b[j] / (j * b[j - 1])
    nu[j] = - (j - 1) * b[j] / (j * b[j - 2])



def solve_width_pressure_RKL2(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon):

    C_EltTip = np.copy(C[np.ix_(EltTip[partlyFilledTip],
                                EltTip[partlyFilledTip])])  # keeping the tip element entries to restore current
    #  tip correction. This is done to avoid copying the full elasticity matrix.

    # filling fraction correction for element in the tip region
    FillF = FillFrac[partlyFilledTip]
    for e in range(0, len(partlyFilledTip)):
        r = FillF[e] - .25
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r
        C[EltTip[partlyFilledTip[e]], EltTip[partlyFilledTip[e]]] *= (1. + ac * np.pi / 4.)

    dt_CFL = 5 * fluid_properties.viscosity * min(Fr_lstTmStp.mesh.hx, Fr_lstTmStp.mesh.hy) ** 3 / \
             (mat_properties.Eprime * np.max(Fr_lstTmStp.w) ** 3)
    s = ceil(-0.5 + (8 + 16 * timeStep / dt_CFL) ** 0.5 / 2)
    print("s = " + repr(s))
    # s = 60
    delt_wTip = wTip - Fr_lstTmStp.w[EltTip]
    tip_delw_step = delt_wTip / s
    n_channel = len(Fr_lstTmStp.EltChannel)

    EltCrack_k = np.concatenate((Fr_lstTmStp.EltChannel, EltTip))
    corr_nei = find_neighbor_in_crack(EltCrack_k, Fr_lstTmStp.mesh)

    cond_0 = finiteDiff_operator_laminar(Fr_lstTmStp.w,
                                         EltCrack_k,
                                         Fr_lstTmStp.muPrime,
                                         Fr_lstTmStp.mesh,
                                         InCrack,
                                         corr_nei)

    mu_t_1 = 4 / (3 * (s * s + s - 2))
    # pf_0 = Fr_lstTmStp.pFluid[EltCrack_k]
    pf_0 = np.empty(len(EltCrack_k))
    pf_0[:n_channel] = np.dot(C[np.ix_(Fr_lstTmStp.EltChannel, EltCrack_k)], Fr_lstTmStp.w[EltCrack_k]) + mat_properties.SigmaO[Fr_lstTmStp.EltChannel]
    pf_0[n_channel:] = np.linalg.solve(timeStep * mu_t_1 * cond_0[n_channel:, n_channel:-1], tip_delw_step - np.dot(timeStep * mu_t_1 * cond_0[n_channel:, :n_channel], pf_0[:n_channel]))
    # pf_0 = np.dot(C[np.ix_(EltCrack_k, EltCrack_k)], Fr_lstTmStp.w[EltCrack_k])

    M_0 = np.dot(cond_0[:, :-1], pf_0) + Qin[EltCrack_k] / Fr_lstTmStp.mesh.EltArea

    W_0 = Fr_lstTmStp.w[EltCrack_k]

    W_1 = Fr_lstTmStp.w[EltCrack_k] + timeStep * mu_t_1 * M_0

    W_jm1 = np.copy(W_1)
    W_jm2 = np.copy(W_0)
    tau_M0 = timeStep * M_0
    param_pack = (Fr_lstTmStp.muPrime, Fr_lstTmStp.mesh, InCrack, corr_nei)

    for j in range(2, s + 1):

        W_j = RKL_substep(j, s, W_jm1, W_jm2, W_0, EltCrack_k, n_channel, tip_delw_step, param_pack, C, timeStep, tau_M0, Qin[EltCrack_k], Fr_lstTmStp.EltChannel, mat_properties.SigmaO)
        W_jm2 = W_jm1
        W_jm1 = W_j

    w_s = np.zeros(Fr_lstTmStp.mesh.NumberOfElts)
    pf_s = np.zeros(Fr_lstTmStp.mesh.NumberOfElts)

    w_s[EltCrack_k] = W_j
    pf_s[EltCrack_k] = np.dot(C[np.ix_(EltCrack_k, EltCrack_k)], W_j)

    C[np.ix_(EltTip[partlyFilledTip], EltTip[partlyFilledTip])] = C_EltTip

    return w_s, pf_s, None


def RKL_substep(j, s, W_jm1, W_jm2, W_0, crack, n_channel, tip_delw_step, param_pack, C, tau, tau_M0, Qin, EltChannel, sigmaO):

    # pf = np.dot(C[np.ix_(crack, crack)], W_jm1)
    muPrime, Mesh, InCrack, neiInCrack = param_pack
    w_jm1 = np.zeros(Mesh.NumberOfElts)
    w_jm1[crack] = W_jm1
    cond = finiteDiff_operator_laminar(w_jm1, crack, muPrime, Mesh, InCrack, neiInCrack)
    mu_t = 4 * (2 * j - 1) * b[j] / (j * (s * s + s - 2) * b[j - 1])
    gamma_t = -a[j - 1] * mu_t

    # pf = np.dot(C[np.ix_(crack, crack)], W_jm1)


    pf = np.empty(len(crack))
    pf[:n_channel] = np.dot(C[np.ix_(EltChannel, crack)], W_jm1) + sigmaO[EltChannel]

    S = j * tip_delw_step - mu[j] * W_jm1[n_channel:] - nu[j] * W_jm2[n_channel:] + (mu[j] + nu[j]) * W_0[n_channel:] - gamma_t * tau_M0[n_channel:] - mu_t * tau * np.dot(cond[n_channel:, :n_channel], pf[:n_channel])
    A = mu_t * tau * cond[n_channel:, n_channel:-1]
    pf[n_channel:] = np.linalg.solve(A, S)

    M_jm1 = np.dot(cond[:, :-1], pf) + Qin / Mesh.EltArea

    # W_j[:n_channel] = mu[j] * W_jm1[:n_channel] + nu[j] * W_jm2[:n_channel] + (1 - mu[j] - nu[j]) * W_0[:n_channel] + mu_t * tau * M_jm1 + gamma_t * tau_M0
    W_j = mu[j] * W_jm1 + nu[j] * W_jm2 + (1 - mu[j] - nu[j]) * W_0 + mu_t * tau * M_jm1 + gamma_t * tau_M0
    # W_j[n_channel:] = W0_tip + j * tip_delw_step
    # W_j[n_channel:] = s * tip_delw_step

    return W_j


def find_neighbor_in_crack(EltCrack, mesh):
    # The code below finds the indices(in the EltCrack list) of the neighbours of all the cells in the crack.
    # This is done to avoid costly slicing of the large numpy arrays while making the linear system during the
    # fixed point iterations. For neighbors that are outside the fracture, len(EltCrack) + 1 is returned.
    corr_nei = np.full((len(EltCrack), 4), len(EltCrack), dtype=np.int)
    for i, elem in enumerate(EltCrack):
        corresponding = np.where(EltCrack == mesh.NeiElements[elem, 0])[0]
        if len(corresponding) > 0:
            corr_nei[i, 0] = corresponding
        corresponding = np.where(EltCrack == mesh.NeiElements[elem, 1])[0]
        if len(corresponding) > 0:
            corr_nei[i, 1] = corresponding
        corresponding = np.where(EltCrack == mesh.NeiElements[elem, 2])[0]
        if len(corresponding) > 0:
            corr_nei[i, 2] = corresponding
        corresponding = np.where(EltCrack == mesh.NeiElements[elem, 3])[0]
        if len(corresponding) > 0:
            corr_nei[i, 3] = corresponding

    return corr_nei

def tip_edges(EltRibbon, EltTip, mesh):
    corr_tElts = []
    for elt in EltRibbon:
        t_edges = []
        for neighbor in mesh.NeiElements[elt]:
            tip_elt = np.where(neighbor == EltTip)[0]
            if len(tip_elt) > 0:
                t_edges.append(tip_elt[0])
        corr_tElts.append(t_edges)
    return corr_tElts


def solve_width_pressure_RKL2_2(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, EltTip, partlyFilledTip, C,
                         FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode, Vel, corr_ribbon):

    C_EltTip = np.copy(C[np.ix_(EltTip[partlyFilledTip],
                                EltTip[partlyFilledTip])])  # keeping the tip element entries to restore current
    #  tip correction. This is done to avoid copying the full elasticity matrix.

    # filling fraction correction for element in the tip region
    FillF = FillFrac[partlyFilledTip]
    for e in range(0, len(partlyFilledTip)):
        r = FillF[e] - .25
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r
        C[EltTip[partlyFilledTip[e]], EltTip[partlyFilledTip[e]]] *= (1. + ac * np.pi / 4.)


    dt_CFL = 5 * fluid_properties.viscosity * min(Fr_lstTmStp.mesh.hx, Fr_lstTmStp.mesh.hy) ** 3 / \
             (mat_properties.Eprime * np.max(Fr_lstTmStp.w) ** 3)
    s = ceil(-0.5 + (8 + 16 * timeStep / dt_CFL) ** 0.5 / 2)
    # s = 599
    delt_wTip = wTip - Fr_lstTmStp.w[EltTip]
    tip_delw_step = delt_wTip / s
    n_channel = len(Fr_lstTmStp.EltChannel)

    EltCrack_k = np.concatenate((Fr_lstTmStp.EltChannel, EltTip))
    corr_nei = find_neighbor_in_crack(EltCrack_k, Fr_lstTmStp.mesh)

    corr_rbn_crk = np.zeros(len(EltTip), dtype=int)
    for i in range(len(EltTip)):
        corr_rbn_crk[i] = np.where(EltCrack_k == corr_ribbon[i])[0]

    cond_0 = finiteDiff_operator_laminar(Fr_lstTmStp.w,
                                         EltCrack_k,
                                         Fr_lstTmStp.muPrime,
                                         Fr_lstTmStp.mesh,
                                         InCrack,
                                         corr_nei)

    # pf_0 = Fr_lstTmStp.pFluid[EltCrack_k]
    pf_0 = np.dot(C[np.ix_(EltCrack_k, EltCrack_k)], Fr_lstTmStp.w[EltCrack_k])
    M_0 = np.dot(cond_0[:, :-1], pf_0) + Qin[EltCrack_k] / Fr_lstTmStp.mesh.EltArea

    W_0 = Fr_lstTmStp.w[EltCrack_k]
    # W_1 = np.zeros(len(EltCrack_k))
    mu_t_1 = 4 / (3 * (s * s + s - 2))
    W_1 = Fr_lstTmStp.w[EltCrack_k] + timeStep * mu_t_1 * M_0
    # W_1[n_channel:] = Fr_lstTmStp.w[EltTip] + tip_delw_step

    W_1_tip = Fr_lstTmStp.w[EltTip] + tip_delw_step
    dwc = W_1[n_channel:] - W_1_tip
    W_1[corr_rbn_crk] += dwc
    W_1[n_channel:] = W_1_tip

    W_jm1 = np.copy(W_1)
    W_jm2 = np.copy(W_0)
    tau_M0 = timeStep * M_0
    param_pack = (Fr_lstTmStp.muPrime, Fr_lstTmStp.mesh, InCrack, corr_nei)
    sum_mut = copy.copy(mu_t_1)
    sum_gamma = 0

    for j in range(2, s + 1):

        # C_EltTip = np.copy(C[np.ix_(EltTip[partlyFilledTip],
        #                             EltTip[partlyFilledTip])])  # keeping the tip element entries to restore current
        # #  tip correction. This is done to avoid copying the full elasticity matrix.
        #
        # # filling fraction correction for element in the tip region
        # FillF = j / s * FillFrac[partlyFilledTip]
        # for e in range(0, len(partlyFilledTip)):
        #     r = FillF[e] - .25
        #     if r < 0.1:
        #         r = 0.1
        #     ac = (1 - r) / r
        #     C[EltTip[partlyFilledTip[e]], EltTip[partlyFilledTip[e]]] *= (1. + ac * np.pi / 4.)


        W_j, sum_mut, sum_gamma = RKL_substep_2(j, s, W_jm1, W_jm2, W_0, EltCrack_k, n_channel, tip_delw_step, param_pack, C, timeStep, tau_M0, Qin, wTip, sum_mut, sum_gamma, corr_rbn_crk)
        W_jm2 = W_jm1
        W_jm1 = W_j

        # C[np.ix_(EltTip[partlyFilledTip], EltTip[partlyFilledTip])] = C_EltTip

    w_s = np.zeros(Fr_lstTmStp.mesh.NumberOfElts)
    pf_s = np.zeros(Fr_lstTmStp.mesh.NumberOfElts)

    w_s[EltCrack_k] = W_j
    pf_s[EltCrack_k] = np.dot(C[np.ix_(EltCrack_k, EltCrack_k)], W_j)

    C[np.ix_(EltTip[partlyFilledTip], EltTip[partlyFilledTip])] = C_EltTip

    return w_s, pf_s, None


def RKL_substep_2(j, s, W_jm1, W_jm2, W_0, crack, n_channel, tip_delw_step, param_pack, C, tau, tau_M0, Qin, W0_tip, sum_mut, sum_gamma, corr_rbn_crk):

    pf = np.dot(C[np.ix_(crack, crack)], W_jm1)
    muPrime, Mesh, InCrack, neiInCrack = param_pack
    w_jm1 = np.zeros(Mesh.NumberOfElts)
    w_jm1[crack] = W_jm1
    cond = finiteDiff_operator_laminar(w_jm1, crack, muPrime, Mesh, InCrack, neiInCrack)
    M_jm1 = np.dot(cond[:, :-1], pf) + Qin[crack] / Mesh.EltArea
    mu_t = 4 * (2 * j - 1) * b[j] / (j * (s * s + s - 2) * b[j - 1])
    gamma_t = -a[j - 1] * mu_t
    W_j = mu[j] * W_jm1 + nu[j] * W_jm2 + (1 - mu[j] - nu[j]) * W_0 + mu_t * tau * M_jm1 + gamma_t * tau_M0
    # sum_mut += mu_t
    # sum_gamma += gamma_t
    W_j_t = W0_tip + j * tip_delw_step

    dwc = W_j[n_channel:] - W_j_t
    W_j[corr_rbn_crk] += dwc
    W_j[n_channel:] = W_j_t

    # W_j[n_channel:] = s * tip_delw_step

    return W_j, sum_mut, sum_gamma

@profile
def solve_with_RKL2_neg(Eprime, *args):

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    viscosity = muPrime / 12
    dt_CFL = 5 * np.min(viscosity) * min(Mesh.hx, Mesh.hy) ** 3 / \
             (Eprime * np.max(wLastTS) ** 3)
    s = ceil(-0.5 + (8 + 16 * dt / dt_CFL) ** 0.5 / 2)
    print("s = " + repr(s))

    delt_wTip = imposed_val - wLastTS[to_impose]
    tip_delw_step = delt_wTip / s

    # act_tip = np.concatenate((active, to_impose))
    act_tip_val = np.concatenate((wc_to_impose, tip_delw_step))
    n_ch = len(to_solve)
    # n_act = len(active)
    # n_tip = len(imposed_val)
    # n_total = n_ch + n_act + n_tip

    ch_indxs = np.arange(n_ch)

    # act_indxs = n_ch + np.arange(n_act)
    # tip_indxs = n_ch + n_act + np.arange(n_tip)


    cond_0_lil = finiteDiff_operator_laminar(wLastTS,
                                         EltCrack,
                                         muPrime,
                                         Mesh,
                                         InCrack,
                                         neiInCrack,
                                         sparse_flag=True)
    cond_0 = cond_0_lil.tocsr()
    mu_t_1 = 4 / (3 * (s * s + s - 2))
    # pf_0 = Fr_lstTmStp.pFluid[EltCrack]
    pf_0 = np.empty(len(EltCrack))
    pf_0[ch_indxs] = np.dot(C[np.ix_(to_solve, EltCrack)], wLastTS[EltCrack]) + \
                       sigma0[to_solve]
    pf_0[n_ch:] = np.linalg.solve(dt * mu_t_1 * (cond_0[n_ch:, n_ch:-1]).toarray(),
                                       act_tip_val - dt * mu_t_1 * cond_0[n_ch:, :][:, :n_ch].dot(pf_0[:n_ch]))
    # pf_0 = np.dot(C[np.ix_(EltCrack, EltCrack)], wLastTS[EltCrack])

    M_0 = cond_0[:, :-1].dot(pf_0) + Q[EltCrack] / Mesh.EltArea

    W_0 = wLastTS[EltCrack]

    W_1 = wLastTS[EltCrack] + dt * mu_t_1 * M_0

    W_jm1 = np.copy(W_1)
    W_jm2 = np.copy(W_0)
    tau_M0 = dt * M_0
    param_pack = (muPrime, Mesh, InCrack, neiInCrack)
    C_red = np.asfortranarray(C[np.ix_(to_solve, EltCrack)])
    for j in range(2, s + 1):
        print('step = ' + repr(j))
        W_j = RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, EltCrack, n_ch, tip_delw_step, param_pack, C_red, dt,
                          tau_M0, Q[EltCrack], to_solve, sigma0, act_tip_val)
        W_jm2 = W_jm1
        W_jm1 = W_j

    # w_s = np.zeros(Mesh.NumberOfElts)
    # pf_s = np.zeros(Mesh.NumberOfElts)
    #
    # w_s[EltCrack] = W_j
    # pf_s[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], W_j)
    sol = W_j - wLastTS[EltCrack]


    return sol, None

@profile
def RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, crack, n_channel, tip_delw_step, param_pack, C_, tau, tau_M0, Qin, EltChannel,
                sigmaO, imposed_value):
    # pf = np.dot(C[np.ix_(crack, crack)], W_jm1)
    muPrime, Mesh, InCrack, neiInCrack = param_pack
    w_jm1 = np.zeros(Mesh.NumberOfElts)
    cp_W_jm1 = np.copy(W_jm1)
    cp_W_jm1[W_jm1 < 1e-6] = 1e-6
    w_jm1[crack] = cp_W_jm1
    cond_lil = finiteDiff_operator_laminar(w_jm1, crack, muPrime, Mesh, InCrack, neiInCrack, sparse_flag=True)
    cond = cond_lil.tocsr()
    mu_t = 4 * (2 * j - 1) * b[j] / (j * (s * s + s - 2) * b[j - 1])
    gamma_t = -a[j - 1] * mu_t

    # pf = np.dot(C[np.ix_(crack, crack)], W_jm1)

    pf = np.empty(len(crack))
    # pf[:n_channel] = pardot(C_, W_jm1, 4, 2) + sigmaO[EltChannel]
    # pf[:n_channel] = np.dot(C_, W_jm1) + sigmaO[EltChannel]
    pf[:n_channel] = pardot_matrix_vector(C_, W_jm1, 4) + sigmaO[EltChannel]
    imposed_value[-len(tip_delw_step):] = j * tip_delw_step
    S = imposed_value - mu[j] * W_jm1[n_channel:] - nu[j] * W_jm2[n_channel:] + (mu[j] + nu[j]) * W_0[
                            n_channel:] - gamma_t * tau_M0[n_channel:] - mu_t * tau * np.dot(
                            (cond[n_channel:, :][:, :n_channel]).toarray(), pf[:n_channel])

    A = mu_t * tau * (cond[n_channel:, :][:, n_channel:-1]).toarray()
    pf[n_channel:] = np.linalg.solve(A, S)

    M_jm1 = cond[:, :-1].dot(pf) + Qin / Mesh.EltArea


    # W_j[:n_channel] = mu[j] * W_jm1[:n_channel] + nu[j] * W_jm2[:n_channel] + (1 - mu[j] - nu[j]) * W_0[:n_channel] + mu_t * tau * M_jm1 + gamma_t * tau_M0
    W_j = mu[j] * W_jm1 + nu[j] * W_jm2 + (1 - mu[j] - nu[j]) * W_0 + mu_t * tau * M_jm1 + gamma_t * tau_M0
    # W_j[n_channel:] = W0_tip + j * tip_delw_step
    # W_j[n_channel:] = s * tip_delw_step

    return W_j

import threading

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def do_dot(a, b, out):
    # np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    out[:] = np.dot(a, b)  # less efficient because the output is stored in a temporary array?


def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    if len(b.shape) == 1:
        b = b.reshape((b.shape[0], 1))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func,
                                  args=(a_blocks[i, 0, :, :],
                                        b_blocks[0, j, :, :],
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out
@profile
def pardot_matrix_vector(a, b, nblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks
    # print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0]), dtype=a.dtype)

    n_in_block = a.shape[0] // nblocks
    out_blocks = out[:n_in_block * nblocks].reshape((nblocks, n_in_block))
    a_blocks = a[:n_in_block * nblocks, :].reshape((nblocks, n_in_block, a.shape[1]))

    threads = []
    for i in range(nblocks):
        th = threading.Thread(target=dot_func,
                              args=(a_blocks[i, :, :],
                                    b,
                                    out_blocks[i, :]))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    out[n_in_block * nblocks:] = np.dot(a[n_in_block * nblocks:, :], b)

    return out
