# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Sep 6 16:53:19 2019.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020. All rights
reserved. See the LICENSE.TXT file for more details.
"""

from elastohydrodynamic_solver import finiteDiff_operator_laminar, FiniteDiff_operator_turbulent_implicit, Gravity_term
import numpy as np
import logging
from math import ceil
import sys
from properties import instrument_start, instrument_close

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

# @profile
def solve_width_pressure_RKL2(Eprime, GPU, n_threads, perf_node, *args):
    log = logging.getLogger('PyFrac.solve_width_pressure_RKL2')
    perfNode_RKL = instrument_start("linear system solve", perf_node)

    (to_solve, to_impose, wLastTS, pfLastTS, imposed_val, EltCrack, Mesh, dt, Q, C, muPrime, rho, InCrack, LeakOff,
     sigma0, turb, dgrain, gravity, active, wc_to_impose, wc, cf, neiInCrack) = args

    viscosity = muPrime / 12
    dt_CFL = 5 * np.min(viscosity) * min(Mesh.hx, Mesh.hy) ** 3 / (Eprime * np.max(wLastTS) ** 3)
    s = ceil(-0.5 + (8 + 16 * dt / dt_CFL) ** 0.5 / 2)
    log.info("no. of sub-steps = " + repr(s))

    delt_wTip = imposed_val - wLastTS[to_impose]
    tip_delw_step = delt_wTip / s

    act_tip_val = np.concatenate((wc_to_impose, tip_delw_step))
    n_ch = len(to_solve)
    ch_indxs = np.arange(n_ch)

    if turb:
        raise SystemExit("RKL scheme with turbulence is not yet implemented!")
    else:
        cond_0_lil = finiteDiff_operator_laminar(wLastTS,
                                             EltCrack,
                                             muPrime,
                                             Mesh,
                                             InCrack,
                                             neiInCrack,
                                             sparse_flag=True)
    cond_0 = cond_0_lil.tocsr()
    mu_t_1 = 4 / (3 * (s * s + s - 2))

    if gravity:
        w_0 = np.zeros(Mesh.NumberOfElts)
        w_0[EltCrack] = wLastTS[EltCrack]
        G = Gravity_term(w_0,
                         EltCrack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)[EltCrack]

    else:
        G = np.zeros((len(EltCrack),))

    if GPU:
        if 'cupy' not in sys.modules:
            import cupy as cp
        C_red = cp.asarray(C[np.ix_(to_solve, EltCrack)])
    else:
        C_red = C[np.ix_(to_solve, EltCrack)]

    Lk_rate = LeakOff / dt
    W_0 = wLastTS[EltCrack]
    pf_0 = np.empty(len(EltCrack))
    pf_0[ch_indxs] = np.dot(C[np.ix_(to_solve, EltCrack)], wLastTS[EltCrack]) + sigma0[to_solve]
    pf_0[n_ch:] = np.linalg.solve(dt * mu_t_1 * (cond_0[n_ch:, n_ch:-1]).toarray(),
                                    act_tip_val - dt * mu_t_1 * (cond_0[n_ch:, :][:, :n_ch].dot(pf_0[:n_ch]) +
                                    G[n_ch:] + (Q[EltCrack[n_ch:]] - Lk_rate[EltCrack[n_ch:]]) / Mesh.EltArea))

    M_0 = cond_0[:, :-1].dot(pf_0) + (Q[EltCrack] - Lk_rate[EltCrack]) / Mesh.EltArea + G
    W_1 = wLastTS[EltCrack] + dt * mu_t_1 * M_0
    W_jm1 = np.copy(W_1)
    W_jm2 = np.copy(W_0)
    tau_M0 = dt * M_0
    param_pack = (muPrime, Mesh, InCrack, neiInCrack)

    Lk_rate_cr = Lk_rate[EltCrack]
    Q_ = Q[EltCrack]
    for j in range(2, s + 1):
        # todo: need to rethink how to pass arguments. Its a mess now
        W_j = RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, EltCrack, n_ch,
                              tip_delw_step, param_pack, C_red, dt,
                              tau_M0, Q_, to_solve, sigma0, act_tip_val,
                              gravity, rho, Lk_rate_cr, GPU, n_threads,
                              turb)
        W_jm2 = W_jm1
        W_jm1 = W_j

    sol = W_j - wLastTS[EltCrack]

    if perf_node is not None:
        instrument_close(perf_node, perfNode_RKL, None, len(W_j), True, False, None)
        perfNode_RKL.iterations = s
        perf_node.RKL_data.append(perfNode_RKL)

    return sol, s

# @profile
#todo: this function is a mess in terms of arguments. Need to think how to pass them. The idea originally was to
# pass all agruments in the form of arrays and not as objects to keep the port to C++ easy.
def RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, crack, n_channel, tip_delw_step, param_pack, C, tau, tau_M0, Qin,
                    EltChannel, sigmaO, imposed_value, gravity, rho, LeakOff, GPU_flag, n_threads, turb):

    muPrime, Mesh, InCrack, neiInCrack = param_pack
    w_jm1 = np.zeros(Mesh.NumberOfElts)
    cp_W_jm1 = np.copy(W_jm1)
    cp_W_jm1[W_jm1 < 1e-6] = 1e-6
    w_jm1[crack] = cp_W_jm1

    if turb:
        raise SystemExit("RKL scheme with turbulence is not yet implemented!")
    else:
        cond_lil = finiteDiff_operator_laminar(w_jm1,
                                               crack,
                                               muPrime,
                                               Mesh,
                                               InCrack,
                                               neiInCrack,
                                               sparse_flag=True)

    cond = cond_lil.tocsr()
    mu_t = 4 * (2 * j - 1) * b[j] / (j * (s * s + s - 2) * b[j - 1])
    gamma_t = -a[j - 1] * mu_t
    if gravity:
        G = Gravity_term(w_jm1,
                         crack,
                         muPrime,
                         Mesh,
                         InCrack,
                         rho)[crack]
    else:
        G = np.zeros((len(crack),))

    pf = np.empty(len(crack))

    if GPU_flag:
        W_jm1_cp = cp.asarray(W_jm1)
        pn = cp.dot(C, W_jm1_cp)
        pf[:n_channel] = cp.asnumpy(pn) + sigmaO[EltChannel]
    else:
        pf[:n_channel] = pardot_matrix_vector(C, W_jm1, n_threads) + sigmaO[EltChannel]

    imposed_value[-len(tip_delw_step):] = j * tip_delw_step
    M_jm1_tip = np.dot((cond[n_channel:, :][:, :n_channel]).toarray(),
                       pf[:n_channel]) + G[n_channel:] + (Qin[n_channel:] - LeakOff[n_channel:])/Mesh.EltArea
    S = imposed_value - mu[j] * W_jm1[n_channel:] - nu[j] * W_jm2[n_channel:] + (mu[j] + nu[j]) * W_0[
                            n_channel:] - gamma_t * tau_M0[n_channel:] - mu_t * tau * M_jm1_tip
    A = mu_t * tau * (cond[n_channel:, :][:, n_channel:-1]).toarray()
    pf[n_channel:] = np.linalg.solve(A, S)

    M_jm1 = cond[:, :-1].dot(pf) + G + (Qin - LeakOff) / Mesh.EltArea

    W_j = mu[j] * W_jm1 + nu[j] * W_jm2 + (1 - mu[j] - nu[j]) * W_0 + mu_t * tau * M_jm1 + gamma_t * tau_M0

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
    log = logging.getLogger('PyFrac.pardot')
    n_jobs = nblocks * mblocks
    log.info('running {} jobs in parallel'.format(n_jobs))

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
# @profile
def pardot_matrix_vector(a, b, nblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks
    # log=logging.getLogger('PyFrac.pardot_matrix_vector')
    # log.info('running {} jobs in parallel'.format(n_jobs))

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
