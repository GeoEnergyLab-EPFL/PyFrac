# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from numba import njit, prange
from utilities.utility import get_distance_components


def load_isotropic_elasticity_matrix(Mesh, Ep, C_precision=np.float32):
    """
    Evaluate the elasticity matrix for the whole mesh.
    Arguments:
        Mesh (object CartesianMesh):    -- a mesh object describing the domain.
        Ep (float):                     -- plain strain modulus.
    Returns:
        ndarray-float:                  -- the elasticity matrix.
    """

    """
    a and b are the half breadth and height of a cell
     ___________________________________
    |           |           |           |
    |           |           |           |
    |     .     |     .     |     .     |
    |           |           |           |
    |___________|___________|___________|
    |           |     ^     |           |
    |           |   b |     |           |
    |     .     |     .<--->|     .     |
    |           |        a  |           |
    |___________|___________|___________|
    |           |           |           |
    |           |           |           |
    |     .     |     .     |     .     |
    |           |           |           |
    |___________|___________|___________|

    """

    a = Mesh.hx / 2.
    b = Mesh.hy / 2.
    Ne = Mesh.NumberOfElts
    const = (Ep / (8. * np.pi))

    C = np.empty([Ne, Ne], dtype=C_precision)

    for i in range(0, Ne):
        x = Mesh.CenterCoor[i, 0] - Mesh.CenterCoor[:, 0]
        y = Mesh.CenterCoor[i, 1] - Mesh.CenterCoor[:, 1]

        apx = a + x
        amx = a - x
        bmy = b - y
        bpy = b + y

        SQ_apx = np.square(apx)
        SQ_amx = np.square(amx)
        SQ_bpy = np.square(bpy)
        SQ_bmy = np.square(bmy)

        C[i] = const * (
                np.sqrt(SQ_amx + SQ_bmy) / (amx * bmy) +
                np.sqrt(SQ_apx + SQ_bmy) / (apx * bmy) +
                np.sqrt(SQ_amx + SQ_bpy) / (amx * bpy) +
                np.sqrt(SQ_apx + SQ_bpy) / (apx * bpy))
    return C


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def get_toeplitzCoe_isotropic(nx, ny, hx, hy, matprop, C_precision):
    """
    Let us make some definitions:
    cartesian mesh             := a structured rectangular mesh of (nx,ny) cells of rectaungular shape

                                        |<------------nx----------->|
                                    _    ___ ___ ___ ___ ___ ___ ___
                                    |   | . | . | . | . | . | . | . |
                                    |   |___|___|___|___|___|___|___|
                                    ny  | . | . | . | . | . | . | . |
                                    |   |___|___|___|___|___|___|___|   y
                                    |   | . | . | . | . | . | . | . |   |
                                    -   |___|___|___|___|___|___|___|   |____x

                                   the cell centers are marked by .

    set of unique distances    := given a set of cells in a cartesian mesh, consider the set of unique distances
                                  between any pair of cell centers.
    set of unique coefficients := given a set of unique distances then consider the interaction coefficients
                                  obtained from them

    C_toeplitz_coe             := An array of size (nx*ny), populated with the unique coefficients.

    Matematically speaking:
    for i in (0,ny) and j in (0,nx) take the set of combinations (i,j) such that [i^2 y^2 + j^2 x^2]^1/2 is unique
    """
    C_toeplitz_coe = np.empty(ny * nx, dtype=C_precision)
    xindrange = np.arange(nx)
    xrange = xindrange * hx

    Ep = matprop[0]
    const = (Ep / (8. * np.pi))

    a = hx / 2.
    b = hy / 2.

    for i in prange(ny):
        y = i * hy
        amx = a - xrange
        apx = a + xrange
        bmy = b - y
        bpy = b + y
        SQ_amx = np.square(amx)
        SQ_bmy = np.square(bmy)
        SQ_apx = np.square(apx)
        SQ_bpy = np.square(bpy)

        C_toeplitz_coe[i * nx:(i + 1) * nx] = const * (np.sqrt(SQ_amx + SQ_bmy)   / (amx * bmy)
                                                       + np.sqrt(SQ_apx + SQ_bmy) / (apx * bmy)
                                                       + np.sqrt(SQ_amx + SQ_bpy) / (amx * bpy)
                                                       + np.sqrt(SQ_apx + SQ_bpy) / (apx * bpy))
    return C_toeplitz_coe[0], C_toeplitz_coe


def get_isotropic_el_self_eff(hx, hy, Ep):
    """
    Evaluate the self effect term (diagonal value) for isotropic elasticity.
    Arguments:
        hx (float):                     -- x size of a mesh cell
        hy (float):                     -- y size of a mesh cell
        Ep (float):                     -- plain strain modulus.
    Returns:
        ndarray-float:                  -- the diagonal term.
    """

    a = hx / 2.  # Lx/nx-1
    b = hy / 2.  # Ly/ny-1
    bb = b * b
    aa = a * a
    sqrt_aa_p_bb = np.sqrt(aa + bb) / (a * b)
    return sqrt_aa_p_bb * Ep / (2. * np.pi)

@njit(nogil=True, cache=True, parallel = True)
def get_R0_normal_traction_at(xy_obs, xy_crack, w_crack, Ep, hx, hy):
    """

    :param xy_obs: array of coordinates x, y of the points where the stress is needed
    :param xy_crack: array of coordinates x, y of the points where w is known
    :param w: crack opening
    :param Ep: plane strain Young's modulus
    :param hx: mesh size in x direction
    :param hy: mesh size in x direction
    :return: traction at the xy_obs points
    """
    const = (Ep / (8. * np.pi))
    n_xy_obs = xy_obs.shape[0]
    n_xy_crack = xy_crack.shape[0]
    normal_trac = np.zeros((n_xy_obs, n_xy_crack))
    total_comp = n_xy_obs * n_xy_crack
    a = hx / 2.
    b = hy / 2.

    for global_ind in prange(total_comp):
        ind_obs = global_ind // n_xy_crack
        ind_crack = global_ind - ind_obs * n_xy_crack
        xy_obs_i = xy_obs[ind_obs,:]
        xy_crack_i = xy_crack[ind_crack,:]
        dx_i, dy_i = get_distance_components(xy_obs_i[0], xy_obs_i[1], xy_crack_i[0], xy_crack_i[1])

        amx = a - dx_i
        apx = a + dx_i
        bmy = b - dy_i
        bpy = b + dy_i
        SQ_amx = np.square(amx)
        SQ_bmy = np.square(bmy)
        SQ_apx = np.square(apx)
        SQ_bpy = np.square(bpy)

        normal_trac[ind_obs,ind_crack] = w_crack[ind_crack] * const * (np.sqrt(SQ_amx + SQ_bmy) / (amx * bmy)
                                                       + np.sqrt(SQ_apx + SQ_bmy) / (apx * bmy)
                                                       + np.sqrt(SQ_amx + SQ_bpy) / (amx * bpy)
                                                       + np.sqrt(SQ_apx + SQ_bpy) / (apx * bpy))
    return np.sum(normal_trac, axis=1)