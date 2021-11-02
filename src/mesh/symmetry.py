# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri August 09 16:30:21 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
import logging

def get_symetric_elements(mesh, elements):
    """ This function gives the four symmetric elements in each of the quadrant for the given element list."""

    symetric_elts = np.empty((len(elements), 4), dtype=int)
    for i in range(len(elements)):
        i_x = elements[i] % mesh.nx
        i_y = elements[i] // mesh.nx

        symetric_x = mesh.nx - i_x - 1
        symetric_y = mesh.ny - i_y - 1

        symetric_elts[i] = np.asarray([i_y * mesh.nx + i_x,
                                      symetric_y * mesh.nx + i_x,
                                      i_y * mesh.nx + symetric_x,
                                      symetric_y * mesh.nx + symetric_x])

    return symetric_elts

# -----------------------------------------------------------------------------------------------------------------------


def get_active_symmetric_elements(mesh):
    """
    This functions gives the elements in the first quadrant, including the elements intersecting the x and y
    axes lines.
    """

    # elements in the quadrant with positive x and y coordinates
    pos_qdrnt = np.intersect1d(np.where(mesh.CenterCoor[:, 0] > mesh.hx / 2)[0],
                              np.where(mesh.CenterCoor[:, 1] > mesh.hy / 2)[0])

    boundary_x = np.intersect1d(np.where(abs(mesh.CenterCoor[:, 1]) < 1e-12)[0],
                                np.where(mesh.CenterCoor[:, 0] > mesh.hx / 2)[0])
    boundary_y = np.intersect1d(np.where(abs(mesh.CenterCoor[:, 0]) < 1e-12)[0],
                                np.where(mesh.CenterCoor[:, 1] > mesh.hy / 2)[0])

    all_elts = np.concatenate((pos_qdrnt, boundary_x))
    all_elts = np.concatenate((all_elts, boundary_y))
    all_elts = np.concatenate((all_elts, mesh.CenterElts))

    return all_elts, pos_qdrnt, boundary_x, boundary_y

# ----------------------------------------------------------------------------------------------------------------------


def corresponding_elements_in_symmetric(mesh):
    """
    This function returns the corresponding elements in symmetric fracture.
    """

    correspondence = np.empty((mesh.NumberOfElts, ), dtype=int)
    all_elmnts, pos_qdrnt, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

    sym_elts = get_symetric_elements(mesh, pos_qdrnt)
    for i in range(len(pos_qdrnt)):
        correspondence[sym_elts[i]] = i

    sym_bound_x = get_symetric_elements(mesh, boundary_x)
    for i in range(len(boundary_x)):
        correspondence[sym_bound_x[i]] = i + len(pos_qdrnt)

    sym_bound_y = get_symetric_elements(mesh, boundary_y)
    for i in range(len(boundary_y)):
        correspondence[sym_bound_y[i]] = i + len(pos_qdrnt) + len(boundary_x)

    correspondence[mesh.CenterElts[0]] = len(pos_qdrnt) + len(boundary_x) + len(boundary_y)

    return correspondence

# ----------------------------------------------------------------------------------------------------------------------