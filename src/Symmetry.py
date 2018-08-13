# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri August 09 16:30:21 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.
"""
import numpy as np

def get_symetric_elements(mesh, elements):

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
    elements = np.intersect1d(np.where(mesh.CenterCoor[:, 0] > mesh.hx / 2)[0],
                              np.where(mesh.CenterCoor[:, 1] > mesh.hy / 2)[0])

    boundary_x = np.intersect1d(np.where(abs(mesh.CenterCoor[:, 1]) < 1e-12)[0],
                                np.where(mesh.CenterCoor[:, 0] > mesh.hx / 2)[0])
    boundary_y = np.intersect1d(np.where(abs(mesh.CenterCoor[:, 0]) < 1e-12)[0],
                                np.where(mesh.CenterCoor[:, 1] > mesh.hy / 2)[0])

    all_elts = np.concatenate((elements, boundary_x))
    all_elts = np.concatenate((all_elts, boundary_y))
    all_elts = np.concatenate((all_elts, mesh.CenterElts))
    return all_elts, elements, boundary_x, boundary_y


def corresponding_elements_in_symmetric(mesh):

    correspondence = np.empty((mesh.NumberOfElts, ), dtype=int)
    all_elmnts, elements, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

    sym_elts = get_symetric_elements(mesh, elements)
    for i in range(len(elements)):
        correspondence[sym_elts[i]] = i

    sym_bound_x = get_symetric_elements(mesh, boundary_x)
    for i in range(len(boundary_x)):
        correspondence[sym_bound_x[i]] = i + len(elements)

    sym_bound_y = get_symetric_elements(mesh, boundary_y)
    for i in range(len(boundary_y)):
        correspondence[sym_bound_y[i]] = i + len(elements) + len(boundary_x)

    correspondence[mesh.CenterElts[0]] = len(elements) + len(boundary_x) + len(boundary_y)

    return correspondence





def elasticity_matrix_symmetric(C, mesh):

    all_elmnts, elements, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

    no_elements = len(elements) + len(boundary_x) + len(boundary_y) + 1
    C_sym = np.zeros((no_elements, no_elements), dtype=np.float32)

    indx_boun_x = len(elements)
    indx_boun_y = indx_boun_x + len(boundary_x)
    indx_cntr_elm = indx_boun_y + len(boundary_y)

    sym_elements = get_symetric_elements(mesh, elements)
    sym_elem_xbound = get_symetric_elements(mesh, boundary_x)
    sym_elem_ybound = get_symetric_elements(mesh, boundary_y)


    # influence on elements
    for i in range(len(elements)):
        C_sym[i, 0: indx_boun_x] = C[elements[i], sym_elements[:, 0]] + \
                                   C[elements[i], sym_elements[:, 1]] + \
                                   C[elements[i], sym_elements[:, 2]] + \
                                   C[elements[i], sym_elements[:, 3]]

    for i in range(len(elements)):
        C_sym[i, indx_boun_x: indx_boun_y] = C[elements[i], sym_elem_xbound[:, 0]] + \
                                             C[elements[i], sym_elem_xbound[:, 3]]

    for i in range(len(elements)):
        C_sym[i, indx_boun_y: indx_cntr_elm] = C[elements[i], sym_elem_ybound[:, 0]] + \
                                               C[elements[i], sym_elem_ybound[:, 1]]

    C_sym[0:indx_boun_x, -1] = C[elements, mesh.CenterElts[0]]


    # influence on x boundary elements
    for i in range(len(boundary_x)):
        C_sym[i + indx_boun_x, 0: indx_boun_x] = C[boundary_x[i], sym_elements[:, 0]] + \
                                                 C[boundary_x[i], sym_elements[:, 1]] + \
                                                 C[boundary_x[i], sym_elements[:, 2]] + \
                                                 C[boundary_x[i], sym_elements[:, 3]]

    for i in range(len(boundary_x)):
        C_sym[i + indx_boun_x, indx_boun_x: indx_boun_y] = C[boundary_x[i], sym_elem_xbound[:, 0]] + \
                                                           C[boundary_x[i], sym_elem_xbound[:, 3]]

    for i in range(len(boundary_x)):
        C_sym[i + indx_boun_x, indx_boun_y: indx_cntr_elm] = C[boundary_x[i], sym_elem_ybound[:, 0]] + \
                                                             C[boundary_x[i], sym_elem_ybound[:, 1]]

    C_sym[indx_boun_x: indx_boun_y, -1] = C[boundary_x, mesh.CenterElts[0]]


    # influence on y boundary elements
    for i in range(len(boundary_y)):
        C_sym[i + indx_boun_y, 0: indx_boun_x] = C[boundary_y[i], sym_elements[:, 0]] + \
                                                 C[boundary_y[i], sym_elements[:, 1]] + \
                                                 C[boundary_y[i], sym_elements[:, 2]] + \
                                                 C[boundary_y[i], sym_elements[:, 3]]

    for i in range(len(boundary_y)):
        C_sym[i + indx_boun_y, indx_boun_x: indx_boun_y] = C[boundary_y[i], sym_elem_xbound[:, 0]] + \
                                                           C[boundary_y[i], sym_elem_xbound[:, 3]]

    for i in range(len(boundary_y)):
        C_sym[i + indx_boun_y, indx_boun_y: indx_cntr_elm] = C[boundary_y[i], sym_elem_ybound[:, 0]] + \
                                                             C[boundary_y[i], sym_elem_ybound[:, 1]]

    C_sym[indx_boun_y: indx_cntr_elm, -1] = C[boundary_y, mesh.CenterElts[0]]


    #influence on center element
    C_sym[-1, 0: len(elements)] = C[mesh.CenterElts[0], sym_elements[:, 0]] + \
                                  C[mesh.CenterElts[0], sym_elements[:, 1]] + \
                                  C[mesh.CenterElts[0], sym_elements[:, 2]] + \
                                  C[mesh.CenterElts[0], sym_elements[:, 3]]

    C_sym[-1, indx_boun_x: indx_boun_y] = C[mesh.CenterElts[0], sym_elem_xbound[:, 0]] + \
                                          C[mesh.CenterElts[0], sym_elem_xbound[:, 3]]

    C_sym[-1, indx_boun_y: indx_cntr_elm] = C[mesh.CenterElts[0], sym_elem_ybound[:, 0]] + \
                                            C[mesh.CenterElts[0], sym_elem_ybound[:, 1]]

    C_sym[-1, -1] = C[mesh.CenterElts[0], mesh.CenterElts[0]]

    return C_sym