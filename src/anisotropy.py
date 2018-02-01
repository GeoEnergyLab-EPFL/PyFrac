#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details.
#


import numpy as np

from src.LevelSet import reconstruct_front
from src.VolIntegral import Integral_over_cell

def toughness_at_tip_CellCenter(ribbon_elts, channel_elts, mesh, mat_prop, sgnd_dist):
    """
    This function gives the scaled toughness(Kprime) at the closest tip point from the cell centers of the ribbon cells.
    The function is different from the toughness_at_tip as it calculates the closest tip from cell centers and not from
    the zero vertex.
    Arguments:
        ribbon_elts (ndarray-int): list of ribbon elements
        mesh (CartesianMesh object): The cartesian mesh object
        mat_prop (MaterialProperties object):    Material properties:
        sgnd_dist (ndarray-float): level set data

    Returns:
        ndarray-float : Kprime at the closest tip point from the center of the given ribbon cells
    """

    # dist = -sgnd_dist
    # alpha = np.zeros((ribbon_elts.size,), dtype=np.float64)

    neighbors = mesh.NeiElements[ribbon_elts]
    (elt_tip, l_tip, alpha_tip, CellStatus) = reconstruct_front(sgnd_dist,
                                                                channel_elts,
                                                                mesh)
    FillFrac = Integral_over_cell(elt_tip,
                                  alpha_tip,
                                  l_tip,
                                  mesh,
                                  'A') / mesh.EltArea

    partly_filled = np.where(FillFrac < 0.999999)[0]

    elt_tip = elt_tip[partly_filled]
    l_tip = l_tip[partly_filled]
    alpha_tip = alpha_tip[partly_filled]

    to_delete = np.array([], dtype=int)
    zero_alpha = np.where(alpha_tip == 0.)[0]
    for i in zero_alpha:
        lftneigb_in_zero = np.where(elt_tip[zero_alpha] == mesh.NeiElements[elt_tip[i], 0])[0]
        if lftneigb_in_zero.size > 0:
            if l_tip[zero_alpha[lftneigb_in_zero]] > l_tip[i]:
                to_delete = np.append(to_delete, zero_alpha[lftneigb_in_zero])
            else:
                to_delete = np.append(to_delete, i)
        rgtneigb_in_zero = np.where(elt_tip[zero_alpha] == mesh.NeiElements[elt_tip[i], 1])[0]
        if rgtneigb_in_zero.size > 0:
            if l_tip[zero_alpha[rgtneigb_in_zero]] > l_tip[i]:
                to_delete = np.append(to_delete, zero_alpha[rgtneigb_in_zero])
            else:
                to_delete = np.append(to_delete, i)

    ninety_alpha = np.where(alpha_tip == np.pi / 2)[0]
    for i in ninety_alpha:
        btmneigb_in_ninety = np.where(elt_tip[ninety_alpha] == mesh.NeiElements[elt_tip[i], 2])[0]
        if btmneigb_in_ninety.size > 0:
            if l_tip[ninety_alpha[btmneigb_in_ninety]] > l_tip[i]:
                to_delete = np.append(to_delete, ninety_alpha[btmneigb_in_ninety])
            else:
                to_delete = np.append(to_delete, i)
        topneigb_in_ninety = np.where(elt_tip[ninety_alpha] == mesh.NeiElements[elt_tip[i], 3])[0]
        if topneigb_in_ninety.size > 0:
            if l_tip[ninety_alpha[topneigb_in_ninety]] > l_tip[i]:
                to_delete = np.append(to_delete, ninety_alpha[topneigb_in_ninety])
            else:
                to_delete = np.append(to_delete, i)
    to_delete = np.unique(to_delete)

    elt_tip = np.delete(elt_tip, to_delete)
    l_tip = np.delete(l_tip, to_delete)
    alpha_tip = np.delete(alpha_tip, to_delete)

    if np.isnan(alpha_tip).any():
        is_nan = np.where(np.isnan(alpha_tip))[0]
        for i in is_nan:
            enclosing = mesh.NeiElements[elt_tip[i]]
            enclosing = np.array([enclosing[0],
                                  enclosing[2] - 1,
                                  enclosing[3] - 1,
                                  enclosing[2],
                                  enclosing[3],
                                  enclosing[2] + 1,
                                  enclosing[3] + 1,
                                  enclosing[1]])
            for j in range(5):
                lft_in_tip = np.where(elt_tip == enclosing[j])[0]
                if lft_in_tip.size > 0 and not np.isnan(alpha_tip[lft_in_tip]):
                    break
            for j in range(7, 2, -1):
                rgt_in_tip = np.where(elt_tip == enclosing[j])[0]
                if rgt_in_tip.size > 0 and not np.isnan(alpha_tip[rgt_in_tip]):
                    break
            if rgt_in_tip.size > 1 or lft_in_tip.size > 1:
                print("found double")
            alpha_tip[i] = (alpha_tip[rgt_in_tip] + alpha_tip[lft_in_tip]) / 2
            print("corrected alpha tip")

    zero_vertex_tip = find_zero_vertex(elt_tip, sgnd_dist, mesh)
    smthed_tip, a, b, c, pnt_lft, pnt_rgt, neig_lft, neig_rgt = construct_polygon(elt_tip,
                                                                                  l_tip,
                                                                                  alpha_tip,
                                                                                  mesh,
                                                                                  zero_vertex_tip)
    if np.isnan(smthed_tip).any():
        return np.nan
    zr_vrtx_smthed_tip = find_zero_vertex(smthed_tip, sgnd_dist, mesh)
    alpha = find_projection(ribbon_elts,
                            smthed_tip,
                            zr_vrtx_smthed_tip,
                            a,
                            b,
                            c,
                            pnt_lft[:, 0],
                            pnt_lft[:, 1],
                            pnt_rgt[:, 0],
                            pnt_rgt[:, 1],
                            neig_lft,
                            neig_rgt,
                            mesh)

    if mat_prop.anisotropic:
        return mat_prop.KprimeFunc(alpha)
    else:
        dist = -sgnd_dist
        x = np.zeros((len(ribbon_elts),), )
        y = np.zeros((len(ribbon_elts),), )

        # evaluating the closest tip points
        for i in range(0, len(ribbon_elts)):
            if zero_vertex[i] == 0:

                x[i] = mesh.CenterCoor[ribbon_elts[i], 0] + dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i], 1] + dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 1:

                x[i] = mesh.CenterCoor[ribbon_elts[i], 0] - dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i], 1] + dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 2:

                x[i] = mesh.CenterCoor[ribbon_elts[i], 0] - dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i], 1] - dist[ribbon_elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 3:

                x[i] = mesh.CenterCoor[ribbon_elts[i], 0] + dist[ribbon_elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[ribbon_elts[i], 1] - dist[ribbon_elts[i]] * np.sin(alpha[i])

            if abs(dist[mesh.NeiElements[ribbon_elts[i], 0]] / dist[mesh.NeiElements[ribbon_elts[i], 1]] - 1) < 1e-7:
                if sgnd_dist[neighbors[i, 2]] < sgnd_dist[neighbors[i, 3]]:
                    x[i] = mesh.CenterCoor[ribbon_elts[i], 0]
                    y[i] = mesh.CenterCoor[ribbon_elts[i], 1] + dist[ribbon_elts[i]]
                elif sgnd_dist[neighbors[i, 2]] > sgnd_dist[neighbors[i, 3]]:
                    x[i] = mesh.CenterCoor[ribbon_elts[i], 0]
                    y[i] = mesh.CenterCoor[ribbon_elts[i], 1] - dist[ribbon_elts[i]]

        # returning the Kprime according to the given function
        return mat_prop.KprimeFunc(x, y)


# -----------------------------------------------------------------------------------------------------------------------
def find_zero_vertex(Elts, level_set, mesh):
    zero_vertex = np.zeros((len(Elts),), dtype=int)
    for i in range(0, len(Elts)):
        neighbors = mesh.NeiElements[Elts]

        if level_set[neighbors[i, 0]] <= level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] <= level_set[
            neighbors[i, 3]]:
            zero_vertex[i] = 0
        elif level_set[neighbors[i, 0]] > level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] <= level_set[
            neighbors[i, 3]]:
            zero_vertex[i] = 1
        elif level_set[neighbors[i, 0]] > level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] > level_set[
            neighbors[i, 3]]:
            zero_vertex[i] = 2
        elif level_set[neighbors[i, 0]] <= level_set[neighbors[i, 1]] and level_set[neighbors[i, 2]] > level_set[
            neighbors[i, 3]]:
            zero_vertex[i] = 3

    return zero_vertex


def toughness_at_tip_zeroVertex(elts, mesh, mat_prop, alpha, l, zero_vrtx):
    if mat_prop.anisotropic:
        return mat_prop.KprimeFunc(alpha)
    else:
        x = np.zeros((len(elts),), )
        y = np.zeros((len(elts),), )
        for i in range(0, len(elts)):
            if zero_vrtx[i] == 0:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 0], 0] + l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 0], 1] + l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 1:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 0] - l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 1] + l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 2:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 0] - l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 1] - l[i] * np.sin(alpha[i])
            elif zero_vrtx[i] == 3:
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 0] + l[i] * np.cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 1] - l[i] * np.sin(alpha[i])

        return mat_prop.KprimeFunc(x, y)


def find_projection(elt_ribbon, elt_tip, zr_vrtx_tip, a_tip, b_tip, c_tip, x_lft, y_lft, x_rgt, y_rgt, neig_lft,
                    neig_rgt, mesh):
    closest_tip_cell = np.zeros((len(elt_ribbon),), dtype=np.int)
    dist_ribbon = np.zeros((len(elt_ribbon),), dtype=np.float64)
    alpha = np.zeros((len(elt_ribbon),), dtype=np.float64)
    for i in range(len(elt_ribbon)):
        dist_front_line = np.zeros((len(elt_tip),), dtype=np.float64)
        point_at_grid_line = np.zeros((len(elt_tip),), dtype=np.uint8)

        for j in range(len(elt_tip)):
            if x_rgt[j] - x_lft[j] == 0:
                xx = mesh.CenterCoor[elt_ribbon[i], 0]
                yy = - c_tip[j]
            else:
                slope_tip_line = (y_rgt[j] - y_lft[j]) / (x_rgt[j] - x_lft[j])
                m = -1. / slope_tip_line

                intrcpt = mesh.CenterCoor[elt_ribbon[i], 1] - m * mesh.CenterCoor[elt_ribbon[i], 0]
                xx = -(intrcpt + c_tip[j]) / (a_tip[j] + m)
                yy = m * xx + intrcpt

            if x_lft[j] > xx or x_rgt[j] < xx or min(y_lft[j], y_rgt[j]) > yy or max(
                    y_lft[j], y_rgt[j]) < yy:
                dist_lft_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_lft[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_lft[j]) ** 2) ** 0.5
                dist_rgt_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_rgt[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_rgt[j]) ** 2) ** 0.5

                dist_front_line[j] = min(dist_lft_pnt, dist_rgt_pnt)
                if dist_lft_pnt < dist_rgt_pnt:
                    point_at_grid_line[j] = 1
                else:
                    point_at_grid_line[j] = 2
            else:
                dist_front_line[j] = abs(
                    mesh.CenterCoor[elt_ribbon[i], 0] * a_tip[j] + mesh.CenterCoor[elt_ribbon[i], 1] + c_tip[j]) / (
                                                                                        a_tip[j] ** 2 + 1) ** 0.5

        closest_tip_cell[i] = np.argmin(dist_front_line)
        if point_at_grid_line[closest_tip_cell[i]] == 0:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha[i] = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
        elif point_at_grid_line[closest_tip_cell[i]] == 1:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_lft[closest_tip_cell[i]]]) / a_tip[neig_lft[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2
        elif point_at_grid_line[closest_tip_cell[i]] == 2:
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_rgt[closest_tip_cell[i]]]) / a_tip[neig_rgt[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2

        dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]

    # zero_angle = np.where(abs(1. - x_lft / x_rgt) < 1e-6)[0]
    zero_angle = np.where(x_lft == x_rgt)[0]
    for i in range(len(zero_angle)):
        if zr_vrtx_tip[zero_angle[i]] == 0 or zr_vrtx_tip[zero_angle[i]] == 3:
            for j in range(3):
                left_in_ribbon = np.where(elt_ribbon == elt_tip[zero_angle[i]] - (j + 1))[0]
                if left_in_ribbon.size > 0:
                    break
            alpha[left_in_ribbon] = 0.0
            dist_ribbon[left_in_ribbon] = abs(
                abs(x_rgt[zero_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[left_in_ribbon], 0]))
        if zr_vrtx_tip[zero_angle[i]] == 1 or zr_vrtx_tip[zero_angle[i]] == 2:
            for j in range(3):
                rgt_in_ribbon = np.where(elt_ribbon == elt_tip[zero_angle[i]] + (j + 1))[0]
                if rgt_in_ribbon.size > 0:
                    break
            alpha[rgt_in_ribbon] = 0.0
            dist_ribbon[rgt_in_ribbon] = abs(
                abs(x_rgt[zero_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[rgt_in_ribbon], 0]))

    ninety_angle = np.where(y_lft == y_rgt)[0]
    for i in range(len(ninety_angle)):
        if zr_vrtx_tip[ninety_angle[i]] == 0 or zr_vrtx_tip[ninety_angle[i]] == 1:
            for j in range(3):
                btm_in_ribbon = np.where(elt_ribbon == elt_tip[ninety_angle[i]] - (j + 1) * mesh.nx)[0]
                if btm_in_ribbon.size > 0:
                    break
            alpha[btm_in_ribbon] = np.pi / 2
            dist_ribbon[btm_in_ribbon] = abs(
                abs(y_rgt[ninety_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[btm_in_ribbon], 1]))
        if zr_vrtx_tip[ninety_angle[i]] == 2 or zr_vrtx_tip[ninety_angle[i]] == 3:
            for j in range(3):
                top_in_ribbon = np.where(elt_ribbon == elt_tip[ninety_angle[i]] + (j + 1) * mesh.nx)[0]
                if top_in_ribbon.size > 0:
                    break
            alpha[top_in_ribbon] = np.pi / 2
            dist_ribbon[top_in_ribbon] = abs(
                abs(y_rgt[ninety_angle[i]]) - abs(mesh.CenterCoor[elt_ribbon[top_in_ribbon], 1]))

    if np.isnan(alpha).any():
        print("found nan")
    return alpha


def construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip):
    slope = np.empty((len(elt_tip),), dtype=np.float64)
    pnt_on_line = np.empty((len(elt_tip), 2), dtype=np.float64)
    for i in range(len(elt_tip)):

        if zero_vertex_tip[i] == 0:
            slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 0]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 1:
            slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 1]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 2:
            slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 2]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] - l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])
        elif zero_vertex_tip[i] == 3:
            slope[i] = np.tan(np.pi / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 3]
            pnt_on_line[i] = np.array([mesh.VertexCoor[zr_vrtx_global, 0] + l_tip[i] * np.cos(alpha_tip[i]),
                                       mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * np.sin(alpha_tip[i])])

    zero_angle = np.where(alpha_tip == 0)[0]
    for i in zero_angle:
        if zero_vertex_tip[i] == 0 or zero_vertex_tip[i] == 1:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0]) ** 2 + (
            pnt_on_line[:, 1] - pnt_on_line[i, 1] - mesh.hy) ** 2) ** 0.5
            closest = np.argmin(dist_from_added)
            if dist_from_added[closest] < (mesh.hx ** 2 + mesh.hy ** 2) ** 0.5 / 10:
                np.delete(pnt_on_line, closest)
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0], pnt_on_line[i, 1] + mesh.hy])))

        if zero_vertex_tip[i] == 2 or zero_vertex_tip[i] == 3:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0]) ** 2 + (
                pnt_on_line[:, 1] - pnt_on_line[i, 1] + mesh.hy) ** 2) ** 0.5
            closest = np.argmin(dist_from_added)
            if dist_from_added[closest] < (mesh.hx ** 2 + mesh.hy ** 2) ** 0.5 / 10:
                np.delete(pnt_on_line, closest)
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0], pnt_on_line[i, 1] - mesh.hy])))
    ninety_angle = np.where(alpha_tip == np.pi / 2)[0]
    for i in ninety_angle:
        if zero_vertex_tip[i] == 0 or zero_vertex_tip[i] == 3:
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0] + mesh.hx, pnt_on_line[i, 1]])))
        if zero_vertex_tip[i] == 1 or zero_vertex_tip[i] == 2:
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0] - mesh.hx, pnt_on_line[i, 1]])))

    grid_lines_x = np.unique(mesh.VertexCoor[:, 0])
    grid_lines_y = np.unique(mesh.VertexCoor[:, 1])
    polygon = np.empty((0, 2), dtype=np.float64)

#### closest point algorithm

    for i in range(pnt_on_line.size):
        remaining = np.copy(pnt_on_line)
        nxt = pnt_on_line[i]
        remaining = np.delete(remaining, i, 0)
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        direction = np.asarray([nxt[0] - remaining[nxt_indx, 0], nxt[1] - remaining[nxt_indx, 1]])
        nxt = remaining[nxt_indx]
        remaining = np.delete(remaining, nxt_indx, 0)
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        direction_sec = np.asarray([nxt[0] - remaining[nxt_indx, 0], nxt[1] - remaining[nxt_indx, 1]])
        if (np.sign(direction) == np.sign(direction_sec))[0] and (np.sign(direction) == np.sign(direction_sec))[1]:
            first = np.copy(pnt_on_line[0])
            pnt_on_line[0] = np.copy(pnt_on_line[i])
            pnt_on_line[i] = np.copy(first)
            break

    remaining = np.copy(pnt_on_line)
    pnt_in_order = np.array([remaining[0]])
    nxt = pnt_on_line[0]
    remaining = np.delete(remaining, 0, 0)
    while remaining.size > 0:
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        nxt = remaining[nxt_indx]
        remaining = np.delete(remaining, nxt_indx, 0)
        pnt_in_order = np.vstack((pnt_in_order, nxt))

    i = 0
    while i <= pnt_in_order.shape[0] - 1:
        i_next = (i + 1) % pnt_in_order.shape[0]
        if pnt_in_order[i, 0] <= pnt_in_order[i_next, 0]:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] >= grid_lines_x, pnt_in_order[i, 0] < grid_lines_x))[0]
        else:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] <= grid_lines_x, pnt_in_order[i, 0] > grid_lines_x))[
                0]

        if grd_lns_btw_pnts_x.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
                pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_x:
                x_p = grid_lines_x[j]
                y_p = slope * (x_p - pnt_in_order[i_next, 0]) + pnt_in_order[i_next, 1]
                polygon = np.vstack((polygon, np.array([x_p, y_p])))

        if pnt_in_order[i, 1] <= pnt_in_order[i_next, 1]:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] >= grid_lines_y, pnt_in_order[i, 1] < grid_lines_y))[
                0]
        else:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] <= grid_lines_y, pnt_in_order[i, 1] > grid_lines_y))[
                0]

        if grd_lns_btw_pnts_y.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
                pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_y:
                y_p = grid_lines_y[j]

                x_p = (y_p - pnt_in_order[i_next, 1]) / slope + pnt_in_order[i_next, 0]
                polygon = np.vstack((polygon, np.array([x_p, y_p])))
        i += 1
    # polygon = np.round(polygon,6)
    polygon = np.vstack({tuple(row) for row in polygon})

    tip_smoothed = np.array([], dtype=np.int)
    smthed_tip_points_left = np.empty((0, 2), dtype=np.float64)
    smthed_tip_points_rgt = np.empty((0, 2), dtype=np.float64)
    for i in range(mesh.NumberOfElts):
        in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0]  # - mesh.hx*1e-6
        in_cell = np.logical_and(in_cell,
                                 polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0])  # + 1e-6*mesh.hx)
        in_cell = np.logical_and(in_cell,
                                 polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1])  # - 1e-6*mesh.hy)
        in_cell = np.logical_and(in_cell,
                                 polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1])  # + 1e-6*mesh.hy)

        cell_pnt = np.where(in_cell)[0]
        if cell_pnt.size > 2:
            # return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            dist = (polygon[cell_pnt[0], 0] - polygon[cell_pnt, 0]) ** 2 + (polygon[cell_pnt[0], 1] - polygon[
                cell_pnt, 1]) ** 2
            farthest = np.argmax(dist)
            to_delete = np.array([], dtype=np.int)
            for m in range(1, cell_pnt.size):
                if m != farthest:
                    to_delete = np.append(to_delete, cell_pnt[m])
            polygon = np.delete(polygon, to_delete, 0)

            in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0]  # - mesh.hx*1e-6
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0])  # + 1e-6*mesh.hx)
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1])  # - 1e-6*mesh.hy)
            in_cell = np.logical_and(in_cell,
                                     polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1])  # + 1e-6*mesh.hy)
            cell_pnt = np.where(in_cell)[0]

        if cell_pnt.size > 1:
            tip_smoothed = np.append(tip_smoothed, i)
            if polygon[cell_pnt[0], 0] <= polygon[cell_pnt[1], 0]:
                smthed_tip_points_left = np.vstack((smthed_tip_points_left, polygon[cell_pnt[0]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[1]]))
            else:
                smthed_tip_points_left = np.vstack((smthed_tip_points_left, polygon[cell_pnt[1]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[0]]))

    smthed_tip_lines_slope = (smthed_tip_points_rgt[:, 1] - smthed_tip_points_left[:, 1]) / (
        smthed_tip_points_rgt[:, 0] - smthed_tip_points_left[:, 0])
    smthed_tip_lines_a = -smthed_tip_lines_slope
    smthed_tip_lines_b = np.ones((len(tip_smoothed),), dtype=np.float64)
    smthed_tip_lines_c = -(smthed_tip_points_rgt[:, 1] - smthed_tip_lines_slope * smthed_tip_points_rgt[:, 0])

    zero_angle = np.where(smthed_tip_points_left[:, 0] == smthed_tip_points_rgt[:, 0])[0]
    smthed_tip_lines_b[zero_angle] = 0.
    smthed_tip_lines_a[zero_angle] = 1.
    smthed_tip_lines_c[zero_angle] = -smthed_tip_points_rgt[zero_angle, 0]

    tip_lft_neghb = np.zeros((len(tip_smoothed),), dtype=np.int)
    tip_rgt_neghb = np.empty((len(tip_smoothed),), dtype=np.int)
    for i in range(len(tip_smoothed)):
        equal = smthed_tip_points_rgt == smthed_tip_points_left[i]
        left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
        if left_nei.size != 1:
            equal = smthed_tip_points_left == smthed_tip_points_left[i]
            left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
            if left_nei.size == 2:
                tip_lft_neghb[i] = left_nei[np.where(left_nei != i)[0]]
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                # tip_lft_neghb[i] = i
        else:
            tip_lft_neghb[i] = left_nei


        equal = smthed_tip_points_left == smthed_tip_points_rgt[i]
        rgt_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
        if rgt_nei.size != 1:
            equal = smthed_tip_points_rgt == smthed_tip_points_rgt[i]
            rgt_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
            if rgt_nei.size == 2:
                tip_rgt_neghb[i] = rgt_nei[np.where(rgt_nei != i)[0]]
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                # tip_rgt_neghb[i] = i
        else:
            tip_rgt_neghb[i] = rgt_nei

    return tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_b, smthed_tip_lines_c, smthed_tip_points_left, \
           smthed_tip_points_rgt, tip_lft_neghb, tip_rgt_neghb
