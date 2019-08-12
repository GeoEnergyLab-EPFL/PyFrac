# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from level_set import reconstruct_front_LS_gradient
from volume_integral import Integral_over_cell


def projection_from_ribbon(ribbon_elts, channel_elts, mesh, sgnd_dist):
    """
    This function finds the projection of the ribbon cell centers on to the fracture front. It is returned as the angle
    inscribed by the perpendiculars drawn on the front from the ribbon cell centers

    Arguments:
        ribbon_elts (ndarray-int)               -- list of ribbon elements
        mesh (CartesianMesh object)             -- The cartesian mesh object
        mat_prop (MaterialProperties object)    -- Material properties:
        sgnd_dist (ndarray-float)               -- level set data

    Returns:
        alpha (ndarray-float)                   -- the angle inscribed by the perpendiculars drawn on the front from
                                                   the ribbon cell centers.
    """

    # reconstruct front to get tip cells from the given level set

    (elt_tip, l_tip, alpha_tip, CellStatus) = reconstruct_front_LS_gradient(sgnd_dist,
                                                            np.setdiff1d(np.arange(mesh.NumberOfElts,), channel_elts),
                                                            channel_elts,
                                                            mesh)
    # get the filling fraction to find partially filled tip cells
    FillFrac = Integral_over_cell(elt_tip,
                                  alpha_tip,
                                  l_tip,
                                  mesh,
                                  'A') / mesh.EltArea

    # taking partially filled as the current tip
    partly_filled = np.where(FillFrac < 0.999999)[0]
    elt_tip = elt_tip[partly_filled]
    l_tip = l_tip[partly_filled]
    alpha_tip = alpha_tip[partly_filled]

    zero_vertex_tip = find_zero_vertex(elt_tip, sgnd_dist, mesh)
    # construct the polygon
    smthed_tip, a, b, c, pnt_lft, pnt_rgt, neig_lft, neig_rgt = construct_polygon(elt_tip,
                                                                                  l_tip,
                                                                                  alpha_tip,
                                                                                  mesh,
                                                                                  zero_vertex_tip)
    if np.isnan(smthed_tip).any():
        # if cannot be found
        return np.nan

    zr_vrtx_smthed_tip = find_zero_vertex(smthed_tip, sgnd_dist, mesh)
    alpha = find_angle(ribbon_elts,
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

    return alpha

#-----------------------------------------------------------------------------------------------------------------------
# todo: the function is not written cleanly and is not readable
def find_angle(elt_ribbon, elt_tip, zr_vrtx_tip, a_tip, b_tip, c_tip, x_lft, y_lft, x_rgt, y_rgt, neig_lft,
                    neig_rgt, mesh):
    """
    This function calculates the angle inscribed by the perpendiculars on the given polygon. The polygon is provided
    in the form of equations of edges of the polygon (with the form ax+by+c=0) and the left and right points of the
    front line in the given tip elements.
    """

    closest_tip_cell = np.zeros((len(elt_ribbon),), dtype=np.int)
    dist_ribbon = np.zeros((len(elt_ribbon),), dtype=np.float64)
    alpha = np.zeros((len(elt_ribbon),), dtype=np.float64)
    for i in range(len(elt_ribbon)):
        # min dist from the front lines of a ribbon cells
        dist_front_line = np.zeros((len(elt_tip),), dtype=np.float64)

        point_at_grid_line = np.zeros((len(elt_tip),), dtype=np.uint8)

        # loop over tip cells for the current ribbon cell
        for j in range(len(elt_tip)):

            if x_rgt[j] - x_lft[j] == 0:
                # if parallel to y-axis
                xx = mesh.CenterCoor[elt_ribbon[i], 0]
                yy = - c_tip[j]
            else:
                slope_tip_line = (y_rgt[j] - y_lft[j]) / (x_rgt[j] - x_lft[j])
                m = -1. / slope_tip_line # slope perp to the tip line

                intrcpt = mesh.CenterCoor[elt_ribbon[i], 1] - m * mesh.CenterCoor[elt_ribbon[i], 0] #intercept
                # x-coordinate of the point where the perpendicular intersects the drawn perpendicular
                xx = -(intrcpt + c_tip[j]) / (a_tip[j] + m)
                # y-coordinate of the point where the perpendicular intersects the drawn perpendicular
                yy = m * xx + intrcpt

            if x_lft[j] > xx or x_rgt[j] < xx or min(y_lft[j], y_rgt[j]) > yy or max(
                    y_lft[j], y_rgt[j]) < yy:
                # if the intersection point is out of the tip cell
                dist_lft_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_lft[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_lft[j]) ** 2) ** 0.5
                dist_rgt_pnt = ((mesh.CenterCoor[elt_ribbon[i], 0] - x_rgt[j]) ** 2
                                + (mesh.CenterCoor[elt_ribbon[i], 1] - y_rgt[j]) ** 2) ** 0.5
                # take the distance of either the left or the right point depending upon which is closer to the ribbon
                dist_front_line[j] = min(dist_lft_pnt, dist_rgt_pnt)
                # save which (right of left) point on the front line is closer to the ribbon cell center
                if dist_lft_pnt < dist_rgt_pnt:
                    point_at_grid_line[j] = 1
                else:
                    point_at_grid_line[j] = 2
            else:
                # if the intersection point of the front line and the perpendicular drawn from the zero vertex is the
                # closest point to the riboon cell center
                dist_front_line[j] = abs(mesh.CenterCoor[elt_ribbon[i], 0] * a_tip[j] + mesh.CenterCoor[elt_ribbon[i],
                                                1] + c_tip[j]) / (a_tip[j] ** 2 + 1) ** 0.5 # distance calculated by
                                                                                            # min distance to a line
                                                                                            # from a point formula

        closest_tip_cell[i] = np.argmin(dist_front_line)

        if point_at_grid_line[closest_tip_cell[i]] == 0:
            # if the closest point is the intersection point of the perpendicular
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            # finding angle using arc cosine
            alpha[i] = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
        elif point_at_grid_line[closest_tip_cell[i]] == 1:
            # if the closest point is the left most point on the front line
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_lft[closest_tip_cell[i]]]) / a_tip[neig_lft[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2
        elif point_at_grid_line[closest_tip_cell[i]] == 2:
            # if the closest point is the right most point on the front line
            y = mesh.CenterCoor[elt_ribbon[i], 1]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            x = (-y - c_tip[neig_rgt[closest_tip_cell[i]]]) / a_tip[neig_rgt[closest_tip_cell[i]]]
            alpha_nei = np.arccos(
                round(dist_front_line[closest_tip_cell[i]] / abs(x - mesh.CenterCoor[elt_ribbon[i], 0]), 5))
            alpha[i] = (alpha_closest + alpha_nei) / 2

        dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]

    # the code below finds the ribbon cells directly below or above the tip cells with ninety degrees angle and sets
    # them to have ninety degrees angle as well. Similarly, the ribbon cells directly on the left or right of the tip
    # cells with zero degrees angle are set to have zero angles.
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


    return alpha

#-----------------------------------------------------------------------------------------------------------------------


def construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip):
    """
    This function construct a polygon from the given non-continous front. The polygon is constructed by joining the
    intersection of the perpendiculars drawn on the front with the front lines. The points closest to each other are
    joined and the intersection of the grid lines with these lines are taken as the vertices of the polygon.
    """

    slope = np.empty((len(elt_tip),), dtype=np.float64)
    pnt_on_line = np.empty((len(elt_tip), 2), dtype=np.float64) # point where the perpendicular drawn on the front
                                                                # intersects the front
    # loop over tip cells to find the intersection point
    for i in range(len(elt_tip)):
        if zero_vertex_tip[i] == 0:
            # if the perpendicular is drawn from the bottom left vertex
            slope[i] = np.tan(-(np.pi / 2 - alpha_tip[i])) # slope of a line perpendicular to the front line
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 0] # bottom left vertex
            # coordinates of the intersection point
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

    # the code below make sure that, for the cells with zero or ninety degrees angle, there are points that are
    # exactly left/right or above/below of each other so that joining them will make a line that is parallel to the x
    # or y axes respectively.
    zero_angle = np.where(alpha_tip == 0)[0]
    for i in zero_angle:
        if zero_vertex_tip[i] == 0 or zero_vertex_tip[i] == 1:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0]) ** 2 + (pnt_on_line[:, 1] - pnt_on_line[i, 1] -
                                                                               mesh.hy) ** 2) ** 0.5
            closest = np.argmin(dist_from_added)
            if dist_from_added[closest] < (mesh.hx ** 2 + mesh.hy ** 2) ** 0.5 / 10:
                np.delete(pnt_on_line, closest)
            pnt_on_line = np.vstack((pnt_on_line, np.array([pnt_on_line[i, 0], pnt_on_line[i, 1] + mesh.hy])))

        if zero_vertex_tip[i] == 2 or zero_vertex_tip[i] == 3:
            dist_from_added = ((pnt_on_line[:, 0] - pnt_on_line[i, 0]) ** 2 + (pnt_on_line[:, 1] - pnt_on_line[i, 1] +
                                                                               mesh.hy) ** 2) ** 0.5
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

    grid_lines_x = np.unique(mesh.VertexCoor[:, 0]) # the x-coordinate of the points on grid lines parallel to y-axis
    grid_lines_y = np.unique(mesh.VertexCoor[:, 1]) # the y-coordinate of the points on grid lines parallel to x-axis
    polygon = np.empty((0, 2), dtype=np.float64)

    # # The code below is a hack. It selects the starting point for the closest point algorithm which joins the points
    # # to construct the polygon. It basically finds the point which has the direction change in the next two closest
    # # points and set it as the starting point.
    # for i in range(pnt_on_line.size):
    #     remaining = np.copy(pnt_on_line)# remaining set of points
    #     nxt = pnt_on_line[i] # current point which is to be joined
    #     remaining = np.delete(remaining, i, 0) # remove from the remaining set
    #     dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
    #     nxt_indx = np.argmin(dist_from_remnng) # index of the closest point
    #     direction = np.asarray([nxt[0] - remaining[nxt_indx, 0], nxt[1] - remaining[nxt_indx, 1]])
    #     nxt = remaining[nxt_indx]
    #     remaining = np.delete(remaining, nxt_indx, 0)
    #     dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
    #     nxt_indx = np.argmin(dist_from_remnng)
    #     direction_sec = np.asarray([nxt[0] - remaining[nxt_indx, 0], nxt[1] - remaining[nxt_indx, 1]])
    #     if (np.sign(direction) == np.sign(direction_sec))[0] and (np.sign(direction) == np.sign(direction_sec))[1]:
    #         first = np.copy(pnt_on_line[0])
    #         pnt_on_line[0] = np.copy(pnt_on_line[i])
    #         pnt_on_line[i] = np.copy(first)
    #         break

    # closest point algorithm giving the points in order to construct a polygon
    remaining = np.copy(pnt_on_line) # remaining points to be joined
    pnt_in_order = np.array([remaining[0]]) # the points of the polygon given in order
    nxt = pnt_on_line[0]
    remaining = np.delete(remaining, 0, 0)
    while remaining.size > 0:
        dist_from_remnng = ((remaining[:, 0] - nxt[0]) ** 2 + (remaining[:, 1] - nxt[1]) ** 2) ** 0.5
        nxt_indx = np.argmin(dist_from_remnng)
        nxt = remaining[nxt_indx]
        remaining = np.delete(remaining, nxt_indx, 0)
        pnt_in_order = np.vstack((pnt_in_order, nxt))

    # the code below finds the grid lines between two consecutive points. The vertices of the polygon are found by
    # the intersection of the grid lines and the line joining consecutive points (found by the closest point algorithm
    # above).
    i = 0
    while i <= pnt_in_order.shape[0] - 1:
        i_next = (i + 1) % pnt_in_order.shape[0] # to make it cyclic (joining the first point to the last)
        # find the grid lines parallel to y-axis between the closest points under consideration
        if pnt_in_order[i, 0] <= pnt_in_order[i_next, 0]:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] >= grid_lines_x, pnt_in_order[i, 0] < grid_lines_x))[0]
        else:
            grd_lns_btw_pnts_x = np.where(
                np.logical_and(pnt_in_order[i_next, 0] <= grid_lines_x, pnt_in_order[i, 0] > grid_lines_x))[
                0]
        # if there is a grid line between the points
        if grd_lns_btw_pnts_x.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
                pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_x:
                x_p = grid_lines_x[j]
                y_p = slope * (x_p - pnt_in_order[i_next, 0]) + pnt_in_order[i_next, 1]
                # add the intersection point to the polygon
                polygon = np.vstack((polygon, np.array([x_p, y_p])))

        # find the grid lines parallel to x-axis between the closest points under consideration
        if pnt_in_order[i, 1] <= pnt_in_order[i_next, 1]:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] >= grid_lines_y, pnt_in_order[i, 1] < grid_lines_y))[
                0]
        else:
            grd_lns_btw_pnts_y = np.where(
                np.logical_and(pnt_in_order[i_next, 1] <= grid_lines_y, pnt_in_order[i, 1] > grid_lines_y))[
                0]
        # if there is a grid line between the points
        if grd_lns_btw_pnts_y.size > 0:
            slope = (pnt_in_order[i_next, 1] - pnt_in_order[i, 1]) / (
                pnt_in_order[i_next, 0] - pnt_in_order[i, 0])
            for j in grd_lns_btw_pnts_y:
                y_p = grid_lines_y[j]
                x_p = (y_p - pnt_in_order[i_next, 1]) / slope + pnt_in_order[i_next, 0]
                # add the intersection point to the polygon
                polygon = np.vstack((polygon, np.array([x_p, y_p])))
        i += 1

    # remove redundant points
    polygon = np.vstack({tuple(row) for row in polygon})


    tip_smoothed = np.array([], dtype=np.int) # the cells containing the edges of polygon (giving the smoothed front)
    smthed_tip_points_left = np.empty((0, 2), dtype=np.float64) #left points of the tip line in the new tip cells
    smthed_tip_points_rgt = np.empty((0, 2), dtype=np.float64) #right points of the tip line in the new tip cells

    # loop over the cells of the grid to find the cells containing one of the edges of the polygon
    for i in range(mesh.NumberOfElts):
        # find the vertices of the polygon with x-coordinates greater than or equal to x-coordinate of the bottom left
        # vertex of the cell
        in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0]
        in_cell = np.logical_and(in_cell, polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0])
        in_cell = np.logical_and(in_cell, polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1])
        in_cell = np.logical_and(in_cell, polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1])
        # points of the polygon on the edges of the current cell
        cell_pnt = np.where(in_cell)[0]
        if cell_pnt.size > 2:
            # Hack!!! if there is more than two points, find the two furthest
            # return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            dist = (polygon[cell_pnt[0], 0] - polygon[cell_pnt, 0]) ** 2 + (polygon[cell_pnt[0], 1] - polygon[
                cell_pnt, 1]) ** 2
            farthest = np.argmax(dist)
            to_delete = np.array([], dtype=np.int)
            for m in range(1, cell_pnt.size):
                if m != farthest:
                    to_delete = np.append(to_delete, cell_pnt[m])
            # delete the extra points from polygon
            polygon = np.delete(polygon, to_delete, 0)
            # find the two points again
            in_cell = polygon[:, 0] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 0]
            in_cell = np.logical_and(in_cell, polygon[:, 0] <= mesh.VertexCoor[mesh.Connectivity[i, 1], 0])
            in_cell = np.logical_and(in_cell, polygon[:, 1] >= mesh.VertexCoor[mesh.Connectivity[i, 0], 1])
            in_cell = np.logical_and(in_cell, polygon[:, 1] <= mesh.VertexCoor[mesh.Connectivity[i, 3], 1])
            cell_pnt = np.where(in_cell)[0]

        if cell_pnt.size > 1:
            # add the cell to the tip cells
            tip_smoothed = np.append(tip_smoothed, i)
            # add accordingly to the left and right points of the added tip cell
            if polygon[cell_pnt[0], 0] <= polygon[cell_pnt[1], 0]:
                smthed_tip_points_left = np.vstack((smthed_tip_points_left, polygon[cell_pnt[0]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[1]]))
            else:
                smthed_tip_points_left = np.vstack((smthed_tip_points_left, polygon[cell_pnt[1]]))
                smthed_tip_points_rgt = np.vstack((smthed_tip_points_rgt, polygon[cell_pnt[0]]))

    # find the equations(of the form ax+by+c=0) of the front lines in the tip cells
    smthed_tip_lines_slope = (smthed_tip_points_rgt[:, 1] - smthed_tip_points_left[:, 1]) / (
        smthed_tip_points_rgt[:, 0] - smthed_tip_points_left[:, 0])
    smthed_tip_lines_a = -smthed_tip_lines_slope
    smthed_tip_lines_b = np.ones((len(tip_smoothed),), dtype=np.float64)
    smthed_tip_lines_c = -(smthed_tip_points_rgt[:, 1] - smthed_tip_lines_slope * smthed_tip_points_rgt[:, 0])

    # equation of the line with 90 degree angle
    zero_angle = np.where(smthed_tip_points_left[:, 0] == smthed_tip_points_rgt[:, 0])[0]
    smthed_tip_lines_b[zero_angle] = 0.
    smthed_tip_lines_a[zero_angle] = 1.
    smthed_tip_lines_c[zero_angle] = -smthed_tip_points_rgt[zero_angle, 0]

    # find the left neighbor of the tip cells in the tip
    tip_lft_neghb = np.zeros((len(tip_smoothed),), dtype=np.int)
    tip_rgt_neghb = np.empty((len(tip_smoothed),), dtype=np.int)
    for i in range(len(tip_smoothed)):
        equal = smthed_tip_points_rgt == smthed_tip_points_left[i]
        left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
        if left_nei.size != 1:
            # Hack!!! find the cell with the same left point
            equal = smthed_tip_points_left == smthed_tip_points_left[i]
            left_nei = np.where(np.logical_and(equal[:, 0], equal[:, 1]))[0]
            if left_nei.size == 2:
                tip_lft_neghb[i] = left_nei[np.where(left_nei != i)[0]]
            else:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                # tip_lft_neghb[i] = i
        else:
            tip_lft_neghb[i] = left_nei

        # find the eight neighbor of the tip cells in the tip
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

#-----------------------------------------------------------------------------------------------------------------------


def projection_from_ribbon_LS_gradient(ribbon_elts, tip_elts, mesh, sgnd_dist):
    """
    This function finds the projection of the ribbon cell centers on to the fracture front from the gradient of the
    level set. It is returned as the angle inscribed by the perpendiculars drawn on the front from the ribbon cell
    centers.

    Arguments:
        ribbon_elts (ndarray-int)               -- list of ribbon elements
        mesh (CartesianMesh object)             -- The cartesian mesh object
        mat_prop (MaterialProperties object)    -- Material properties:
        sgnd_dist (ndarray-float)               -- level set data

    Returns:
        alpha (ndarray-float)                   -- the angle inscribed by the perpendiculars drawn on the front from
                                                   the ribbon cell centers.
    """

    n_vertex = np.zeros((len(tip_elts), 2), float)
    n_centre = np.zeros((len(ribbon_elts), 2), float)
    Coor_vertex = np.zeros((len(tip_elts), 2), float)
    alpha = np.zeros((len(ribbon_elts),), dtype=np.float64)

    zero_vertex = find_zero_vertex(tip_elts,
                                      sgnd_dist,
                                      mesh)
    for i in range(len(tip_elts)):
        # neighbors
        #     6     3    7
        #     0    elt   1
        #     4    2     5
         neighbors_tip = np.zeros(8, dtype=int)
         neighbors_tip[:4] = mesh.NeiElements[tip_elts[i]]
         neighbors_tip[4] = mesh.NeiElements[neighbors_tip[2]][0]
         neighbors_tip[5] = mesh.NeiElements[neighbors_tip[2]][1]
         neighbors_tip[6] = mesh.NeiElements[neighbors_tip[3]][0]
         neighbors_tip[7] = mesh.NeiElements[neighbors_tip[3]][1]

        # Vertex
        #     3         2
        #     0         1
         if zero_vertex[i]==0:
              gradx = -((sgnd_dist[neighbors_tip[0]]+sgnd_dist[neighbors_tip[4]])/2 - (
                      sgnd_dist[tip_elts[i]]+sgnd_dist[neighbors_tip[2]])/2) / mesh.hx
              grady = ((sgnd_dist[neighbors_tip[0]]+sgnd_dist[tip_elts[i]])/2 - (
                      sgnd_dist[neighbors_tip[4]]+sgnd_dist[neighbors_tip[2]])/2) / mesh.hy
              Coor_vertex[i,0] = mesh.CenterCoor[tip_elts[i], 0]-mesh.hx/2
              Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] - mesh.hy / 2
         elif zero_vertex[i] == 1:
              gradx = ((sgnd_dist[neighbors_tip[1]] + sgnd_dist[neighbors_tip[5]]) / 2 - (
                        sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[2]]) / 2) / mesh.hx
              grady = ((sgnd_dist[neighbors_tip[1]] + sgnd_dist[tip_elts[i]]) / 2 - (
                          sgnd_dist[neighbors_tip[5]] + sgnd_dist[neighbors_tip[2]]) / 2) / mesh.hy
              Coor_vertex[i, 0] = mesh.CenterCoor[tip_elts[i], 0] + mesh.hx / 2
              Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] - mesh.hy / 2
         elif zero_vertex[i] == 2:
              gradx = ((sgnd_dist[neighbors_tip[1]] + sgnd_dist[neighbors_tip[7]]) / 2 - (
                    sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hx
              grady = -((sgnd_dist[neighbors_tip[1]] + sgnd_dist[tip_elts[i]]) / 2 - (
                  sgnd_dist[neighbors_tip[3]] + sgnd_dist[neighbors_tip[7]]) / 2) / mesh.hy
              Coor_vertex[i, 0] = mesh.CenterCoor[tip_elts[i], 0] + mesh.hx / 2
              Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] + mesh.hy / 2
         elif zero_vertex[i] == 3:
               gradx =-((sgnd_dist[neighbors_tip[6]] + sgnd_dist[neighbors_tip[0]]) / 2 - (
                    sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[3]]) / 2) /mesh.hx
               grady = ((sgnd_dist[neighbors_tip[0]] + sgnd_dist[tip_elts[i]]) / 2 - (
                    sgnd_dist[neighbors_tip[6]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hy
               Coor_vertex[i, 0] = mesh.CenterCoor[tip_elts[i], 0] - mesh.hx / 2
               Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] + mesh.hy / 2
         n_vertex[i, 0] = gradx / (gradx ** 2 + grady ** 2) ** 0.5
         n_vertex[i, 1] = grady / (gradx ** 2 + grady ** 2) ** 0.5

    for i in range(len(ribbon_elts)):

         actvElts = np.where((2 * abs(mesh.CenterCoor[ribbon_elts[i], 0] - Coor_vertex[:, 0]) - mesh.hx < mesh.hx/10) &
                            (2 * abs(mesh.CenterCoor[ribbon_elts[i], 1] - Coor_vertex[:, 1]) - mesh.hy < mesh.hy/10))[0]

         n_centre[i, 0] = np.mean(n_vertex[actvElts, 0])
         n_centre[i, 1] = np.mean(n_vertex[actvElts, 1])
         alpha[i] = np.abs(np.arcsin(n_centre[i, 1]))

    return alpha

#-----------------------------------------------------------------------------------------------------------------------


def find_zero_vertex(Elts, level_set, mesh):
    """
    This function finds the zero-vertex (the vertex opposite to the propagation direction) from where the perpendicular
    is drawn on the front.

    Arguments:
        Elts (ndarray)              -- the given elements for which the zero-vertex is to be found.
        level_set (ndarray)         -- the level set data (distance from front of the elements of the grid).
        mesh (ndarray)              -- the mesh given by CartesianMesh object.

    Returns:
        zero_vertex (ndarray)       -- the zero vertex list
    """

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





def get_toughness_from_cellCenter(alpha, sgnd_dist=None, elts=None, mat_prop=None, mesh=None):
    """
    This function returns the toughness given the angle inscribed from the cell centers on the front. both the cases
    of heterogenous or anisotropic toughness are taken care off.
    """

    if mat_prop.anisotropic_K1c:
        try:
            return mat_prop.K1cFunc(alpha)
        except TypeError:
            SystemExit("For anisotropic toughness, the function taking the angle and returning the toughness is to "
                       "be provided")
    else:
        dist = -sgnd_dist
        x = np.zeros((len(elts),), )
        y = np.zeros((len(elts),), )

        neighbors = mesh.NeiElements[elts]
        zero_vertex = find_zero_vertex(elts,
                                       sgnd_dist,
                                       mesh)
        # evaluating the closest tip points
        for i in range(0, len(elts)):
            if zero_vertex[i] == 0:

                x[i] = mesh.CenterCoor[elts[i], 0] + dist[elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 1:

                x[i] = mesh.CenterCoor[elts[i], 0] - dist[elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 2:

                x[i] = mesh.CenterCoor[elts[i], 0] - dist[elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]] * np.sin(alpha[i])

            elif zero_vertex[i] == 3:

                x[i] = mesh.CenterCoor[elts[i], 0] + dist[elts[i]] * np.cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]] * np.sin(alpha[i])

            # assume the angle is zero if the distance of the left and right neighbor is extremely close
            if abs(dist[mesh.NeiElements[elts[i], 0]] / dist[mesh.NeiElements[elts[i], 1]] - 1) < 1e-7:
                if sgnd_dist[neighbors[i, 2]] < sgnd_dist[neighbors[i, 3]]:
                    x[i] = mesh.CenterCoor[elts[i], 0]
                    y[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]]
                elif sgnd_dist[neighbors[i, 2]] > sgnd_dist[neighbors[i, 3]]:
                    x[i] = mesh.CenterCoor[elts[i], 0]
                    y[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]]
            # assume the angle is 90 degrees if the distance of the bottom and top neighbor is extremely close
            if abs(dist[mesh.NeiElements[elts[i], 2]] / dist[mesh.NeiElements[elts[i], 3]] - 1) < 1e-7:
                if sgnd_dist[neighbors[i, 0]] < sgnd_dist[neighbors[i, 1]]:
                    x[i] = mesh.CenterCoor[elts[i], 0] + dist[elts[i]]
                    y[i] = mesh.CenterCoor[elts[i], 1]
                elif sgnd_dist[neighbors[i, 0]] > sgnd_dist[neighbors[i, 1]]:
                    x[i] = mesh.CenterCoor[elts[i], 0] - dist[elts[i]]
                    y[i] = mesh.CenterCoor[elts[i], 1]

        # returning the Kprime according to the given function
        K1c = np.empty((len(elts), ), dtype=np.float64)
        for i in range(len(elts)):
            try:
                K1c[i] = mat_prop.K1cFunc(x[i], y[i])
            except TypeError:
                SystemExit("For precise space dependant toughness, the function taking the coordinates and returning"
                           "the toughness is to be provided.")
        return K1c

#-----------------------------------------------------------------------------------------------------------------------


def get_toughness_from_zeroVertex(elts, mesh, mat_prop, alpha, l, zero_vrtx):
    """
    This function returns the toughness given the angle inscribed from the zero-vertex on the front. both the cases
    of heterogenous or anisotropic toughness are taken care off.
    """

    if mat_prop.K1cFunc is None:
        return mat_prop.K1c[elts]

    if mat_prop.anisotropic_K1c:
        return mat_prop.K1cFunc(alpha)
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

        # returning the Kprime according to the given function
        K1c = np.empty((len(elts),), dtype=np.float64)
        for i in range(len(elts)):
            K1c[i] = mat_prop.K1cFunc(x[i], y[i])

        return K1c

#-----------------------------------------------------------------------------------------------------------------------


def TI_plain_strain_modulus(alpha, Cij):
    """
    This function computes the plain strain elasticity modulus in transverse isotropic medium. The modulus is a function
    of the orientation of the fracture front with respect to the bedding plane. This functions is used for the tip
    inversion and for evaluation of the fracture volume for the case of TI elasticity.

    Arguments:
        alpha (ndarray-float)             -- the angle inscribed by the perpendiculars drawn on the front from the \
                                             ribbon cell centers.
        Cij (ndarray-float)               -- the TI stiffness matrix in the canonical basis

    Returns:
        E' (ndarray-float)               -- plain strain TI elastic modulus.
    """

    C11 = Cij[0, 0]
    C12 = Cij[0, 1]
    C13 = Cij[0, 2]
    C33 = Cij[2, 2]
    C44 = Cij[3, 3]

    # we use the same notation for the elastic paramateres as S. Fata et al. (2013).

    alphag = (C11 * (C11-C12) * np.cos(alpha) ** 4 + (C11 * C13
                 - C12 * (C13 + 2 * C44)) * (np.cos(alpha) * np.sin(alpha)) ** 2
                 - (C13 ** 2 - C11 * C33 + 2 * C13 * C44) * np.sin(alpha) ** 4
                 + C11 * C44 * np.sin(2 * alpha) ** 2) / (C11 * (C11 - C12) * np.cos(alpha) ** 2
                                                          + 2 * C11 * C44 * np.sin(alpha) ** 2)

    gammag = ((C11 * np.cos(alpha) ** 4 + 2 * C13 * (np.cos(alpha) * np.sin(alpha)) ** 2
                 + C33 * np.sin(alpha) ** 4 + C44 * np.sin(2 * alpha) ** 2) / C11) ** 0.5

    deltag = ((C11 - C12) * (C11 + C12) * np.cos(alpha) ** 4
                 + 2 * (C11 - C12) * C13 * (np.cos(alpha) * np.sin(alpha)) ** 2
                 + (- C13 ** 2 + C11 * C33) * np.sin(alpha) ** 4
                 + C11 * C44 * np.sin(2 * alpha) ** 2) / (C11 * (2 * (alphag + gammag)) ** 0.5)

    Eprime = 2 * deltag / gammag

    return Eprime

#-----------------------------------------------------------------------------------------------------------------------