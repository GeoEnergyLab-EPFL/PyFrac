# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Aug 09 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
import logging
from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix
import math
import sys

# internal imports
from level_set.discontinuous_front_reconstruction import reconstruct_front, UpdateLists
from level_set.FMM import fmm
from level_set.continuous_front_reconstruction import reconstruct_front_continuous, UpdateListsFromContinuousFrontRec, \
    get_xy_from_Ffront, get_cells_in_neighborhood, \
    ray_tracing_numpy

from tip.volume_integral import Integral_over_cell
from solid.elasticity_isotropic_symmetric import self_influence
from linear_solvers.linear_iterative_solver import iteration_counter
from scipy.optimize import least_squares


def get_eliptical_survey_cells(mesh, a, b, center=None):
    """
    This function would provide the ribbon of cells on the inside of the perimeter of an ellipse with the given
    lengths of the major and minor axes. A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object):        -- a CartesianMesh class object describing the grid.
        a (float):                          -- the length of the major axis of the provided ellipse.
        b (float):                          -- the length of the minor axis of the provided ellipse.
        inj_point (list or ndarray):        -- the coordinates [x, y] of the injection point.

    Returns:
        - surv_cells (ndarray)              -- the list of cells on the inside of the perimeter of the given\
                                               ellipse.
        - surv_dist (ndarray)               -- the list of corresponding distances of the surv_cells to the fracture\
                                               tip.
        - inner_cells (ndarray)             -- the list of cells inside the given ellipse.
    """
    if center is None:
        center = np.asarray([0, 0])

    # distances of the cell vertices
    dist_vertx = ((mesh.VertexCoor[:, 0] - center[0])/ a) ** 2 + ((mesh.VertexCoor[:, 1] - center[1]) / b) ** 2 - 1.
    # vertices that are inside the ellipse
    vertices = dist_vertx[mesh.Connectivity] < 0

    #cells with all four vertices inside
    log_and = np.logical_and(np.logical_and(vertices[:, 0], vertices[:, 1]),
                             np.logical_and(vertices[:, 2],vertices[:, 3]))
    inner_cells = np.where(log_and)[0]
    if len(inner_cells) == 0:
        raise SystemError("The given ellipse is too small compared to mesh!")

    dist = np.zeros((inner_cells.size,), dtype=np.float64)
    # get minimum distance from center of the inner cells
    for i in range(0, inner_cells.size):
        dist[i] = Distance_ellipse(a,
                                   b,
                                   mesh.CenterCoor[inner_cells[i], 0] - center[0],
                                   mesh.CenterCoor[inner_cells[i], 1] - center[1])

    cell_len = (mesh.hx * mesh.hx + mesh.hy * mesh.hy) ** 0.5  # one cell diagonal length
    ribbon = np.where(dist <= 2 * cell_len)[0]
    surv_cells = inner_cells[ribbon]
    surv_dist = dist[ribbon]

    # if center is not None:
    #     surv_cells, tmp = shift_injection_point(inj_point[0],
    #                                              inj_point[1],
    #                                              mesh,
    #                                              active_elts=surv_cells)
    #     inner_cells, tmp = shift_injection_point(inj_point[0],
    #                                              inj_point[1],
    #                                              mesh,
    #                                              active_elts=inner_cells)

    return surv_cells, surv_dist, inner_cells

#-----------------------------------------------------------------------------------------------------------------------


def get_radial_survey_cells(mesh, r, center=None, external_crack=False):
    """
    This function would provide the ribbon of cells and their distances to the front on the inside (or outside) of the
    perimeter of a circle with the given radius. A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object):        -- a CartesianMesh class object describing the grid.
        r (float):                          -- the radius of the circle.
        inj_point (list or ndarray):        -- the coordinates [x, y] of the injection point.
        external_crack (bool):              -- True if you would like the fracture to be an external crack.

    Returns:
        - surv_cells (ndarray)              -- the list of cells on the inside of the perimeter of the given circle.\
                                               In case of external_crack=True the list of cells outside of the perimeter.
        - surv_dist (ndarray)               -- the list of corresponding distances of the surv_cells to the fracture\
                                               tip.
        - inner_cells (ndarray)             -- the list of cells inside the given circle.
    """
    if center is None:
        center = np.asarray([0, 0])

    # distances of the cell vertices
    dist_vertx = (((mesh.VertexCoor[:, 0] - center[0])) ** 2 + ((mesh.VertexCoor[:, 1] - center[1])) ** 2 ) \
                  ** (1 / 2) / r - 1.

    # vertices that are inside the ellipse
    vertices = dist_vertx[mesh.Connectivity] <= 0

    # cells with all four vertices inside
    log_and = np.logical_and(np.logical_and(vertices[:, 0], vertices[:, 1]),
                             np.logical_and(vertices[:, 2], vertices[:, 3]))

    inner_cells = np.where(log_and)[0]
    dist = r - ((mesh.CenterCoor[inner_cells, 0] - center[0]) ** 2
                + (mesh.CenterCoor[inner_cells, 1] - center[1]) ** 2) ** 0.5

    if len(inner_cells) == 0:
        raise SystemError("The given radius is too small!")

    cell_len = 2 * (mesh.hx * mesh.hx + mesh.hy * mesh.hy) ** 0.5  # one cell diagonal length
    ribbon = np.where(dist <= cell_len)[0]
    surv_cells = inner_cells[ribbon]
    surv_dist = dist[ribbon]

    if external_crack:
        # vertices that are outside the ellipse
        vertices_out = dist_vertx[mesh.Connectivity] >= 0

        # cells with all four vertices outside
        log_and_out = np.logical_and(np.logical_and(vertices_out[:, 0], vertices_out[:, 1]),
                                     np.logical_and(vertices_out[:, 2], vertices_out[:, 3]))

        outer_cells = np.where(log_and_out)[0]
        dist_outer = -r + ((mesh.CenterCoor[outer_cells, 0] - center[0]) ** 2
                    + (mesh.CenterCoor[outer_cells, 1] - center[1]) ** 2) ** 0.5

        # mesh.domainLimits[ bottom top left right ]
        if mesh.domainLimits[0] > center[1] -r : #bottom
            raise SystemError("The given circle lies outside of the mesh")
        if mesh.domainLimits[1] < center[1] +r : #top
            raise SystemError("The given circle lies outside of the mesh")
        if mesh.domainLimits[2] > center[0] -r : #left
            raise SystemError("The given circle lies outside of the mesh")
        if mesh.domainLimits[3] < center[0] +r : #right
            raise SystemError("The given circle lies outside of the mesh")

        cell_len = 2 * (mesh.hx * mesh.hx + mesh.hy * mesh.hy) ** 0.5  # one cell diagonal length
        ribbon = np.where(dist_outer <= cell_len)[0]
        surv_cells = outer_cells[ribbon]
        surv_dist = dist_outer[ribbon]

        # from utility import plot_as_matrix
        # K = np.zeros((mesh.NumberOfElts,), )
        # K[surv_cells] = surv_dist
        # plot_as_matrix(K, mesh)
    return surv_cells, surv_dist, inner_cells

# ----------------------------------------------------------------------------------------------------------------------

def get_rectangular_survey_cells(mesh, length, height, center=None):
    """
    This function would provide the ribbon of cells on the inside of the perimeter of a rectangle with the given
    lengths and height. A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object):        -- a CartesianMesh class object describing the grid.
        length (float):                     -- the half length of the rectangle.
        height (float):                     -- the height of the rectangle.
        inj_point (list or ndarray):        -- the coordinates [x, y] of the injection point.

    Returns:
        - surv_cells (ndarray)              -- the list of cells on the inside of the perimeter of the given rectangle.
        - surv_dist (ndarray)               -- the list of corresponding distances of the surv_cells to the fracture\
                                               tip.
        - inner_cells (ndarray)             -- the list of cells inside the given ellipse.
    """

    if center is None:
        center = np.asarray([0, 0])

    inner_cells = np.intersect1d(np.where(abs(mesh.CenterCoor[np.ix_(np.arange(0, len(mesh.CenterCoor)), [0])]
                                              - center[0]) < length)[0],
                                 np.where(abs(mesh.CenterCoor[np.ix_(np.arange(0, len(mesh.CenterCoor)), [1])]
                                              - center[1]) < height / 2)[0])
    max_x = max(mesh.CenterCoor[inner_cells, 0])
    min_x = min(mesh.CenterCoor[inner_cells, 0])
    max_y = max(mesh.CenterCoor[inner_cells, 1])
    min_y = min(mesh.CenterCoor[inner_cells, 1])
    ribbon_max_x = np.where(abs(mesh.CenterCoor[np.ix_(inner_cells, [0])] - max_x) < 100 * sys.float_info.epsilon)[0]
    ribbon_min_x = np.where(abs(mesh.CenterCoor[np.ix_(inner_cells, [0])] - min_x) < 100 * sys.float_info.epsilon)[0]
    ribbon_max_y = np.where(abs(mesh.CenterCoor[np.ix_(inner_cells, [1])] - max_y) < 100 * sys.float_info.epsilon)[0]
    ribbon_min_y = np.where(abs(mesh.CenterCoor[np.ix_(inner_cells, [1])] - min_y) < 100 * sys.float_info.epsilon)[0]

    surv_cells = np.append(inner_cells[ribbon_max_x], inner_cells[ribbon_max_y])
    surv_cells = np.append(surv_cells, inner_cells[ribbon_min_x])
    surv_cells = np.append(surv_cells, inner_cells[ribbon_min_y])
    surv_cells = np.unique(surv_cells)

    surv_dist = np.zeros((len(surv_cells),), dtype=np.float64)

    for i in range(len(surv_cells)):
        surv_dist[i] = np.min([length - float(abs(mesh.CenterCoor[surv_cells[i], 0] - center[0])),
                              height / 2 - float(abs(mesh.CenterCoor[surv_cells[i], 1] - center[1]))])

    if len(inner_cells) == 0:
        raise SystemError("The given rectangular region is too small compared to the mesh!")

    return surv_cells, surv_dist, inner_cells

# ----------------------------------------------------------------------------------------------------------------------
def reduce_based_on_interval(edges_, coord_pt1, coord_pt2 ,coords_):
    """
      Based on the coord, one can exclude any intersection between the segment defined by pt1 and pt2 and the
      vertical (or horizhontal) segment

      o-----|------|----|----------> coord
         coord    pt1  pt2
                       pt2
            o         /
            |        /
            |       /
            |     pt1
            o

    """
    to_keep = []
    for i in range(len(coords_)):
        coord = coords_[i]
        if ((coord - coord_pt1) * (coord - coord_pt2)) < 0.:
            to_keep.append(i)
    return edges_[to_keep]

def get_intersections(mesh_new,Ffront_old):
    """
    :param mesh_new:
    :param Ffront_old:
    :return:
    """
    Ffront_new = []
    EltTip = []

    edges_int = []
    x_int_tot = []
    y_int_tot = []
    h_or_v = []

    for segment in Ffront_old:

        # get the coordinates of the extremes of the points
        x1 = segment[0]; x2 = segment[2]
        y1 = segment[1]; y2 = segment[3]

        # get the distance between the two points
        Lseg = np.sqrt((x1-x2)**2 + (y1-y2)**2)**(0.5)

        # get a band of cells where the old front is passing
        dist_max = np.maximum(1.2 * Lseg, mesh_new.cellDiag)
        elem_around_1st_vertex = mesh_new.get_cells_inside_circle(dist_max, [segment[0], segment[1]])
        elem_around_2nd_vertex = mesh_new.get_cells_inside_circle(dist_max, [segment[2], segment[3]])

        # take the elements in the intersection of the two circles
        elems = np.unique(elem_around_1st_vertex + elem_around_2nd_vertex)

        ## plot only for debugging ##
        #fig = plot_cell_lists(mesh_new, elems, fig=None, mycolor='g', mymarker="_", shiftx=0.01, shifty=0.01, annotate_cellName=False, grid=True)
        #fig = plot_just_xy_points([x1,x2], [y1,y2], fig, joinPoints=True, color='red')
        ## ----------------------- ##

        # The segment has no intersection with the new mesh. So we simply jump to the next segment
        if len(elems) == 0:
            continue
        # take the list unique vertexes of these elements
        vertexes = np.unique(mesh_new.Connectivity[elems].flatten())

        # take the list of unique horizhontal edges that might be intersected
        # connNodesEdges is [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
        edges = mesh_new.Connectivitynodesedges[vertexes]
        edges_v = (edges[:,[0,2]]).flatten()
        edges_h = (edges[:,[1,3]]).flatten()

        # take the list of unique edges that might be intersected
        edges_v = np.unique(edges_v)
        edges_h = np.unique(edges_h)

        # take the x coord of the vertical edges and the y coord of the horizontal edges
        x_v = mesh_new.VertexCoor[mesh_new.Connectivityedgesnodes[edges_v,0]][:,0]
        y_h = mesh_new.VertexCoor[mesh_new.Connectivityedgesnodes[edges_h,0]][:,1]

        # reduce the list of edges based on the fact that pt1 and pt2 can not be on the same side of one edge
        edges_v = reduce_based_on_interval(edges_v, x1, x2, x_v)
        edges_h = reduce_based_on_interval(edges_h, y1, y2, y_h)

        # take the x coord of the vertical edges and the y coord of the horizontal edges
        x_v = mesh_new.VertexCoor[mesh_new.Connectivityedgesnodes[edges_v,0]][:,0]
        y_h = mesh_new.VertexCoor[mesh_new.Connectivityedgesnodes[edges_h,0]][:,1]

        # get intersections between the edges and the segments, even if those are on the on the edge prolongation
        if len(edges_h) > 0 :
            #   - intersections with the horizontal edge
            if y2-y1 != 0. :
                alpha = (y_h - y1) * Lseg / (y2-y1)
                x_int = alpha * (x2 - x1) / Lseg + x1
            else:
                x_int = None

            # check if x_int lies in the range of the edge
            e = 0 ; indx_to_keep = []
            for edge in edges_h:
                A = mesh_new.Connectivityedgesnodes[edge][0]
                B = mesh_new.Connectivityedgesnodes[edge][1]
                xA = mesh_new.VertexCoor[A,0]
                xB = mesh_new.VertexCoor[B,0]
                if (x_int[e] - xA)*(x_int[e] - xB)<=0:
                    indx_to_keep.append(e)
                e = e + 1
            edges_h = edges_h[indx_to_keep]
            x_int = x_int[indx_to_keep]
            y_h = y_h[indx_to_keep]
            ## plot only for debugging ##
            #fig = plot_just_xy_points(x_int, y_h, fig, joinPoints=True, color='blue')
            ## ----------------------- ##
        else: x_int = []

        if len(edges_v) > 0 :
            #   - intersections with the vertical edge
            if x2-x1 != 0. :
                alpha = (x_v - x1) * Lseg / (x2-x1)
                y_int = alpha * (y2 - y1) / Lseg + y1
            else:
                y_int = None

            # check if y_int lies in the range of the edge
            e = 0 ; indx_to_keep = []
            for edge in edges_v:
                A = mesh_new.Connectivityedgesnodes[edge][0]
                B = mesh_new.Connectivityedgesnodes[edge][1]
                yA = mesh_new.VertexCoor[A,1]
                yB = mesh_new.VertexCoor[B,1]
                if (y_int[e] - yA)*(y_int[e] - yB)<=0:
                    indx_to_keep.append(e)
                e = e + 1
            edges_v = edges_v[indx_to_keep]
            y_int = y_int[indx_to_keep]
            x_v = x_v[indx_to_keep]
            ## plot only for debugging ##
            #fig = plot_just_xy_points(x_v, y_int, fig, joinPoints=True, color='blue')
            ## ----------------------- ##
        else:
            y_int = []

        # store all info
        if len(x_v)>0:
            edges_int = edges_int + edges_v.tolist()
            x_int_tot = x_int_tot + x_v.tolist()
            y_int_tot = y_int_tot + y_int.tolist()
            h_or_v = h_or_v + [1] * len(edges_v)
        if len(y_h)>0:
            edges_int = edges_int + edges_h.tolist()
            x_int_tot = x_int_tot + x_int.tolist()
            y_int_tot = y_int_tot + y_h.tolist()
            h_or_v = h_or_v + [0] * len(edges_h)


    ## plot only for debugging ##
    #fig = plot_cell_lists(mesh_new, [], fig=None, mycolor='g', mymarker="_", shiftx=0.01, shifty=0.01, annotate_cellName=False, grid=True)
    #x, y = get_xy_from_Ffront(Ffront_old)
    #fig = plot_just_xy_points(x, y, fig, joinPoints=True, color='red')
    #fig = plot_just_xy_points(x_int_tot, y_int_tot, fig, joinPoints=False, color='blue')
    ## ----------------------- ##

    # all arguments that can be returned:
    #       return edges_int, x_int_tot, y_int_tot, h_or_v

    return x_int_tot, y_int_tot

def get_nodes_to_fictitius_cell(mesh_new, x, y):
    fictitius_cells = []
    for i in range(len(x)):
        # find the minimum distance
        cell_name = np.argmin( ((mesh_new.CenterCoor[:,0]-x[i])**2. + (mesh_new.CenterCoor[:,1]-y[i])**2.)**0.5 ).tolist()
        xmin = mesh_new.CenterCoor[cell_name, 0]
        ymin = mesh_new.CenterCoor[cell_name, 1]

        neighborhood_ID = get_cells_in_neighborhood(cell_name, mesh_new)
        """
                                             0 1 2 3 4 5 6 7 8   
        you are in cell i and get the cells [a,b,c,d,e,f,g,h,i]
          _   _   _   _   _
        | _ | _ | _ | _ | _ |
        | _ | e | a | f | _ |
        | _ | _ | _ | _ | _ |
        | _ | d | i | b | _ |
        | _ | _ | _ | _ | _ |
        | _ | h | c | g | _ |
        | _ | _ | _ | _ | _ |
        """

        fictitius_cell = None
        # define the fictitius_cell
        if   (xmin - x[i] > 0) and (ymin - y[i] > 0):
            """
             ( )      ( )
                center
             (*)      ( )
            """
            fictitius_cell = neighborhood_ID[7] # h
        elif (xmin - x[i] > 0) and (ymin - y[i] < 0):
            """
             (*)      ( )
                center
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[3]  # d
        elif (xmin - x[i] < 0) and (ymin - y[i] > 0):
            """
             ( )      ( )
                center
             ( )      (*)
            """
            fictitius_cell = neighborhood_ID[2]  # c
        elif (xmin - x[i] < 0) and (ymin - y[i] < 0):
            """
             ( )      (*)
                center
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[8]  # i
        elif (xmin - x[i] > 0) and (ymin - y[i] == 0):
            """
             ( )      ( )
             (*)center
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[7]  # h
        elif (xmin - x[i] < 0) and (ymin - y[i] == 0):
            """
             ( )      ( )
                center(*)
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[2]  # c
        elif (xmin - x[i] == 0) and (ymin - y[i] > 0):
            """
             ( )      ( )
                center
             ( ) (*)  ( )
            """
            fictitius_cell = neighborhood_ID[7]  # h
        elif (xmin - x[i] == 0) and (ymin - y[i] < 0):
            """
             ( ) (*)  ( )
                center
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[3]  # d
        elif (xmin - x[i] == 0) and (ymin - y[i] == 0):
            """
             ( )      ( )
                c(*)r
             ( )      ( )
            """
            fictitius_cell = neighborhood_ID[8]  # i
        else:
            SystemExit("value not allowed")

        # append the cell
        if fictitius_cell is not None: fictitius_cells.append(fictitius_cell)

    ## plot only for debugging ##
    # fig = plot_cell_lists(mesh_new, [], fig=None, mycolor='g', mymarker="_", shiftx=0.01, shifty=0.01,
    #                       annotate_cellName=False, grid=True)
    # fig = plot_just_xy_points(x, y, fig, joinPoints=False, color='red')
    #
    # fig = plot_just_xy_points(mesh_new.CenterCoor[np.asarray(fictitius_cells),0],
    #                           mesh_new.CenterCoor[np.asarray(fictitius_cells),1], fig, joinPoints=False, color='blue')
    ## ----------------------- ##

    return fictitius_cells

class Bilinear_int():
  def __init__(self, mesh_new, x, y, fictitius_cells_names):
      self.x_ = x
      self.y_ = y
      self.fictitius_cells_names_ = fictitius_cells_names
      self.mesh_new_ = mesh_new

      # check if the number of equations is sufficient
      self.nrow = len(x)
      self.unique_fc = np.unique(fictitius_cells_names)
      LS_unknowns = []
      for fc in self.unique_fc:
          """
              0 1 2 3 4 5 6 7 8   
             [a,b,c,d,e,f,g,h,i]
             take [8, 1, 5, 0]
          """
          [a,b,c,d,e,f,g,h,i] = get_cells_in_neighborhood(fc, mesh_new)
          LS_unknowns = LS_unknowns + [i, b, f, a]
      del a,b,c,d,e,f,g,h,i
      LS_unknowns = np.unique(LS_unknowns)
      self.LS_unknowns = LS_unknowns
      self.ncol = len(LS_unknowns)
      self.map_IDtoDOF = dict(zip(LS_unknowns, range(self.ncol)))
      self.checkdimension()

      # initializing the matrix
      data, row_ind, col_ind = self.fillM()
      self.M_ = csc_matrix((data, (row_ind, col_ind)), shape = (self.nrow, self.ncol))

  def fillM(self):
      dA = self.mesh_new_.hx * self.mesh_new_.hy
      data = []
      row_ind = []
      col_ind = []
      for row in range(self.nrow):
          x = self.x_[row]
          y = self.y_[row]
          """
                   _   _ (x2,y2)   
                 | a | f |  
                 | _ | _ | 
                 | i | b |  
            --(y1,x1)--------------> x
          """
          fc_name = self.fictitius_cells_names_[row]
          [a, b, c, d, e, f, g, h, i] = get_cells_in_neighborhood(fc_name, self.mesh_new_)
          i_dof = self.map_IDtoDOF[i]
          b_dof = self.map_IDtoDOF[b]
          f_dof = self.map_IDtoDOF[f]
          a_dof = self.map_IDtoDOF[a]
          x_1 = self.mesh_new_.CenterCoor[i,0]
          y_1 = self.mesh_new_.CenterCoor[i,1]
          x_2 = self.mesh_new_.CenterCoor[f,0]
          y_2 = self.mesh_new_.CenterCoor[f,1]
          """
          Writing the value of the LS at one point as a wheighted mean:
            LS(x,y) = w11*LS_11 + w12*LS_12 + w21*LS_21 + w22*LS_22
          and with the weights satisfying the following system:
          | 1    1    1    1    |  |w11|      |1 |
          | x1   x1   x2   x2   |  |w21|      |x |
          | y1   y2   y1   y2   |  |w12|  =   |y |
          | x1y1 x1y2 x2y1 x2y2 |  |w22|      |xy|
          referring to the positions on the picture above
          """
          row_ind.append(row)
          col_ind.append(i_dof)
          data.append((x_2 - x) * (y_2 - y) /dA)

          row_ind.append(row)
          col_ind.append(a_dof)
          data.append((x_2 - x) * (y - y_1) /dA)

          row_ind.append(row)
          col_ind.append(b_dof)
          data.append((x - x_1) * (y_2 - y) /dA)

          row_ind.append(row)
          col_ind.append(f_dof)
          data.append((x - x_1) * (y - y_1) /dA)
      return data, row_ind, col_ind


  def checkdimension(self):
      log = logging.getLogger('PyFrac.remeshing.checkdimension')
      if self.nrow < self.ncol:
          log.error("The size of the system does not allow for a well posed least square problem: too few equations")

  def residual(self, x):
      return self.M_.dot(x) # - zeros() (the rhs is an array of 0)

def get_bounds(LS_unknowns, mesh_new, Ffront_old):
    x = np.asarray([mesh_new.CenterCoor[LS_unknowns, 0]])
    y = np.asarray([mesh_new.CenterCoor[LS_unknowns, 1]])
    x_fr_old, y_fr_old = get_xy_from_Ffront(Ffront_old)
    poly = np.column_stack((x_fr_old, y_fr_old))
    answers_in_out = ray_tracing_numpy(x, y, poly)
    upper_bound = []
    lower_bound = []
    x0 = []
    for answer_in in answers_in_out:
        if answer_in:
            upper_bound.append(0.)
            lower_bound.append(-mesh_new.cellDiag)
            x0.append(-mesh_new.cellDiag/2.)
        else:
            upper_bound.append(mesh_new.cellDiag)
            lower_bound.append(0.)
            x0.append(+mesh_new.cellDiag/2.)
    return upper_bound, lower_bound, x0

def get_ribbon_and_channel(mesh_new, Ffront_old):
    # 0) define if each vertex is outside or inside the odl front
    x = np.asarray([mesh_new.VertexCoor[:, 0]])
    y = np.asarray([mesh_new.VertexCoor[:, 1]])
    x_fr_old, y_fr_old = get_xy_from_Ffront(Ffront_old)
    poly = np.column_stack((x_fr_old, y_fr_old))
    answers_in_out = ray_tracing_numpy(x, y, poly)

    tip = []
    channel = []

    #1) get cell type: tip, channel, out
    for i in range(mesh_new.NumberOfElts):
        vertexes = mesh_new.Connectivity[i]
        answers_in_out_el = np.sum(answers_in_out[vertexes])
        if answers_in_out_el == 4:
            channel.append(i)
        elif answers_in_out_el < 4 and answers_in_out_el > 0:
            tip.append(i)

    #2) intersect tip_nei and channel to get ribbon
    ribbon = np.intersect1d(channel, np.unique((mesh_new.NeiElements[tip]).flatten()))

    # CHECK THE SOLUTION
    # from utility import plot_as_matrix
    # K = np.zeros((mesh_new.NumberOfElts,), )
    # K[tip] = 1
    # K[ribbon] = 2
    # plot_as_matrix(K, mesh_new)

    return ribbon, np.asarray(channel)

def generate_footprint_from_Ffront(mesh_new,Ffront_old):
    # the following routine assumes Ffront to be a closed polygon
    # the following routine does not work - yet - with coalescing or fractures that can disappear - yet

    # 1) find the intersections of the old Ffront with the new mesh
    x_new, y_new = get_intersections(mesh_new, Ffront_old)
    x_old, y_old = get_xy_from_Ffront(Ffront_old)

    # 2) join old cells and new cells
    x = x_new + x_old
    y = y_new + y_old

    # 3) get fictitius cells
    # see front reconstruction for its definition
    fictitius_cells_names = get_nodes_to_fictitius_cell(mesh_new, x, y)

    # 4) initialize residual function & check number of equations
    bilinear_int = Bilinear_int(mesh_new, x, y, fictitius_cells_names)

    # 5) check that the number of equations (rows) is sufficient, i.e. > number of rows
    if bilinear_int.nrow < bilinear_int.ncol:
        SystemExit("the number of points describing the fracture front is insufficient to define its location")

    # 6) get bounds (if levelset is greater or smaller than zero)
    upper, lower, x0 = get_bounds(bilinear_int.LS_unknowns, mesh_new, Ffront_old)

    # 7) solve the least square problem
    LS_res = least_squares(bilinear_int.residual, x0=x0, bounds=(lower, upper))
    if LS_res['status'] != 1:
        SystemExit("the solution for the LS was not found")

    # CHECK THE SOLUTION
    # from utility import plot_as_matrix
    # K = np.zeros((mesh_new.NumberOfElts,), )
    # K[bilinear_int.LS_unknowns] = LS_res.x
    # plot_as_matrix(K, mesh_new)

    # 8) define the ribbon cells as all the cells that have all the vertexes inside the front at the last time step
    EltRibbon, EltChannel = get_ribbon_and_channel(mesh_new, Ffront_old)

    # 9) compute the LS everywhere
    #       - Creating a fmm structure to solve the level set
    fmmStruct = fmm(mesh_new)
    fmmStruct.solveFMM((-LS_res.x, bilinear_int.LS_unknowns), EltChannel, mesh_new)
    outside_elts = np.setdiff1d(np.arange(mesh_new.NumberOfElts), EltChannel)
    fmmStruct.solveFMM((LS_res.x, bilinear_int.LS_unknowns), outside_elts, mesh_new)

    #       -  The solution stored in the object is the calculated level set. we need however to change the sign as to have
    #           negative inside and positive outside.
    sgndDist = -fmmStruct.LS
    sgndDist[outside_elts] = -sgndDist[outside_elts]
    sgndDist[bilinear_int.LS_unknowns] = LS_res.x # just to be sure than we are a sign change where we know it

    #       -  We define a front region and a pstv_region needed to construct the front.
    front_region = np.arange(mesh_new.NumberOfElts)
    pstv_region = np.where(sgndDist[front_region] >= - mesh_new.cellDiag)[0]

    # 10) reconstruct the front
    recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = False

    EltTip, \
    listofTIPcellsONLY, \
    l, \
    alpha, \
    CellStatus, \
    newRibbon, \
    zrVertx_k_with_fully_traversed, \
    zrVertx_k_without_fully_traversed, \
    correct_size_of_pstv_region,\
    sgndDist_k_temp, Ffront_new, number_of_fronts, fronts_dictionary = reconstruct_front_continuous(sgndDist,
                                                                  front_region[pstv_region],
                                                                  EltRibbon,
                                                                  EltChannel,
                                                                  mesh_new,
                                                                  recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                                                    oldfront=Ffront_old)
    if not correct_size_of_pstv_region[0]:
        SystemExit("the region where the level set should be known does not allow for front reconstruction")

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac = Integral_over_cell(EltTip,
                                  alpha,
                                  l,
                                  mesh_new,
                                  'A',
                                  projMethod='LS_continuousfront') / mesh_new.EltArea

    return EltChannel, EltTip, np.hstack((EltChannel, EltTip)), \
           EltRibbon,  zrVertx_k_with_fully_traversed,  CellStatus, \
           l,  alpha,  FillFrac,  sgndDist, \
           Ffront_new,  number_of_fronts,  fronts_dictionary

# ----------------------------------------------------------------------------------------------------------------------

def generate_footprint(mesh, surv_cells, inner_region, dist_surv_cells, projMethod):
    """
    This function takes the survey cells and their distances from the front and generate the footprint of a fracture
    using the fast marching method.

    Arguments:
        mesh (CartesianMesh):       -- a CartesianMesh class object describing the grid.
        surv_cells (ndarray):       -- list of survey cells from which the distances from front are provided
        inner_region (ndarray):     -- list of cells enclosed by the survey cells
        dist_surv_cells (ndarray):  -- distances of the provided survey cells from the front

    Returns:
        - EltChannel (ndarray-int)    -- list of cells in the channel region.
        - EltTip (ndarray-int)        -- list of cells in the Tip region.
        - EltCrack (ndarray-int)      -- list of cells in the crack region.
        - EltRibbon (ndarray-int)     -- list of cells in the Ribbon region.
        - ZeroVertex (ndarray-float)  -- Vertex from which the perpendicular is drawn on the front in a cell(can have\
                                         value from 0 to 3, where 0 signify bottom left, 1 signifying bottom right, 2\
                                         signifying top right and 3 signifying top left vertex).
        - CellStatus (ndarray-int)    -- specifies which region each element currently belongs to (0 for Crack, 1 for\
                                         channel, 2 for tip and 3 for ribbon).
        - l (ndarray-float)           -- length of perpendicular on the fracture front (see Pierce 2015, Computation\
                                         Methods Appl. Mech).
        - alpha (ndarray-float)       -- angle prescribed by perpendicular on the fracture front (see Pierce 2015,\
                                         Computation Methods Appl. Mech)
        - FillF (ndarray-float)       -- filling fraction of each tip cell.
        - sgndDist (ndarray-float)    -- signed minimun distance from fracture front of each cell in the domain.
    """

    # Creating a fmm structure to solve the level set
    fmmStruct = fmm(mesh)

    # We define the survey cells as the known elements and solve from there inwards (inside the fracture). To do
    # so, we need a sign change on the level set (positive inside)
    fmmStruct.solveFMM((-dist_surv_cells, surv_cells),
                       np.hstack((np.setdiff1d(np.arange(mesh.NumberOfElts), inner_region), surv_cells)), mesh)

    # We define the survey cells as the known elements and solve from there outwards to the domain boundary.
    toEval = np.hstack((surv_cells, inner_region))
    fmmStruct.solveFMM((dist_surv_cells, surv_cells), toEval, mesh)

    # The solution stored in the object is the calculated level set. we need however to change the sign as to have
    # negative inside and positive outside.
    sgndDist = fmmStruct.LS
    sgndDist[toEval] = -sgndDist[toEval]

    band = np.arange(mesh.NumberOfElts)
    # costruct the front
    if projMethod == 'LS_continousfront':
        correct_size_of_pstv_region = [False, False, False]
        recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = False
        while not correct_size_of_pstv_region[0]:
            EltTip_tmp, \
            listofTIPcellsONLY, \
            l_tmp, \
            alpha_tmp, \
            CellStatus, \
            newRibbon, \
            ZeroVertex_with_fully_traversed, \
            ZeroVertex, \
            correct_size_of_pstv_region,\
            sgndDist_k_temp, Ffront, number_of_fronts, fronts_dictionary = reconstruct_front_continuous(sgndDist,
                                                                           band,
                                                                           surv_cells,
                                                                           inner_region,
                                                                           mesh,
                                                                           recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                           oldfront=None)
            if correct_size_of_pstv_region[1] or correct_size_of_pstv_region[2]:
                raise ValueError('The mesh is to small for the proposed initiation')

            if not correct_size_of_pstv_region[0]:
                raise SystemExit('FRONT RECONSTRUCTION ERROR: it is not possible to initialize the front with the given distances to the front')
        sgndDist = sgndDist_k_temp
        del correct_size_of_pstv_region

    else:
        (EltTip_tmp, l_tmp, alpha_tmp, CSt) = reconstruct_front(sgndDist, band, inner_region, mesh)
        Ffront = 'It will be computed later by the method process_fracture_front()'
        number_of_fronts=None


    # get the filling fraction of the tip cells
    FillFrac_tmp = Integral_over_cell(EltTip_tmp,
                              alpha_tmp,
                              l_tmp,
                              mesh,
                              'A') / mesh.EltArea

    # generate cell lists
    if projMethod == 'LS_continousfront':
        (EltChannel,
         EltTip,
         EltCrack,
         EltRibbon,
         CellStatus,
         fully_traversed) = UpdateListsFromContinuousFrontRec(newRibbon,
                                                         sgndDist,
                                                         inner_region,
                                                         EltTip_tmp,
                                                         listofTIPcellsONLY,
                                                         mesh)
    else:
        (EltChannel,
         EltTip,
         EltCrack,
         EltRibbon,
         ZeroVertex,
         CellStatus,
         fully_traversed) = UpdateLists(inner_region,
                                   EltTip_tmp,
                                   FillFrac_tmp,
                                   sgndDist,
                                   mesh)
        fronts_dictionary = None
        #todo: implement volume control with two different pressures in the fractures in the case of proj_method = 'ILSA_orig'

    # removing fully traversed cells from the tip cells and other lists
    newTip_indices = np.arange(len(EltTip_tmp))[np.in1d(EltTip_tmp, EltTip)]
    l = l_tmp[newTip_indices]
    alpha = alpha_tmp[newTip_indices]
    FillFrac = FillFrac_tmp[newTip_indices]

    if EltChannel.size <= EltRibbon.size:
        raise SystemExit("No channel elements. The initial radius is probably too small!")


    return EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, l, alpha, FillFrac, sgndDist, Ffront, number_of_fronts, fronts_dictionary

#-----------------------------------------------------------------------------------------------------------------------


def get_width_pressure(mesh, EltCrack, EltTip, FillFrac, C, w=None, p=None, volume=None, symmetric=False, useBlockToeplizCompression=False,
                       volumeControlHMAT=False,
                       Eprime=None,
                       boundaryEffect = None,
                       gmres_tol = 1e-12,
                       gmres_maxiter = 1000,
                       prescribe_w_and_pnet = False):
    """
    This function calculates the width and pressure depending on the provided data. If only volume is provided, the
    width is calculated as a static fracture with the given footprint. Else, the pressure or width are calculated
    according to the given elasticity matrix.

    Arguments:
        mesh (CartesianMesh):   -- a CartesianMesh class object describing the grid.
        EltCrack (ndarray):     -- list of cells in the crack region.
        EltTip (ndarray):       -- list of cells in the Tip region.
        FillFrac (ndarray):     -- filling fraction of each tip cell. Used for correction.
        C (ndarray):            -- The elasticity matrix.
        w (ndarray):            -- the provided width for each cell, can be None if not available.
        p (ndarray):            -- the provided pressure for each cell, can be None if not available.
        volume (ndarray):       -- the volume of the fracture, can be None if not available.
        symmetric (bool):       -- if True, the fracture will be considered strictly symmetric and only one quadrant
                                   will be simulated.
        Eprime (float):         -- the plain strain elastic modulus.

    Returns:
        - w_calculated (ndarray)    -- the calculated width.
        - p_calculated (ndarray)    -- the calculated pressure.
    """
    log = logging.getLogger('PyFrac.initialization')

    if prescribe_w_and_pnet:
        tracFromBoundary = None # currently not implemented with prescribed w and pnet
        if not isinstance(p, np.ndarray):
            p_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
            p_calculated[EltCrack] = np.full((EltCrack.size, ), p, dtype=np.float64)
        else:
            p_calculated = p

        if not isinstance(w, np.ndarray):
            w_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
            w_calculated[EltCrack] = np.full((EltCrack.size, ), w, dtype=np.float64)
        else:
            w_calculated = w

        return w_calculated, p_calculated, tracFromBoundary

    else:
        if w is None and p is None and volume is None:
            raise ValueError("At least one of the three variables w, p and volume has to be provided.")

        if p is None:
            p_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        elif not isinstance(p, np.ndarray):
            p_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
            p_calculated[EltCrack] = np.full((EltCrack.size, ), p, dtype=np.float64)
        else:
            p_calculated = p

        if w is None:
            w_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        elif not isinstance(p, np.ndarray):
            w_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
            w_calculated[EltCrack] = np.full((EltCrack.size,), w, dtype=np.float64)
        elif w.size != mesh.NumberOfElts and not w is None:
            raise ValueError("The given width should be an ndarray with the size equal to the ndarrayumber of cells in mesh!")
        else:
            w_calculated = w
            
        if not w is None and not p is None:
            return w_calculated, p_calculated, None

        if symmetric and not useBlockToeplizCompression and not volumeControlHMAT:

            CrackElts_sym = mesh.corresponding[EltCrack]
            CrackElts_sym = np.unique(CrackElts_sym)

            EltTip_sym = mesh.corresponding[EltTip]
            EltTip_sym = np.unique(EltTip_sym)

            FillF_mesh = np.zeros((mesh.NumberOfElts,), )
            FillF_mesh[EltTip] = FillFrac
            FillF_sym = FillF_mesh[mesh.activeSymtrc[EltTip_sym]]
            self_infl = self_influence(mesh, Eprime)

            C_EltTip = np.copy(C[np.ix_(EltTip_sym, EltTip_sym)])  # keeping the tip element entries to restore current tip correction. This is
            # done to avoid copying the full elasticity matrix.

            # filling fraction correction for element in the tip region
            for e in range(len(EltTip_sym)):
                r = FillF_sym[e] - .25
                if r < 0.1:
                    r = 0.1
                ac = (1 - r) / r
                C[EltTip_sym[e], EltTip_sym[e]] += ac * np.pi / 4. * self_infl

            # known p
            if w is None and not p is None:
                w_sym_EltCrack = np.linalg.solve(C[np.ix_(CrackElts_sym, CrackElts_sym)],
                                                 p_calculated[mesh.activeSymtrc[CrackElts_sym]])
                for i in range(len(w_sym_EltCrack)):
                    w_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = w_sym_EltCrack[i]

            # known w
            if w is not None and p is None:
                p_sym_EltCrack = np.dot(C[np.ix_(CrackElts_sym, CrackElts_sym)], w[mesh.activeSymtrc[CrackElts_sym]])
                for i in range(len(p_sym_EltCrack)):
                    p_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = p_sym_EltCrack[i]

            # w and p both unknown
            if w is None and p is None:
                # calculate the width and pressure by considering fracture as a static fracture.
                C_Crack = C[np.ix_(CrackElts_sym, CrackElts_sym)]

                A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
                weights = mesh.volWeights[CrackElts_sym]
                weights = np.concatenate((weights, np.array([0.0])))
                A = np.vstack((A, weights))

                b = np.zeros((len(EltCrack) + 1,), dtype=np.float64)
                b[-1] = volume / mesh.EltArea

                sol = np.linalg.solve(A, b)

                w_calculated[EltCrack] = sol[np.arange(EltCrack.size)]
                p_calculated[EltCrack] = sol[EltCrack.size]

            # recover original C (without filling fraction correction)
            C[np.ix_(EltTip_sym, EltTip_sym)] = C_EltTip

        elif useBlockToeplizCompression:
            C_Crack = C[np.ix_(EltCrack, EltCrack)]
            EltTip_positions = np.where(np.in1d(EltCrack,EltTip))[0]

            # filling fraction correction for element in the tip region
            r = FillFrac - .25
            indx = np.where(np.less(r,0.1))[0]
            r[indx] = 0.1
            ac = (1 - r) / r
            C_Crack[EltTip_positions,EltTip_positions]=C_Crack[EltTip_positions,EltTip_positions] * (1. + ac * np.pi / 4.)

            # known p
            if w is None and not p is None:
                if boundaryEffect is not None:
                    # we must compute the effect of the boundary
                    toll = 10.**(-9)
                    error = 1

                    # compute the trial opening - no boundary effect
                    w_calculated[EltCrack] = np.linalg.solve(C_Crack, p_calculated[EltCrack] )
                    iter = 0
                    while error > toll and iter < 20:
                        #correct the get the rhs correction
                        tracFromBoundary = boundaryEffect.getTraction(w_calculated,EltCrack)
                        # from utility import plot_as_matrix
                        # K = tracFromBoundary
                        # plot_as_matrix(K, mesh)

                        # compute the error of the compl

                        # compute the trial opening - no boundary effect
                        w_calculated[EltCrack] = np.linalg.solve(C_Crack, p_calculated[EltCrack] - tracFromBoundary[EltCrack])

                        #compute the error of the compl
                        error = boundaryEffect.getSystemError(w_calculated, p_calculated, EltCrack)
                        iter = iter + 1
                        print(error)

                else:
                    w_calculated[EltCrack] = np.linalg.solve(C_Crack, p_calculated[EltCrack])

                    # w_calculated[EltCrack] = np.linalg.solve(C[np.ix_(EltCrack, EltCrack)], p_calculated[EltCrack])
                    # # prepare preconditioner
                    # EHL_iLU = spilu(csc_matrix(C._get9stencilC(EltCrack)), drop_tol=0., fill_factor=1)
                    # Aprec = APrec(EHL_iLU)
                    # counter = gmres_counter()  # to obtain the number of iteration and residual
                    # C._set_domain_IDX(EltCrack)
                    # C._set_codomain_IDX(EltCrack)
                    # # sol_GMRES = gmres(C,
                    # #                   p_calculated[EltCrack],
                    # #                   M=Aprec,
                    # #                   atol=10.e-14,
                    # #                   tol=1.e-9,
                    # #                   maxiter=1000,
                    # #                   callback=counter,
                    # #                   restart=1000)
                    #
                    # sol_GMRES = bicgstab(C,
                    #                   p_calculated[EltCrack],
                    #                   M=Aprec,
                    #                   atol=10.e-14,
                    #                   tol=1.e-9,
                    #                   maxiter=1000,
                    #                   callback=counter)
                    # if sol_GMRES[1] > 0:
                    #     log.warning("EHL system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
                    # elif sol_GMRES[1] == 0:
                    #     log.debug(" --> GMRES EHL converged after " + str(counter.niter) + " iter. ")

            # known w
            if not w is None and p is None:
                p_calculated[EltCrack] = np.dot(C_Crack, w[EltCrack])

            # w and p both unknown
            if w is None and p is None:
                # calculate the width and pressure by considering fracture as a static fracture.
                A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
                A = np.vstack((A, np.ones((1, EltCrack.size + 1), dtype=np.float64)))
                A[-1, -1] = 0

                b = np.zeros((len(EltCrack)+1, ), dtype=np.float64)
                b[-1] = volume / mesh.EltArea

                sol = np.linalg.solve(A, b)

                w_calculated[EltCrack] = sol[np.arange(EltCrack.size)]
                p_calculated[EltCrack] = sol[EltCrack.size]

        elif volumeControlHMAT:
            C._set_tipcorr(FillFrac, np.asarray(EltTip))
            C._set_domain_and_codomain_IDX(EltCrack, EltCrack, same_domain_and_codomain = True)

            # known p
            if w is None and not p is None:
                # solving the system using no preconditioner
                rhs = p_calculated[EltCrack]
                counter = iteration_counter(log)  # to obtain the number of iteration and residual
                sol_GMRES = gmres(C, rhs, tol=gmres_tol, maxiter=gmres_maxiter, callback=counter)

                # check convergence
                #todo assess the convergence against the true residual (not the one with respect to the preconditioned rhs)
                if sol_GMRES[1] > 0:
                    log.warning("WARNING: Volume control system did NOT converge after " + str(sol_GMRES[1]) + " iterations!")
                    rel_err = np.linalg.norm(C._matvec(sol_GMRES[0]) - (rhs)) / np.linalg.norm(rhs)
                    log.warning("         error of the solution: " + str(rel_err))
                elif sol_GMRES[1] == 0:
                    rel_err = np.linalg.norm(C._matvec(sol_GMRES[0]) - (rhs)) / np.linalg.norm(rhs)
                    log.debug(
                        " --> GMRES BOUNDARY EFF. converged after " + str(counter.niter) + " iter. & rel err is " + str(
                            rel_err))

                w_calculated[EltCrack] = sol_GMRES[0]
                C.enable_tip_corr = False

            # known w
            if not w is None and p is None:
                raise ValueError("ERROR: case not yet implemented")

            # w and p both unknown
            if w is None and p is None:
                raise ValueError("ERROR: case not yet implemented")
        else:
            C_EltTip = np.copy(C[np.ix_(EltTip, EltTip)])  # keeping the tip element entries to restore current tip correction. This is
                                                  # done to avoid copying the full elasticity matrix.

            # filling fraction correction for element in the tip region
            for e in range(0, len(EltTip)):
                r = FillFrac[e] - .25
                if r < 0.1:
                    r = 0.1
                ac = (1 - r) / r
                C[EltTip[e], EltTip[e]] = C[EltTip[e], EltTip[e]] * (1. + ac * np.pi / 4.)



            if w is None and not p is None:
                w_calculated[EltCrack] = np.linalg.solve(C[np.ix_(EltCrack, EltCrack)], p_calculated[EltCrack])

            if not w is None and p is None:
                p_calculated[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], w[EltCrack])

            # calculate the width and pressure by considering fracture as a static fracture.
            if w is None and p is None:

                C_Crack = C[np.ix_(EltCrack, EltCrack)]

                A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
                A = np.vstack((A, np.ones((1, EltCrack.size + 1), dtype=np.float64)))
                A[-1, -1] = 0

                b = np.zeros((len(EltCrack)+1, ), dtype=np.float64)
                b[-1] = volume / mesh.EltArea

                sol = np.linalg.solve(A, b)

                w_calculated[EltCrack] = sol[np.arange(EltCrack.size)]
                p_calculated[EltCrack] = sol[EltCrack.size]

            # recover original C (without filling fraction correction)
            C[np.ix_(EltTip, EltTip)] = C_EltTip
        if boundaryEffect is None:
            return w_calculated, p_calculated, None
        else:
            return w_calculated, p_calculated, tracFromBoundary


#-----------------------------------------------------------------------------------------------------------------------

def g(a, b, x0, y0, la):
    return pow(a * x0 / (pow(a, 2) + la), 2) + pow(b * y0 / (pow(b, 2) + la), 2) - 1


#-----------------------------------------------------------------------------------------------------------------------

def Distance_ellipse(a, b, x0, y0):
    """
    This function calculates the smallest distance of a point from the given ellipse.

    Arguments:
        a (float):       -- the length of the major axis of the ellipse.
        b (float):       -- the length of the minor axis of the ellipse.
        x0 (float):      -- the x coordinate of the point from which the distance is to be found
        y0 (float):      -- the y coordinate of the point from which the distance is to be found

    Returns:
        D (float):       -- the shortest distance of the point from the ellipse.
    """

    # todo check! written by Weihan
    # a>b ellipse parameters, (x0,y0) is the center of the cell

    x0 = abs(x0)
    y0 = abs(y0)
    if (x0 < 1e-12 and y0 < 1e-12):
        D = b

    elif (x0 <1e-12  and y0 > 0):
        D = abs(y0 - b)

    elif (y0 <1e-12  and x0 > 0):
        if (x0 < (pow(a, 2) - pow(b, 2)) / a):
            # D=b*math.sqrt(1-pow(x0,2)/(pow(a,2)-pow(b,2)))
            xellipse = pow(a, 2) * x0 / (pow(a, 2) - pow(b, 2))
            yellipse = b * math.sqrt(1 - pow(xellipse / a, 2))
            D = math.sqrt(pow(x0 - xellipse, 2) + pow(yellipse, 2))
        else:
            D = abs(x0 - a)

    else:
        lamin = -pow(b, 2) + b * y0
        lamax = -pow(b, 2) + math.sqrt(pow(a * x0, 2) + pow(b * y0, 2))

        while (abs(g(a, b, x0, y0, lamin)) > 1e-6 or abs(g(a, b, x0, y0, lamax)) > 1e-6):
            lanew = (lamin + lamax) / 2

            if (g(a, b, x0, y0, lanew) < 0):
                lamax = lanew
            else:
                lamin = lanew

        la = (lamin + lamax) / 2
        xellipse = pow(a, 2) * x0 / (pow(a, 2) + la)
        yellipse = pow(b, 2) * y0 / (pow(b, 2) + la)
        D = math.sqrt(pow(x0 - xellipse, 2) + pow(y0 - yellipse, 2))

    return D


#-----------------------------------------------------------------------------------------------------------------------

def Distance_square(lx, ly, x, y):
    """
    The shortest distance of a point from a square
    """

    return abs(min([lx-x, lx+x, ly-y, ly+y]))


#-----------------------------------------------------------------------------------------------------------------------

class InitializationParameters:
    """
    This class store the initialization parameters.

    Args:
        geometry (Geometry):        -- Geometry class object describing the geometry of the fracture.
        regime (str):               -- the propagation regime of the fracture. Possible options are the following:

                                        - 'M'     -- radial fracture in viscosity dominated regime.
                                        - 'Mt'    -- radial fracture in viscosity dominated regime with leak-off.
                                        - 'K'     -- radial fracture in toughness dominated regime.
                                        - 'Kt'    -- radial fracture in toughness dominated regime with leak-off.
                                        - 'PKN'   -- PKN fracture.
                                        - 'E_K'   -- elliptical fracture propagating in toughness dominated regime.\
                                                     The solution is equivalent to a particular anisotropic toughness \
                                                     case described in Zia and Lecampion, 2018.
                                        - 'E_E'   -- the elliptical solution with transverse isotropic material \
                                                     properties (see Moukhtari and Lecampion, 2019).
                                        - 'MDR'   -- viscosity dominated solution for turbulent flow. The friction \
                                                     factor is calculated using MDR asymptote (see Zia and Lecampion\
                                                     2019).
        time (float):                   -- the time since the start of injection.
        width (ndarray):                -- the initial width of the fracture. The size should be equal to the number of
                                           elements in the mesh.
        net_pressure (float/ndarray):   -- the initial net pressure of the fracture. It can be either uniform for the static
                                           fracture or an ndarray.
        fracture_volume (float):        -- total initial volume of the fracture.
        tip_velocity (float/ndarray):   -- the velocity of the tip. It can be a float for radial fractures propagating
                                           with steady velocity or an ndarray equal to the size of tip elements list
                                           giving velocity of the corresponding tip elements.
        elasticity_matrix (ndarray):    -- the BEM elasticity matrix. See Zia & Lecampion 2019.

    """
    def __init__(self, geometry=None, regime='M', time=None, width=None, net_pressure=None, fracture_volume=None,
                 tip_velocity=None, elasticity_matrix=None, prescribe_w_and_pnet = False):
        self.geometry = geometry
        self.regime = regime
        self.time = time
        self.width = width
        self.netPressure = net_pressure
        self.fractureVolume = fracture_volume
        self.tipVelocity = tip_velocity
        self.C = elasticity_matrix
        self.prescribe_w_and_pnet = prescribe_w_and_pnet

        self.check_consistency()


    def check_consistency(self):
        """
        This function checks if the given parameters are consistent with each other.
        """
        log = logging.getLogger('PyFrac.InitializationParameters.check_consistency')

        compatible_regimes = {
            'radial': ['M', 'Mt', 'K', 'Kt', 'MDR', 'static', 'static-radial-K'],
            'height contained': ['PKN', 'KGD_K', 'static'],
            'elliptical': ['E_E', 'E_K', 'static'],
            'level set': ['static']
            }

        try:
            if self.regime not in compatible_regimes[self.geometry.shape]:
                err_string = "Initialization is not supported for the given regime and geometrical shape.\nBelow is " \
                             "the list of compatible regimes and shapes (see documentation for description of " \
                             "the regimes):\n\n"
                for keys, values in compatible_regimes.items():
                    err_string = err_string + repr(keys) + ':\t' + repr(values) + '\n'
                raise ValueError(err_string)
        except KeyError:
            err_string = "The given geometrical shape is not supported!\nSee the list below for supported shapes:\n"
            for keys, values in compatible_regimes.items():
                err_string = err_string + repr(keys) + '\n'
            raise ValueError(err_string)

        errors_analytical = {
            'radial': "Either time or radius is to be provided for radial fractures!",
            'height containedPKN': "Either time or length is to be provided for PKN type fractures. The height of the "
                                   "fracture is required in both cases!",
            'height containedKGD_K': "Either time or length is to be provided for toughness dominated KGD type "
                                     "fractures. The height of the fracture is required in both cases!",
            'ellipticalE_K': "Either time or length of minor axis is required to initialize the elliptical "
                             "fracture in toughness dominated regime!",
            'ellipticalE_E': "Either time or minor axis length along with the major to minor axes length ratio (gamma) " 
                             "is to be provided to initialize in transverse isotropic material!",
            }

        errors_static = {
            'radial': "Radius is to be provided for static radial fractures!",
            'height contained': "Length and height are required to initialize height contained fractures!",
            'elliptical': "The length of minor axis and the aspect ratio (Geometry.gamma) is required to initialize the"
                          " static elliptical fracture!",
            'level set': "To initialize according to a level set, the survey cells (Geometry.surveyCells) and their "
                         "distances (Geometry.tipDistances) along with \n the cells enclosed by the survey cells"
                         " (geometry.innerCells) are required!",
        }

        error = False
        # checks for analytical solutions
        if self.regime != 'static':
            if self.time is None:
                if self.geometry.shape == 'radial' and self.geometry.radius is None:
                    raise ValueError(errors_analytical[self.geometry.shape])
                if self.geometry.shape == 'height contained':
                    if self.geometry.fractureLength is None or self.geometry.fractureHeight is None:
                        error = True
                if self.geometry.shape == 'elliptical':
                    if self.regime == 'E_K' and self.geometry.minorAxis is None:
                        error = True
                    if self.regime == 'E_E':
                        if self.geometry.minorAxis is None or self.geometry.gamma is None:
                            error = True
            else:
                if self.geometry.shape == 'height contained':
                    if self.geometry.fractureHeight is None:
                        error = True
                if self.geometry.shape == 'elliptical':
                    if self.regime == 'E_E' and self.geometry.gamma is None:
                        error = True

            if error:
                raise ValueError(errors_analytical[self.geometry.shape + self.regime])

        # checks for static fracture
        else:
            if self.geometry.shape == 'radial' and self.geometry.radius is None:
                error = True
            elif self.geometry.shape == 'height contained':
                if self.geometry.fractureLength is None or self.geometry.fractureHeight is None:
                    error = True
            elif self.geometry.shape == 'elliptical':
                if self.geometry.minorAxis is None or self.geometry.gamma is None :
                    error = True
            elif self.geometry.shape == 'level set':
                if self.geometry.surveyCells is None or self.geometry.tipDistances is None or \
                            self.geometry.innerCells is None:
                    error = True

            if error:
                raise ValueError(errors_static[self.geometry.shape])

            if self.prescribe_w_and_pnet and (self.width is not None and self.netPressure is not None and self.C is None):
                log.warning('You are starting the simulation prescribing initial width and pressure, make sure it is what you want.')
            else:
                if (self.width is None and self.netPressure is None and self.fractureVolume is None) or self.C is None:
                    raise ValueError("The following parameters are required to initialize a static fracture:\n"
                                     "\t\t -- width or net pressure or total volume of the fracture\n"
                                     "\t\t -- the elasticity matrix")


#-----------------------------------------------------------------------------------------------------------------------

class Geometry:
    """
    This class defines the geometry of the fracture to be initialized.

    Args:
        shape (string):             -- string giving the geometrical shape of the fracture. Possible options are:

                                        - 'radial'
                                        - 'height contained'
                                        - 'elliptical'
                                        - 'level set'
        radius (float):             -- the radius of the radial fracture.
        fracture_length (float):    -- the half length of the fracture.
        fracture_height (float):    -- the height of the height contained fracture.
        minor_axis (float):         -- length of minor axis for elliptical fracture shape.
        gamma (float):              -- ratio of the length of the major axis to the minor axis. It should be more than
                                        one.
        survey_cells (ndarray):     -- the cells from which the distances to the fracture tip are provided.
        tip_distances (ndarray):    -- the minimum distances of the corresponding cells provided in the survey_cells to
                                       the tip of the fracture.
        inner_cells (ndarray):      -- the cells enclosed by the cells given in the survey_cells (inclusive). In other
                                       words, the cells inside the fracture.
        center (ndarray):           -- location of the center of the geometry.

    """

    def __init__(self, shape=None, radius=None, fracture_length=None, fracture_height=None, minor_axis=None,
                 gamma=None, survey_cells=None, tip_distances=None, inner_cells=None, center=None):
        self.shape = shape
        self.radius = radius
        self.fractureLength = fracture_length
        self.fractureHeight = fracture_height
        self.minorAxis = minor_axis
        if gamma is not None:
            if gamma < 1.:
                raise ValueError("The aspect ratio (ratio of the length of major axis to the minor axis) should be more"
                                 " than one")
        self.gamma = gamma
        self.surveyCells = survey_cells
        self.tipDistances = tip_distances
        self.innerCells = inner_cells
        self.center = center

# ----------------------------------------------------------------------------------------------------------------------

    def get_length_dimension(self):

        if self.shape == 'radial':
            length = self.radius
        elif self.shape == 'elliptical':
            length = self.minorAxis
        elif self.shape == 'height contained':
            length = self.fractureLength
        return length

# ----------------------------------------------------------------------------------------------------------------------
    def set_length_dimension(self, length):

        if self.shape == 'radial':
            self.radius = length
        elif self.shape == 'elliptical':
            self.minorAxis = length
        elif self.shape == 'height contained':
            self.fractureLength = length

# ----------------------------------------------------------------------------------------------------------------------
    def get_center(self):
        if self.center == None:
            return [0., 0.]
        else:
            return self.center



# ----------------------------------------------------------------------------------------------------------------------

def get_survey_points(geometry, mesh, source_coord=None):
    """
    This function provided the survey cells, corresponding distances to the front and the enclosed cells for the given
    geometry.
    """

    if geometry.center is None:
        center = source_coord
    else:
        center =geometry.center

    if geometry.shape == 'radial':
        if geometry.radius > min(mesh.Lx, mesh.Ly):
            raise ValueError("The radius of the radial fracture is larger than domain!")
        surv_cells, surv_dist, inner_cells = get_radial_survey_cells(mesh,
                                                                    geometry.radius,
                                                                    center)
    elif geometry.shape == 'elliptical':
        a = geometry.minorAxis * geometry.gamma
        if geometry.minorAxis > mesh.Ly or a > mesh.Lx:
            raise ValueError("The axes length of the elliptical fracture is larger than domain!")
        elif geometry.minorAxis < 2 * mesh.hy:
            raise ValueError("The fracture is very small compared to the mesh cell size!")
        surv_cells, surv_dist, inner_cells = get_eliptical_survey_cells(mesh,
                                                                        a,
                                                                        geometry.minorAxis,
                                                                        center)
    elif geometry.shape == 'height contained':
        if geometry.fractureLength > mesh.Lx or geometry.fractureHeight > mesh.Ly:
            raise ValueError("The fracture is larger than domain!")
        elif geometry.fractureLength < 2 * mesh.hx or geometry.fractureHeight < 2 * mesh.hy:
            raise ValueError("The fracture is very small compared to the mesh cell size!")
        surv_cells, surv_dist, inner_cells = get_rectangular_survey_cells(mesh,
                                                                          geometry.fractureLength,
                                                                          geometry.fractureHeight,
                                                                          center)
    elif geometry.shape == 'level set':
        surv_cells = geometry.surveyCells
        surv_dist = geometry.tipDistances
        inner_cells = geometry.innerCells
    else:
        raise ValueError("The given footprint shape is not supported!")

    return surv_cells, surv_dist, inner_cells

