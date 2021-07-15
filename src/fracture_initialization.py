# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Aug 09 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np
import math
import sys
from level_set import SolveFMM, reconstruct_front, UpdateLists
from volume_integral import Integral_over_cell
from symmetry import self_influence
from continuous_front_reconstruction import reconstruct_front_continuous, UpdateListsFromContinuousFrontRec



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

    sgndDist = np.full((mesh.NumberOfElts,), 1e50)
    sgndDist[surv_cells] = -dist_surv_cells

    # rest of the cells outside the survey cell ring
    EltRest = np.setdiff1d(np.arange(mesh.NumberOfElts), inner_region)

    # fast marching to get level set
    SolveFMM(sgndDist,
             surv_cells,
             inner_region,
             mesh,
             EltRest,
             inner_region)

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
                       Eprime=None):
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

    if w is None and p is None and volume is None:
        raise ValueError("Atleast one of the three variables w, p and volume has to be provided.")

    if p is None:
        p_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    elif not isinstance(p, np.ndarray):
        p_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        p_calculated[EltCrack] = np.full((EltCrack.size, ), p, dtype=np.float64)
    else:
        p_calculated = p

    if w is None:
        w_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    elif not isinstance(w, np.ndarray):
        w_calculated = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        w_calculated[EltCrack] = np.full((EltCrack.size, ), w, dtype=np.float64)
    else:
        w_calculated = w

    if not w is None and not p is None:
        return w_calculated, p_calculated

    if symmetric and not useBlockToeplizCompression:

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

        if w is None and not p is None:
            w_sym_EltCrack = np.linalg.solve(C[np.ix_(CrackElts_sym, CrackElts_sym)],
                                             p_calculated[mesh.activeSymtrc[CrackElts_sym]])
            for i in range(len(w_sym_EltCrack)):
                w_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = w_sym_EltCrack[i]

        if w is not None and p is None:
            p_sym_EltCrack = np.dot(C[np.ix_(CrackElts_sym, CrackElts_sym)], w[mesh.activeSymtrc[CrackElts_sym]])
            for i in range(len(p_sym_EltCrack)):
                p_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = p_sym_EltCrack[i]

        # calculate the width and pressure by considering fracture as a static fracture.
        if w is None and p is None:
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

        if w is None and not p is None:
            w_calculated[EltCrack] = np.linalg.solve(C_Crack, p_calculated[EltCrack])

        if not w is None and p is None:
            p_calculated[EltCrack] = np.dot(C_Crack, w[EltCrack])

        # calculate the width and pressure by considering fracture as a static fracture.
        if w is None and p is None:

            A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
            A = np.vstack((A, np.ones((1, EltCrack.size + 1), dtype=np.float64)))
            A[-1, -1] = 0

            b = np.zeros((len(EltCrack)+1, ), dtype=np.float64)
            b[-1] = volume / mesh.EltArea

            sol = np.linalg.solve(A, b)

            w_calculated[EltCrack] = sol[np.arange(EltCrack.size)]
            p_calculated[EltCrack] = sol[EltCrack.size]

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

    return w_calculated, p_calculated


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
                 tip_velocity=None, elasticity_matrix=None):
        self.geometry = geometry
        self.regime = regime
        self.time = time
        self.width = width
        self.netPressure = net_pressure
        self.fractureVolume = fracture_volume
        self.tipVelocity = tip_velocity
        self.C = elasticity_matrix

        self.check_consistency()


    def check_consistency(self):
        """
        This function checks if the given parameters are consistent with each other.
        """

        compatible_regimes = {
            'radial': ['M', 'Mt', 'K', 'Kt', 'MDR', 'static'],
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

