# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Aug 09 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np
import math
from src.Utility import radius_level_set
from src.LevelSet import SolveFMM, reconstruct_front, UpdateLists
from src.VolIntegral import Integral_over_cell


def get_circular_survey_cells(mesh, initRad):
    """
    This function would provide the ribbon of cells on the inside of the perimeter of a circle with the given radius.
    A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object)         -- a CartesianMesh class object describing the grid.
        initRad (float)                     -- the radius of the circle closest to which the survey cells are to be
                                               provided.

    Returns:
        surv_cells (ndarray)                -- the list of cells on the inside of the perimeter of a circle with the
                                               given radius.
        inner_cells (ndarray)               -- the list of cells inside the given circle.
    """

    # level set value at middle of the elements
    phiMid = np.empty([mesh.NumberOfElts, 1], dtype=float)
    for e in range(0, mesh.NumberOfElts):
        phiMid[e] = radius_level_set(mesh.CenterCoor[e], initRad)
    # level set value at vertices of the element
    phiVertices = np.empty([len(mesh.VertexCoor), 1], dtype=float)
    for i in range(0, len(mesh.VertexCoor)):
        phiVertices[i] = radius_level_set(mesh.VertexCoor[i], initRad)
        # finding elements containing at least one vertices inside the fracture, i.e. with a value of the level <0
        # avoiding loop on elements....

    # array of Length (number of elements) containig the sum of vertices with neg level set value)
    psum = np.sum(phiVertices[mesh.Connectivity[:]] < 0, axis=1)
    # indices of tip element which by definition have less than 4 but at least 1 vertices inside the level set
    EltTip = (np.where(np.logical_and(psum > 0, psum < 4)))[0]
    inner_cells = (np.where(psum == 4))[0]  # indices of channel element / fully cracked

    # find the ribbon elements: Channel Elements having at least
    # on common vertices with a Tip element
    #
    # loop on ChannelElement, and on TipElement
    testribbon = np.empty([len(inner_cells), 1], dtype=float)
    for e in range(0, len(inner_cells)):
        for i in range(0, len(EltTip)):
            if (len(np.intersect1d(mesh.Connectivity[inner_cells[e]], mesh.Connectivity[EltTip[i]])) > 0):
                testribbon[e] = 1
                break
            else:
                testribbon[e] = 0
    surv_cells = inner_cells[(np.reshape(testribbon, len(inner_cells)) == 1)]

    return surv_cells, inner_cells

#-----------------------------------------------------------------------------------------------------------------------


def get_eliptical_survey_cells(mesh, a, b):
    """
        This function would provide the ribbon of cells on the inside of the perimeter of an ellipse with the given
        lengths of the major and minor axes. A list of all the cells inside the fracture is also provided.

        Arguments:
            mesh (CartesianMesh object)         -- a CartesianMesh class object describing the grid.
            a (float)                           -- the length of the major axis of the provided ellipse.
            b (float)                           -- the length of the minor axis of the provided ellipse.

        Returns:
            surv_cells (ndarray)                -- the list of cells on the inside of the perimeter of the given
                                                   ellipse.
            inner_cells (ndarray)               -- the list of cells inside the given ellipse.
    """

    # distances of the cell vertices
    dist_vertx = (mesh.VertexCoor[:,0]/a)**2 + (mesh.VertexCoor[:,1]/b)**2 - 1.
    # vertices that are inside the ellipse
    vertices = dist_vertx[mesh.Connectivity]<0

    #cells with all four vertices inside
    log_and = np.logical_and(np.logical_and(vertices[:,0],vertices[:,1]),np.logical_and(vertices[:,2],vertices[:,3]))
    channel = np.where(log_and)[0]

    # todo: Hack !!! returning channel cells as inner cells also
    return channel, channel

#-----------------------------------------------------------------------------------------------------------------------


def generate_footprint(mesh, surv_cells, inner_region, dist_surv_cells):
    """
    This function takes the survey cells and their distances from the front and generate the footprint of a fracture
    using the fast marching method.

    Arguments:
        mesh (CartesianMesh object) -- a CartesianMesh class object describing the grid.
        surv_cells (ndarray)        -- list of survey cells from which the distances from front are provided
        inner_region (ndarray)      -- list of cells enclosed by the survey cells
        dist_surv_cells (ndarray)   -- distances of the provided survey cells from the front

    Returns:
        EltChannel (ndarray-int)    -- list of cells in the channel region.
        EltTip (ndarray-int)        -- list of cells in the Tip region.
        EltCrack (ndarray-int)      -- list of cells in the crack region.
        EltRibbon (ndarray-int)     -- list of cells in the Ribbon region.
        ZeroVertex (ndarray-float)  -- Vertex from which the perpendicular is drawn on the front in a cell(can have
                                       value from 0 to 3, where 0 signify bottom left, 1 signifying bottom right, 2
                                       signifying top right and 3 signifying top left vertex).
        CellStatus (ndarray-int)    -- specifies which region each element currently belongs to (0 for Crack, 1 for
                                       channel, 2 for tip and 3 for ribbon).
        l (ndarray-float)           -- length of perpendicular on the fracture front (see Pierce 2015, Computation
                                       Methods Appl. Mech).
        alpha (ndarray-float)       -- angle prescribed by perpendicular on the fracture front (see Pierce 2015,
                                       Computation Methods Appl. Mech)
        FillF (ndarray-float)       -- filling fraction of each tip cell.
        sgndDist (ndarray-float)    -- signed minimun distance from fracture front of each cell in the domain.
    """

    sgndDist = np.full((mesh.NumberOfElts,), 1e10)
    sgndDist[surv_cells] = -dist_surv_cells

    # rest of the cells outside the survey cell ring
    EltRest = np.arange(mesh.NumberOfElts)
    for i in range(len(inner_region)):
        EltRest = np.delete(EltRest, np.where(EltRest == inner_region[i])[0])

    # fast marching to get level set
    SolveFMM(sgndDist,
             surv_cells,
             inner_region,
             mesh,
             EltRest,
             inner_region)

    # costruct the front
    (EltTip_tmp, l_tmp, alpha_tmp, CSt) = reconstruct_front(sgndDist, inner_region, mesh)

    # get the filling fraction of the tip cells
    FillFrac_tmp = Integral_over_cell(EltTip_tmp,
                              alpha_tmp,
                              l_tmp,
                              mesh,
                              'A') / mesh.EltArea

    # generate cell lists
    (EltChannel,
     EltTip,
     EltCrack,
     EltRibbon,
     ZeroVertex,
     CellStatus) = UpdateLists(inner_region,
                               EltTip_tmp,
                               FillFrac_tmp,
                               sgndDist,
                               mesh)

    # removing fully traversed cells from the tip cells and other lists
    newTip_indices = np.arange(len(EltTip_tmp))[np.in1d(EltTip_tmp, EltTip)]
    l = l_tmp[newTip_indices]
    alpha = alpha_tmp[newTip_indices]
    FillFrac = FillFrac_tmp[newTip_indices]

    if EltChannel.size <= EltRibbon.size:
        raise SystemExit("No channel elements. The initial radius is probably too small")


    return EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, l, alpha, FillFrac, sgndDist

#-----------------------------------------------------------------------------------------------------------------------


def get_width_pressure(mesh, EltCrack, EltTip, FillFrac, C, w=None, p=None, volume=None):
    """
    This function calculates the width and pressure depending on the provided data. If only volume is provided, the
    width is calculated as a static fracture with the given footprint. Else, the pressure or width are calculated
    according to the given elasticity matrix.

    Arguments:
        mesh (CartesianMesh object) -- a CartesianMesh class object describing the grid.
        EltCrack (ndarray-int)      -- list of cells in the crack region.
        EltTip (ndarray-int)        -- list of cells in the Tip region.
        FillFrac (ndarray-float)    -- filling fraction of each tip cell. Used for correction
        C (ndarray-float)           -- The elasticity matrix
        w (ndarray-float)           -- the provided width for each cell, can be None if not available.
        p (ndarray-float)           -- the provided pressure for each cell, can be None if not available.
        volume (ndarray-float)      -- the volume of the fracture, can be None if not available.

    Returns:
        w_calculated (ndarray-float)-- the calculated width
        p_calculated (ndarray-float)-- the calculated pressure
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
    elif w.size != mesh.NumberOfElts and not w is None:
        raise ValueError("The given width should be an ndarray with the size equal to the number of cells in mesh!")
    else:
        w_calculated = w

    if not w is None and not p is None:
        return w_calculated, p_calculated

    C_EltTip = C[0][np.ix_(EltTip, EltTip)]  # keeping the tip element entries to restore current tip correction. This is
                                          # done to avoid copying the full elasticity matrix.

    # filling fraction correction for element in the tip region
    for e in range(0, len(EltTip)):
        r = FillFrac[e] - .25
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r
        C[0][EltTip[e], EltTip[e]] = C[0][EltTip[e], EltTip[e]] * (1. + ac * np.pi / 4.)



    if w is None and not p is None:
        w_calculated[EltCrack] = np.linalg.solve(C[0][np.ix_(EltCrack, EltCrack)], p_calculated[EltCrack])

    if not w is None and p is None:
        p_calculated[EltCrack] = np.dot(C[0][np.ix_(EltCrack, EltCrack)], w[EltCrack])

    # calculate the width and pressure by considering fracture as a static fracture.
    if w is None and p is None:

        C_Crack = C[0][np.ix_(EltCrack, EltCrack)]

        A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
        A = np.vstack((A, np.ones((1, EltCrack.size + 1), dtype=np.float64)))
        A[-1, -1] = 0

        b = np.zeros((len(EltCrack)+1, ), dtype=np.float64)
        b[-1] = volume / mesh.EltArea

        sol = np.linalg.solve(A, b)

        w_calculated[EltCrack] = sol[np.arange(EltCrack.size)]
        p_calculated[EltCrack] = sol[EltCrack.size]

    # recover original C (without filling fraction correction)
    C[0][np.ix_(EltTip, EltTip)] = C_EltTip

    return w_calculated, p_calculated


#-----------------------------------------------------------------------------------------------------------------------

def g(a, b, x0, y0, la):
    return pow(a * x0 / (pow(a, 2) + la), 2) + pow(b * y0 / (pow(b, 2) + la), 2) - 1

def Distance_ellipse(a, b, x0, y0):
    """
    This function calculates the smallest distance of a point from the given ellipse.

    Arguments:
        a (float)       -- the length of the major axis of the ellipse.
        b (float)       -- the length of the minor axis of the ellipse.
        x0 (float)      -- the x coordinate of the point from which the distance is to be found
        y0 (float)      -- the y coordinate of the point from which the distance is to be found

    Returns:
        D (float)       -- the shortest distance of the point from the ellipse.
    """

    # a>b ellipse parameters, (x0,y0) is the center of the cell
    x0 = abs(x0)
    y0 = abs(y0)
    if (x0<1e-12 and y0<1e-12):
        D = b
        xellipse = 0
        yellipse = b

    elif (x0 <1e-12  and y0 > 0):
        D = abs(y0 - b)
        xellipse = 0
        yellipse = b

    elif (y0 <1e-12  and x0 > 0):
        if (x0 < (pow(a, 2) - pow(b, 2)) / a):
            # D=b*math.sqrt(1-pow(x0,2)/(pow(a,2)-pow(b,2)))
            xellipse = pow(a, 2) * x0 / (pow(a, 2) - pow(b, 2))
            yellipse = b * math.sqrt(1 - pow(xellipse / a, 2))
            D = math.sqrt(pow(x0 - xellipse, 2) + pow(yellipse, 2))
        else:
            D = abs(x0 - a)
            xellipse = a
            yellipse = 0

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


def Distance_square(lx, ly, x, y):
    """
    The shortest distance of a point from a square
    """

    return abs(min([lx-x, lx+x, ly-y, ly+y]))
