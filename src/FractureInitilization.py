# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Aug 09 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np
import math
import skfmm
from src.HFAnalyticalSolutions import shift_injection_point
from src.LevelSet import SolveFMM, reconstruct_front, UpdateLists
from src.VolIntegral import Integral_over_cell
from src.Symmetry import *


def get_eliptical_survey_cells(mesh, a, b):
    """
    This function would provide the ribbon of cells on the inside of the perimeter of an ellipse with the given
    lengths of the major and minor axes. A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object):        -- a CartesianMesh class object describing the grid.
        a (float):                          -- the length of the major axis of the provided ellipse.
        b (float):                          -- the length of the minor axis of the provided ellipse.

    Returns:
        - surv_cells (ndarray)              -- the list of cells on the inside of the perimeter of the given\
                                               ellipse.
        - inner_cells (ndarray)             -- the list of cells inside the given ellipse.
    """

    # distances of the cell vertices
    dist_vertx = ((mesh.VertexCoor[:, 0])/ a) ** 2 + ((mesh.VertexCoor[:, 1]) / b) ** 2 - 1.
    # vertices that are inside the ellipse
    vertices = dist_vertx[mesh.Connectivity] < 0

    #cells with all four vertices inside
    log_and = np.logical_and(np.logical_and(vertices[:, 0], vertices[:, 1]),
                             np.logical_and(vertices[:, 2],vertices[:, 3]))

    inner_cells = np.where(log_and)[0]
    # todo: Hack !!! returning all inner cells as survey cells also
    surv_cells = np.copy(inner_cells)

    return surv_cells, inner_cells

#-----------------------------------------------------------------------------------------------------------------------


def generate_footprint(mesh, surv_cells, inner_region, dist_surv_cells, inj_point):
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
    (EltTip_tmp, l_tmp, alpha_tmp, CSt) = reconstruct_front(sgndDist, band, inner_region, mesh)

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
        raise SystemExit("No channel elements. The initial radius is probably too small!")


    return EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, l, alpha, FillFrac, sgndDist

#-----------------------------------------------------------------------------------------------------------------------


def get_width_pressure(mesh, EltCrack, EltTip, FillFrac, C, w=None, p=None, volume=None, symmetric=False,
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
    elif w.size != mesh.NumberOfElts and not w is None:
        raise ValueError("The given width should be an ndarray with the size equal to the number of cells in mesh!")
    else:
        w_calculated = w

    if not w is None and not p is None:
        return w_calculated, p_calculated

    if symmetric:

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

    # a>b ellipse parameters, (x0,y0) is the center of the cell
    x0 = abs(x0)
    y0 = abs(y0)
    if (x0 < 1e-12 and y0 < 1e-12):
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
