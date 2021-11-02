# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 19:01:22 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
import logging
from scipy.optimize import fsolve

def SolveFMM(levelSet, EltRibbon, EltChannel, mesh, farAwayPstv, farAwayNgtv):
    """
    solve Eikonal equation to get level set.

    Arguments:
        levelSet (ndarray-float):           -- level set to be evaluated and updated.
        EltRibbon (ndarray-int):            -- cells with given distance from the front.
        EltChannel (ndarray-int):           -- cells enclosed by the given cells
        mesh (CartesianMesh object):        -- mesh object
        farAwayNgtv (ndarray-float):        -- the cells inwards from ribbon cells for which the distance from front
                                               is to be evaluated
        farAwayPstv (ndarray-float):        -- the cells outwards from ribbon cells for which the distance from front
                                               is to be evaluated

    Returns:
        Note:
            Does not return anything. The levelSet is updated in place.
    """
    log = logging.getLogger('PyFrac.SolveFMM')
    # todo: This method is inefficient. It can be implemented with heap for better efficiency

    # for Elements radialy outward from ribbon cells
    Alive = EltRibbon.tolist()
    NarrowBand = EltRibbon.tolist()
    FarAway = np.setdiff1d(farAwayPstv, NarrowBand).tolist()

    Alive_status = np.full((mesh.NumberOfElts, ), False, dtype=bool)
    NarrowBand_status = np.full((mesh.NumberOfElts,), False, dtype=bool)
    FarAway_status = np.full((mesh.NumberOfElts,), False, dtype=bool)
    Alive_status[Alive] = True
    NarrowBand_status[NarrowBand] = True
    FarAway_status[FarAway] = True

    # the maximum distance any point can have from another in the current mesh. This distance is used to detect the
    # cells that are not yet traversed, i.e. having infinity distance

    beta = mesh.hx / mesh.hy
    while len(NarrowBand) > 0:
        Smallest = NarrowBand[levelSet[NarrowBand].argmin()]
        neighbors = mesh.NeiElements[Smallest]

        for neighbor in neighbors:
            if not Alive_status[neighbor]:
                if FarAway_status[neighbor]:
                    NarrowBand.append(neighbor)
                    NarrowBand_status[neighbor] = True
                    FarAway.remove(neighbor)
                    FarAway_status[neighbor] = False

                NeigxMin = min(levelSet[mesh.NeiElements[neighbor, 0]], levelSet[mesh.NeiElements[neighbor, 1]])
                NeigyMin = min(levelSet[mesh.NeiElements[neighbor, 2]], levelSet[mesh.NeiElements[neighbor, 3]])
                if NeigxMin >= 1e50 and NeigyMin >= 1e50 :
                    log.warning("You are trying to compute the level set in a cell where all the neighbours have infinite distance to the front")
                    # A possible fix of this situation could be leave apart the cell and come back later
                    # remember that as soon as one neighbour has non infinite level set we can solve the LS via fast macing method
                delT = NeigyMin - NeigxMin

                theta_sq = mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delT ** 2
                if theta_sq > 0:
                    levelSet[neighbor] = (NeigxMin + beta ** 2 * NeigyMin + theta_sq**0.5) / (1 + beta ** 2)
                else:  # the distance is to be taken from the horizontal or vertical neighbouring cell as it is the only
                       # distance available
                    levelSet[neighbor] = min(NeigyMin + mesh.hy, NeigxMin + mesh.hx)

        Alive.append(Smallest)
        Alive_status[Smallest] = True
        NarrowBand.remove(Smallest)
        NarrowBand_status[Smallest] = False

    # todo !!! hack - find out why this is required
    if (levelSet[farAwayPstv] >= 1e50).any():
        unevaluated = np.where(levelSet[farAwayPstv] >= 1e50)[0]

        for i in range(len(unevaluated)):
            neighbors = mesh.NeiElements[farAwayPstv[unevaluated[i]]]
            Eikargs = (levelSet[neighbors[0]], levelSet[neighbors[1]], levelSet[neighbors[2]],
                       levelSet[neighbors[3]], 1, mesh.hx, mesh.hy)  # arguments for the Eikonal equation function
            guess = np.max(levelSet[neighbors])  # initial starting guess for the numerical solver
            levelSet[farAwayPstv[unevaluated[i]]] = fsolve(Eikonal_Res, guess, args=Eikargs)  # numerical solver


    # for elements radialy inward from ribbon cells. The sign of the level set values(tip asymptote) in the ribbon cells
    # is inverted to run the fast marching algorithm. The sign is finally inverted back to assign the value in the level
    # set to be returned.
    if len(farAwayNgtv) > 0:
        RibbonInwardElts = np.setdiff1d(EltChannel, EltRibbon)
        positive_levelSet = 1e50 * np.ones((mesh.NumberOfElts,), np.float64)
        positive_levelSet[EltRibbon] = -levelSet[EltRibbon]
        Alive = EltRibbon.tolist()
        NarrowBand = EltRibbon.tolist()
        FarAway = np.setdiff1d(farAwayNgtv, NarrowBand).tolist()

        Alive_status = np.full((mesh.NumberOfElts,), False, dtype=bool)
        NarrowBand_status = np.full((mesh.NumberOfElts,), False, dtype=bool)
        FarAway_status = np.full((mesh.NumberOfElts,), False, dtype=bool)
        Alive_status[Alive] = True
        NarrowBand_status[NarrowBand] = True
        FarAway_status[FarAway] = True

        while len(NarrowBand) > 0:

            Smallest = NarrowBand[positive_levelSet[NarrowBand].argmin()]
            neighbors = mesh.NeiElements[Smallest]

            for neighbor in neighbors:
                if not Alive_status[neighbor]:
                    if FarAway_status[neighbor]:
                        NarrowBand.append(neighbor)
                        NarrowBand_status[neighbor] = True
                        FarAway.remove(neighbor)
                        FarAway_status[neighbor] = False

                    NeigxMin = min(positive_levelSet[mesh.NeiElements[neighbor, 0]],
                                   positive_levelSet[mesh.NeiElements[neighbor, 1]])
                    NeigyMin = min(positive_levelSet[mesh.NeiElements[neighbor, 2]],
                                   positive_levelSet[mesh.NeiElements[neighbor, 3]])
                    if NeigxMin >= 1e50 and NeigyMin >= 1e50:
                        log.warning(
                            "You are trying to compute the level set in a cell where all the neighbours have infinite distance to the front")
                        # A possible fix of this situation could be leave apart the cell and come back later
                        # remember that as soon as one neighbour has non infinite level set we can solve the LS via fast macing method
                    beta = mesh.hx / mesh.hy
                    delT = NeigyMin - NeigxMin
                    theta_sq = mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delT ** 2
                    if theta_sq > 0:
                        positive_levelSet[neighbor] = (NeigxMin + beta ** 2 * NeigyMin + theta_sq**0.5) / (
                                                                                                1 + beta ** 2)
                    else:   # the distance is to be taken from the horizontal or vertical neighbouring cell as it is the
                            # only distance available
                        positive_levelSet[neighbor] = min(NeigyMin + mesh.hy, NeigxMin + mesh.hx)

            Alive.append(Smallest)
            Alive_status[Smallest] = True
            NarrowBand.remove(Smallest)
            NarrowBand_status[Smallest] = False

        # assigning adjusted value to the level set to be returned
        levelSet[RibbonInwardElts] = -positive_levelSet[RibbonInwardElts]

    # todo !!! hack - find out why this is required
    if (abs(levelSet[farAwayNgtv]) >= 1e50).any():
        unevaluated = np.where(abs(levelSet[farAwayNgtv]) >= 1e50)[0]

        for i in range(len(unevaluated)):
            neighbors = mesh.NeiElements[farAwayNgtv[unevaluated[i]]]
            Eikargs = (levelSet[neighbors[0]], levelSet[neighbors[1]], levelSet[neighbors[2]],
                       levelSet[neighbors[3]], 1, mesh.hx, mesh.hy)  # arguments for the eikonal equation function
            guess = np.max(levelSet[neighbors])  # initial starting guess for the numerical solver
            levelSet[farAwayNgtv[unevaluated[i]]] = fsolve(Eikonal_Res, guess, args=Eikargs)  # numerical solver
# from visualization import plot_fracture_variable_as_image
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# A = np.full(mesh.NumberOfElts, -1.)
# A[farAwayPstv] = levelSet[farAwayPstv]
# A[farAwayNgtv] = levelSet[farAwayNgtv]
# A[EltRibbon] = levelSet[EltRibbon]
#
# for i in range(mesh.NumberOfElts):
#     if A[i] < -10000 or A[i] > 10000:
#         A[i] = -1
# fig = plot_fracture_variable_as_image(A, mesh, fig=fig)
# ax = fig.get_axes()[0]
# x_center = mesh.CenterCoor[Alive, 0]
# y_center = mesh.CenterCoor[Alive, 1]
# for i, txt in enumerate(Alive):
#     ax.annotate(txt, (x_center[i], y_center[i]))

def Eikonal_Res(Tij, *args):
    """quadratic Eikonal equation residual to be used by numerical root finder"""

    (Tleft, Tright, Tbottom, Ttop, Fij, dx, dy) = args
    return np.nanmax([(Tij - Tleft) / dx, 0]) ** 2 + np.nanmin([(Tright - Tij) / dx, 0]) ** 2 + np.nanmax(
        [(Tij - Tbottom) / dy, 0]) ** 2 + \
           np.nanmin([(Ttop - Tij) / dy, 0]) ** 2 - Fij ** 2


# -----------------------------------------------------------------------------------------------------------------------
