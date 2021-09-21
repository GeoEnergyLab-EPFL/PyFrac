# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 19:01:22 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
import numpy as np
import logging
import warnings
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
#-----------------------------------------------------------------------------------------------------------------------


def reconstruct_front(dist, bandElts, EltChannel, mesh):
    """
    Track the fracture front, the length of the perpendicular drawn on the fracture and the angle inscribed by the
    perpendicular. The angle is calculated using the formulation given by Pierce and Detournay 2008.
    
    Arguments:
        dist (ndarray):         -- the signed distance of the cells from the fracture front.
        bandElts (ndarray):     -- the band of elements to which the search is limited.
        EltChannel (ndarray):   -- list of Channel elements.
        mesh (CartesianMesh):   -- the mesh of the fracture.
    """

    # Elements that are not in channel
    EltRest = np.setdiff1d(bandElts, EltChannel)
    ElmntTip = np.asarray([], int)
    l = np.asarray([])
    alpha = np.asarray([])

    for i in range(0, len(EltRest)):
        neighbors = mesh.NeiElements[EltRest[i]]

        minx = min(dist[neighbors[0]], dist[neighbors[1]])
        miny = min(dist[neighbors[2]], dist[neighbors[3]])
        # distance of the vertex (zero vertex, i.e. rotated distance) of the current cell from the front
        Pdis = -(minx + miny) / 2

        # if the vertex distance is positive, meaning the fracture has passed the vertex
        if Pdis >= 0:
            ElmntTip = np.append(ElmntTip, EltRest[i])
            l = np.append(l, Pdis)

            # calculate angle imposed by the perpendicular on front (see Peirce & Detournay 2008)
            delDist = miny - minx
            beta = mesh.hx / mesh.hy
            theta = (mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delDist ** 2) ** 0.5
            # angle calculate with inverse of cosine trigonometric function
            a1 = np.arccos((theta + beta ** 2 * delDist) / (mesh.hx * (1 + beta ** 2)))
            # angle calculate with inverse of sine trigonometric function
            sinalpha = beta * (theta - delDist) / (mesh.hx * (1 + beta ** 2))
            a2 = np.arcsin(sinalpha)

            # !!!Hack. this check of zero or 90 degree angle works better
            warnings.filterwarnings("ignore")
            if abs(1 - dist[neighbors[0]] / dist[neighbors[1]]) < 1e-5:
                a2 = np.pi / 2
            elif abs(1 - dist[neighbors[2]] / dist[neighbors[3]]) < 1e-5:
                a2 = 0.

            #todo hack!!!
            # checks to remove numerical noise in angle calculation
            if a2 >= 0 and a2 <= np.pi / 2:
                alpha = np.append(alpha, a2)
            elif a1 >= 0 and a1 <= np.pi / 2:
                alpha = np.append(alpha, a1)
            elif a2 < 0 and a2 > -1e-6:
                alpha = np.append(alpha, 0)
            elif a2 > np.pi / 2 and a2 < np.pi / 2 + 1e-6:
                alpha = np.append(alpha, np.pi / 2)
            elif a1 < 0 and a1 > -1e-6:
                alpha = np.append(alpha, 0)
            elif a1 > np.pi / 2 and a1 < np.pi / 2 + 1e-6:
                alpha = np.append(alpha, np.pi / 2)
            else:
                if abs(1 - dist[neighbors[0]] / dist[neighbors[1]]) < 0.1:
                    alpha = np.append(alpha, np.pi / 2)
                elif abs(1 - dist[neighbors[2]] / dist[neighbors[3]]) < 0.1:
                    alpha = np.append(alpha, 0)
                else:
                    alpha = np.append(alpha, np.nan)
    # from utility import plot_as_matrix
    # K = np.zeros((mesh.NumberOfElts,), )
    # K[ElmntTip] = alpha
    # plot_as_matrix(K, mesh)
    nan = np.where(np.isnan(alpha))[0]
    if len(nan) > 0:
        alpha_mesh = np.full((mesh.NumberOfElts,), np.nan)
        alpha_mesh[ElmntTip] = alpha
        for i in range(len(nan)):
            neighbors = mesh.NeiElements[ElmntTip[nan[i]]]
            neig_in_tip = np.intersect1d(ElmntTip, neighbors)
            alpha_neig = alpha_mesh[neig_in_tip]
            alpha_neig = np.delete(alpha_neig, np.where(np.isnan(alpha_neig))[0])
            alpha[nan[i]] = np.mean(alpha_neig)

    CellStatusNew = np.zeros(mesh.NumberOfElts, int)
    CellStatusNew[EltChannel] = 1
    CellStatusNew[ElmntTip] = 2

    return ElmntTip, l, alpha, CellStatusNew

# -----------------------------------------------------------------------------------------------------------------------


def reconstruct_front_LS_gradient(dist, EltBand, EltChannel, mesh):
    """
    Track the fracture front, the length of the perpendicular drawn on the fracture and the angle inscribed by the
    perpendicular. The angle is calculated from the gradient of the level set.

    Arguments:
        dist (ndarray):         -- the signed distance of the cells from the fracture front.
        EltBand (ndarray):      -- the band of elements to which the search is limited.
        EltChannel (ndarray):   -- list of Channel elements.
        mesh (CartesianMesh):   -- the mesh of the fracture.
    """

    # Elements that are not in channel
    EltRest = np.setdiff1d(EltBand, EltChannel)
    ElmntTip = np.asarray([], int)
    l = np.asarray([])
    alpha = np.asarray([])

    for i in range(0, len(EltRest)):
        neighbors = mesh.NeiElements[EltRest[i]]

        minx = min(dist[neighbors[0]], dist[neighbors[1]])
        miny = min(dist[neighbors[2]], dist[neighbors[3]])
        # distance of the vertex (zero vertex, i.e. rotated distance) of the current cell from the front
        Pdis = -(minx + miny) / 2

        # if the vertex distance is positive, meaning the fracture has passed the vertex
        if Pdis >= 0:
            ElmntTip = np.append(ElmntTip, EltRest[i])
            l = np.append(l, Pdis)

            # neighbors
            #     6     3    7
            #     0    elt   1
            #     4    2     5
            neighbors_tip = np.zeros(8, dtype=int)
            neighbors_tip[:4] = mesh.NeiElements[EltRest[i]]
            neighbors_tip[4] = mesh.NeiElements[neighbors_tip[2]][0]
            neighbors_tip[5] = mesh.NeiElements[neighbors_tip[2]][1]
            neighbors_tip[6] = mesh.NeiElements[neighbors_tip[3]][0]
            neighbors_tip[7] = mesh.NeiElements[neighbors_tip[3]][1]

            # zero Vertex
            #     3         2
            #     0         1
            if dist[neighbors_tip[0]] <= dist[neighbors_tip[1]] and dist[neighbors_tip[2]] <= dist[
                neighbors_tip[3]]:
                # if zero vertex is 0:
                gradx = -((dist[neighbors_tip[0]] + dist[neighbors_tip[4]]) / 2 - (
                    dist[EltRest[i]] + dist[neighbors_tip[2]]) / 2) / mesh.hx
                grady = ((dist[neighbors_tip[0]] + dist[EltRest[i]]) / 2 - (
                    dist[neighbors_tip[4]] + dist[neighbors_tip[2]]) / 2) / mesh.hy

            elif dist[neighbors_tip[0]] > dist[neighbors_tip[1]] and dist[neighbors_tip[2]] <= dist[
                    neighbors_tip[3]]:
                # if zero vertex is 1:
                gradx = ((dist[neighbors_tip[1]] + dist[neighbors_tip[5]]) / 2 - (
                    dist[EltRest[i]] + dist[neighbors_tip[2]]) / 2) / mesh.hx
                grady = ((dist[neighbors_tip[1]] + dist[EltRest[i]]) / 2 - (
                    dist[neighbors_tip[5]] + dist[neighbors_tip[2]]) / 2) / mesh.hy

            elif dist[neighbors_tip[0]] > dist[neighbors_tip[1]] and dist[neighbors_tip[2]] > dist[
                    neighbors_tip[3]]:
                # if zero vertex is 2:
                gradx = ((dist[neighbors_tip[1]] + dist[neighbors_tip[7]]) / 2 - (
                    dist[EltRest[i]] + dist[neighbors_tip[3]]) / 2) / mesh.hx
                grady = -((dist[neighbors_tip[1]] + dist[EltRest[i]]) / 2 - (
                    dist[neighbors_tip[3]] + dist[neighbors_tip[7]]) / 2) / mesh.hy

            elif dist[neighbors_tip[0]] <= dist[neighbors_tip[1]] and dist[neighbors_tip[2]] > dist[
                    neighbors_tip[3]]:
                # if zero vertex is 3:
                gradx = -((dist[neighbors_tip[6]] + dist[neighbors_tip[0]]) / 2 - (
                    dist[EltRest[i]] + dist[neighbors_tip[3]]) / 2) / mesh.hx
                grady = ((dist[neighbors_tip[0]] + dist[EltRest[i]]) / 2 - (
                    dist[neighbors_tip[6]] + dist[neighbors_tip[3]]) / 2) / mesh.hy

            alpha = np.append(alpha, np.abs(np.arcsin(grady / (gradx ** 2 + grady ** 2) ** 0.5)))

    CellStatusNew = np.zeros(mesh.NumberOfElts, int)
    CellStatusNew[EltChannel] = 1
    CellStatusNew[ElmntTip] = 2

    return ElmntTip, l, alpha, CellStatusNew


# ----------------------------------------------------------------------------------------------------------------------

def UpdateLists(EltsChannel, EltsTipNew, FillFrac, levelSet, mesh):
    """
    This function update the Element lists, given the element lists from the last time step. EltsTipNew list can have 
    partially filled and fully filled elements. The function update lists accordingly.
    
    Arguments:
        EltsChannel (ndarray):      -- channel elements list.
        EltsTipNew (ndarray):       -- list of the new tip elements, including fully filled cells that were tip
                                       cells in the last time step.
        FillFrac (ndarray):         -- filling fraction of the new tip cells.
        levelSet (ndarray):         -- current level set.
        mesh (CartesianMesh):       -- the mesh of the fracture.
        
    Returns:
        - eltsChannel (ndarray):    -- new channel elements list.
        - eltsTip (ndarray):        -- new tip elements list.
        - eltsCrack(ndarray):       -- new crack elements list.
        - eltsRibbon (ndarray):     -- new ribbon elements list.
        - zeroVrtx (ndarray):       -- list specifying the zero vertex of the tip cells. (can have value from 0 to\
                                       3, where 0 signify bottom left, 1 signifying bottom right, 2 signifying top\
                                       right and 3 signifying top left vertex).
        - CellStatusNew (ndarray):  -- specifies which region each element currently belongs to.
    """
    log = logging.getLogger('PyFrac.UpdateLists')
    # new tip elements contain only the partially filled elements
    eltsTip = EltsTipNew[np.where(FillFrac <= 0.9999)]

    # Tip elements flag to avoid search on each iteration
    inTip = np.zeros((mesh.NumberOfElts,), bool)
    inTip[eltsTip] = True
    i = 0

    #todo: the while below is probably inserting a bug - found it with poor resolution and volume control
    while i < len(eltsTip):  # to remove a special case encountered in sharp edges and rectangular cells
        neighbors = mesh.NeiElements[eltsTip[i]]
        if inTip[neighbors[0]] and inTip[neighbors[3]] and inTip[neighbors[3] - 1]:
            conjoined = np.asarray([neighbors[0], neighbors[3], neighbors[3] - 1, eltsTip[i]])
            mindist = np.argmin(mesh.distCenter[conjoined])
            inTip[conjoined[mindist]] = False
            eltsTip = np.delete(eltsTip, np.where(eltsTip == conjoined[mindist]))
            i -= 1
        i += 1

    # new channel elements
    newEltChannel = np.setdiff1d(EltsTipNew, eltsTip)

    eltsChannel = np.append(EltsChannel, newEltChannel)
    eltsCrack = np.append(eltsChannel, eltsTip)
    eltsRibbon = np.array([], int)
    zeroVrtx = np.zeros((len(eltsTip),), int)  # Vertex from where the perpendicular is drawn

    # All the inner cells neighboring tip cells are added to ribbon cells
    for i in range(0, len(eltsTip)):
        neighbors = mesh.NeiElements[eltsTip[i]]

        if levelSet[neighbors[0]] <= levelSet[neighbors[1]]:
            eltsRibbon = np.append(eltsRibbon, neighbors[0])
            drctx = -1
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[1])
            drctx = 1

        if levelSet[neighbors[2]] <= levelSet[neighbors[3]]:
            eltsRibbon = np.append(eltsRibbon, neighbors[2])
            drcty = -1
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[3])
            drcty = 1

        # Assigning zero vertex according to the direction of propagation
        if drctx < 0 and drcty < 0:
            zeroVrtx[i] = 0
        elif drctx > 0 and drcty < 0:
            zeroVrtx[i] = 1
        elif drctx < 0 and drcty > 0:
            zeroVrtx[i] = 3
        elif drctx > 0 and drcty > 0:
            zeroVrtx[i] = 2

    eltsRibbon = np.setdiff1d(eltsRibbon, eltsTip)
    if np.any(levelSet[eltsRibbon]>0):
        log.debug("Probably there is a bug here....")
    # plot for checking
    # from continuous_front_reconstruction import plot_cell_lists
    # fig = plot_cell_lists(mesh, eltsTip, mymarker='.', mycolor='red')
    # fig = plot_cell_lists(mesh, eltsRibbon, fig=fig, mymarker='.', mycolor='b', shiftx=0.08)

    # Cells status list store the status of all the cells in the domain
    CellStatusNew = np.zeros(mesh.NumberOfElts, int)
    CellStatusNew[eltsChannel] = 1
    CellStatusNew[eltsTip] = 2
    CellStatusNew[eltsRibbon] = 3

    return eltsChannel, eltsTip, eltsCrack, eltsRibbon, zeroVrtx, CellStatusNew, newEltChannel

    # -----------------------------------------------------------------------------------------------------------------------

def Eikonal_Res(Tij, *args):
    """quadratic Eikonal equation residual to be used by numerical root finder"""

    (Tleft, Tright, Tbottom, Ttop, Fij, dx, dy) = args
    return np.nanmax([(Tij - Tleft) / dx, 0]) ** 2 + np.nanmin([(Tright - Tij) / dx, 0]) ** 2 + np.nanmax(
        [(Tij - Tbottom) / dy, 0]) ** 2 + \
           np.nanmin([(Ttop - Tij) / dy, 0]) ** 2 - Fij ** 2


def get_cells_inside_circle(mesh, r, center):
    """
    This function would provide a list of all the cells inside of the
    perimeter of a circle with the given radius. A list of all the cells inside the fracture is also provided.

    Arguments:
        mesh (CartesianMesh object):        -- a CartesianMesh class object describing the grid.
        r (float):                          -- the radius of the circle.
        center (list or ndarray):           -- the coordinates [x, y] of the center.

    Returns:
        - inner_cells (ndarray)             -- the list of cells inside the given circle.
    """


    # distances of the cell centers
    dist_vertx = (((mesh.CenterCoor[:, 0] - center[0])) ** 2 + ((mesh.CenterCoor[:, 1] - center[1])) ** 2) \
                 ** (1 / 2) / r - 1.


    inner_cells = np.where(dist_vertx <= 0)[0]


    return  inner_cells.tolist()


# -----------------------------------------------------------------------------------------------------------------------

def  get_front_region(mesh, EltRibbon, sgndDist_k_EltRibbon):
    """
    This function returns a list of elements that form a band where the location of the tip is expected to be.
    Args:
        mesh:
        EltRibbon:
        sgndDist_k_EltRibbon:
        EltChannel_lstTmStp:

    Returns:

    """
    front_region = []
    advancing_fast = []
    # take the cells in a circle drown from each ribbon cell with radius equal to the distance to the front from the tip inversion
    for i in range(len(EltRibbon)):
        cell_i = EltRibbon[i]
        if np.abs(sgndDist_k_EltRibbon[i]) > mesh.hx and np.abs(sgndDist_k_EltRibbon[i]) > mesh.hy :
            radius_i = np.abs(sgndDist_k_EltRibbon[i]) + 2.5 * (mesh.hx ** 2 + mesh.hy ** 2) ** 0.5
            advancing_fast.append(cell_i)
        else:
            radius_i = np.abs(sgndDist_k_EltRibbon[i]) + 1.5 * (mesh.hx ** 2 + mesh.hy ** 2) ** 0.5
        center_i = mesh.CenterCoor[cell_i]
        new_cells = get_cells_inside_circle(mesh, radius_i, center_i)
        for j in new_cells:
            front_region.append(j)
    front_region = np.unique(front_region)

    # take out the ribbon
    front_region =np.setdiff1d(front_region, EltRibbon)

    # from utility import plot_as_matrix
    # K = np.zeros((mesh.NumberOfElts,), )
    # K[front_region] = 1
    # K[EltRibbon] = 2
    # plot_as_matrix(K, mesh)
    return front_region