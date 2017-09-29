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
from src.VolIntegral import VolumeIntegral


def get_circular_survey_cells(mesh, initRad):

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
    EltChannel = (np.where(psum == 4))[0]  # indices of channel element / fully cracked

    # find the ribbon elements: Channel Elements having at least
    # on common vertices with a Tip element
    #
    # loop on ChannelElement, and on TipElement
    testribbon = np.empty([len(EltChannel), 1], dtype=float)
    for e in range(0, len(EltChannel)):
        for i in range(0, len(EltTip)):
            if (len(np.intersect1d(mesh.Connectivity[EltChannel[e]], mesh.Connectivity[EltTip[i]])) > 0):
                testribbon[e] = 1
                break
            else:
                testribbon[e] = 0
    EltRibbon = EltChannel[(np.reshape(testribbon, len(EltChannel)) == 1)]  # EltChannel is (N,) testribbon is (N,1)

    return EltRibbon, EltChannel


def generate_footprint(mesh, surv_cells, inner_region, dist_surv_cells):

    sgndDist = np.full((mesh.NumberOfElts,), 1e10)
    sgndDist[surv_cells] = -dist_surv_cells

    EltRest = np.arange(mesh.NumberOfElts)
    for i in range(len(inner_region)):
        EltRest = np.delete(EltRest, np.where(EltRest == inner_region[i])[0])

    SolveFMM(sgndDist,
             surv_cells,
             inner_region,
             mesh,
             EltRest,
             inner_region)

    (EltTip, l, alpha, CSt) = reconstruct_front(sgndDist, inner_region, mesh)

    FillFrac = VolumeIntegral(EltTip,
                              alpha,
                              l,
                              mesh,
                              'A') / mesh.EltArea
    (EltChannel,
     EltTip_updated,
     EltCrack,
     EltRibbon,
     ZeroVertex,
     CellStatus) = UpdateLists(inner_region,
                               EltTip,
                               FillFrac,
                               sgndDist,
                               mesh)

    newTip_indices = np.arange(len(EltTip))[np.in1d(EltTip, EltTip_updated)]
    l_updated = l[newTip_indices]
    alpha_updated = alpha[newTip_indices]
    FillFrac_updated = FillFrac[newTip_indices]

    if EltChannel.size <= EltRibbon.size:
        raise SystemExit("No channel elements. The initial radius is probably too small")


    return EltChannel, EltTip_updated, EltCrack, EltRibbon, ZeroVertex, \
            CellStatus, l_updated, alpha_updated, FillFrac_updated, sgndDist


def initial_width_pressure(mesh, EltCrack, EltTip, FillFrac, C, w=None, p=None, volume=None):

    if w is None and p is None and volume is None:
        raise SystemExit("Atleast one of the three variables w, p and volume has to be provided.")

    if p is None:
        p_to_return = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    elif not isinstance(p, np.ndarray):
        p_to_return = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
        p_to_return[EltCrack] = np.full((EltCrack.size, ), p, dtype=np.float64)
    else:
        p_to_return = p

    if w is None:
        w_to_return = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    elif w.size != mesh.NumberOfElts and not w is None:
        raise SystemExit("The given width should be an ndarray with the size equal to the size of the mesh")
    else:
        w_to_return = w

    if not w is None and not p is None:
        return w_to_return, p_to_return

    C_EltTip = C[np.ix_(EltTip, EltTip)]  # keeping the tip element entries to restore current
    #  tip correction. This is done to avoid copying the full elasticity matrix.

    # filling fraction correction for element in the tip region
    for e in range(0, len(EltTip)):
        r = FillFrac[e] - .25
        if r < 0.1:
            r = 0.1
        ac = (1 - r) / r
        C[EltTip[e], EltTip[e]] = C[EltTip[e], EltTip[e]] * (1. + ac * np.pi / 4.)



    if w is None and not p is None:
        w_to_return[EltCrack] = np.linalg.solve(C[np.ix_(EltCrack, EltCrack)], p_to_return[EltCrack])

    if not w is None and p is None:
        p_to_return[EltCrack] = np.dot(C[np.ix_(EltCrack, EltCrack)], w[EltCrack])

    if w is None and p is None:

        C_Crack = C[np.ix_(EltCrack, EltCrack)]

        A = np.hstack((C_Crack, -np.ones((EltCrack.size, 1), dtype=np.float64)))
        A = np.vstack((A, np.ones((1, EltCrack.size + 1), dtype=np.float64)))
        A[-1, -1] = 0

        b = np.zeros((len(EltCrack)+1, ), dtype=np.float64)
        b[-1] = volume / mesh.EltArea

        sol = np.linalg.solve(A, b)

        w_to_return[EltCrack] = sol[np.arange(EltCrack.size)]
        p_to_return[EltCrack] = sol[EltCrack.size]

    # recover original C (without filling fraction correction)
    C[np.ix_(EltTip, EltTip)] = C_EltTip

    return w_to_return, p_to_return


#-----------------------------------------------------------------------------------------------------------------------

def g(a, b, x0, y0, la):
    return pow(a * x0 / (pow(a, 2) + la), 2) + pow(b * y0 / (pow(b, 2) + la), 2) - 1

def Distance_ellipse(a, b, x0, y0):
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

    return abs(min([lx-x, lx+x, ly-y, ly+y]))


def get_eliptical_survey_cells(mesh, a, b):
    dist_vertx = (mesh.VertexCoor[:,0]/a)**2 + (mesh.VertexCoor[:,1]/b)**2 - 1.
    vertices = dist_vertx[mesh.Connectivity]<0
    log_and = np.logical_and(np.logical_and(vertices[:,0],vertices[:,1]),np.logical_and(vertices[:,2],vertices[:,3]))
    channel = np.where(log_and)[0]
    # for i in range(mesh.NumberOfElts):
    #     mesh.Connec
    #
    # srv_cells = np.array([], dtype=np.int)
    # for i in range(mesh.nx,mesh.NumberOfElts-mesh.nx):
    #     if np.sign(dist[i]) != np.sign(dist[i-1]) or np.sign(dist[i]) != np.sign(dist[i-mesh.nx]):
    #         srv_cells = np.append(srv_cells, i)
    #
    # channel = np.where(dist<=0.)[0]
    return channel, channel