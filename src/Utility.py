# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 17:18:37 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import copy
from src.TipInversion import TipAsymInversion

import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def radius_level_set(xy, R):
    """
    signed distance from a circle; (<0 inside circle , >0 outside, - zero at the circle)
    Arguments:
        xy (ndarray-flat):      array with x and y coordinate values of the cell centers. The x coordinates are given in
                                the frist column and y coordinate is given in the second column
        R (float):              radius of the circle
    
    Returns:
        ndarray-float:          signed distance from the given circle for each cell given row wise by the argument xy
    """
    if len(xy) > 2:
        # for arrays
        return np.linalg.norm(xy, 2, 1) - R  # norm(xy)=(x^2+y^2)^1/2    -R
    else:
        # for single entries
        return np.linalg.norm(xy, 2) - R

#-----------------------------------------------------------------------------------------------------------------------

def Neighbors(elem, nx, ny):
    """
    Neighbouring elements of an element within the mesh . Boundary elements have themselves as neighbor
    Arguments:
        elem (int): element whose neighbor are to be found
        nx (int):   number of elements in x direction
        ny (int):   number of elements in y direction
        
    Returns:
        int:        left neighbour
        int:        right neighbour
        int:        bottom neighbour
        int:        top neighbour
    """

    j = elem // nx
    i = elem % nx

    if i == 0:
        left = elem
    else:
        left = j * nx + i - 1

    if i == nx - 1:
        right = elem
    else:
        right = j * nx + i + 1

    if j == 0:
        bottom = elem
    else:
        bottom = (j - 1) * nx + i

    if j == ny - 1:
        up = elem
    else:
        up = (j + 1) * nx + i

    return (left, right, bottom, up)

# ----------------------------------------------------------------------------------------------------------------------

def PrintDomain(Matrix, mesh, Elem = None):
    """
    3D plot of all elements given in the form of a list;
    Arguments:
        Elem(ndarray-int):          list of elements to be plotted
        Matrix(ndarray-float):      values to be plotted, should be equal in size to the first argument(Elem)
        mesh(CartesianMesh object): mesh object
    """
    if Elem == None:
        Elem = np.arange(mesh.NumberOfElts)

    # if len(Matrix.shape)==1:
    #     Matrix = np.reshape(Matrix, (mesh.ny, mesh.nx))

    tmp = np.zeros((mesh.NumberOfElts,))
    tmp[Elem] = Matrix
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(mesh.CenterCoor[:, 0], mesh.CenterCoor[:, 1], tmp, cmap=cm.jet, linewidth=0.2)
    plt.show()
    plt.pause(0.01)


# ----------------------------------------------------------------------------------------------------------------------
def plot_Reynolds_number(Fr, ReyNum, edge):

    # figr = Fr.plot_fracture("complete", "footPrint")
    # ax = figr.axes[0]
    figr = plt.figure()
    ax = figr.add_subplot(111)
    ReMesh = np.resize(ReyNum[edge, :], (Fr.mesh.ny, Fr.mesh.nx))
    x = np.linspace(-Fr.mesh.Lx, Fr.mesh.Lx, Fr.mesh.nx)
    y = np.linspace(-Fr.mesh.Ly, Fr.mesh.Ly, Fr.mesh.ny)
    xv, yv = np.meshgrid(x, y)
    # cax = ax.contourf(xv, yv, ReMesh, levels=[0, 100, 2100, 10000])
    cax = ax.matshow(ReMesh)
    figr.colorbar(cax)
    plt.title("Reynolds number")
    plt.show()

    return figr

#-----------------------------------------------------------------------------------------------------------------------

def plot_as_matrix(data, mesh, fig=None):

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)

    ReMesh = np.resize(data, (mesh.ny, mesh.nx))
    # x = np.linspace(-mesh.Lx, mesh.Lx, mesh.nx)
    # y = np.linspace(mesh.Ly, mesh.Ly, mesh.ny)

    # delta_x = x[1] - x[0]
    # extnt_x =  [x[0] - delta_x / 2, x[-1] + delta_x / 2]
    # delta_y = y[1] - y[0]
    # extnt_y = [y[0] - delta_y / 2, y[-1] + delta_y / 2]
    #
    # cax = ax.imshow(ReMesh, aspect='auto', interpolation='none',
    #            extent=extnt_x + extnt_y, origin='lower')

    cax = ax.matshow(ReMesh)
    fig.colorbar(cax)
    plt.show()

    return fig

#-----------------------------------------------------------------------------------------------------------------------
def ReadFracture(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

#-----------------------------------------------------------------------------------------------------------------------

def find_regime(w, Fr, Material_properties, sim_properties, timeStep, Kprime, asymptote_universal):

    sim_parameters_tmp = copy.deepcopy(sim_properties)
    sim_parameters_tmp.tipAsymptote = 'K'
    asymptote_toughness = TipAsymInversion(w,
                                               Fr,
                                               Material_properties,
                                               sim_parameters_tmp,
                                               timeStep,
                                               Kprime_k=Kprime)
    sim_parameters_tmp.tipAsymptote = 'M'
    asymptote_viscosity = TipAsymInversion(w,
                                             Fr,
                                             Material_properties,
                                             sim_parameters_tmp,
                                             timeStep)

    # regime = np.full((asymptote_universal.size,), np.nan, dtype=np.float64)
    # regime[np.where(abs(1. - asymptote_universal/asymptote_toughness) < 1.e-6)[0]] = 0.
    # regime[np.where(abs(1. - asymptote_universal / asymptote_viscosity) < 1.e-6)[0]] = 1.
    #
    #
    # tough_to_visc = np.where(asymptote_toughness < asymptote_universal)[0]
    # regime[tough_to_visc] = (asymptote_universal[tough_to_visc] - asymptote_toughness[tough_to_visc] ) / (
    #                         asymptote_toughness[tough_to_visc] - asymptote_viscosity[tough_to_visc])
    # visc_to_tough = np.where(asymptote_viscosity < asymptote_universal)[0]
    # regime[visc_to_tough] = 1. - abs((asymptote_universal[visc_to_tough] - asymptote_viscosity[visc_to_tough]) /
    #                                  (asymptote_toughness[visc_to_tough] - asymptote_viscosity[visc_to_tough]))

    regime = 1. - abs(asymptote_viscosity - asymptote_universal) / abs(asymptote_viscosity - asymptote_toughness)
    # regime[np.where(regime < 0.)[0]] = 0.


    return regime
