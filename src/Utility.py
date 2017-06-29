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

from src.VolIntegral import Pdistance

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

def plot_as_matrix(data, mesh):
    figr = plt.figure()
    ax = figr.add_subplot(111)
    ReMesh = np.resize(data, (mesh.ny, mesh.nx))
    x = np.linspace(-mesh.Lx, mesh.Lx, mesh.nx)
    y = np.linspace(mesh.Ly, mesh.Ly, mesh.ny)
    xv, yv = np.meshgrid(x, y)
    # cax = ax.contourf(xv, yv, ReMesh, levels=[0, 100, 2100, 10000])
    cax = ax.matshow(ReMesh)
    figr.colorbar(cax)
    plt.show()

    return figr

#-----------------------------------------------------------------------------------------------------------------------
def ReadFracture(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
