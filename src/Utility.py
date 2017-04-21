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

def PrintDomain(Elem, Matrix, mesh):
    """
    3D plot of all elements given in the form of a list;
    Arguments:
        Elem(ndarray-int):          list of elements to be plotted
        Matrix(ndarray-float):      values to be plotted, should be equal in size to the first argument(Elem)
        mesh(CartesianMesh object): mesh object
    """

    tmp = np.zeros((mesh.NumberOfElts,))
    tmp[Elem] = Matrix
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(mesh.CenterCoor[:, 0], mesh.CenterCoor[:, 1], tmp, cmap=cm.jet, linewidth=0.2)
    plt.show()
    plt.pause(0.01)


# ----------------------------------------------------------------------------------------------------------------------

def PlotMeshFractureTrace(Mesh, EltTip, EltChannel, EltRibbon, I, J, Ranalytical, mat_properties, Identify):
    """
    Plot fracture trace and different regions of fracture
    """

    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots()
    ax.set_xlim([-Mesh.Lx, Mesh.Lx])
    ax.set_ylim([-Mesh.Ly, Mesh.Ly])

    patches = []
    for i in range(Mesh.NumberOfElts):
        polygon = Polygon(np.reshape(Mesh.VertexCoor[Mesh.Connectivity[i], :], (4, 2)), True)
        patches.append(polygon)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)

    # todo: A proper mechanism to mark element with different material properties has to be looked into
    # marking those elements that have sigmaO different than the sigmaO at the center
    markedElts = []
    if mat_properties != None:
        markedElts = np.where(mat_properties.SigmaO != np.mean(mat_properties.SigmaO[Mesh.CenterElts]))

    # applying different colors for different types of elements
    colors = 100. * np.full(len(patches), 0.4)
    colors[markedElts] = 50
    colors[EltTip] = 70.
    colors[EltChannel] = 10.
    colors[EltRibbon] = 90.
    colors[Identify] = 0.

    p.set_array(np.array(colors))
    ax.add_collection(p)

    # Plot the analytical solution
    if Ranalytical>0.:
        circle = plt.Circle((0, 0), radius=Ranalytical)
        circle.set_ec('r')
        circle.set_fill(False)
        ax.add_patch(circle)

    # print Element numbers on the plot for elements to be identified
    for i in range(len(Identify)):
        ax.text(Mesh.CenterCoor[Identify[i], 0] - Mesh.hx / 4, Mesh.CenterCoor[Identify[i], 1] - Mesh.hy / 4,
                repr(Identify[i]), fontsize=10)

    #todo !!!Hack: gets very large values sometime, needs to be resolved
    for e in range(0, len(I)):
        if max(abs(I[e, :] - J[e, :])) < 3 * (Mesh.hx ** 2 + Mesh.hy ** 2) ** 0.5: # if
            plt.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), '.-k')

    plt.axis('equal')

    # maximize the plot window
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    return fig


# ----------------------------------------------------------------------------------------------------------------------
def plot_Reynolds_number(Fr, ReyNum, edge):

    figr = Fr.plot_fracture("complete", "footPrint")
    ax = figr.axes[0]
    ReMesh = np.resize(ReyNum[edge, :], (Fr.mesh.ny, Fr.mesh.nx))
    x = np.linspace(-Fr.mesh.Lx, Fr.mesh.Lx, Fr.mesh.nx)
    y = np.linspace(-Fr.mesh.Ly, Fr.mesh.Ly, Fr.mesh.ny)
    xv, yv = np.meshgrid(x, y)
    cax = ax.contour(xv, yv, ReMesh, levels=[0, 100, 2100, 10000])
    figr.colorbar(cax)
    plt.show()

    return figr

#-----------------------------------------------------------------------------------------------------------------------
def ReadFracture(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
