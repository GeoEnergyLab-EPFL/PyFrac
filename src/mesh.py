# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 11:51:00 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# import
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.colors import to_rgb
from matplotlib.collections import PatchCollection
from properties import PlotProperties

from visualization import zoom_factory, to_precision, text3d
from symmetry import *
import numpy as np
import logging

class CartesianMesh:
    """Class defining a Cartesian Mesh.

    The constructor creates a uniform Cartesian mesh centered at (0,0) and having the dimensions of [-Lx,Lx]*[-Ly,Ly].

    Args:
        nx,ny (int):                -- number of elements in x and y directions respectively.
        Lx,Ly (float):              -- lengths in x and y directions respectively.
        symmetric (bool):           -- if true, additional variables (see list of attributes) will be evaluated for
                                       symmetric fracture solver.

    Attributes:
        Lx,Ly (float):                    -- length of the domain in x and y directions respectively. The rectangular domain
                                             have a total length of 2*Lx in the x direction and 2*Ly in the y direction. Both
                                             the positive and negative halves are included.
        nx,ny (int):                      -- number of elements in x and y directions respectively.
        hx,hy (float):                    -- grid spacing in x and y directions respectively.
        VertexCoor  (ndarray):            -- [x,y] Coordinates of the vertices.
        CenterCoor  (ndarray):            -- [x,y] coordinates of the center of the elements.
        NumberOfElts (int):               -- total number of elements in the mesh.
        EltArea (float):                  -- area of each element.
        Connectivity (ndarray):           -- connectivity array giving four vertices of an element in the following order
                                             [bottom left, bottom right, top right, top left]
        Connectivityelemedges (ndarray):  -- connectivity array giving four edges of an element in the following order
                                             [bottom, right, top, left]
        Connectivityedgeselem (ndarray):  -- connectivity array giving two elements that are sharing an edge
        Connectivityedgesnodes (ndarray): -- connectivity array giving two vertices of an edge
        Connectivitynodesedges (ndarray): -- connectivity array giving four edges of a node in the following order
                                             [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
        Connectivitynodeselem (ndarray):  -- connectivity array giving four elements of a node in the following order
                                             [bottom left, bottom right, top right, top left]
        NeiElements (ndarray):            -- Giving four neighboring elements with the following order:[left, right,
                                             bottom, up].
        distCenter (ndarray):             -- the distance of the cells from the center.
        CenterElts (ndarray):             -- the element in the center (the cell with the injection point).

        domainLimits (ndarray):           -- the limits of the domain

    Note:
        The attributes below are only evaluated if symmetric solver is used.

    Attributes:
        corresponding (ndarray): -- the index of the corresponding symmetric cells in the set of active cells
                                    (activeSymtrc) for each cell in the mesh.
        symmetricElts (ndarray): -- the set of four symmetric cells in the mesh for each of the cell.
        activeSymtrc (ndarray):  -- the set of cells that are active in the mesh. Only these cells will be solved
                                    and the solution will be replicated in the symmetric cells.
        posQdrnt (ndarray):      -- the set of elements in the positive quadrant not including the boundaries.
        boundary_x (ndarray):    -- the elements intersecting the positive x-axis line.
        boundary_y (ndarray):    -- the elements intersecting the positive y-axis line.
        volWeights (ndarray):    -- the weights of the active elements in the volume of the fracture. The cells in the
                                    positive quadrant, the boundaries and the injection cell have the weights of 4, 2
                                    and 1 respectively.

    """

    def __init__(self, Lx, Ly, nx, ny, symmetric=False):
        """ 
        Creates a uniform Cartesian mesh centered at zero and having the dimensions of [-Lx, Lx]*[-Ly, Ly].

        Args:
            nx,ny (int):        -- number of elements in x and y directions respectively
            Lx,Ly (float):      -- lengths in x and y directions respectively
            symmetric (bool):   -- if true, additional variables (see list of attributes) will be evaluated for
                                    symmetric fracture solver.

        """
        log = logging.getLogger('PyFrac.mesh')
        if not isinstance(Lx, list):
            self.Lx = Lx
            xlims = np.asarray([-Lx, Lx])
        else:
            self.Lx = abs(Lx[0]-Lx[1]) / 2
            xlims = np.asarray([Lx[0], Lx[1]])

        if not isinstance(Ly, list):
            self.Ly = Ly
            ylims = np.asarray([-Ly, Ly])
        else:
            self.Ly = abs(Ly[0]-Ly[1]) / 2
            ylims = np.asarray([Ly[0], Ly[1]])

        self.domainLimits = np.hstack((ylims, xlims))


        # Check if the number of cells is odd to see if the origin would be at the mid point of a single cell
        if nx % 2 == 0:
            log.warning("Number of elements in x-direction are even. Using " + repr(nx+1) + " elements to have origin at a "
                                                                                      "cell center...")
            self.nx = nx+1
        else:
            self.nx = nx

        if ny % 2 == 0:
            log.warning("Number of elements in y-direction are even. Using " + repr(ny+1) + " elements to have origin at a "
                                                                                      "cell center...")
            self.ny = ny+1
        else:
            self.ny = ny

        self.hx = 2. * self.Lx / (self.nx - 1)
        self.hy = 2. * self.Ly / (self.ny - 1)

        x = np.linspace(self.domainLimits[2] - self.hx / 2., self.domainLimits[3] + self.hx / 2., self.nx + 1)
        y = np.linspace(self.domainLimits[0] - self.hy / 2., self.domainLimits[1] + self.hy / 2., self.ny + 1)

        xv, yv = np.meshgrid(x, y)  # coordinates of the vertex of each elements

        a = np.resize(xv, ((self.nx + 1) * (self.ny + 1), 1))
        b = np.resize(yv, ((self.nx + 1) * (self.ny + 1), 1))

        self.VertexCoor = np.reshape(np.stack((a, b), axis=-1), (len(a), 2))

        self.NumberofNodes = (self.nx+1) * (self.ny+1)
        self.NumberOfElts = self.nx * self.ny
        self.EltArea = self.hx * self.hy


        """
        We create a list of cell names thaht are colose to the boundary of the mesh. See the example below.
        In that case the list will contain the elements identified with a x.
        The list of elements will be called Frontlist
        
         _____________________________
        |    |    |    |    |    |    |
        |____|____|____|____|____|____|
        |    | x  |  x |  x |  x |    |
        |____|____|____|____|____|____|
        |    | x  |    |    |  x |    |
        |____|____|____|____|____|____|
        |    | x  |    |    |  x |    |
        |____|____|____|____|____|____|
        |    | x  |  x |  x |  x |    |
        |____|____|____|____|____|____|
        |    |    |    |    |    |    |
        |____|____|____|____|____|____|            
        """
        self.Frontlist=[]
        self.Frontlist=self.Frontlist + list(range(self.nx+1,2*self.nx-1))
        self.Frontlist=self.Frontlist + list(range((self.ny-3)*(self.nx)+self.nx + 1, (self.ny-3)*(self.nx)+2 * self.nx - 1))
        for i in range(1,self.ny-3):
            self.Frontlist.append(  self.nx+1+i*self.nx)
            self.Frontlist.append(2*self.nx-2+i*self.nx)



        """
         Giving four neighbouring elements in the following order: [left,right,bottom,up]
         ______ ______ _____ 
        |      | top  |     |
        |______|______|_____|
        |left  |  i   |right|
        |______|______|_____|
        |      |bottom|     |
        |______|______|_____|
        """
        Nei = np.zeros((self.NumberOfElts, 4), int)
        for i in range(0, self.NumberOfElts):
            Nei[i, :] = np.asarray(self.Neighbors(i, self.nx, self.ny))
        self.NeiElements = Nei


        """
         conn is the connectivity array giving four vertices of an element in the following order
         ______ ______ _____ 
        |      |      |     |
        |______3______2_____|
        |      |  i   |     |
        |______0______1_____|
        |      |      |     |
        |______|______|_____|
        """

        """
         connElemEdges is a connectivity array: for each element is listing the name of its 4 edges
         connEdgesElem is a connectivity array: for each edge is listing the name of its 2 neighbouring elements
        """


        #
        # connEdgesNodes is a connectivity array: for each edge is listing the name of its 2 end nodes

        # connNodesElem is a connectivity array: for each node is listing the 4 elements that share that
        # connNodesEdges:
        #            0
        #            |
        #         1__o__3    o is the node and the order in  connNodesEdges is [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
        #            |
        #            2
        numberofedges = (2 * self.nx * self.ny + self.nx + self.ny)  # Peruzzo 2019
        conn = np.empty([self.NumberOfElts, 4], dtype=int)
        booleconnEdgesNodes = np.zeros([numberofedges, 1], dtype=int)  # Peruzzo 2019
        connEdgesNodes = np.empty([numberofedges, 2], dtype=int)  # Peruzzo 2019
        connElemEdges = np.empty([self.NumberOfElts, 4], dtype=int)  # Peruzzo 2019
        connEdgesElem = np.full([numberofedges, 2], np.nan, dtype=np.int)  # Peruzzo 2019
        connNodesEdges = np.full([self.NumberofNodes, 4], np.nan, dtype=int)  # Peruzzo 2019
        connNodesElem = np.full([self.NumberofNodes, 4], np.nan, dtype=int)  # Peruzzo 2019
        k = 0
        for j in range(0, self.ny):
            for i in range(0, self.nx):
                conn[k, 0] = (i + j * (self.nx + 1))
                conn[k, 1] = (i + 1) + j * (self.nx + 1)
                conn[k, 2] = i + 1 + (j + 1) * (self.nx + 1)
                conn[k, 3] = i + (j + 1) * (self.nx + 1)
                connElemEdges[k, 0] = (j * (2 * self.nx + 1) + i)  # BottomEdge - Peruzzo 2019
                connElemEdges[k, 1] = (j * (2 * self.nx + 1) + self.nx + i + 1)  # RightEdge  - Peruzzo 2019
                connElemEdges[k, 2] = ((j + 1) * (2 * self.nx + 1) + i)  # topEdge    - Peruzzo 2019
                connElemEdges[k, 3] = (j * (2 * self.nx + 1) + self.nx + i)  # LeftEdge   - Peruzzo 2019
                connEdgesElem[connElemEdges[k, 0], :] = [k, Nei[k, 2]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 1], :] = [k, Nei[k, 1]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 2], :] = [k, Nei[k, 3]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 3], :] = [k, Nei[k, 0]]  # Peruzzo 2019
                # How neighbours are sorted within Nei: [left, right, bottom, up]
                for s in range(0, 4):  # Peruzzo 2019
                    index = connElemEdges[k, s]  # Peruzzo 2019
                    if booleconnEdgesNodes[index] == 0:  # Peruzzo 2019
                        booleconnEdgesNodes[index] = 1  # Peruzzo 2019
                        if s < 3:  # Peruzzo 2019
                            connEdgesNodes[index, :] = [conn[k, s], conn[k, s + 1]]  # Peruzzo 2019
                        else:  # Peruzzo 2019
                            connEdgesNodes[index, :] = [conn[k, s], conn[k, 0]]  # Peruzzo 2019
                if i == (self.nx - 1) or j == (self.ny - 1) or i == 0 or j == 0:  # start Peruzzo 2019
                    if i == (self.nx - 1) and j != (self.ny - 1) and i != 0 and j != 0:  # right row of cells
                        # for each top left node
                        connNodesEdges[conn[k, 3], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # leftedge
                        # BottomEdgeOfTopLeftNeighboursElem
                        connNodesEdges[conn[k, 3], 2] = ((j + 1) * (2 * self.nx + 1) + (i - 1))
                        # RightEdgeOfTopLeftNeighboursElem
                        connNodesEdges[conn[k, 3], 3] = ((j + 1) * (2 * self.nx + 1) + self.nx + (i - 1) + 1)
                        connNodesEdges[conn[k, 1], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # rightedge
                        # RightEdgeOfBottomNeighboursElem
                        connNodesEdges[conn[k, 1], 2] = ((j - 1) * (2 * self.nx + 1) + self.nx + i + 1)
                        connNodesEdges[conn[k, 1], 3] = connElemEdges[k, 0]  # bottomedge #repeated
                        # connNodesElem:
                        #    |   |
                        # ___a___o
                        #    |   |
                        # ___o___b
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 1], 0] = Nei[k, 2]  # element: bottom left
                        connNodesElem[conn[k, 1], 1] = Nei[k, 2]  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 2] = Nei[k, 2]  # element: top right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top left (current k)
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 3], 0] = Nei[k, 0]  # element: bottom left
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 3], 2] = Nei[k, 3]  # element: top right
                        connNodesElem[conn[k, 3], 3] = Nei[k - 1, 3]  # element: top left

                    elif i != (self.nx - 1) and j == (self.ny - 1) and i != 0 and j != 0:  # top row of cells
                        # for each bottom left node
                        connNodesEdges[conn[k, 0], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 0], 1] = connElemEdges[k, 3]  # leftedge
                        # TopEdgeOfBottomLeftNeighboursElem
                        connNodesEdges[conn[k, 0], 2] = (((j - 1) + 1) * (2 * self.nx + 1) + (i - 1))
                        # RightEdgeOfBottomLeftNeighboursElem
                        connNodesEdges[conn[k, 0], 3] = ((j - 1) * (2 * self.nx + 1) + self.nx + (i - 1) + 1)
                        connNodesEdges[conn[k, 2], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # rightedge
                        # TopEdgeOfRightNeighboursElem
                        connNodesEdges[conn[k, 2], 2] = ((j + 1) * (2 * self.nx + 1) + (i + 1))
                        connNodesEdges[conn[k, 2], 3] = connElemEdges[k, 1]  # rightedge #repeated
                        # connNodesElem:
                        # ___o___b___
                        #    |   |
                        # ___a___o___
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 0], 0] = Nei[k - 1, 2]  # element: bottom left #repeated
                        connNodesElem[conn[k, 0], 1] = Nei[k, 2]  # element: bottom right #repeated
                        connNodesElem[conn[k, 0], 2] = k  # element: top right (current k)
                        connNodesElem[conn[k, 0], 3] = Nei[k, 0]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 0] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 2], 1] = Nei[k, 1]  # element: bottom right
                        connNodesElem[conn[k, 2], 2] = Nei[k + 1, 3]  # element: top right
                        connNodesElem[conn[k, 2], 3] = Nei[k, 3]  # element: top left

                    elif i != (self.nx - 1) and j != (self.ny - 1) and i == 0 and j != 0:  # left row of cells
                        # for each bottom right node
                        connNodesEdges[conn[k, 1], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # rightedge
                        # TopEdgeOfBottomRightNeighboursElem
                        connNodesEdges[conn[k, 1], 2] = (((j - 1) + 1) * (2 * self.nx + 1) + (i + 1))
                        # LeftEdgeOfBottomRightNeighboursElem
                        connNodesEdges[conn[k, 1], 3] = ((j - 1) * (2 * self.nx + 1) + self.nx + (i + 1))
                        connNodesEdges[conn[k, 3], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # leftedge
                        # LeftEdgeOfTopNeighboursElem
                        connNodesEdges[conn[k, 3], 2] = ((j + 1) * (2 * self.nx + 1) + self.nx + i)
                        connNodesEdges[conn[k, 3], 3] = connElemEdges[k, 2]  # topedge #repeated
                        # connNodesElem:
                        # |   |
                        # a___o___
                        # |   |
                        # o___b___
                        # |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 1], 0] = Nei[k, 2]  # element: bottom left
                        connNodesElem[conn[k, 1], 1] = Nei[k + 1, 2]  # element: bottom right
                        connNodesElem[conn[k, 1], 2] = Nei[k, 1]  # element: top right
                        connNodesElem[conn[k, 1], 3] = k  # element: top left  (current k)
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 3], 0] = k  # element: bottom left  #repeated
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 3], 2] = Nei[k, 3]  # element: top right
                        connNodesElem[conn[k, 3], 3] = Nei[k, 3]  # element: top left  #repeated

                    elif i != (self.nx - 1) and j != (self.ny - 1) and i != 0 and j == 0:  # bottom row of cells
                        # for each top right node
                        connNodesEdges[conn[k, 2], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # rightedge
                        # BottomEdgeOfTopRightNeighboursElem
                        connNodesEdges[conn[k, 2], 2] = ((j + 1) * (2 * self.nx + 1) + (i + 1))
                        # LeftEdgeOfTopRightNeighboursElem
                        connNodesEdges[conn[k, 2], 3] = ((j + 1) * (2 * self.nx + 1) + self.nx + (i + 1))
                        connNodesEdges[conn[k, 0], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 0], 1] = connElemEdges[k, 3]  # leftedge
                        # BottomEdgeOfLeftNeighboursElem
                        connNodesEdges[conn[k, 0], 2] = (j * (2 * self.nx + 1) + (i - 1))
                        connNodesEdges[conn[k, 0], 3] = connElemEdges[k, 3]  # leftedge # repeated
                        # connNodesElem:
                        #    |   |
                        # ___o___b___
                        #    |   |
                        # ___a___o___
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 0], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 0], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 0], 2] = k  # element: top right (current k)
                        connNodesElem[conn[k, 0], 3] = Nei[k, 0]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 0] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 2], 1] = Nei[k, 1]  # element: bottom right
                        connNodesElem[conn[k, 2], 2] = Nei[k + 1, 1]  # element: top right
                        connNodesElem[conn[k, 2], 3] = Nei[k, 3]  # element: top left

                    elif i == (self.nx - 1) and j == (self.ny - 1):  # corner cell: top right
                        connNodesEdges[conn[k, 2], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # rightedge
                        connNodesEdges[conn[k, 2], 2] = connElemEdges[k, 2]  # topedge   #repeated
                        connNodesEdges[conn[k, 2], 3] = connElemEdges[k, 1]  # rightedge #repeated
                        connNodesEdges[conn[k, 1], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # rightedge
                        connNodesEdges[conn[k, 1], 2] = (
                                    (j - 1) * (2 * self.nx + 1) + self.nx + i + 1)  # RightEdgeBottomCell
                        connNodesEdges[conn[k, 1], 3] = connElemEdges[k, 0]  # bottomedge  #repeated
                        # connNodesElem:
                        # ___o___b
                        #    |   |
                        # ___o___a
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 1], 0] = Nei[k, 2]  # element: bottom left
                        connNodesElem[conn[k, 1], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 2] = k  # element: top right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top left (current k)
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 0] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 2], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 2], 2] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 2], 3] = k  # element: top left  #repeated

                    elif i == (self.nx - 1) and j == 0:  # corner cell: bottom right
                        connNodesEdges[conn[k, 1], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # rightedge
                        connNodesEdges[conn[k, 1], 2] = connElemEdges[k, 0]  # bottomedge #repeated
                        connNodesEdges[conn[k, 1], 3] = connElemEdges[k, 1]  # rightedge  #repeated
                        connNodesEdges[conn[k, 0], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 0], 1] = connElemEdges[k, 3]  # leftedge
                        connNodesEdges[conn[k, 0], 2] = (j * (2 * self.nx + 1) + (i - 1))  # BottomEdgeLeftCell
                        connNodesEdges[conn[k, 0], 3] = connElemEdges[k, 3]  # leftedge  #repeated
                        # connNodesElem:
                        #    |   |
                        # ___o___o
                        #    |   |
                        # ___a___b
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 0], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 0], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 0], 2] = k  # element: top right (current k)
                        connNodesElem[conn[k, 0], 3] = Nei[k, 0]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 1], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 1], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 2] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top left  (current k)

                    elif i == 0 and j == (self.ny - 1):  # corner cell: top left
                        connNodesEdges[conn[k, 3], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # leftedge
                        connNodesEdges[conn[k, 3], 2] = connElemEdges[k, 2]  # topedge #repeated
                        connNodesEdges[conn[k, 3], 3] = connElemEdges[k, 3]  # leftedge #repeated
                        connNodesEdges[conn[k, 2], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # rightedge
                        connNodesEdges[conn[k, 2], 2] = ((j + 1) * (2 * self.nx + 1) + (i + 1))  # TopEdgeRightCell
                        connNodesEdges[conn[k, 2], 3] = connElemEdges[k, 1]  # rightedge #repeated
                        # connNodesElem:
                        # b___a___
                        # |   |
                        # o___o___
                        # |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 2], 0] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 2], 1] = Nei[k, 1]  # element: bottom right
                        connNodesElem[conn[k, 2], 2] = k  # element: top right #repeated
                        connNodesElem[conn[k, 2], 3] = k  # element: top left #repeated
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 3], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 3], 2] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 3], 3] = k  # element: top left  #repeated

                    elif i == 0 and j == 0:  # corner cell: bottom left
                        connNodesEdges[conn[k, 0], 0] = connElemEdges[k, 0]  # bottomedge
                        connNodesEdges[conn[k, 0], 1] = connElemEdges[k, 3]  # leftedge
                        connNodesEdges[conn[k, 0], 2] = connElemEdges[k, 0]  # bottomedge #repeated
                        connNodesEdges[conn[k, 0], 3] = connElemEdges[k, 3]  # leftedge #repeated
                        connNodesEdges[conn[k, 3], 0] = connElemEdges[k, 2]  # topedge
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # leftedge
                        connNodesEdges[conn[k, 3], 2] = ((j + 1) * (2 * self.nx + 1) + self.nx + i)  # LeftEdgeTopCell
                        connNodesEdges[conn[k, 3], 3] = connElemEdges[k, 2]  # topedge #repeated
                        # connNodesElem:
                        #
                        # |   |
                        # a___o___
                        # |   |
                        # b___o___
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 3], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 3], 2] = Nei[k, 3]  # element: top right #repeated
                        connNodesElem[conn[k, 3], 3] = k  # element: top left #repeated
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 0], 0] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 0], 1] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 0], 2] = k  # element: top right  (current k)
                        connNodesElem[conn[k, 0], 3] = k  # element: top left  #repeated
                else:
                    node = conn[k, 1]  # for each bottom right node of the elements not near the mesh boundaryes
                    connNodesEdges[node, 0] = connElemEdges[k, 1]  # rightedge
                    connNodesEdges[node, 1] = connElemEdges[k, 0]  # bottomedge
                    connNodesEdges[node, 2] = connElemEdges[Nei[k, 2], 1]  # leftedgeBottomNeighboursElem
                    connNodesEdges[node, 3] = connElemEdges[Nei[k + 1, 2], 2]  # bottomedgeLeftNeighboursElem
                    # connNodesElem:
                    # note:  NeiElements(ndarray): [left, right,bottom, up]
                    # o___o___o
                    # | 3k| 2 |     k is the current element
                    # o___x___o     x is the current node
                    # | 0 | 1 |
                    # o___o___o
                    #
                    connNodesElem[node, 0] = Nei[k, 2]  # element: bottom left with respect to the node x
                    connNodesElem[node, 1] = Nei[k + 1, 2]  # element: bottom right with respect to the node x
                    connNodesElem[node, 2] = Nei[k, 1]  # element: top right  with respect to the node x
                    connNodesElem[node, 3] = k  # element: top left (current k) with respect to the node x
                    # end Peruzzo 2019

                k = k + 1

        self.Connectivity = conn
        self.Connectivityelemedges = connElemEdges  # Peruzzo 2019
        self.Connectivityedgeselem = connEdgesElem  # Peruzzo 2019
        self.Connectivityedgesnodes = connEdgesNodes  # Peruzzo 2019
        self.Connectivitynodesedges = connNodesEdges  # Peruzzo 2019
        self.Connectivitynodeselem = connNodesElem  # Peruzzo 2019

        # coordinates of the center of the mesh
        centerMesh = np.asarray([(self.domainLimits[2] + self.domainLimits[3])/2,
                                 (self.domainLimits[1] + self.domainLimits[0])/2])

        # coordinates of the center of the elements
        CoorMid = np.empty([self.NumberOfElts, 2], dtype=float)
        for e in range(0, self.NumberOfElts):
            t = np.reshape(self.VertexCoor[conn[e]], (4, 2))
            CoorMid[e] = np.mean(t, axis=0)
        self.CenterCoor = CoorMid

        self.distCenter = ((CoorMid[:, 0] - centerMesh[0]) ** 2 + (CoorMid[:, 1] - centerMesh[1]) ** 2) ** 0.5



        # the element in the center (used for fluid injection)
        # todo: No it is not necessarily where we inject!
        self.CenterElts = np.intersect1d(np.where(abs(self.CenterCoor[:, 0] - centerMesh[0]) < self.hx/2),
                                         np.where(abs(self.CenterCoor[:, 1] - centerMesh[1]) < self.hy/2))
        if len(self.CenterElts) != 1:
            self.CenterElts = self.NumberOfElts / 2
            log.debug("Mesh with no center element. To be looked into")
            #todo
            #raise ValueError("Mesh with no center element. To be looked into")

        if symmetric:
            self.corresponding = corresponding_elements_in_symmetric(self)
            self.symmetricElts = get_symetric_elements(self, np.arange(self.NumberOfElts))
            self.activeSymtrc, self.posQdrnt, self.boundary_x, self.boundary_y = get_active_symmetric_elements(self)

            self.volWeights = np.full((len(self.activeSymtrc), ), 4., dtype=np.float32)
            self.volWeights[len(self.posQdrnt): -1] = 2.
            self.volWeights[-1] = 1.


    # -----------------------------------------------------------------------------------------------------------------------

    def locate_element(self, x, y):
        """
        This function gives the cell containing the given coordinates. Numpy nan is returned if the cell is not in
        the mesh.

        Args:
            x (float):  -- the x coordinate of the given point.
            y (float):  -- the y coordinate of the given point.

        Returns:
            elt (int):  -- the element containing the given coordinates.

        """
        log = logging.getLogger('PyFrac.locate_element')
        if x >= self.domainLimits[3] + self.hx / 2 or y >= self.domainLimits[1] + self.hy / 2\
                or x <= self.domainLimits[2] - self.hx / 2 or y <= self.domainLimits[0] - self.hy / 2:
            log.warning("Point is outside domain.")
            return np.nan

        precision = 0.1*np.sqrt(np.finfo(float).eps)

        return np.intersect1d(np.where(abs(self.CenterCoor[:, 0] - x) < self.hx / 2 + precision),
                       np.where(abs(self.CenterCoor[:, 1] - y) < self.hy / 2 + precision))

#-----------------------------------------------------------------------------------------------------------------------

    def Neighbors(self, elem, nx, ny):
        """
        Neighbouring elements of an element within the mesh. Boundary elements have themselves as neighbor.

        Args:
            elem (int):         -- element whose neighbor are to be found.
            nx (int):           -- number of elements in x direction.
            ny (int):           -- number of elements in y direction.

        Returns:
            (tuple): A tuple containing the following:

                | left (int)     -- left neighbour.
                | right (int)    -- right neighbour.
                | bottom (int)   -- bottom neighbour.
                | top (int)      -- top neighbour.

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

        return left, right, bottom, up

# ----------------------------------------------------------------------------------------------------------------------


    def plot(self, material_prop=None, backGround_param=None, fig=None, plot_prop=None):
        """
        This function plots the mesh in 2D. If the material properties is given, the cells will be color coded
        according to the parameter given by the backGround_param argument.

        Args:
            material_prop (MaterialProperties):  -- a MaterialProperties class object
            backGround_param (string):           -- the cells of the grid will be color coded according to the value
                                                    of the parameter given by this argument. Possible options are
                                                    'sigma0' for confining stress, 'K1c' for fracture toughness and
                                                    'Cl' for leak off.
            fig (Figure):                        -- A figure object to superimpose.
            plot_prop (PlotProperties):          -- A PlotProperties object giving the properties to be utilized for
                                                    the plot.

        Returns:
            (Figure):                            -- A Figure object to superimpose.

        """

        if fig is None:
            fig, ax = plt.subplots()
        else:
            plt.figure(fig.number)
            plt.subplot(111)
            ax = fig.get_axes()[0]

        # set the four corners of the rectangular mesh
        ax.set_xlim([self.domainLimits[2] - self.hx / 2, self.domainLimits[3] + self.hx / 2])
        ax.set_ylim([self.domainLimits[0] - self.hy / 2, self.domainLimits[1] + self.hy / 2])

        # add rectangle for each cell
        patches = []
        for i in range(self.NumberOfElts):
            polygon = mpatches.Polygon(np.reshape(self.VertexCoor[self.Connectivity[i], :], (4, 2)), True)
            patches.append(polygon)

        if plot_prop is None:
            plot_prop = PlotProperties()
            plot_prop.alpha = 0.65
            plot_prop.lineColor = '0.5'
            plot_prop.lineWidth = 0.2

        p = PatchCollection(patches,
                            cmap=plot_prop.colorMap,
                            alpha=plot_prop.alpha,
                            edgecolor=plot_prop.meshEdgeColor,
                            linewidth=plot_prop.lineWidth)

        # applying color according to the prescribed parameter
        if material_prop is not None and backGround_param is not None:
            min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                        backGround_param)
            # plotting color bar
            sm = plt.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                       norm=plt.Normalize(vmin=min_value, vmax=max_value))
            sm._A = []
            clr_bar = fig.colorbar(sm, alpha=0.65)
            clr_bar.set_label(parameter)

        else:
            colors = np.full((self.NumberOfElts,), 0.5)

        p.set_array(np.array(colors))
        ax.add_collection(p)
        plt.axis('equal')

        return fig


#-----------------------------------------------------------------------------------------------------------------------

    def plot_3D(self, material_prop=None, backGround_param=None, fig=None, plot_prop=None):
        """
        This function plots the mesh in 3D. If the material properties is given, the cells will be color coded
        according to the parameter given by the backGround_param argument.

        Args:
            material_prop (MaterialProperties):  -- a MaterialProperties class object
            backGround_param (string):           -- the cells of the grid will be color coded according to the value
                                                    of the parameter given by this argument. Possible options are
                                                    'sigma0' for confining stress, 'K1c' for fracture toughness and
                                                    'Cl' for leak off.
            fig (Figure):                        -- A figure object to superimpose.
            plot_prop (PlotProperties):          -- A PlotProperties object giving the properties to be utilized for
                                                    the plot.

        Returns:
            (Figure):                            -- A Figure object to superimpose.

        """
        log = logging.getLogger('PyFrac.plot3D')
        if backGround_param is not None and material_prop is None:
            raise ValueError("Material properties are required to plot the background parameter.")
        if material_prop is not None and backGround_param is None:
            log.warning("back ground parameter not provided. Plotting confining stress...")
            backGround_param = 'sigma0'

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlim([self.domainLimits[2] * 1.2, self.domainLimits[3] * 1.2])
            ax.set_ylim([self.domainLimits[0] * 1.2, self.domainLimits[1] * 1.2])
            scale = 1.1
            zoom_factory(ax, base_scale=scale)
        else:
            ax = fig.get_axes()[0]

        if plot_prop is None:
            plot_prop = PlotProperties()
        if plot_prop.textSize is None:
            plot_prop.textSize = max(self.Lx / 15, self.Ly / 15)

        log.info("Plotting mesh in 3D...")
        if material_prop is not None and backGround_param is not None:
            min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                        backGround_param)

        # add rectangle for each cell
        for i in range(self.NumberOfElts):
            rgb_col = to_rgb(plot_prop.meshColor)
            if backGround_param is not None:
                face_color = (rgb_col[0] * colors[i], rgb_col[1] * colors[i], rgb_col[2] * colors[i], 0.5)
            else:
                face_color = (rgb_col[0], rgb_col[1], rgb_col[2], 0.5)

            rgb_col = to_rgb(plot_prop.meshEdgeColor)
            edge_color = (rgb_col[0], rgb_col[1], rgb_col[2], 0.2)
            cell = mpatches.Rectangle((self.CenterCoor[i, 0] - self.hx / 2,
                                       self.CenterCoor[i, 1] - self.hy / 2),
                                       self.hx,
                                       self.hy,
                                       ec=edge_color,
                                       fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)

        if backGround_param is not None and material_prop is not None:
            make_3D_colorbar(self, material_prop, backGround_param, ax, plot_prop)

        self.plot_scale_3d(ax, plot_prop)

        ax.grid(False)
        ax.set_frame_on(False)
        ax.set_axis_off()
        set_aspect_equal_3d(ax)
        return fig


#-----------------------------------------------------------------------------------------------------------------------

    def plot_scale_3d(self, ax, plot_prop):
        """
        This function plots the scale of the fracture by adding lines giving the length dimensions of the fracture.

        """
        log = logging.getLogger('PyFrac.plot_scale_3d')
        log.info("\tPlotting scale...")

        Path = mpath.Path

        rgb_col = to_rgb(plot_prop.meshLabelColor)
        edge_color = (rgb_col[0], rgb_col[1], rgb_col[2], 1.)

        codes = []
        verts = []
        verts_x = np.linspace(self.domainLimits[2], self.domainLimits[3], 7)
        verts_y = np.linspace(self.domainLimits[0], self.domainLimits[1], 7)
        tick_len = max(self.hx / 2, self.hy / 2)
        for i in range(7):
            codes.append(Path.MOVETO)
            elem = self.locate_element(verts_x[i], self.domainLimits[0])
            verts.append((self.CenterCoor[elem, 0], self.domainLimits[0] - self.hy / 2))
            codes.append(Path.LINETO)
            verts.append((self.CenterCoor[elem, 0], self.domainLimits[0] + tick_len))
            x_val = to_precision(np.round(self.CenterCoor[elem, 0], 5), plot_prop.dispPrecision)
            text3d(ax,
                   (self.CenterCoor[elem, 0] - plot_prop.dispPrecision * plot_prop.textSize / 3,
                    self.domainLimits[0] - self.hy / 2 - plot_prop.textSize,
                    0),
                   x_val,
                   zdir="z",
                   size=plot_prop.textSize,
                   usetex=plot_prop.useTex,
                   ec="none",
                   fc=edge_color)

            codes.append(Path.MOVETO)
            elem = self.locate_element(self.domainLimits[2], verts_y[i])
            verts.append((self.domainLimits[2] - self.hx / 2, self.CenterCoor[elem, 1][0]))
            codes.append(Path.LINETO)
            verts.append((self.domainLimits[2] + tick_len, self.CenterCoor[elem, 1][0]))
            y_val = to_precision(np.round(self.CenterCoor[elem, 1], 5), plot_prop.dispPrecision)
            text3d(ax,
                   (self.domainLimits[2] - self.hx / 2 - plot_prop.dispPrecision * plot_prop.textSize,
                    self.CenterCoor[elem, 1] - plot_prop.textSize / 2,
                    0),
                   y_val,
                   zdir="z",
                   size=plot_prop.textSize,
                   usetex=plot_prop.useTex,
                   ec="none",
                   fc=edge_color)

        log.info("\tAdding labels...")
        text3d(ax,
               (0.,
                -self.domainLimits[2] - plot_prop.textSize * 3,
                0),
               'meters',
               zdir="z",
               size=plot_prop.textSize,
               usetex=plot_prop.useTex,
               ec="none",
               fc=edge_color)

        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,
                                   lw=plot_prop.lineWidth,
                                   facecolor='none',
                                   edgecolor=edge_color)
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch)

#-----------------------------------------------------------------------------------------------------------------------


    def identify_elements(self, elements, fig=None, plot_prop=None, plot_mesh=True, print_number=True):
        """
        This functions identify the given set of elements by highlighting them on the grid. the function plots
        the grid and the given set of elements.

        Args:
            elements (ndarray):             -- the given set of elements to be highlighted.
            fig (Figure):                   -- A figure object to superimpose.
            plot_prop (PlotProperties):     -- A PlotProperties object giving the properties to be utilized for
                                               the plot.
            plot_mesh (bool):               -- if False, grid will not be plotted and only the edges of the given
                                               elements will be plotted.
            print_number (bool):            -- if True, numbers of the cell will also be printed along with outline.

        Returns:
            (Figure):                       -- A Figure object that can be used superimpose further plots.

        """

        if plot_prop is None:
            plot_prop = PlotProperties()

        if plot_mesh:
            fig = self.plot(fig=fig)

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]

        # set the four corners of the rectangular mesh
        ax.set_xlim([self.domainLimits[2] - self.hx / 2, self.domainLimits[3] + self.hx / 2])
        ax.set_ylim([self.domainLimits[0] - self.hy / 2, self.domainLimits[1] + self.hy / 2])

        # add rectangle for each cell
        patch_list = []
        for i in elements:
            polygon = mpatches.Polygon(np.reshape(self.VertexCoor[self.Connectivity[i], :], (4, 2)), True)
            patch_list.append(polygon)

        p = PatchCollection(patch_list,
                            cmap=plot_prop.colorMap,
                            edgecolor=plot_prop.lineColor,
                            linewidth=plot_prop.lineWidth,
                            facecolors='none')
        ax.add_collection(p)

        if print_number:
            # print Element numbers on the plot for elements to be identified
            for i in range(len(elements)):
                ax.text(self.CenterCoor[elements[i], 0] - self.hx / 4, self.CenterCoor[elements[i], 1] -
                        self.hy / 4, repr(elements[i]), fontsize=plot_prop.textSize)

        return fig

#-----------------------------------------------------------------------------------------------------------------------


def make_3D_colorbar(mesh, material_prop, backGround_param, ax, plot_prop):
    """
    This function makes the color bar on 3D mesh plot using rectangular patches with color gradient from gray to the
    color given by the plot properties. The minimum and maximum values are taken from the given parameter in the
    material properties.

    """
    log = logging.getLogger('PyFrac.make_3D_colorbar')
    log.info("\tMaking colorbar...")

    min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                backGround_param)
    rgb_col_mesh = to_rgb(plot_prop.meshEdgeColor)
    edge_color = (rgb_col_mesh[0],
                  rgb_col_mesh[1],
                  rgb_col_mesh[2],
                  0.2)

    color_range = np.linspace(0, 1., 11)
    y = np.linspace(-mesh.Ly, mesh.Ly, 11)
    dy = y[1] - y[0]
    for i in range(11):
        rgb_col = to_rgb(plot_prop.meshColor)
        face_color = (rgb_col[0] * color_range[i],
                      rgb_col[1] * color_range[i],
                      rgb_col[2] * color_range[i],
                      0.5)
        cell = mpatches.Rectangle((mesh.Lx + 4 * mesh.hx,
                                   y[i]),
                                  2 * dy,
                                  dy,
                                  ec=edge_color,
                                  fc=face_color)
        ax.add_patch(cell)
        art3d.pathpatch_2d_to_3d(cell)

    rgb_col_txt = to_rgb(plot_prop.meshLabelColor)
    txt_color = (rgb_col_txt[0],
                 rgb_col_txt[1],
                 rgb_col_txt[2],
                 1.0)
    text3d(ax,
           (mesh.Lx + 4 * mesh.hx, y[9] + 3 * dy, 0),
           parameter,
           zdir="z",
           size=plot_prop.textSize,
           usetex=plot_prop.useTex,
           ec="none",
           fc=txt_color)
    y = [y[0], y[5], y[10]]
    values = np.linspace(min_value, max_value, 11)
    values = [values[0], values[5], values[10]]
    for i in range(3):
        disp_val = to_precision(values[i], plot_prop.dispPrecision)
        text3d(ax,
               (mesh.Lx + 4 * mesh.hx + 2 * dy, y[i] + dy / 2, 0),
               disp_val,
               zdir="z",
               size=plot_prop.textSize,
               usetex=plot_prop.useTex,
               ec="none",
               fc=txt_color)

#-----------------------------------------------------------------------------------------------------------------------


def process_material_prop_for_display(material_prop, backGround_param):
    """
    This function generates the appropriate variables to display the color coded mesh background.

    """

    colors = np.full((len(material_prop.SigmaO),), 0.5)

    if backGround_param in ['confining stress', 'sigma0']:
        max_value = max(material_prop.SigmaO) / 1e6
        min_value = min(material_prop.SigmaO) / 1e6
        if max_value - min_value > 0:
            colors = (material_prop.SigmaO / 1e6 - min_value) / (max_value - min_value)
        parameter = "confining stress ($MPa$)"
    elif backGround_param in ['fracture toughness', 'K1c']:
        max_value = max(material_prop.K1c) / 1e6
        min_value = min(material_prop.K1c) / 1e6
        if max_value - min_value > 0:
            colors = (material_prop.K1c / 1e6 - min_value) / (max_value - min_value)
        parameter = "fracture toughness ($Mpa\sqrt{m}$)"
    elif backGround_param in ['leak off coefficient', 'Cl']:
        max_value = max(material_prop.Cl)
        min_value = min(material_prop.Cl)
        if max_value - min_value > 0:
            colors = (material_prop.Cl - min_value) / (max_value - min_value)
        parameter = "Leak off coefficient"
    elif backGround_param is not None:
        raise ValueError("Back ground color identifier not supported!\n"
                         "Select one of the following:\n"
                         "-- \'confining stress\' or \'sigma0\'\n"
                         "-- \'fracture toughness\' or \'K1c\'\n"
                         "-- \'leak off coefficient\' or \'Cl\'")

    return min_value, max_value, parameter, colors


#-----------------------------------------------------------------------------------------------------------------------

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
