# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 11:51:00 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# import
import numpy as np
from src.Utility import Neighbors
import sys



class CartesianMesh:
    """ Class defining a Cartesian Mesh.

        Instance variables:
            Lx,Ly (float)           -- length of the domain in x and y directions respectively. The rectangular domain
                                       have a total length of 2xLx in the x direction and 2xLy in the y direction if
                                       both the positive and negative halves are included.
            nx,ny (int)             -- number of elements in x and y directions respectively
            hx,hy (float)           -- grid spacing in x and y directions respectively
            VertexCoor  (ndarray)   -- [x,y] Coordinates of the vertices
            CenterCoor  (ndarray)   -- [x,y] coordinates of the center of the elements
            NumberOfElts (int)      -- total number of elements in the mesh
            EltArea (float)         -- area of each element
            Connectivity (ndarray)  -- connectivity array giving four vertices of an element in the following order
                                        3         2
                                        0         1
            NeiElements (ndarray)   -- Giving four neighbouring elements with the following order:[left,right,bottom,up]
            distCenter (ndarray)    -- the distance of the cells from the center
            CenterElts (ndarray)    -- the element(s) in the center (the cell with the injection point)
            
        Methods:
            __init__()      -- create a uniform Cartesian mesh centered  [-Lx,Lx]*[-Ly,Ly]
            remesh()        -- remesh the grid uniformly by increasing the domain lengths with the given factor in both
                               x and y directions
    """

    def __init__(self, Lx, Ly, nx, ny):
        """ 
        Creates a uniform Cartesian mesh centered at zero and having the dimensions of [-Lx,Lx]*[-Ly,Ly]
        Arguments:
            nx,ny (int)     -- number of elements in x and y directions respectively
            Lx,Ly (float)   -- lengths in x and y directions respectively
        """

        self.Lx = Lx
        self.Ly = Ly

        # Check if the number of cells is odd to see if the origin would be at the mid point of a single cell
        if nx % 2 == 0:
            print("Number of elements in x-direction are even. Using " + repr(nx+1) + " elements to have origin at a "
                                                                                      "cell center...")
            self.nx = nx+1
        else:
            self.nx = nx

        if ny % 2 == 0:
            print("Number of elements in y-direction are even. Using " + repr(ny+1) + " elements to have origin at a "
                                                                                      "cell center...")
            self.ny = ny+1
        else:
            self.ny = ny


        self.hx = 2. * Lx / (self.nx - 1)
        self.hy = 2. * Ly / (self.ny - 1)

        x = np.linspace(-Lx - self.hx / 2., Lx + self.hx / 2., self.nx + 1)
        y = np.linspace(-Ly - self.hy / 2., Ly + self.hy / 2., self.ny + 1)

        xv, yv = np.meshgrid(x, y)  # coordinates of the vertex of each elements

        a = np.resize(xv, ((self.nx + 1) * (self.ny + 1), 1))
        b = np.resize(yv, ((self.nx + 1) * (self.ny + 1), 1))

        self.VertexCoor = np.reshape(np.stack((a, b), axis=-1), (len(a), 2))

        self.NumberOfElts = self.nx * self.ny
        self.EltArea = self.hx * self.hy

        # Connectivity array giving four vertices of an element in the following order
        #     3         2
        #     0   -     1
        conn = np.empty([self.NumberOfElts, 4], dtype=int)
        k = 0
        for j in range(0, self.ny):
            for i in range(0, self.nx):
                conn[k, 0] = (i + j * (self.nx + 1))
                conn[k, 1] = (i + 1) + j * (self.nx + 1)
                conn[k, 2] = i + 1 + (j + 1) * (self.nx + 1)
                conn[k, 3] = i + (j + 1) * (self.nx + 1)
                k = k + 1

        self.Connectivity = conn

        # coordinates of the center of the elements
        CoorMid = np.empty([self.NumberOfElts, 2], dtype=float)
        for e in range(0, self.NumberOfElts):
            t = np.reshape(self.VertexCoor[conn[e]], (4, 2))
            CoorMid[e] = np.mean(t, axis=0)
        self.CenterCoor = CoorMid

        self.distCenter = (CoorMid[:, 0] ** 2 + CoorMid[:, 1] ** 2) ** 0.5

        # Giving four neighbouring elements in the following order: [left,right,bottom,up]
        Nei = np.zeros((self.NumberOfElts, 4), int)
        for i in range(0, self.NumberOfElts):
            Nei[i, :] = np.asarray(Neighbors(i, self.nx, self.ny))
        self.NeiElements = Nei

        # the element in the center (used for fluid injection)
        (minx, miny) = (min(abs(self.CenterCoor[:, 0])), min(abs(self.CenterCoor[:, 1])))
        self.CenterElts = np.intersect1d(np.where(abs(self.CenterCoor[:, 0]) < self.hx/2),
                                         np.where(abs(self.CenterCoor[:, 1]) < self.hy/2))
        if self.CenterElts.size != 1:
            #todo
            raise ValueError("Mesh with no center element. To be looked into")

    def locate_element(self, x, y):
        """
        This function gives the cell containing the given coordinates
        Arguments:
            x (float)  -- the x coordinate of the given point
            y (float)  -- the y coordinate of the given point

        Returns:
            elt (int)  -- the element containing the given coordinates
        """
        elt = np.intersect1d(np.where(abs(self.CenterCoor[:, 0] - x) <= self.hx/2.+sys.float_info.epsilon)[0],
                           np.where(abs(self.CenterCoor[:, 1] - y) <= self.hy/2.+sys.float_info.epsilon)[0])
        if elt.size>1:
            print("two found")
            return elt[0]
        return elt

#-----------------------------------------------------------------------------------------------------------------------

