# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 11:51:00 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""
# import
import numpy as np
from importlib.machinery import SourceFileLoader

from src.Utility import Neighbors



class CartesianMesh:
    """ Class defining a Cartesian Mesh.

        instance variables:
            Lx,Ly           -- length in x and y directions respectively
            nx,ny           -- number of elements in x and y directions respectively
            hx,hy           -- grid spacing in x and y directions respectively            
            VertexCoor      -- [x,y] Coordinates of the vertices
            NumberOfElts    -- total number of elements in the mesh
            EltArea         -- Area of each element
            Connectivity    -- Connectivity array giving four vertices of an element in the following order
                                3         2
                                0         1
            CenterCoor      -- [x,y] coordinates of the center of the elements
            NeiElements     -- Giving four neigbouring elements with the following order: [left,right,bottom,up]
            CenterElts      -- the element(s) in the center (used primerily for fluid injection)
            
        methods:
            __init__()      -- create a uniform Cartesian mesh centered  [-Lx,Lx]*[-Ly,Ly]
            remesh()        -- remesh the grid uniformly with a given factor to grid spacing 
    """

    def __init__(self, Lx, Ly, nx, ny):
        """ 
        Constructor
        create a uniform Cartesian mesh centered  [-Lx,Lx]*[-Ly,Ly]
        input:
            nx,ny -- number of elt in x and y directions respectively
            Lx,Ly -- lengths in x and y directions respectively
        """

        self.Lx = Lx
        self.Ly = Ly

        # Check if odd to have origin at mid points of a single element
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

        ### Connectivity array giving four vertices of an element in the following order
        #     3         2
        #     0   -     1
        conn = np.empty([self.NumberOfElts, 4], dtype=int)  # to ensure an array of integers.
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

        # the element(s) in the center (used for fluid injection)
        (minx, miny) = (min(abs(self.CenterCoor[:, 0])), min(abs(self.CenterCoor[:, 1])))
        self.CenterElts = np.intersect1d(np.where(abs(self.CenterCoor[:, 0]) - minx < 1e-10),
                                         np.where(abs(self.CenterCoor[:, 1]) - miny < 1e-10))

    def locate_element(self, x, y):
        e = np.intersect1d(np.where(abs(self.CenterCoor[:, 0]) - x < 1e-10),
                           np.where(abs(self.CenterCoor[:, 1]) - y < 1e-10))
        # todo: form mesh according to the injection point
        return e

#############################################
