# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# TODO: For now we need a PyFrac mesh object --> Generalize
# TODO: Implement the possibility of adding a Velocity field
# TODO: allow to pass a function to solve for (not necessarily the distance funciton of the level set)

# imports
import heapq
import numpy as np

class fmm:
    """
    Class to solve the fast marching method (fmm) on a given mesh.

    Args:
        mesh (CartesianMesh)    -- a CartesianMesh class object describing the grid.

    Attributes:
        neiElems (ndarray)      -- Contains the neighbouring element for every element in the mesh.
        status (ndarray)        -- Value of -1 if far away, 0 if in the neighbourhood and, 1 if known (-2 for locations
                                   where no solution is calculated)
        LS (ndarray)            -- Solution of the Eikonal equation
        heapStruct (tuple)      -- Tuple containing the level set and the index of the elements in the neighbourhood.
                               heapified structure to get faster acces to smallest element.

    """

    def __init__(self, mesh):
        """
        Constructor of an fmm object.

        """

        # We set the status of all elements to far away
        self.Status = np.full((mesh.NumberOfElts,), -2, dtype=int)

        # we store the neighbouring elements
        self.neiElems = mesh.NeiElements

        # We initialize the level set as infinity
        self.LS = np.full((mesh.NumberOfElts,), np.inf)

    def n2k(self, newKnown):
        """
            Change the status of cells from being a neighbour (0) to known (1)

            Arguments:
                newKnown (ndarray)  -- contains the element id of the cells to switch from neighbour to known.

            Note: Does automatically change the status of the neighbours of the newKnown cells to neighbours (no longer
                  far field) by calling the function self.f2n(newKnown)
        """
        # Setting the status of the cells where we know the level set to 1
        self.Status[newKnown] = 1
        # Calling the function self.f2n(newKnown) to set the status of the new neighbours to 0.
        self.f2n(newKnown)

    def f2n(self, newKnown):
        """
            Change the status of cells from being far away (-1) to a neighbour (0)

            Arguments:
                newKnown (ndarray)  -- contains the element id of the cells to switch from neighbour to known.

            Note: Autmoatically called by the function self.n2k(newKnown) to update the list of neighbours.
        """
        # We get the neighbours from the list
        neighbors = self.get_n(newKnown)
        # We check which neighbours are wheter known (1) nor not to evaluate (-2)
        mask = (self.Status[neighbors] != 1) * (self.Status[neighbors] != -2)
        # We change the status of the new neighbours to 0
        self.Status[neighbors[mask]] = 0

    def get_n(self, elements):
        """
            Function to get all neighbours of a cell.

            Arguments:
                elements (ndarray)                  -- contains the element id of the cells to switch from neighbour to
                                                       known.

            Return:
                self.neiElems[elements] (ndarray)   -- contains the four neighbours for every element in newKnown.

            Note: Autmoatically called by the function self.f2n(newKnown) to update the list of neighbours.
        """
        return self.neiElems[elements]

    def addLs(self, newLs):
        """
            Function to update the solution of the eikonal equation for a set of cells.

            Arguments:
                newLs (ndarray) -- contains the new value of the solution and the element id of the corresponding cell.

        """
        self.LS[newLs[1]] = newLs[0]

    def calcLs(self, calcLS, mesh):
        """
            Function to calculate the new level set of a set of points.

            Arguments:
                calcLS (ndarray)        -- contains the new value of the solution and the element id of the
                                           corresponding cell.
                mesh (CartesianMesh)    -- cartesian mesh object of PyFrac

            Return:
                [newLs, calcLS] (list)  -- A list containing the value of the calculated solution newLs and the id of
                                        -- cells calcLS.

            Note: only calculates the value of the solution of the eikonal equation. Does not update it automatically.

        """

        # Calculating the minimum value of the solution in x (theta_1) and y (theta_2)
        theta_1 = self.LS[self.get_n(calcLS)[:, [0, 1]]].min(axis=1)
        theta_2 = self.LS[self.get_n(calcLS)[:, [2, 3]]].min(axis=1)

        # Initialize the solution vector
        newLS = np.full((len(calcLS),), -1.)

        # calculate the cell aspect ratio beta and the parameter theta^2
        beta = mesh.hx / mesh.hy
        theta_sq = mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * (theta_2 - theta_1) ** 2

        # we check where theta^2 is bigger than zero (mask) and where not (nmask)
        mask = theta_sq > 0
        nmask = np.invert(mask)

        # assigne the value of the solution to the eikonal equation according to the fact if theta^2 > or < 0
        newLS[mask] = np.asarray((theta_1 + beta ** 2 * theta_2 + theta_sq ** 0.5) / (1 + beta ** 2))[mask]
        newLS[nmask] = np.asarray([theta_2 + mesh.hy, theta_1 + mesh.hx]).min(axis=0)[nmask]

        # Returning a list with the newly calculated solution and the cells.
        return [newLS, calcLS]

    def solveFMM(self, known, toEvaluate, Mesh):
        """
            Function to solve the Eikonal equation using a FMM on the provided Mesh

            Arguments:
                known (ndarray)         -- contains the value and the id of the cells where the solution to the eikonal
                                           equation is known.
                toEvaluate (ndarray)    -- contains the id of all elements where the solution should be calculated
                Mesh (CartesianMesh)    -- cartesian mesh object of PyFrac

            Note: The function does not return anything but updates the attribute LS of the object. The solution to the
                  eikonal equation is thus stored in the attribute LS of the object. Non evaluated cells have a solution
                  of np.inf().

        """

        # we set the status of the elements according to the input data
        # Note: We do not check if we already have a calculated / known solution at those points. Solutions already
        #       stored and in known or evaluated here will get overriden.
        self.Status[toEvaluate] = -1    # cells we want to evaluate (-1)
        self.Status[known[1]] = 1       # cells we know the solution (1)
        self.LS[known[1]] = known[0]    # solution at the known cells

        # We generate the heap structure to have the FMM in n*Log10(n). The first elements in the heap are the locations
        # where the solution is known.
        self.heapStruct = list(map(tuple, np.asarray((known[0], known[1])).T))
        heapq.heapify(self.heapStruct)

        # Now we loop as long as we have cells in the neighbourhood
        while len(self.heapStruct) != 0:
            # We pop the smallest element of the heap
            evEl = heapq.heappop(self.heapStruct)
            evN = int(evEl[1]) # index of the element with the smallest element

            # We calculate the LS of the neighbors to the smallest object
            newNLS = self.calcLs(self.neiElems[evN], Mesh)

            # We search for the elements were we already have a calculated solution
            updN = (self.Status[newNLS[1]] == 0) * (newNLS[1] != evN)
            # if elements are int he neighbourhood and already has a solution, we update to the smallest solution
            if updN.any():
                # we check if one of the calculated solutions is smaller than the original solution
                toUpdate = self.LS[newNLS[1][updN]] > newNLS[0][updN]
                # if a newly calculated solution is smaller we update the solution vecotr and the heap
                if toUpdate.any():
                    # updating the solution
                    self.addLs([newNLS[0][updN][toUpdate], newNLS[1][updN][toUpdate]])

                    # find the index of the updated solution to update the heap
                    updH = np.where(np.in1d(list(zip(*self.heapStruct))[1], newNLS[1][updN][toUpdate]))[0]

                    # generate the tuples to introduce them into the heap
                    iter = 0
                    toUpdate = list(map(tuple, np.asarray((self.LS[newNLS[1][updN][toUpdate]],
                                                            newNLS[1][updN][toUpdate])).T))
                    # update the heap
                    for item in toUpdate:
                        self.heapStruct[updH[iter]] = item
                        iter += 1

            # Check for the new elements where we want to calculate the solution
            newN = self.Status[newNLS[1]] == -1
            # update the solution and add the elements to the heap
            if newN.any():
                self.addLs([newNLS[0][newN], newNLS[1][newN]])

                # push all new elements to the heap
                toPush = list(map(tuple, np.asarray((newNLS[0][newN], newNLS[1][newN])).T))
                for item in toPush:
                    heapq.heappush(self.heapStruct, item)

            # add all neighbors of the evaluated element to the ones to bee evaluated
            self.n2k(evN)