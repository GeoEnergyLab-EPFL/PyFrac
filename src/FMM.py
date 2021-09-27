import logging
import heapq

import numpy as np

class fmm:
    """
    Class to solve the fast marching method (fmm) on a grid.

    Attributes:
        indices (ndarray)   -- Contains the indices of all elements to solve for
        status (ndarray)    -- Value of -1 if far away, 0 if in the neighbourhood and, 1 if known
        LS (ndarray)        -- Value of the level set
        heapStruct (tuple)  -- Tuple containing the level set and the index of the elements in the neighbourhood.
                               heapified structure to get faster acces to smallest element.

    """

    def __init__(self, known, toEvaluate, mesh):

        # We set the status of all elements to far away
        self.Status = np.full((mesh.NumberOfElts,), -2, dtype=int)
        # We set the status of the known elements to known
        self.Status[toEvaluate] = -1
        self.Status[known[1]] = 0

        # we store the neighbouring elements
        self.neiElems = mesh.NeiElements

        # We initialize the level set as infinity
        self.LS = np.full((mesh.NumberOfElts,), np.inf)
        self.LS[known[1]] = known[0]

        self.heapStruct = list(map(tuple, np.asarray((known[0], known[1])).T))
        heapq.heapify(self.heapStruct)


    def n2k(self, newKnown):
        self.Status[newKnown] = 1
        self.f2n(newKnown)

    def f2n(self, newKnown):
        neighbors = self.get_n(newKnown)
        mask = (self.Status[neighbors] != 1) * (self.Status[neighbors] != -2)
        self.Status[neighbors[mask]] = 0

    def get_n(self, newKnown):
        return self.neiElems[newKnown]

    def addLs(self, newLs):
        self.LS[newLs[1]] = newLs[0]

    def calcLs(self, calcLS, mesh):

        # if type(calcLS) == np.ndarray:
        #     calcLS = calcLS.astype(int)
        # else:
        #     calcLS = int(calcLS)

        theta_1 = self.LS[self.neiElems[calcLS][:, [0, 1]]].min(axis=1)
        theta_2 = self.LS[self.neiElems[calcLS][:, [2, 3]]].min(axis=1)

        newLS = np.full((len(calcLS),), -1.)

        beta = mesh.hx / mesh.hy
        theta_sq = mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * (theta_1 - theta_2) ** 2

        mask = theta_sq > 0
        nmask = np.invert(mask)

        newLS[mask] = np.asarray((theta_1 + beta ** 2 * theta_2 + theta_sq ** 0.5) / (1 + beta ** 2))[mask]
        newLS[nmask] = np.asarray([theta_1 + mesh.hy, theta_2 + mesh.hx]).min(axis=0)[nmask]

        #return np.asarray([newLS, calcLS])
        return [newLS, calcLS]

    def solveFMM(self, Mesh):

        while (self.Status == 0).any():
            # We do the heap
            evEl = heapq.heappop(self.heapStruct)

            evN = int(evEl[1])

            # we calculate the LS of the neighbors to the smallest object
            newNLS = self.calcLs(self.neiElems[evN], Mesh)

            # for these elements we already had a level set so we update the heap
            updN = (self.Status[newNLS[1]] == 0) * (newNLS[1] != evN)
            if updN.any():
                # only update the ones where we have a change!
                toUpdate = self.LS[newNLS[1][updN]] > newNLS[0][updN]
                if toUpdate.any():
                    self.addLs([newNLS[0][updN][toUpdate], newNLS[1][updN][toUpdate]])
                    updH = np.where(np.in1d(list(zip(*self.heapStruct))[1], newNLS[1][updN][toUpdate]))[0]

                    iter = 0
                    toUpdate = list(map(tuple, np.asarray((self.LS[newNLS[1][updN][toUpdate]],
                                                            newNLS[1][updN][toUpdate])).T))
                    for item in toUpdate:
                        self.heapStruct[updH[iter]] = item
                        iter += 1

            # enter all of these new Nodes
            newN = self.Status[newNLS[1]] == -1
            if newN.any():
                self.addLs([newNLS[0][newN], newNLS[1][newN]])

            self.n2k(evN)

            # push all new and pop the smallest
            toPush = list(map(tuple, np.asarray((newNLS[0][newN], newNLS[1][newN])).T))
            for item in toPush:
                heapq.heappush(self.heapStruct, item)

        return self.LS