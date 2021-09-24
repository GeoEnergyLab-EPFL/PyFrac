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

    def __init__(self, args):

        (known, toEvaluate, mesh) = args

        self.indices = toEvaluate

        # We set the status of all elements to far away
        self.Status = np.full((len(self.indices),), -1, dtype=int)
        # We set the status of the known elements to known
        self.Status[known[1]] = 0

        # we store the neighbouring elements
        self.neiElems = mesh.NeiElements[toEvaluate]

        # We initialize the level set as infinity
        self.LS = np.full((len(toEvaluate),), np.inf)
        self.LS[known[1]] = known[0]

        self.heapStruct = list(map(tuple, np.asarray((known[0], known[1])).T))
        heapq.heapify(self.heapStruct)


    def n2k(self, newKnown):
        self.Status[newKnown] = 1
        self.f2n(newKnown)

    def f2n(self, newKnown):
        neighbors = self.get_n(newKnown)
        self.Status[neighbors[self.Status[neighbors] != 1]] = 0

    def get_n(self, newKnown):
        return self.neiElems[newKnown]

    def addLs(self, newLs):
        self.LS[newLs[1]] = newLs[0]

    def calcLs(self, args):

        (calcLS, mesh) = args

        if type(calcLS) == np.ndarray:
            calcLS = calcLS.astype(int)
        else:
            calcLS = int(calcLS)

        theta_1 = self.LS[self.neiElems[calcLS][:, [0, 1]]].min(axis=1)
        theta_2 = self.LS[self.neiElems[calcLS][:, [2, 3]]].min(axis=1)

        newLS = np.full((len(calcLS),), -1, dtype=float)

        beta = mesh.hx / mesh.hy
        theta_sq = mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * (theta_1 - theta_2) ** 2

        newLS[theta_sq > 0] = np.asarray((theta_1 + beta ** 2 * theta_2 + theta_sq ** 0.5)
                              / (1 + beta ** 2))[theta_sq > 0]
        newLS[theta_sq <= 0] = np.asarray([theta_1 + mesh.hy, theta_2 + mesh.hx]).min(axis=0)[theta_sq <= 0]

        return np.asarray([newLS, calcLS])

    def solveFMM(self, Mesh):

        log = logging.getLogger('PyFrac.fmm')

        # We do the first heap
        evEl = heapq.heappop(self.heapStruct)

        while min(self.Status) != 1:

            # we calculate the LS of the neighbors to the smallest object
            newNLS = self.calcLs((self.neiElems[int(evEl[1])], Mesh))

            # for these elements we already had a level set so we update the heap
            updN = self.Status[newNLS[1].astype(int)] == 0
            if updN.any():
                self.addLs([np.asarray([self.LS[newNLS[1][updN].astype(int)], newNLS[0][updN]]).min(axis=0),
                            newNLS[1][updN].astype(int)])
                # here some type problem....
                updH = np.where(np.in1d(list(zip(*self.heapStruct))[1], newNLS[1][updN]))[0].astype()
                self.heapStruct[np.asarray(updH).astype(int)] = \
                    list(map(tuple, np.asarray((self.LS[newNLS[1][updN].astype(int)], newNLS[1][updN])).T))


            # enter all of these new Nodes
            newN = self.Status[newNLS[1].astype(int)] == -1
            if newN.any():
                self.addLs([np.asarray([self.LS[newNLS[1][newN].astype(int)], newNLS[0][newN]]).min(axis=0),
                            newNLS[1][newN].astype(int)])

            self.n2k(int(evEl[1]))

            # push all new and pop the smallest
            for item in list(map(tuple, np.asarray((newNLS[0][newN], newNLS[1][newN])).T)):
                heapq.heappush(self.heapStruct, item)

            evEl = heapq.heappop(self.heapStruct)

        return self.LS