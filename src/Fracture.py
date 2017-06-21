# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import warnings


# local import ....

from src.Utility import *
from src.HFAnalyticalSolutions import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Properties import *
from src.CartesianMesh import *


# todo : merge the __init__ with the actual initialization

class Fracture():
    """ Class defining propagating fracture;
        
        Instance variables:

            muPrime (ndarray-float):    12 viscosity (mu' parameter) for each cell in the domain
            w (ndarray-float):          fracture opening (width)
            p (ndarray-float):          fracture pressure
            time (float):               time since the start of injection
            EltChannel (ndarray-int):   list of cells currently in the channel region
            EltCrack (ndarray-int):     list of cells currently in the crack region
            EltRibbon (ndarray-int):    list of cells currently in the Ribbon region
            EltTip (ndarray-int):       list of cells currently in the Tip region
            v (ndarray-float):          propagation velocity for each cell in the tip cells
            alpha (ndarray-float):      angle prescribed by perpendicular on the fracture front
                                        (see Pierce 2015, Computation Methods Appl. Mech)
            l (ndarray-float):          length of perpendicular on the fracture front
                                        (see Pierce 2015, Computation Methods Appl. Mech)
            ZeroVertex (ndarray-float): Vertex from which the perpendicular is drawn (can have value from 0 to 3, where
                                        0 signify bottom left, 1 signifying bottom right, 2 signifying top right and 3
                                        signifying top left vertex)
            FillF (ndarray-float):      filling fraction of each tip cell
            CellStatus (ndarray-int):   specifies which region each element currently belongs to
            initRad (float):            starting radius
            initTime (float):           starting time
            sgndDist (ndarray-float):   signed minimun distance from fracture front of each cell in the domain
            Q (ndarray-float):          injection rate into each cell of the domain
            FractEvol (ndarray-float):  array containing the coordinates of the individual fracture front lines;
                                        used for printing fracture evolution through time
            InCrack (ndarray-int):      array specifying whether the cell is inside or outside the fracture.
            Front                       array of with x,y locations of the intersection of the fracture front with the background mesh edges
            Volume                      real - fracture volume
                                 
        functions:
            plot_fracture:               plot given variable of the fracture
            PrintFractureTrace:         plot current regions and front position of the fracture

            
    """

    def __init__(self, Mesh, initValue, initType, regime, solid, fluid, injection, simulProp):
        """ Initialize the fracture according to the given initial value and the propagation regime. Either initial 
        radius or time can be given as the initial value. The function sets up the fracture front and other fracture
        parameters according to the given regime at the given time or radius.
            
            Arguments:
                initValue (float):      initial value
                initType (string):      type of the initial variable provided. Possible values:
                                            time    -- indicating the given value is initial time
                                            radius  -- indicating the given value is initial radius
                regime (string):        Possible values:
                                            K   -- indicating toughness dominated regime, without leak off
                                            M   -- indicating viscosity dominated regime, without leak off
                                            Mt  -- indicating viscosity dominated regime, with leak off
        """

        self.mesh = Mesh
        if initType == 'time':
            self.time = initValue
            if regime == 'K':
                (self.initRad, self.p, self.w, v) = K_vertex_solution_t_given(solid.Kprime, solid.Eprime,
                                                                    injection.injectionRate[1,0], self.mesh, initValue)
            elif regime == 'M':
                (self.initRad, self.p, self.w, v) = M_vertex_solution_t_given(solid.Eprime, injection.injectionRate[1,0],
                                                                                fluid.muPrime, self.mesh, initValue)
            elif regime == 'Mt':
                (self.initRad, self.p, self.w, v) = MT_vertex_solution_t_given(solid.Eprime, np.mean(solid.Cprime),
                                                    injection.injectionRate[1,0], fluid.muPrime, self.mesh, initValue)
            else:
                print('regime ' + regime + ' not supported')
                return
        elif initType == 'radius':
            self.initRad = initValue
            if regime == 'K':
                (self.time, self.p, self.w, v) = K_vertex_solution_r_given(solid.Kprime, solid.Eprime,
                                                                    injection.injectionRate[1,0], self.mesh, initValue)
            elif regime == 'M':
                (self.time, self.p, self.w, v) = M_vertex_solution_R_given(solid.Eprime, injection.injectionRate[1,0],
                                                                    fluid.muPrime, self.mesh, initValue)
            elif regime == 'Mt':
                (self.time, self.p, self.w, v) = Mt_vertex_solution_r_given(solid.Eprime, np.mean(solid.Cprime),
                                                                    injection.injectionRate[1,0],
                                                                    fluid.muPrime, self.mesh, initValue)
            else:
                print('regime ' + regime + ' not supported')
                return
            self.initTime = self.time
        else:
            print('initType ' + initType + ' not supported')



        # level set value at middle of the elements
        phiMid = np.empty([self.mesh.NumberOfElts, 1], dtype=float)
        for e in range(0, self.mesh.NumberOfElts):
            phiMid[e] = radius_level_set(self.mesh.CenterCoor[e], self.initRad)
        # level set value at vertices of the element
        phiVertices = np.empty([len(self.mesh.VertexCoor), 1], dtype=float)
        for i in range(0, len(self.mesh.VertexCoor)):
            phiVertices[i] = radius_level_set(self.mesh.VertexCoor[i], self.initRad)
            # finding elements containing at least one vertices inside the fracture, i.e. with a value of the level <0
            # avoiding loop on elements....

        # array of Length (number of elements) containig the sum of vertices with neg level set value)
        psum = np.sum(phiVertices[self.mesh.Connectivity[:]] < 0, axis=1)
        # indices of tip element which by definition have less than 4 but at least 1 vertices inside the level set
        EltTip = (np.where(np.logical_and(psum > 0, psum < 4)))[0]
        EltCrack = (np.where(psum > 0))[0]  # # indices of cracked element
        EltChannel = (np.where(psum == 4))[0]  # indices of channel element / fully cracked

        # find the ribbon elements: Channel Elements having at least
        # on common vertices with a Tip element
        #
        # loop on ChannelElement, and on TipElement
        testribbon = np.empty([len(EltChannel), 1], dtype=float)
        for e in range(0, len(EltChannel)):
            for i in range(0, len(EltTip)):
                if (len(np.intersect1d(self.mesh.Connectivity[EltChannel[e]], self.mesh.Connectivity[EltTip[i]])) > 0):
                    testribbon[e] = 1
                    break
                else:
                    testribbon[e] = 0
        EltRibbon = EltChannel[(np.reshape(testribbon, len(EltChannel)) == 1)]  # EltChannel is (N,) testribbon is (N,1)

        # Get the initial Filling fraction as well as location of the intersection of the crack front with the edges
        #                               of the mesh
        # we loop over all the tip element  (partially fractured element)

        EltArea = self.mesh.EltArea
        # a vector containing the filling fraction of each Tip Elements     -> should be of all elements
        FillF = np.empty([len(EltTip)], dtype=float)
        # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - I point
        I = np.empty([len(EltTip), 2], dtype=float)
        # a vector containing the coordinantes of the intersection of the front with the edges of each Tip Element - J point
        J = np.empty([len(EltTip), 2], dtype=float)

        for i in range(0, len(EltTip)):

            ptsV = self.mesh.VertexCoor[self.mesh.Connectivity[EltTip[i]]]  #
            # level set value at the vertices of this element
            levelV = np.reshape(phiVertices[self.mesh.Connectivity[EltTip[i]]], 4)
            s = np.argsort(levelV)  # sort the level set
            furthestin = s[0]  # vertex the furthest inside the fracture
            InsideFrac = 1 * (levelV < 0.)  # array of 0 and 1

            if np.sum(InsideFrac) == 1:
                # case 1 vertex in the fracture
                Ve = np.where(InsideFrac == 1)[0]  # corresponding vertex indices in the reference element
                x = np.sqrt(
                    self.initRad ** 2 - (ptsV[Ve, 1][0]) ** 2)  # zero of the level set in x direction (same y as Ve)
                y = np.sqrt(
                    self.initRad ** 2 - (ptsV[Ve, 0][0]) ** 2)  # zero of the level set in y direction (same x as Ve)
                # note the np.around(,8)  to avoid running into numerical precision issue
                if (x < np.around(ptsV[0, 0], 8)) | (x > np.around(ptsV[1, 0], 8)):
                    x = -x
                if (y < np.around(ptsV[0, 1], 8)) | (y > np.around(ptsV[3, 1], 8)):
                    y = -y

                if (Ve == 0 | Ve == 2):
                    # case it is 0 node or 2 node
                    I[i] = np.array([x, ptsV[Ve, 1][0]])
                    J[i] = np.array([ptsV[Ve, 0][0], y])
                else:
                    J[i] = np.array([x, ptsV[Ve, 1][0]])
                    I[i] = np.array([ptsV[Ve, 0][0], y])

                # the filling fraction is simple to compute - area of the triangle Ve-IJ - divided by EltArea
                FillF[i] = 0.5 * np.linalg.norm(I[i] - ptsV[Ve]) * np.linalg.norm(J[i] - ptsV[Ve]) / EltArea

            if np.sum(InsideFrac) == 2:
                # case of 2 vertices inside the fracture (and 2 outside)
                Ve = np.where(InsideFrac == 1)[0]
                if (np.sum(Ve == np.array([0, 1])) == 2) | (np.sum(Ve == np.array([2, 3])) == 2):
                    # case where the front is mostly horizontal i.e. [0-1] or [2,3]
                    y1 = np.sqrt(self.initRad ** 2 - (ptsV[Ve[0], 0]) ** 2)
                    y2 = np.sqrt(self.initRad ** 2 - (ptsV[Ve[1], 0]) ** 2)
                    if (y1 < np.around(ptsV[0, 1], 8)) | (y1 > np.around(ptsV[3, 1], 8)):
                        y1 = -y1
                    if (y2 < np.around(ptsV[0, 1], 8)) | (y2 > np.around(ptsV[3, 1], 8)):
                        y2 = -y2
                    if (furthestin == 0) | (furthestin == 2):
                        I[i] = np.array([(ptsV[Ve[0], 0]), y1])
                        J[i] = np.array([(ptsV[Ve[1], 0]), y2])
                        FillF[i] = 0.5 * (np.linalg.norm(I[i] - ptsV[Ve[0]]) + np.linalg.norm(J[i] - ptsV[Ve[1]])) \
                                   * (np.linalg.norm(ptsV[Ve[0]] - ptsV[Ve[1]])) / EltArea
                    else:
                        J[i] = np.array([(ptsV[Ve[0], 0]), y1])
                        I[i] = np.array([(ptsV[Ve[1], 0]), y2])
                        FillF[i] = 0.5 * (np.linalg.norm(I[i] - ptsV[Ve[1]]) + np.linalg.norm(J[i] - ptsV[Ve[0]])) \
                                   * (np.linalg.norm(ptsV[Ve[0]] - ptsV[Ve[1]])) / EltArea
                else:
                    # case where the front is mostly vertical i.e. [0-3] or [1,2]
                    x1 = np.sqrt(self.initRad ** 2 - (ptsV[Ve[0], 1]) ** 2)
                    x2 = np.sqrt(self.initRad ** 2 - (ptsV[Ve[1], 1]) ** 2)
                    if (x1 < (np.around(ptsV[0, 0], 8))) | (x1 > (np.around(ptsV[1, 0], 8))):
                        x1 = -x1
                    if (x2 < np.around(ptsV[0, 0], 8)) | (x2 > np.around(ptsV[1, 0], 8)):
                        x2 = -x2
                    if (furthestin == 0) | (furthestin == 2):
                        I[i] = np.array([x1, (ptsV[Ve[0], 1])])
                        J[i] = np.array([x2, (ptsV[Ve[1], 1])])
                        FillF[i] = 0.5 * (np.linalg.norm(I[i] - ptsV[Ve[0]]) + np.linalg.norm(J[i] - ptsV[Ve[1]])) \
                                   * (np.linalg.norm(ptsV[Ve[0]] - ptsV[Ve[1]])) / EltArea
                    else:
                        J[i] = np.array([x1, (ptsV[Ve[0], 1])])
                        I[i] = np.array([x2, (ptsV[Ve[1], 1])])
                        FillF[i] = 0.5 * (np.linalg.norm(I[i] - ptsV[Ve[1]]) + np.linalg.norm(J[i] - ptsV[Ve[0]])) \
                                   * (np.linalg.norm(ptsV[Ve[0]] - ptsV[Ve[1]])) / EltArea

            if np.sum(InsideFrac) == 3:
                # only one vertices outside the fracture
                # we redo the same than for case 1 but Ve now corresponds to the only vertex outside the fracture
                Ve = np.where(InsideFrac == 0)[0]
                x = np.sqrt(self.initRad ** 2 - (ptsV[Ve, 1][0]) ** 2)
                y = np.sqrt(self.initRad ** 2 - (ptsV[Ve, 0][0]) ** 2)
                if (x < np.around(ptsV[0, 0], 8)) | (x > np.around(ptsV[1, 0], 8)):
                    x = -x
                if (y < np.around(ptsV[0, 1], 8)) | (y > np.around(ptsV[3, 1], 8)):
                    y = -y
                if (Ve == 0 | Ve == 2):
                    # case it is
                    J[i] = np.array([x, ptsV[Ve, 1][0]])
                    I[i] = np.array([ptsV[Ve, 0][0], y])
                else:
                    I[i] = np.array([x, ptsV[Ve, 1][0]])
                    J[i] = np.array([ptsV[Ve, 0][0], y])

                FillF[i] = 1. - 0.5 * np.linalg.norm(I[i] - ptsV[Ve]) * np.linalg.norm(J[i] - ptsV[Ve]) / EltArea

        # Type of each cell (1 for channel cells, 2 for tip cells, 3 for ribbon cells and 0 for rest)
        CellStatus = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        CellStatus[:] = 0
        CellStatus[EltChannel] = 1
        CellStatus[EltTip] = 2
        CellStatus[EltRibbon] = 3

        # local viscosity
        self.muPrime = np.full((Mesh.NumberOfElts,), fluid.muPrime, dtype=np.float64)

        # assign 1 for all cells inside the fracture
        InCrack = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        InCrack[EltCrack] = 1

        # initializing signed distance according to the initial radius
        self.sgndDist = radius_level_set(self.mesh.CenterCoor, self.initRad)

        # todo !!! Hack: tip elements are evaluated again with the front reconstructing function to avoid discrepancy
        (self.EltTip, self.l, self.alpha, CSt) = reconstruct_front(self.sgndDist, EltChannel, self.mesh)
        # filling fraction list adjusted according to the tip cells given by the front reconstructing function
        self.FillF = FillF[np.arange(EltTip.shape[0])[np.in1d(EltTip, self.EltTip)]]

        # check if initial radius is large enough to have exclusively channel elements
        if EltChannel.size <= EltRibbon.size:
            raise SystemExit("No channel elements. The initial radius is propably too small")
        (self.EltChannel, self.EltRibbon, self.EltCrack) = (EltChannel, EltRibbon, EltCrack)

        self.Ffront = np.concatenate((I, J), axis=1)
        self.CellStatus = CellStatus
        self.InCrack = InCrack
        self.v = v * np.ones((len(self.l)), float) # uniform velocity for all the tip elements

        # assigning ZeroVertex 0, 1, 2, or 3 according to the sign of the cell center coordinates. For example, cells
        # with both the x and y axis positive will get 0(signifying the bottom left vertex) as the zero vertex.
        self.ZeroVertex = np.zeros((len(self.EltTip),), int)
        for i in range(0, len(self.EltTip)):
            if self.mesh.CenterCoor[self.EltTip[i], 0] <= 0 and self.mesh.CenterCoor[self.EltTip[i], 1] <= 0:
                self.ZeroVertex[i] = 2
            elif self.mesh.CenterCoor[self.EltTip[i], 0] >= 0 and self.mesh.CenterCoor[self.EltTip[i], 1] <= 0:
                self.ZeroVertex[i] = 3
            elif self.mesh.CenterCoor[self.EltTip[i], 0] <= 0 and self.mesh.CenterCoor[self.EltTip[i], 1] >= 0:
                self.ZeroVertex[i] = 1
            elif self.mesh.CenterCoor[self.EltTip[i], 0] >= 0 and self.mesh.CenterCoor[self.EltTip[i], 1] >= 0:
                self.ZeroVertex[i] = 0

        # assigning nan for cells which are not in the fracture yet
        self.Tarrival = np.full((self.mesh.NumberOfElts,), np.nan, dtype=np.float64)

        # using Mtilde solution to initialize arrival time
        self.Tarrival[self.EltChannel] = (solid.Cprime[self.EltChannel] ** 2 * self.mesh.distCenter[self.EltChannel] ** 4 *
                                          np.pi ** 4 / injection.injectionRate[1,0] ** 2 / 4)

        self.Leakedoff = np.zeros((self.mesh.NumberOfElts,), dtype=float)
        # calculate leaked off volume for the channel elements using Carter leak off (see e.g. Dontsov and Peirce, 2008)
        self.Leakedoff[self.EltChannel] = 2 * solid.Cprime[self.EltChannel] * self.mesh.EltArea * (self.time -
                                                                                            self.Tarrival[
                                                                                            self.EltChannel]) ** 0.5
        # calculate leaked off volume for the tip cells by integrating Carter leak off expression (see Dontsov and Peirce, 2008)
        self.Leakedoff[self.EltTip] = 2 * solid.Cprime[self.EltTip] * VolumeIntegral(self.EltTip,
                                                                                     self.alpha,
                                                                                     self.l,
                                                                                     self.mesh,
                                                                                     'Lk',
                                                                                     solid,
                                                                                     self.muPrime,
                                                                                     self.v)

        # fracture evolution data
        self.FractEvol = np.empty((1, 4), float)
        self.process_fracture_front()

        self.FractureVolume = np.sum(self.w)*(Mesh.EltArea)

        # saving initial state of fracture and properties if the output flags are set
        if simulProp.plotFigure:
            fig = self.plot_fracture('complete', 'footPrint', mat_Properties=solid)
            plt.show()

        if simulProp.saveToDisk:
            self.SaveFracture(simulProp.outFileAddress + "file_" + repr(0))
            prop = (solid, fluid, injection, simulProp)
            with open(simulProp.outFileAddress + "properties", 'wb') as output:
                pickle.dump(prop, output, -1)

        # todo change it to data file?
        f = open('log', 'w+')
        from time import gmtime, strftime
        f.write('log file, program run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n\n')

    ###############################################################################

    def plot_fracture(self, Elem_Identifier, Parameter_Identifier, analytical=0, identify=[], mat_Properties=None):
        """
        Plots the given parameter of the specified  cells;
        
            Arguments:
                Elem_Identifier(string):        elements to be printed; possible options:
                                                    complete
                                                    channel
                                                    crack
                                                    ribbon
                Parameter_Identifier(string):   parameter to be ploted; possible options:
                                                    width
                                                    pressure
                                                    viscosity
                                                    footPrint
                analytical (float):             radius of fracture footprint calculated analytically.
                                                not plotted if not given. (or Zero ?)
                evol (boolean):                 fracture evolution plot flag. Set to true will print fracture
                                                evolution with time.
                identify (ndarray):             plot the cells in the provided list with cell number and different color
                                                to identify. This option can be used in debugging.
                perpendicular (bool):           if true, perpendicular from the zero vertex on the fracture fron will be
                                                drawn. This can be used for debugging.
        """

        if Elem_Identifier == 'complete':
            Elts = np.arange(self.mesh.NumberOfElts)
        elif Elem_Identifier == 'channel':
            Elts = self.EltChannel
        elif Elem_Identifier == 'crack':
            Elts = self.EltCrack
        elif Elem_Identifier == 'ribbon':
            Elts = self.EltRibbon
        elif Elem_Identifier == 'tip':
            Elts = self.EltTip
        else:
            print('invalid element identifier')
            return

        values = np.zeros((self.mesh.NumberOfElts), float)
        if Parameter_Identifier == 'width':
            values[Elts] = self.w[Elts]
        elif Parameter_Identifier == 'pressure':
            values[Elts] = self.p[Elts]
        elif Parameter_Identifier == 'muPrime':
            values[Elts] = self.muPrime[Elts]
        elif Parameter_Identifier == 'footPrint':
            fig = self.print_fracture_trace(analytical, identify, mat_Properties)
            return fig
        else:
            print('invalid parameter identifier')
            return None

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.mesh.CenterCoor[:, 0], self.mesh.CenterCoor[:, 1], values, cmap=cm.jet, linewidth=0.2)
        return fig

    ######################################

    def process_fracture_front(self):
        """ process fracture front and different regions of the fracture
            Arguments:
                 
        """
        # list of points where fracture front is intersecting the grid lines. 
        intrsct1 = np.zeros((2, len(self.l)))
        intrsct2 = np.zeros((2, len(self.l)))

        # todo: commenting print_fracture_trace function

        for i in range(0, len(self.l)):
            if self.alpha[i] != 0 and self.alpha[i] != math.pi / 2: # for angles greater than zero and less than 90 deg
                # calculate intercept on y axis and gradient
                yIntrcpt = self.l[i] / math.cos(math.pi / 2 - self.alpha[i])
                grad = -1 / math.tan(self.alpha[i])

                if Pdistance(0, self.mesh.hy, grad, yIntrcpt) <= 0:
                    # one point on top horizontal line of the cell
                    intrsct1[0, i] = 0
                    intrsct1[1, i] = yIntrcpt
                else:
                    # one point on left vertical line of the cell
                    intrsct1[0, i] = (self.mesh.hy - yIntrcpt) / grad
                    intrsct1[1, i] = self.mesh.hy

                if Pdistance(self.mesh.hx, 0, grad, yIntrcpt) <= 0:
                    intrsct2[0, i] = -yIntrcpt / grad
                    intrsct2[1, i] = 0
                else:
                    intrsct2[0, i] = self.mesh.hx
                    intrsct2[1, i] = yIntrcpt + grad * self.mesh.hx

            if self.alpha[i] == 0:
                intrsct1[0, i] = self.l[i]
                intrsct1[1, i] = self.mesh.hy
                intrsct2[0, i] = self.l[i]
                intrsct2[1, i] = 0

            if self.alpha[i] == math.pi / 2:
                intrsct1[0, i] = 0
                intrsct1[1, i] = self.l[i]
                intrsct2[0, i] = self.mesh.hx
                intrsct2[1, i] = self.l[i]

            if self.ZeroVertex[i] == 0:
                intrsct1[0, i] = intrsct1[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct1[1, i] = intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]
                intrsct2[0, i] = intrsct2[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct2[1, i] = intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]

            if self.ZeroVertex[i] == 1:
                intrsct1[0, i] = -intrsct1[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct1[1, i] = intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]
                intrsct2[0, i] = -intrsct2[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct2[1, i] = intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]

            if self.ZeroVertex[i] == 3:
                intrsct1[0, i] = intrsct1[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct1[1, i] = -intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]
                intrsct2[0, i] = intrsct2[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct2[1, i] = -intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]

            if self.ZeroVertex[i] == 2:
                intrsct1[0, i] = -intrsct1[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct1[1, i] = -intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]
                intrsct2[0, i] = -intrsct2[0, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 0]
                intrsct2[1, i] = -intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]], 1]

        tmp = np.transpose(intrsct1)
        tmp = np.hstack((tmp, np.transpose(intrsct2)))

        self.Ffront=tmp


    #-------------------------------------------------------------------------------------------------------------------
    def print_fracture_trace(self, rAnalytical, identify, mat_properties, colormap=cm.jet, color='0.5'):
        """ Print fracture front and different regions of the fracture
            Arguments:
                rAnalytical (float):    radius of fracture footprint calculated analytically
                identify (ndarray):     list of elements to be identified (see plot_fracture function)
                Mat_Properties :        solid material properties object (containing the sigma0 on each element)

        """


        fig, ax = plt.subplots()
        ax.set_xlim([-self.mesh.Lx, self.mesh.Lx])
        ax.set_ylim([-self.mesh.Ly, self.mesh.Ly])

        patches = []
        for i in range(self.mesh.NumberOfElts):
            polygon = Polygon(np.reshape(self.mesh.VertexCoor[self.mesh.Connectivity[i], :], (4, 2)), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=colormap, alpha=0.65, edgecolor=color)

        # todo: A proper mechanism to mark element with different material properties has to be looked into
        # marking those elements that have sigmaO or toughness different than the sigmaO or toughness at the center
        markedElts = []


        # applying different colors for different types of elements
        colors = 100. * np.full(len(patches), 0.4)
        colors[self.EltTip] = 70.
        colors[self.EltChannel] = 10.
        colors[self.EltRibbon] = 90.
        colors[identify] = 0.

        if mat_properties != None:
            colors += -100. * (mat_properties.SigmaO) / np.max(mat_properties.SigmaO)
            colors += -100. * (mat_properties.Kprime) / np.max(mat_properties.Kprime)
            
        p.set_array(np.array(colors))
        ax.add_collection(p)

        # Plot the analytical solution
        if rAnalytical > 0.:
            circle = plt.Circle((0, 0), radius=rAnalytical)
            circle.set_ec('r')
            circle.set_fill(False)
            ax.add_patch(circle)

        # print Element numbers on the plot for elements to be identified
        for i in range(len(identify)):
            ax.text(self.mesh.CenterCoor[identify[i], 0] - self.mesh.hx / 4, self.mesh.CenterCoor[identify[i], 1] - self.mesh.hy / 4,
                    repr(identify[i]), fontsize=10)

        I = self.Ffront[:, 0:2]
        J = self.Ffront[:, 2:4]
        # todo !!!Hack: gets very large values sometime, needs to be resolved
        for e in range(0, len(I)):
            if max(abs(I[e, :] - J[e, :])) < 3 * (self.mesh.hx ** 2 + self.mesh.hy ** 2) ** 0.5:  # if
                plt.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), '.-k')

        plt.axis('equal')

        # maximize the plot window
        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()

        return fig

#-----------------------------------------------------------------------------------------------------------------------

    def SaveFracture(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)
