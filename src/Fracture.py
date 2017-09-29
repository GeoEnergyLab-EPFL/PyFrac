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
from src.FractureInitilization import *

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

    def __init__(self, Mesh, init_type, solid, fluid, injection, simulProp, analyt_init_data=None,
                 general_init_data=None):
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
        if init_type == 'analytical':

            (initValue, initType, regime) = analyt_init_data

            if initType == 'time':
                self.time = initValue
                if regime == 'K':
                    (self.initRad, self.p, self.w, self.v) = K_vertex_solution_t_given(solid.Kprime,
                                                                                  solid.Eprime,
                                                                                  injection.injectionRate[1,0],
                                                                                  self.mesh,
                                                                                  initValue)
                elif regime == 'M':
                    (self.initRad, self.p, self.w, self.v) = M_vertex_solution_t_given(solid.Eprime,
                                                                                  injection.injectionRate[1,0],
                                                                                  fluid.muPrime,
                                                                                  self.mesh,
                                                                                  initValue)
                elif regime == 'Mt':
                    (self.initRad, self.p, self.w, self.v) = MT_vertex_solution_t_given(solid.Eprime,
                                                                                   np.mean(solid.Cprime),
                                                                                   injection.injectionRate[1,0],
                                                                                   fluid.muPrime,
                                                                                   self.mesh,
                                                                                   initValue)
                else:
                    print('regime ' + regime + ' not supported')
                    return
            elif initType == 'radius':
                self.initRad = initValue
                if regime == 'K':
                    (self.time, self.p, self.w, self.v) = K_vertex_solution_r_given(solid.Kprime,
                                                                               solid.Eprime,
                                                                               injection.injectionRate[1,0],
                                                                               self.mesh,
                                                                               initValue)
                elif regime == 'M':
                    (self.time, self.p, self.w, self.v) = M_vertex_solution_r_given(solid.Eprime,
                                                                               injection.injectionRate[1,0],
                                                                               fluid.muPrime,
                                                                               self.mesh,
                                                                               initValue)
                elif regime == 'Mt':
                    (self.time, self.p, self.w, self.v) = Mt_vertex_solution_r_given(solid.Eprime,
                                                                                np.mean(solid.Cprime),
                                                                                injection.injectionRate[1,0],
                                                                                fluid.muPrime,
                                                                                self.mesh,
                                                                                initValue)
                else:
                    print('regime ' + regime + ' not supported')
                    return
                self.initTime = self.time
            else:
                raise SystemExit('initType ' + initType + ' not supported in fracture initialization')

            surv_cells, channel_cells = get_circular_survey_cells(self.mesh, self.initRad)
            surv_cells_dist = self.initRad - (Mesh.CenterCoor[surv_cells, 0] ** 2 + Mesh.CenterCoor[
                                                                        surv_cells, 1] ** 2) ** 0.5
            self.EltChannel, self.EltTip, self.EltCrack, self.EltRibbon, self.ZeroVertex, \
            self.CellStatus, self.l, self.alpha, self.FillF, self.sgndDist = generate_footprint(self.mesh,
                                                                                    surv_cells,
                                                                                    channel_cells,
                                                                                    surv_cells_dist)

        elif init_type == 'general':
            (surv_cells, channel_cells, surv_cells_dist, w, p, C, volume, vel) = general_init_data

            self.EltChannel, self.EltTip, self.EltCrack, self.EltRibbon, self.ZeroVertex, \
            self.CellStatus, self.l, self.alpha, self.FillF,self.sgndDist  = generate_footprint(self.mesh,
                                                                                    surv_cells,
                                                                                    channel_cells,
                                                                                    surv_cells_dist)

            self.w, self.p = initial_width_pressure(self.mesh,
                                                    self.EltCrack,
                                                    self.EltTip,
                                                    self.FillF,
                                                    C,
                                                    w,
                                                    p,
                                                    volume)

            # self.w = np.zeros((self.mesh.NumberOfElts,), )
            # self.w[self.EltCrack] = w
            #
            # self.p = np.zeros((self.mesh.NumberOfElts, ), )
            # self.p[self.EltCrack] = p

            self.v = vel * np.ones((self.EltTip.size, ), )
            if volume is None:
                volume = np.sum(self.w) * (Mesh.EltArea)
            self.time = volume/injection.injectionRate[1,0]

        #
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
        # # calculate leaked off volume for the tip cells by integrating Carter leak off expression (see Dontsov and Peirce, 2008)
        # self.Leakedoff[self.EltTip] = 2 * solid.Cprime[self.EltTip] * VolumeIntegral(self.EltTip,
        #                                                                              self.alpha,
        #                                                                              self.l,
        #                                                                              self.mesh,
        #                                                                              'Lk',
        #                                                                              solid,
        #                                                                              self.muPrime,
        #                                                                              self.v)

        # fracture evolution data
        self.process_fracture_front()

        self.FractureVolume = np.sum(self.w)*(Mesh.EltArea)

        self.InCrack = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        self.InCrack[self.EltCrack] = 1

        # local viscosity
        self.muPrime = np.full((Mesh.NumberOfElts,), fluid.muPrime, dtype=np.float64)

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

    def plot_fracture(self, Elem_Identifier, Parameter_Identifier, analytical=None, identify=[], mat_Properties=None):
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

        if mat_properties != None:
            if np.max(mat_properties.SigmaO) > 0:
                colors += -100. * (mat_properties.SigmaO) / np.max(mat_properties.SigmaO)
            if not mat_properties.KprimeFunc is None and not mat_properties.anisotropic:
                Kprime = mat_properties.KprimeFunc(self.mesh.CenterCoor[:,0],self.mesh.CenterCoor[:,1])
            else:
                Kprime = mat_properties.Kprime
            colors += -100. * (Kprime) / np.max(Kprime)

        colors[self.EltTip] = 70.
        colors[self.EltChannel] = 10.
        colors[self.EltRibbon] = 90.
        colors[identify] = 0.

        p.set_array(np.array(colors))
        ax.add_collection(p)

        # Plot the analytical solution
        if not rAnalytical is None:
            if mat_properties.K1c_perp is None:
                circle = plt.Circle((0, 0), radius=rAnalytical)
                circle.set_ec('r')
                circle.set_fill(False)
                ax.add_patch(circle)
            else:
                from matplotlib.patches import Ellipse
                import matplotlib as mpl
                a = (mat_properties.K1c[0] / mat_properties.K1c_perp)**2 * rAnalytical
                ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * rAnalytical, angle=360)
                # ellipse.set_clip_box(ax.bbox)
                ellipse.set_fill(False)
                ellipse.set_ec('r')
                ax.add_patch(ellipse)

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
        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        return fig

#-----------------------------------------------------------------------------------------------------------------------

    def SaveFracture(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)
