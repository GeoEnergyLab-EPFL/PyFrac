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
from scipy.interpolate import griddata


class Fracture():
    """ Class defining propagating fracture;
        
        Instance variables:

            w (ndarray-float)           -- fracture opening (width)
            p (ndarray-float)           -- fracture pressure
            time (float)                -- time since the start of injection
            EltChannel (ndarray-int)    -- list of cells currently in the channel region
            EltCrack (ndarray-int)      -- list of cells currently in the crack region
            EltRibbon (ndarray-int)     -- list of cells currently in the Ribbon region
            EltTip (ndarray-int)        -- list of cells currently in the Tip region
            v (ndarray-float)           -- propagation velocity for each cell in the tip cells
            alpha (ndarray-float)       -- angle prescribed by perpendicular on the fracture front (see Pierce 2015,
                                           Computation Methods Appl. Mech)
            l (ndarray-float)           -- length of perpendicular on the fracture front
                                           (see Pierce 2015, Computation Methods Appl. Mech)
            ZeroVertex (ndarray-float)  -- Vertex from which the perpendicular is drawn (can have value from 0 to 3,
                                           where 0 signify bottom left, 1 signifying bottom right, 2 signifying top
                                           right and 3 signifying top left vertex)
            FillF (ndarray-float)       -- filling fraction of each tip cell
            CellStatus (ndarray-int)    -- specifies which region each element currently belongs to
            initRad (float)             -- starting radius
            initTime (float)            -- starting time
            sgndDist (ndarray-float)    -- signed minimun distance from fracture front of each cell in the domain
            Q (ndarray-float)           -- injection rate into each cell of the domain
            FractEvol (ndarray-float)   -- array containing the coordinates of the individual fracture front lines;
                                           used for printing fracture evolution through time
            InCrack (ndarray-int)       -- array specifying whether the cell is inside or outside the fracture.
            FractureVolume (float)      -- fracture volume
                                 
        functions:
            __init__                    Initialize the fracture according to the given initialization parameters.
            plot_fracture               plot given variable of the fracture
            PrintFractureTrace:         plot current regions and front position of the fracture

            
    """

    def __init__(self, Mesh, init_param, solid=None, fluid=None, injection=None, simulProp=None):
        """ Initialize the fracture according to the given initialization parameters.
            
        Arguments:
            Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
            init_param (tuple)     -- a tuple with the initialization parameters. The number of parameters depends
                                      on the initialization type given by the first element of the tuple.
                                            In case the first element (init_param[0], see below for possible options) is
                                            'PKN', the tuple should contain the following parameters in order:
                                                2. given_type   -- the type of the given value (see below for possible
                                                                   options).
                                                3. given_value  -- the value at which the fracture is initialized.
                                                4. -h           -- the height of the PKN fracture.

                                            In case the first element (init_param[0]) is 'M', 'Mt', 'K', 'Kt' or 'E',
                                            the tuple should contain the following parameters in order:
                                                2. given_type   -- the type of the given value (see below for possible
                                                                   options).
                                                3. given_value  -- the value at which the fracture is initialized.

                                            In case the first element (init_param[0]) is 'G', the tuple should
                                            contain the following parameters in order:
                                                2. surv_cells   -- list of the survey cells giving distance from the
                                                                   fracture front.
                                                3. inner_cells  -- the cells enclosed by the survey cells inside the
                                                                   fracture.
                                                4. surv_dist    -- the dist of the survey cells from the fracture front.
                                                5. w            -- the array giving the width to be initialized for
                                                                   each of the mesh cell. It can be 'None' if not
                                                                   available.
                                                6. p            -- the array giving the pressure to be initialized for
                                                                   each of the mesh cell. It can be 'None' if not
                                                                   available.
                                                7. C            -- the elasticity influence matrix.
                                                8. volume       -- total volume of the fracture. It can be 'None' if not
                                                                   available.
                                                9. vel          -- vel of the propagating front. Maximum of the given
                                                                   velocity will be used to calculate the time step.

                                            The fracture can be initialized according to the following regimes
                                            (specified by the first element of the init_param tuple):
                                                'M'     -- radial fracture in viscosity dominated regime
                                                'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                'K'     -- radial fracture in toughness dominated regime
                                                'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                'E'     -- elliptical fracture in toughness dominated regime
                                                'PKN'   -- PKN fracture
                                                'G'     -- flexible, general purpose initialization

                                            given_type can be one of the following:
                                                't'     -- time at which the fracture is to be initialized.
                                                'l'     -- the length parameter. It will be treated as the fracture
                                                           radius, the minor axis length and the fracture length for the
                                                           cases of a radial fracture, an elliptical fracture and a PKN
                                                           fracture respectively.

            solid (MaterialProperties object)           -- the MaterialProperties object giving the material properties.
            fluid (FluidProperties object)              -- the FluidProperties object giving the fluid properties.
            injection (InjectionProperties object)      -- the InjectionProperties object giving the injection
                                                           properties.
            simulProp (SimulationParameters object)     -- the SimulationParameters object giving the numerical
                                                           parameters to be used in the simulation.
        """

        # the parameter specifying the type of initialization
        init_type = init_param[0]

        if init_type is 'PKN':
            (init_type, given_type, given_value, h) = init_param
        elif init_type in ('M', 'Mt', 'K', 'Kt', 'E'): # radial fracture
            if len(init_param) == 3:
                (init_type, given_type, given_value) = init_param
                h = None
            else:
                raise ValueError("Three parameters are to be provided for initialization type " + repr(init_type) +
                                ". See Fracture class initialization function documentation.")
        elif init_type is 'G': # general purpose initialization
            if len(init_param) == 9:
                (init_type, surv_cells, inner_cells, surv_dist, w, p, C, volume, vel) = init_param
            else:
                raise ValueError("Nine parameters are to be provided for initialization type " + repr(init_type) +
                                ". See Fracture class initialization function documentation.")
        else:
            raise ValueError("Given initialization type '" + init_type + "' is not correct. See Fracture class "
                            "initialization function documentation for possible options.")


        self.mesh = Mesh

        if not init_type is 'G':

            if given_type is 't':
                length = None
                time = given_value
            elif given_type is 'l':
                time = None
                length = given_value
            else:
                raise ValueError("The initial value can only be of type time ('t') or length ('l')")

            # get analytical solution
            self.time, self.initRad, self.p, self.w, vel, actvElts = HF_analytical_sol(init_type,
                                                                      self.mesh,
                                                                      solid.Eprime,
                                                                      injection.injectionRate[1,0],
                                                                      muPrime=fluid.muPrime,
                                                                      Kprime=solid.Kprime[self.mesh.CenterElts][0],
                                                                      Cprime=solid.Cprime[self.mesh.CenterElts][0],
                                                                      length=length,
                                                                      t=time,
                                                                      KIc_min=solid.K1c_perp,
                                                                      h=h)

            if init_type in ('M', 'Mt', 'K', 'Kt'):
                # survey cells and their distances from the front
                surv_cells, inner_cells = get_circular_survey_cells(self.mesh, self.initRad)
                surv_dist = self.initRad - (Mesh.CenterCoor[surv_cells, 0] ** 2 + Mesh.CenterCoor[
                                                                            surv_cells, 1] ** 2) ** 0.5
            elif init_type is 'E':
                a = (solid.Kprime[self.mesh.CenterElts]/((32 / np.pi) ** 0.5) / solid.K1c_perp) ** 2 * self.initRad
                surv_cells, inner_cells = get_eliptical_survey_cells(Mesh, a, self.initRad)
                surv_dist = np.zeros((surv_cells.size, ), dtype=np.float64)
                # get minimum distance from center of the survey cells
                for i in range(0, surv_cells.size):
                     surv_dist[i] = Distance_ellipse(a, self.initRad, Mesh.CenterCoor[surv_cells[i], 0],
                                                                    Mesh.CenterCoor[surv_cells[i], 1])
            elif init_type is 'PKN':
                raise SystemExit("PKN initialization is to be implemented")

        self.EltChannel, self.EltTip, self.EltCrack, self.EltRibbon, self.ZeroVertex, \
        self.CellStatus, self.l, self.alpha, self.FillF, self.sgndDist = generate_footprint(self.mesh,
                                                                                surv_cells,
                                                                                inner_cells,
                                                                                surv_dist)

        if init_type is 'G':
            self.w, self.p = get_width_pressure(self.mesh,
                                                self.EltCrack,
                                                self.EltTip,
                                                self.FillF,
                                                C,
                                                w,
                                                p,
                                                volume)


            if volume is None:
                volume = np.sum(self.w) * (Mesh.EltArea)

            if injection !=None:
                self.time = volume/injection.injectionRate[1,0]

        self.v = vel * np.ones((self.EltTip.size,), )
        # setting arrival time to current time (assuming leak off starts at the time the fracture is initialized)
        self.Tarrival = np.full((self.mesh.NumberOfElts,), np.nan, dtype=np.float64)
        self.Tarrival[self.EltCrack] = self.time

        self.LkOff_vol = np.zeros((self.mesh.NumberOfElts,), dtype=np.float64)
        self.efficiency = 1.

        self.process_fracture_front()

        self.FractureVolume = np.sum(self.w)*(Mesh.EltArea)

        self.InCrack = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        self.InCrack[self.EltCrack] = 1

        # local viscosity
        if fluid != None:
            self.muPrime = np.full((Mesh.NumberOfElts,), fluid.muPrime, dtype=np.float64)

        # regime variable (goes from 0 for fully toughness dominated and one for fully viscosity dominated propagation)
        self.regime = np.vstack((np.ones((self.EltRibbon.size,), dtype=np.float64), self.EltRibbon))

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

    def plot_fracture(self, elts='complete', parameter='footPrint', analytical=None, identify=[],
                      mat_Properties=None):
        """
        Plots the given parameter of the specified cells.
        
            Arguments:
                elts(string):        elements to be printed; possible options:
                                                    complete
                                                    channel
                                                    crack
                                                    ribbon
                parameter(string):   parameter to be ploted; possible options:
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

        if elts == 'complete':
            Elts = np.arange(self.mesh.NumberOfElts)
        elif elts == 'channel':
            Elts = self.EltChannel
        elif elts == 'crack':
            Elts = self.EltCrack
        elif elts == 'ribbon':
            Elts = self.EltRibbon
        elif elts == 'tip':
            Elts = self.EltTip
        else:
            raise ValueError('Invalid element identifier!')

        values = np.zeros((self.mesh.NumberOfElts), float)
        if parameter == 'width':
            values[Elts] = self.w[Elts]
        elif parameter == 'pressure':
            values[Elts] = self.p[Elts]
        elif parameter == 'muPrime':
            values[Elts] = self.muPrime[Elts]
        elif parameter == 'footPrint':
            fig = self.print_fracture_trace(analytical, identify, mat_Properties)
            return fig
        else:
            raise ValueError('invalid parameter identifier')

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
            colors += -100. * (Kprime) / (np.max(Kprime) + 1e-15)

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

#-----------------------------------------------------------------------------------------------------------------------

    def remesh(self, factor, C, material_prop, fluid_prop, inj_prop, sim_prop):

        coarse_mesh = CartesianMesh(factor*self.mesh.Lx, factor*self.mesh.Ly, self.mesh.nx, self.mesh.ny)
        SolveFMM(self.sgndDist,
                 self.EltRibbon,
                 self.EltChannel,
                 self.mesh,
                 [],
                 self.EltChannel)

        sgndDist_coarse = griddata(self.mesh.CenterCoor[self.EltChannel], self.sgndDist[self.EltChannel], coarse_mesh.CenterCoor, method='linear')

        max_diag = (coarse_mesh.hx ** 2 + coarse_mesh.hy ** 2) ** 0.5
        excluding_tip = np.where(sgndDist_coarse <= -max_diag)[0]
        sgndDist_copy = np.copy(sgndDist_coarse)
        sgndDist_coarse = np.full(sgndDist_coarse.shape, 1e10, dtype=np.float64)
        sgndDist_coarse[excluding_tip] = sgndDist_copy[excluding_tip]

        w_coarse = griddata(self.mesh.CenterCoor[self.EltChannel], self.w[self.EltChannel],
                                   coarse_mesh.CenterCoor, method='linear', fill_value=0.)

        # w_coarse[np.isnan(w_coarse)]=0

        #todo: Find the velocity by merging tip cells. Asigning the maximum for now to calculate the correct time step
        v_coarse = max(self.v)

        init_data = ('G',
                     excluding_tip,
                     excluding_tip,
                     -sgndDist_coarse[excluding_tip],
                     w_coarse,
                     None,
                     C,
                     self.FractureVolume,
                     v_coarse)

        saveToDisk_cpy = sim_prop.saveToDisk
        sim_prop.saveToDisk = False
        Fr_coarse = Fracture(coarse_mesh,
                            init_data,
                            solid=material_prop,
                            fluid=fluid_prop,
                            injection=inj_prop,
                            simulProp=sim_prop)


        Fr_coarse.Tarrival[Fr_coarse.EltChannel] = griddata(self.mesh.CenterCoor[self.EltChannel],
                                                            self.Tarrival[self.EltChannel],
                                                            coarse_mesh.CenterCoor[Fr_coarse.EltChannel],
                                                            method='linear')

        Fr_coarse.LkOff_vol[Fr_coarse.EltChannel] = 2 * material_prop.Cprime[Fr_coarse.EltChannel] * (
                                Fr_coarse.time - Fr_coarse.Tarrival[Fr_coarse.EltChannel])**0.5 * coarse_mesh.EltArea
        Fr_coarse.LkOff_vol[Fr_coarse.EltTip] = 2 * material_prop.Cprime[Fr_coarse.EltTip] * Integral_over_cell(
                                                        Fr_coarse.EltTip,
                                                        Fr_coarse.alpha,
                                                        Fr_coarse.l,
                                                        Fr_coarse.mesh,
                                                        'Lk',
                                                        frac=Fr_coarse,
                                                        mat_prop=material_prop,
                                                        Vel=Fr_coarse.v,
                                                        dt=1.e20)
        injected_vol = inj_prop.injectionRate[1, 0] * Fr_coarse.time
        Fr_coarse.efficiency = (injected_vol - sum(Fr_coarse.LkOff_vol[Fr_coarse.EltCrack])) / injected_vol

        Fr_coarse.time = self.time
        sim_prop.saveToDisk = saveToDisk_cpy

        #############
        # material_prop.CPrime = np.full((coarse_mesh.NumberOfElts,), 5e-7, dtype=np.float64)
        # stressed_layer_1 = np.where(coarse_mesh.CenterCoor[:, 1] > 50)[0]
        # stressed_layer_2 = np.where(coarse_mesh.CenterCoor[:, 1] < -50)[0]
        # material_prop.CPrime[stressed_layer_1] = 1e-10
        # material_prop.CPrime[stressed_layer_2] = 1e-10


        return Fr_coarse