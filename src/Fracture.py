# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

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
            muPrime (ndarray)           -- local viscosity parameter
            Ffront (ndarray)            -- a list containing the intersection of the front and grid lines for the tip
                                           cells.
            regime (ndarray)            -- the regime of the ribbon cells (0 to 1, where 0 is fully toughness dominated,
                                           and 1 is fully viscosity dominated; See Zia and Lecampion 2018)
                                 
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
                                                'time'  -- time at which the fracture is to be initialized.
                                                'length'-- the length parameter. It will be treated as the fracture
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

            if given_type is 'time':
                length = None
                time = given_value
            elif given_type is 'length':
                time = None
                length = given_value
            else:
                raise ValueError("The initial value can only be of type time ('time') or length ('length')")

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
        # for general purpose initialization
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
            else:
                # set time to zero if mechanical loading is creating the fracture
                self.time = 0

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
        self.regime = np.vstack((np.full((self.EltRibbon.size,), np.nan, dtype=np.float64), self.EltRibbon))

        # saving initial state of fracture and properties if the output flags are set
        if simulProp.plotFigure:
            fig = self.plot_fracture(mat_properties=solid, sim_properties=simulProp)
            plt.show()

        if simulProp.saveToDisk:
            self.SaveFracture(simulProp.get_outFileAddress() + "fracture_" + repr(0))
            prop = (solid, fluid, injection, simulProp)
            with open(simulProp.get_outFileAddress() + "properties", 'wb') as output:
                pickle.dump(prop, output, -1)

        if simulProp.timeStepLimit is None:
            # setting time step limit according to the initial velocity
            if max(self.v) <= 0:
                simulProp.timeStepLimit = self.time * 0.1
            else:
                simulProp.timeStepLimit = simulProp.tmStpPrefactor * min(Mesh.hx, Mesh.hy) / np.max(
                                                            self.v) * simulProp.tmStpFactLimit

        if simulProp.get_solTimeSeries() is None and simulProp.outputTimePeriod is None:
            simulProp.set_solTimeSeries(2 ** np.linspace(np.log2(self.time + simulProp.timeStepLimit),
                                                      np.log2(simulProp.FinalTime), 15))

        # todo change it to data file?
        f = open('log', 'w+')
        from time import gmtime, strftime
        f.write('log file, program run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n\n')
        f.close()
#-----------------------------------------------------------------------------------------------------------------------


    def plot_fracture(self, parameter='footPrint', elts='complete', analytical=None, identify=[], mat_properties=None,
                      sim_properties=None, fig=None):
        """
        Plots the given parameter of the specified cells.
        
        Arguments:
            elts(string)            -- elements to be printed; possible options:
                                                complete
                                                channel
                                                crack
                                                ribbon
            parameter(string)       -- parameter to be ploted; possible options:
                                                width
                                                pressure
                                                viscosity
                                                footPrint
                                                mesh
            analytical (float)      -- radius of fracture footprint calculated analytically. Not plotter if None.
            identify (ndarray):     -- plot the cells in the provided list with cell number and different color
                                       to identify. This option can be used in debugging.
            mat_properties (MaterialProperties)   -- material properties to colorcode the grid according to the given
                                       parameter in the simulation properties. Can be None.
            sim_properties (SimulationParameters) -- Simulation paramters to define various plotting parameters. Can be
                                       None
            fig (figure)            -- figure object to superimpose the image

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
            fig = self.print_fracture_trace(rAnalytical=analytical,
                                            identify=identify,
                                            mat_properties=mat_properties,
                                            sim_prop=sim_properties,
                                            fig=fig)
            return fig
        elif parameter == 'mesh':
            fig = self.print_fracture_trace(rAnalytical=analytical,
                                            identify=identify,
                                            mat_properties=mat_properties,
                                            sim_prop=sim_properties,
                                            fig=fig,
                                            mesh_only=True)
            return fig
        else:
            raise ValueError('Invalid parameter identifier!')

        if fig is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            ax = fig.gca(projection='3d')

        ax.plot_trisurf(self.mesh.CenterCoor[:, 0],
                        self.mesh.CenterCoor[:, 1],
                        values,
                        cmap=cm.jet,
                        linewidth=0.2)
        return fig

#-----------------------------------------------------------------------------------------------------------------------

    def process_fracture_front(self):
        """
        process fracture front and different regions of the fracture. This function adds the start and endpoints of the
        front lines in each of the tip cell to the Ffront variable of the Fracture class.

        Arguments:

        Returns:

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

#-----------------------------------------------------------------------------------------------------------------------


    def print_fracture_trace(self, rAnalytical, identify, mat_properties, sim_prop=None, colormap=cm.viridis,
                             color='0.5', fig=None, mesh_only=False):
        """
        Print fracture front footprint and other parameters.

        Arguments:
            rAnalytical (float)     -- radius of fracture footprint calculated analytically.
            identify (ndarray)      -- list of elements to be identified (see plot_fracture function).
            mat_Properties          -- solid material properties object (containing the material properties which
                                       can be specified to color code the cells).
            sim_prop                -- the SimulationParameters object specifying different parameters to plot.
            colormap                -- colormap to be used to color code the grid cells.
            color                   -- color of the grid lines (default is grey).
            fig                     -- matplotlib figure object to superimpose the image.
            mesh_only (boolean)     -- if True, onle mesh will be plotted.

        Returns:
            fig                     -- matplotlib figure object.
        """


        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)

        # set the four corners of the rectangular mesh
        ax.set_xlim([-self.mesh.Lx, self.mesh.Lx])
        ax.set_ylim([-self.mesh.Ly, self.mesh.Ly])

        # add rectangle for each cell
        patches = []
        for i in range(self.mesh.NumberOfElts):
            polygon = Polygon(np.reshape(self.mesh.VertexCoor[self.mesh.Connectivity[i], :], (4, 2)), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=colormap, alpha=0.65, edgecolor=color)


        # applying color according to the prescribed parameter
        colors = np.full(len(patches), 0.5)

        if not sim_prop is None:
            if sim_prop.bckColor == 'sigma0':
                max_bck = max(mat_properties.SigmaO)
                min_bck = min(mat_properties.SigmaO)
                if max_bck - min_bck > 0:
                    plt_clrBar = True
                    colors = (mat_properties.SigmaO - min_bck) / (max_bck - min_bck)
                else:
                    plt_clrBar = False
                label = "confining stress"
            elif sim_prop.bckColor == 'Kprime':
                max_bck = max(mat_properties.Kprime)
                min_bck = min(mat_properties.Kprime)
                if max_bck - min_bck > 0:
                    plt_clrBar = True
                    colors = (mat_properties.Kprime - min_bck) / (max_bck - min_bck)
                else:
                    plt_clrBar = False
                label = "toughness (K')"
            elif sim_prop.bckColor == 'Cprime':
                max_bck = max(mat_properties.Cprime)
                min_bck = min(mat_properties.Cprime)
                if max_bck - min_bck > 0:
                    plt_clrBar = True
                    colors = (mat_properties.Cprime - min_bck) / (max_bck - min_bck)
                else:
                    plt_clrBar = False
                label = "leak off coefficient (C')"
            elif not sim_prop.bckColor is None:
                raise ValueError("Back ground color identifier not supported!")

        p.set_array(np.array(colors))
        ax.add_collection(p)

        # mark type of elements
        if (not sim_prop is None) and sim_prop.plotEltType and not mesh_only:
            for i in self.EltTip:
                coord = self.mesh.CenterCoor[i]
                circle = plt.Circle((coord[0], coord[1]),
                                    radius=1/4 * min(self.mesh.hy,self.mesh.hy),
                                    fc='#E52C54')
                ax.add_patch(circle)

            for i in self.EltChannel:
                coord = self.mesh.CenterCoor[i]
                circle = plt.Circle((coord[0], coord[1]),
                                    radius=1/4 * min(self.mesh.hy,self.mesh.hy),
                                    fc='#D16A4E')
                ax.add_patch(circle)

            for i in self.EltRibbon:
                coord = self.mesh.CenterCoor[i]
                circle = plt.Circle((coord[0], coord[1]),
                                    radius=1/4 * min(self.mesh.hy,self.mesh.hy),
                                    fc='#07E81C')
                ax.add_patch(circle)


        # Plot the analytical solution
        if not (sim_prop is None) and sim_prop.plotAnalytical and (not rAnalytical is None) and not mesh_only:

            if sim_prop.analyticalSol in ('M', 'Mt', 'K', 'Kt'):
                circle = plt.Circle((0, 0), radius=rAnalytical)
                circle.set_ec('r')
                circle.set_fill(False)
                ax.add_patch(circle)
            elif sim_prop.analyticalSol is 'E' and (not mat_properties.K1c_perp is None):
                from matplotlib.patches import Ellipse
                import matplotlib as mpl
                a = (mat_properties.K1c[0] / mat_properties.K1c_perp)**2 * rAnalytical
                ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * rAnalytical, angle=360)
                ellipse.set_fill(False)
                ellipse.set_ec('r')
                ax.add_patch(ellipse)
            elif sim_prop.analyticalSol is 'PKN':
                print("PKN is to be implemented.")

        # print Element numbers on the plot for elements to be identified
        for i in range(len(identify)):
            ax.text(self.mesh.CenterCoor[identify[i], 0] - self.mesh.hx / 4, self.mesh.CenterCoor[identify[i], 1] - self.mesh.hy / 4,
                    repr(identify[i]), fontsize=10)

        # print the front lines
        if not mesh_only:
            I = self.Ffront[:, 0:2]
            J = self.Ffront[:, 2:4]
            # todo !!!Hack: gets very large values sometime, needs to be resolved
            for e in range(0, len(I)):
                if max(abs(I[e, :] - J[e, :])) < 3 * (self.mesh.hx ** 2 + self.mesh.hy ** 2) ** 0.5:  # if
                    plt.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), '-k')

        plt.axis('equal')

        if not sim_prop.bckColor is None:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_bck, vmax=max_bck))
            sm._A = []
            clr_bar = plt.colorbar(sm,alpha=0.65)
            clr_bar.set_label(label)

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
        """
        This function compresses the fracture by the given factor once it has reached the end of the mesh. The
        elasticity matrix, the properties objects are also re-adjusted according to the new mesh.

        Arguments:
            factor (float)      -- the factor by which the domain is to be compressed. For example, a factor of 2 will
                                   merge the adjacent four cells to a single cell.
            C (ndarray)         -- the elasticity matrix to be re-evaluated for the new mesh.
            material_prop       -- the material properties to be re-evaluated for the new mesh.
            fluid_prop          -- the fluid properties to be re-evaluated for the new mesh.
            inj_prop            -- the injection properties to be re-evaluated for the new mesh.
            sim_prop            -- the simulation properties.

        Returns:
            Fr_coarse           -- the new fracture after remeshing.
        """

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

        # to avoid plotting and saving of the remeshed fracture
        saveToDisk_cpy = copy.copy(sim_prop.saveToDisk)
        plotFigure_cpy = copy.copy(sim_prop.plotFigure)
        sim_prop.saveToDisk = False
        sim_prop.plotFigure = False

        # re-meshing of the material properties
        material_prop.remesh(coarse_mesh)

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
        sim_prop.saveToDisk = copy.copy(saveToDisk_cpy)
        sim_prop.plotFigure = copy.copy(plotFigure_cpy)

        # update the saved properties
        if sim_prop.saveToDisk:
            prop = (material_prop, fluid_prop, inj_prop, sim_prop)
            with open(sim_prop.get_outFileAddress() + "properties", 'wb') as output:
                pickle.dump(prop, output, -1)

        return Fr_coarse