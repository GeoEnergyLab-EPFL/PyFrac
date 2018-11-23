# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import dill

# local import ....

from src.Utility import *
from src.HFAnalyticalSolutions import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Properties import *
from src.CartesianMesh import *
from src.FractureInitilization import *
from src.Labels import *
from src.PostProcessFracture import *
from scipy.interpolate import griddata
from src.Visualization import *


class Fracture():
    """ Class defining propagating fracture;

    Args:

        Mesh (CartesianMesh):   -- a CartesianMesh class object describing the grid.
        init_param (tuple):     -- a tuple with the initialization parameters. The number of parameters depends
                                   on the initialization type given by the first element of the tuple.

                                    -- In case the first element (init_param[0]) is 'M', 'Mt', 'K', 'Kt' or 'MDR',
                                    the tuple should contain the following parameters in order:
                                        1. (string)     -- specifying the type of initialization(see below for possible options).
                                        2. (string)     -- the type of the given value (see below for possible options).
                                        3. (float)      -- the value at which the fracture is initialized.

                                    -- In case the first element (init_param[0]) is 'G', the tuple should
                                    contain the following parameters in order:
                                        1. (string)     -- specifying the type of initialization(see below for possible options).
                                        2. (ndarray)    -- list of the survey cells giving distance from the fracture front.
                                        3. (ndarray)    -- the cells enclosed by the survey cells inside the fracture.
                                        4. (ndarray)    -- the dist of the survey cells from the fracture front.
                                        5. (ndarray)    -- the array giving the width to be initialized for each of the mesh cell. It can be 'None' if not available.
                                        6. (ndarray)    -- the array giving the pressure to be initialized for each of the mesh cell. It can be 'None' if not available.
                                        7. (ndarray)    -- the elasticity influence matrix.
                                        8. (ndarray)    -- total volume of the fracture. It can be 'None' if not available.
                                        9. (ndarray)    -- vel of the propagating front. Maximum of the given velocity will be used to calculate the time step.

                                    -- In case the first element (init_param[0], see below for possible options)
                                    is 'PKN', the tuple should contain the following parameters in order:
                                        1. (string)     -- specifying the type of initialization(see below for possible options).
                                        2. (string)     -- the type of the given value (see below for possible options).
                                        3. (float)      -- the value at which the fracture is initialized.
                                        4. (float)      -- the height of the PKN fracture.

                                    The fracture can be initialized according to the following regimes
                                    (specified by the first element of the init_param tuple):
                                        | 'M'     -- radial fracture in viscosity dominated regime
                                        | 'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                        | 'K'     -- radial fracture in toughness dominated regime
                                        | 'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                        | 'E'     -- elliptical fracture in toughness dominated regime
                                        | 'PKN'   -- PKN fracture
                                        | 'G'     -- flexible, general purpose initialization

                                    given_type can be one of the following:
                                        'time'      -- time at which the fracture is to be initialized.
                                        'length'    -- the length parameter. It will be treated as the fracture
                                                   radius, the minor axis length and the fracture length for the
                                                   cases of a radial fracture, an elliptical fracture and a PKN
                                                   fracture respectively.

        solid (MaterialProperties):           -- the MaterialProperties object giving the material properties.
        fluid (FluidProperties):              -- the FluidProperties object giving the fluid properties.
        injection (InjectionProperties):      -- the InjectionProperties object giving the injection
                                                       properties.
        simulProp (SimulationParameters):     -- the SimulationParameters object giving the numerical
                                                       parameters to be used in the simulation.

    Attributes:

        w (ndarray) :          -- fracture opening (width)
        p (ndarray):           -- fracture pressure
        time (float):                -- time since the start of injection
        EltChannel (ndarray):    -- list of cells currently in the channel region
        EltCrack (ndarray):      -- list of cells currently in the crack region
        EltRibbon (ndarray):     -- list of cells currently in the Ribbon region
        EltTip (ndarray):        -- list of cells currently in the Tip region
        v (ndarray):           -- propagation velocity for each cell in the tip cells
        alpha (ndarray):       -- angle prescribed by perpendicular on the fracture front (see Pierce 2015,
                                       Computation Methods Appl. Mech)
        l (ndarray):           -- length of perpendicular on the fracture front
                                       (see Pierce 2015, Computation Methods Appl. Mech)
        ZeroVertex (ndarray):  -- Vertex from which the perpendicular is drawn (can have value from 0 to 3,
                                       where 0 signify bottom left, 1 signifying bottom right, 2 signifying top
                                       right and 3 signifying top left vertex)
        FillF (ndarray):       -- filling fraction of each tip cell
        CellStatus (ndarray):    -- specifies which region each element currently belongs to
        initRad (float):             -- starting radius
        initTime (float):            -- starting time
        sgndDist (ndarray):    -- signed minimun distance from fracture front of each cell in the domain
        Q (ndarray-float):           -- injection rate into each cell of the domain
        FractEvol (ndarray):   -- array containing the coordinates of the individual fracture front lines;
                                       used for printing fracture evolution through time
        InCrack (ndarray):       -- array specifying whether the cell is inside or outside the fracture.
        FractureVolume (float):      -- fracture volume
        muPrime (ndarray):           -- local viscosity parameter
        Ffront (ndarray):            -- a list containing the intersection of the front and grid lines for the tip
                                       cells.
        regime (ndarray):            -- the regime of the ribbon cells (0 to 1, where 0 is fully toughness dominated,
                                       and 1 is fully viscosity dominated; See Zia and Lecampion 2018)
        ReynoldsNumber (ndarray):    -- the reynolds number at each edge of the cells in the fracture. The
                                       arrangement is left, right, bottom, top.
        fluidFlux (ndarray):         -- the fluid flux at each edge of the cells in the fracture. The arrangement is
                                       left, right, bottom, top.
        fluidVelocity (ndarray):     -- the fluid velocity at each edge of the cells in the fracture. The
                                       arrangement is left, right, bottom, top.

    functions:
        __init__                    Initialize the fracture according to the given initialization parameters.
        plot_fracture               plot given variable of the fracture
        PrintFractureTrace:         plot current regions and front position of the fracture

            
    """

    def __init__(self, Mesh, init_param, solid=None, fluid=None, injection=None, simulProp=None):
        """ Initialize the fracture according to the given initialization parameters.
            
        Args:
            Mesh (CartesianMesh)   -- a CartesianMesh class object describing the grid.
            init_param (tuple)     -- a tuple with the initialization parameters. The number of parameters depends
                                      on the initialization type given by the first element of the tuple.
                                            In case the first element (init_param[0], see below for possible options) is
                                            'PKN', the tuple should contain the following parameters in order:
                                                2. given_type   -- the type of the given value (see below for possible
                                                                   options).
                                                3. given_value  -- the value at which the fracture is initialized.
                                                4. -h           -- the height of the PKN fracture.

                                            In case the first element (init_param[0]) is 'M', 'Mt', 'K', 'Kt', 'MDR' or 'E',
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

        if init_type in ('PKN', 'KGD_K'):
            (init_type, given_type, given_value, h) = init_param
            gamma = None
        elif init_type in ('E_E', 'E_K'):
            (init_type, given_type, given_value, gamma) = init_param
            h = None
        elif init_type in ('M', 'Mt', 'K', 'Kt', 'MDR'): # radial fracture
            if len(init_param) == 3:
                (init_type, given_type, given_value) = init_param
                h = None
                gamma = None
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
                                                                      Kc_1=solid.Kc1,
                                                                      h=h,
                                                                      density=fluid.density,
                                                                      gamma=gamma)

            if init_type in ('M', 'Mt', 'K', 'Kt', 'MDR'):
                if self.initRad > min(Mesh.Lx, Mesh.Ly):
                    raise ValueError("The radius of the radial fracture is larger than domain!")
                # survey cells and their distances from the front
                surv_cells, inner_cells = get_eliptical_survey_cells(Mesh, self.initRad, self.initRad)
                surv_dist = self.initRad - (Mesh.CenterCoor[surv_cells, 0] ** 2 + Mesh.CenterCoor[
                                                                            surv_cells, 1] ** 2) ** 0.5
            elif init_type in ('E_E', 'E_K'):
                a = self.initRad * gamma
                if self.initRad > Mesh.Ly or a > Mesh.Lx:
                    raise ValueError("The axes length of the elliptical fracture is larger than domain!")
                surv_cells, inner_cells = get_eliptical_survey_cells(Mesh, a, self.initRad)
                surv_dist = np.zeros((surv_cells.size, ), dtype=np.float64)
                # get minimum distance from center of the survey cells
                for i in range(0, surv_cells.size):
                     surv_dist[i] = Distance_ellipse(a, self.initRad, Mesh.CenterCoor[surv_cells[i], 0],
                                                                    Mesh.CenterCoor[surv_cells[i], 1])
            elif 'PKN' in init_type or 'KGD' in init_type :
                if self.initRad > Mesh.Lx or h > Mesh.Ly:
                    raise ValueError("The fracture is larger than domain!")
                inner_cells = np.intersect1d(np.where(abs(Mesh.CenterCoor[:, 0]) < self.initRad)[0],
                                             np.where(abs(Mesh.CenterCoor[:, 1]) < h / 2)[0])
                max_x = max(abs(Mesh.CenterCoor[inner_cells, 0]))
                max_y = max(abs(Mesh.CenterCoor[inner_cells, 1]))
                ribbon_x = np.where(abs(abs(Mesh.CenterCoor[inner_cells, 0]) - max_x) < 100 * sys.float_info.epsilon)[0]
                ribbon_y = np.where(abs(abs(Mesh.CenterCoor[inner_cells, 1]) - max_y) < 100 * sys.float_info.epsilon)[0]
                surv_cells = np.append(inner_cells[ribbon_x], inner_cells[ribbon_y])
                surv_dist = np.zeros((len(surv_cells),), dtype=np.float64)
                surv_dist[0:len(ribbon_x)] = self.initRad - float(abs(Mesh.CenterCoor[inner_cells[ribbon_x[0]], 0]))
                surv_dist[len(ribbon_x):len(surv_cells)] = h / 2 - float(
                    abs(Mesh.CenterCoor[inner_cells[ribbon_y[0]], 1]))
                inner_cells = [x for x in inner_cells if x not in surv_cells]

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
                                                volume,
                                                simulProp.symmetric,
                                                solid.Eprime)


            if volume is None:
                volume = np.sum(self.w) * (Mesh.EltArea)

            if injection !=None:
                self.time = volume/injection.injectionRate[1,0]
            else:
                # set time to zero if mechanical loading is creating the fracture
                self.time = 0

        if vel is not None:
            self.v = vel * np.ones((self.EltTip.size, ), )
        else:
            self.v = vel
        self.sgndDist_last = None
        self.timeStep_last = None
        # setting arrival time to current time (assuming leak off starts at the time the fracture is initialized)
        self.Tarrival = np.full((self.mesh.NumberOfElts,), np.nan, dtype=np.float64)
        self.Tarrival[self.EltCrack] = self.time
        self.LkOff_vol = np.zeros((self.mesh.NumberOfElts,), dtype=np.float64)
        self.efficiency = 1.
        self.FractureVolume = np.sum(self.w) * Mesh.EltArea
        self.injectedVol = np.sum(self.w) * Mesh.EltArea
        self.InCrack = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        self.InCrack[self.EltCrack] = 1

        self.process_fracture_front()

        # local viscosity
        if fluid != None:
            self.muPrime = np.full((Mesh.NumberOfElts,), fluid.muPrime, dtype=np.float64)


        if simulProp.saveReynNumb:
            self.ReynoldsNumber = np.full((4, Mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.ReynoldsNumber = None

        # regime variable (goes from 0 for fully toughness dominated and one for fully viscosity dominated propagation)
        if simulProp.saveRegime:
            self.regime = np.full((Mesh.NumberOfElts, ), np.nan, dtype=np.float32)
        else:
            self.regime = None

        if simulProp.saveFluidFlux:
            self.fluidFlux = np.full((4, Mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidFlux = None

        if simulProp.saveFluidVel:
            self.fluidVelocity = np.full((4, Mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidVelocity = None

        self.closed = np.array([], dtype=int)

        self.TarrvlZrVrtx = np.full((Mesh.NumberOfElts,), np.nan, dtype=np.float64)
        self.TarrvlZrVrtx[self.EltTip] = self.time - self.l / self.v

#-----------------------------------------------------------------------------------------------------------------------


    def plot_fracture(self, variable='complete', mat_properties=None, projection='3D', elements=None,
                       backGround_param=None, plot_prop=None, fig=None, edge=4, contours_at=None, labels=None):
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
            identify (ndarray)      -- plot the cells in the provided list with cell number and different color
                                       to identify. This option can be used in debugging.
            mat_properties (MaterialProperties)   -- material properties to colorcode the grid according to the given
                                       parameter in the simulation properties. Can be None.
            sim_properties (SimulationParameters) -- Simulation paramters to define various plotting parameters. Can be
                                       None
            fig (figure)            -- figure object to superimpose the image

        """

        if variable in unidimensional_variables:
            raise ValueError("The variable does not vary spatially!")

        if variable is 'complete':
            fig = plot_fracture_list([self],
                                       variable='mesh',
                                       mat_properties=mat_properties,
                                       projection=projection,
                                       elements=elements,
                                       backGround_param=backGround_param,
                                       plot_prop=plot_prop,
                                       fig=fig,
                                       edge=edge,
                                       contours_at=contours_at,
                                       labels=labels)

            fig = plot_fracture_list([self],
                                     variable='footprint',
                                     mat_properties=mat_properties,
                                     projection=projection,
                                     elements=elements,
                                     backGround_param=backGround_param,
                                     plot_prop=plot_prop,
                                     fig=fig,
                                     edge=edge,
                                     contours_at=contours_at,
                                     labels=labels)
            variable = 'width'

        plot_non_zero = True
        if projection is '3D':
            plot_non_zero = False

        fig = plot_fracture_list([self],
                           variable=variable,
                           mat_properties=mat_properties,
                           projection=projection,
                           elements=elements,
                           backGround_param=backGround_param,
                           plot_prop=plot_prop,
                           fig=fig,
                           edge=edge,
                           contours_at=contours_at,
                           labels=labels,
                           plot_non_zero=plot_non_zero)

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
    def plot_fracture_slice(self, variable='width', point1=None, point2=None, projection='2D', plot_prop=None,
                            fig=None, edge=4, labels=None, plot_cell_center=False, orientation='horizontal'):
        """


        """

        return plot_fracture_list_slice([self],
                                        variable=variable,
                                        point1=point1,
                                        point2=point2,
                                        plot_prop=plot_prop,
                                        projection=projection,
                                        fig=fig,
                                        edge=edge,
                                        labels=labels,
                                        plot_cell_center=plot_cell_center,
                                        orientation=orientation)

# ------------------------------------------------------------------------------------------------------------------

    def SaveFracture(self, filename):
        with open(filename, 'wb') as output:
            dill.dump(self, output, -1)

# -----------------------------------------------------------------------------------------------------------------------

    def plot_front(self, fig=None, plot_prop=None):

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.axis('equal')
        else:
            ax = fig.get_axes()[0]

        if plot_prop is None:
            plot_prop = PlotProperties()

        I = self.Ffront[:, 0:2]
        J = self.Ffront[:, 2:4]
        # todo !!!Hack: gets very large values sometime, needs to be resolved
        for e in range(0, len(I)):
            if max(abs(I[e, :] - J[e, :])) < 3 * (self.mesh.hx ** 2 + self.mesh.hy ** 2) ** 0.5:
                ax.plot(np.array([I[e, 0], J[e, 0]]),
                        np.array([I[e, 1], J[e, 1]]),
                        plot_prop.lineStyle,
                        color=plot_prop.lineColor)

        if plot_prop.PlotFP_Time:
            tipVrtxCoord = self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip, self.ZeroVertex]]
            r_indx = np.argmax((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + self.l)
            x_coor = self.mesh.CenterCoor[self.EltTip[r_indx], 0] + 0.1 * self.mesh.hx
            y_coor = self.mesh.CenterCoor[self.EltTip[r_indx], 1] + 0.1 * self.mesh.hy
            if plot_prop.textSize is None:
                plot_prop.textSize = max(self.mesh.hx, self.mesh.hx)
            t = to_precision(self.time, plot_prop.dispPrecision) + 's'

            ax.text(x_coor,
                    y_coor,
                    t)

        return fig

#-----------------------------------------------------------------------------------------------------------------------


    def plot_front_3D(self, fig=None, plot_prop=None):

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlim(np.min(self.Ffront), np.max(self.Ffront))
            ax.set_ylim(np.min(self.Ffront), np.max(self.Ffront))
            plt.gca().set_aspect('equal')
            scale = 1.1
            zoom_factory(ax, base_scale=scale)
        else:
            ax = fig.get_axes()[0]

        ax.set_frame_on(False)
        ax.grid(False)
        ax.set_frame_on(False)
        ax.set_axis_off()

        if plot_prop is None:
            plot_prop = PlotProperties()

        I = self.Ffront[:, 0:2]
        J = self.Ffront[:, 2:4]

        # draw front lines
        for e in range(0, len(I)):
            Path = mpath.Path
            path_data = [
                (Path.MOVETO, [I[e, 0], I[e, 1]]),
                (Path.LINETO, [J[e, 0], J[e, 1]])]

            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)
            patch = mpatches.PathPatch(path,
                                       lw=plot_prop.lineWidth,
                                       edgecolor=plot_prop.lineColor)
            ax.add_patch(patch)
            art3d.pathpatch_2d_to_3d(patch)

        return fig

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

        coarse_mesh = CartesianMesh(factor*self.mesh.Lx,
                                    factor*self.mesh.Ly,
                                    self.mesh.nx,
                                    self.mesh.ny,
                                    symmetric=sim_prop.symmetric)

        # interpolate the level set by first advancing and then interpolating
        SolveFMM(self.sgndDist,
                 self.EltRibbon,
                 self.EltChannel,
                 self.mesh,
                 [],
                 self.EltChannel)

        sgndDist_coarse = griddata(self.mesh.CenterCoor[self.EltChannel],
                                   self.sgndDist[self.EltChannel],
                                   coarse_mesh.CenterCoor,
                                   method='linear',
                                   fill_value=1e10)

        # avoid adding tip cells from the fine mesh to get into the channel cells of the coarse mesh
        max_diag = (coarse_mesh.hx ** 2 + coarse_mesh.hy ** 2) ** 0.5
        excluding_tip = np.where(sgndDist_coarse <= -max_diag)[0]
        sgndDist_copy = np.copy(sgndDist_coarse)
        sgndDist_coarse = np.full(sgndDist_coarse.shape, 1e10, dtype=np.float64)
        sgndDist_coarse[excluding_tip] = sgndDist_copy[excluding_tip]

        # enclosing cells for each cell in the grid
        enclosing = np.zeros((self.mesh.NumberOfElts, 8), dtype=int)
        enclosing[:, :4] = self.mesh.NeiElements[:, :]
        enclosing[:, 4] = self.mesh.NeiElements[enclosing[:, 2], 0]
        enclosing[:, 5] = self.mesh.NeiElements[enclosing[:, 2], 1]
        enclosing[:, 6] = self.mesh.NeiElements[enclosing[:, 3], 0]
        enclosing[:, 7] = self.mesh.NeiElements[enclosing[:, 3], 1]

        if factor == 2.:
            intersecting = np.array([], dtype=int)
            for i in range(-int(((self.mesh.ny - 1) / 2 + 1) / 2) + 1, int(((self.mesh.ny - 1) / 2 + 1) / 2)):
                center = self.mesh.CenterElts[0] + i * self.mesh.nx
                row_to_add = np.arange(center - int(((self.mesh.nx - 1) / 2 + 1) / 2) + 1,
                                       center + int(((self.mesh.nx - 1) / 2 + 1) / 2),
                                       dtype=int)
                intersecting = np.append(intersecting, row_to_add)

            corresponding = []
            for i in intersecting:
                corresponding.append(self.mesh.locate_element(coarse_mesh.CenterCoor[i, 0],
                                                            coarse_mesh.CenterCoor[i, 1]))


            corresponding = np.asarray(corresponding, dtype=int)

            w_coarse = np.zeros((coarse_mesh.NumberOfElts, ), dtype=np.float64)
            w_coarse[intersecting] = (self.w[corresponding]
                                        + np.sum(self.w[enclosing[corresponding, :4]] / 2, axis=1) +
                                        np.sum(self.w[enclosing[corresponding, 4:8]] / 4, axis=1)) / 4

            LkOff_vol = np.zeros((coarse_mesh.NumberOfElts,), dtype=np.float64)
            LkOff_vol[intersecting] = (self.LkOff_vol[corresponding]
                                        + np.sum(self.LkOff_vol[enclosing[corresponding, :4]] / 2, axis=1) +
                                        np.sum(self.LkOff_vol[enclosing[corresponding, 4:8]] / 4, axis=1)) / 4


        else:
            w_coarse = griddata(self.mesh.CenterCoor[self.EltChannel],
                                self.w[self.EltChannel],
                                coarse_mesh.CenterCoor,
                                method='linear',
                                fill_value=0.)

        # interpolate last level set by first advancing to the end of the grid and then interpolating
        SolveFMM(self.sgndDist_last,
                 self.EltRibbon,
                 self.EltChannel,
                 self.mesh,
                 [],
                 self.EltChannel)

        sgndDist_last_coarse = griddata(self.mesh.CenterCoor[self.EltChannel],
                                   self.sgndDist_last[self.EltChannel],
                                   coarse_mesh.CenterCoor,
                                   method='linear',
                                   fill_value=1e10)

        # making the initialization paramenters tuple
        init_data = ('G',
                     excluding_tip,
                     excluding_tip,
                     -sgndDist_coarse[excluding_tip],
                     w_coarse,
                     None,
                     C,
                     self.FractureVolume,
                     np.nan)

        # re-meshing of the material properties
        material_prop.remesh(coarse_mesh)

        Fr_coarse = Fracture(coarse_mesh,
                            init_data,
                            solid=material_prop,
                            fluid=fluid_prop,
                            injection=inj_prop,
                            simulProp=sim_prop)

        # evaluate current level set on the coarse mesh
        EltRibbon = np.delete(Fr_coarse.EltRibbon,np.where(sgndDist_copy[Fr_coarse.EltRibbon] >= 1e10)[0])
        EltChannel = np.delete(Fr_coarse.EltChannel, np.where(sgndDist_copy[Fr_coarse.EltChannel] >= 1e10)[0])
        cells_outside = np.arange(coarse_mesh.NumberOfElts)
        cells_outside = np.delete(cells_outside, EltChannel)
        SolveFMM(sgndDist_copy,
                 EltRibbon,
                 EltChannel,
                 coarse_mesh,
                 cells_outside,
                 [])

        # evaluate last level set on the coarse mesh to evaluate velocity of the tip
        EltRibbon = np.delete(Fr_coarse.EltRibbon, np.where(sgndDist_last_coarse[Fr_coarse.EltRibbon] >= 1e10)[0])
        EltChannel = np.delete(Fr_coarse.EltChannel, np.where(sgndDist_last_coarse[Fr_coarse.EltChannel] >= 1e10)[0])
        cells_outside = np.arange(coarse_mesh.NumberOfElts)
        cells_outside = np.delete(cells_outside, EltChannel)
        SolveFMM(sgndDist_last_coarse,
                 EltRibbon,
                 EltChannel,
                 coarse_mesh,
                 cells_outside,
                 [])

        Fr_coarse.v = -(sgndDist_copy[Fr_coarse.EltTip] -
                        sgndDist_last_coarse[Fr_coarse.EltTip]) / self.timeStep_last

        Fr_coarse.Tarrival[Fr_coarse.EltChannel] = griddata(self.mesh.CenterCoor[self.EltChannel],
                                                            self.Tarrival[self.EltChannel],
                                                            coarse_mesh.CenterCoor[Fr_coarse.EltChannel],
                                                            method='linear')

        Fr_coarse.LkOff_vol = LkOff_vol
        Fr_coarse.injectedVol = self.injectedVol
        Fr_coarse.efficiency = (Fr_coarse.injectedVol - sum(Fr_coarse.LkOff_vol[Fr_coarse.EltCrack]))\
                               / Fr_coarse.injectedVol
        Fr_coarse.time = self.time
        Fr_coarse.closed = self.closed

        # update the saved properties
        if sim_prop.saveToDisk:
            prop = (material_prop, fluid_prop, inj_prop, sim_prop)
            with open(sim_prop.get_outputFolder() + "properties", 'wb') as output:
                dill.dump(prop, output, -1)

        return Fr_coarse