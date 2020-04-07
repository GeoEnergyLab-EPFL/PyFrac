# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 16:22:33 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
import dill
import numpy as np
import math
from scipy.interpolate import griddata

# local import
# import fracture_initialization
# import visualization
from level_set import SolveFMM
from volume_integral import Pdistance
from fracture_initialization import get_survey_points, get_width_pressure, generate_footprint
from fracture_initialization import Geometry, InitializationParameters
from HF_analytical import HF_analytical_sol
from visualization import plot_fracture_list, plot_fracture_list_slice, to_precision, zoom_factory
from labels import unidimensional_variables
from properties import PlotProperties


class Fracture:
    """
     Class defining propagating fracture.

    Args:
        mesh (CartesianMesh):                   -- a CartesianMesh class object describing the grid.
        init_param (tuple):                     -- a InitializationParameters class object (see class documentation).
        solid (MaterialProperties):             -- the MaterialProperties object giving the material properties.
        fluid (FluidProperties):                -- the FluidProperties class object giving the fluid properties.
        injection (InjectionProperties):        -- the InjectionProperties class object giving the injection properties.
        simulProp (SimulationParameters):       -- the SimulationParameters class object giving the numerical parameters
                                                   to be used in the simulation.

    Attributes:
        w (ndarray) :               -- fracture opening (width)
        pFluid (ndarray):           -- the fluid pressure in the fracture.
        pNet (ndarray):             -- the net pressure in the fracture.
        time (float):               -- time since the start of injection
        EltChannel (ndarray):       -- list of cells currently in the channel region
        EltCrack (ndarray):         -- list of cells currently in the crack region
        EltRibbon (ndarray):        -- list of cells currently in the Ribbon region
        EltTip (ndarray):           -- list of cells currently in the Tip region
        v (ndarray):                -- propagation velocity for each cell in the tip cells
        alpha (ndarray):            -- angle prescribed by perpendicular on the fracture front (see Pierce 2015,
                                       Computation Methods Appl. Mech)
        l (ndarray):                -- length of perpendicular on the fracture front
                                       (see Pierce 2015, Computation Methods Appl. Mech)
        ZeroVertex (ndarray):       -- Vertex from which the perpendicular is drawn (can have value from 0 to 3,
                                       where 0 signify bottom left, 1 signifying bottom right, 2 signifying top
                                       right and 3 signifying top left vertex)
        FillF (ndarray):            -- filling fraction of each tip cell
        CellStatus (ndarray):       -- specifies which region each element currently belongs to
        sgndDist (ndarray):         -- signed minimum distance from fracture front of each cell in the domain
        InCrack (ndarray):          -- array specifying whether the cell is inside or outside the fracture.
        FractureVolume (float):     -- fracture volume
        muPrime (ndarray):          -- local viscosity parameter
        Ffront (ndarray):           -- a list containing the intersection of the front and grid lines for the tip
                                       cells. Each row contains the x and y coordinates of the two points.
        regime (ndarray):           -- the regime of the ribbon cells (0 to 1, where 0 is fully toughness dominated,
                                       and 1 is fully viscosity dominated; See Zia and Lecampion 2018, IJF)
        ReynoldsNumber (ndarray):   -- the reynolds number at each edge of the cells in the fracture. The
                                       arrangement is left, right, bottom, top.
        fluidFlux (ndarray):        -- the fluid flux at each edge of the cells in the fracture. The arrangement is
                                       left, right, bottom, top.
        fluidVelocity (ndarray):    -- the fluid velocity at each edge of the cells in the fracture. The
                                       arrangement is left, right, bottom, top.
        LkOffTotal (ndarray):       -- total fluid volume leaked off from each of the cell in the mesh
        Tarrival (ndarray):         -- the arrival time of the fracture front for each of the cell in the domain. It
                                       is used to evaluate the leak off using Carter's leak off formulation. The time
                                       is averaged over entering and leaving of the front from a cell.
        TarrvlZrVrtx (ndarray):     -- the time at which the front crosses the zero vertex. This is used to evaluate
                                       leak off in tip cells, i.e. for cells where the front has not left the cell.
        closed (ndarray):           -- the cells which have closed due to leak off or flow back of the fluid.
        injectedVol (float):        -- the total volume that is injected into the fracture.
        sgndDist_last (ndarray):    -- the signed distance of the last time step. Used for re-meshing.
        timeStep_last (float):      -- the last time step. Required for re-meshing.
        source (ndarray):           -- the list of injection cells i.e. the source elements.

    """

    def __init__(self, mesh, init_param, solid=None, fluid=None, injection=None, simulProp=None):
        """
        Initialize the fracture according to the given initialization parameters.

        """

        self.mesh = mesh

        if init_param.regime != 'static':

            # get appropriate length dimension variable
            length = init_param.geometry.get_length_dimension()

            # get analytical solution
            self.time, length, self.pNet, \
            self.w, self.v, actvElts = HF_analytical_sol(init_param.regime,
                                          self.mesh,
                                          solid.Eprime,
                                          injection.injectionRate[1, 0],
                                          inj_point=injection.sourceCoordinates,
                                          muPrime=fluid.muPrime,
                                          Kprime=solid.Kprime[self.mesh.CenterElts][0],
                                          Cprime=solid.Cprime[self.mesh.CenterElts][0],
                                          length=length,
                                          t=init_param.time,
                                          Kc_1=solid.Kc1,
                                          h=init_param.geometry.fractureHeight,
                                          density=fluid.density,
                                          Cij=solid.Cij,
                                          gamma=init_param.geometry.gamma)
            init_param.geometry.set_length_dimension(length)

        surv_cells, surv_dist, inner_cells = get_survey_points(init_param.geometry,
                                                               self.mesh,
                                                               source_coord=injection.sourceCoordinates)

        self.EltChannel, self.EltTip, self.EltCrack, \
        self.EltRibbon, self.ZeroVertex, self.CellStatus, \
        self.l, self.alpha, self.FillF, self.sgndDist, self.Ffront = generate_footprint(self.mesh,
                                                                        surv_cells,
                                                                        inner_cells,
                                                                        surv_dist,
                                                                        simulProp.projMethod)
        # for static fracture initialization
        if init_param.regime == 'static':
            self.w, self.pNet = get_width_pressure(self.mesh,
                                                   self.EltCrack,
                                                   self.EltTip,
                                                   self.FillF,
                                                   init_param.C,
                                                   init_param.width,
                                                   init_param.netPressure,
                                                   init_param.fractureVolume,
                                                   simulProp.symmetric,
                                                   solid.Eprime)

            if init_param.fractureVolume is None:
                volume = np.sum(self.w) * mesh.EltArea
                self.time = volume / injection.injectionRate[1, 0]

            self.v = init_param.tipVelocity

        if self.v is not None:
            if isinstance(self.v, float):
                self.v = self.v * np.ones((self.EltTip.size,), dtype=np.float64)
        else:
            self.v = np.full((self.EltTip.size,), np.nan, dtype=np.float64)
        self.pFluid = np.zeros((self.mesh.NumberOfElts,))
        self.pFluid[self.EltCrack] = self.pNet[self.EltCrack] + solid.SigmaO[self.EltCrack]
        self.sgndDist_last = None
        self.timeStep_last = None
        # setting arrival time to current time (assuming leak off starts at the time the fracture is initialized)
        self.Tarrival = np.full((self.mesh.NumberOfElts,), np.nan, dtype=np.float64)
        self.Tarrival[self.EltCrack] = self.time
        self.LkOff = np.zeros((self.mesh.NumberOfElts,), dtype=np.float64)
        self.LkOffTotal = 0.
        self.efficiency = 1.
        self.FractureVolume = np.sum(self.w) * mesh.EltArea
        self.injectedVol = np.sum(self.w) * mesh.EltArea
        self.InCrack = np.zeros((self.mesh.NumberOfElts,), dtype=np.uint8)
        self.InCrack[self.EltCrack] = 1
        self.wHist = np.copy(self.w)
        self.source = np.intersect1d(injection.sourceElem, self.EltCrack)

        if simulProp.projMethod != 'LS_continousfront':
            self.process_fracture_front()

        # local viscosity
        if fluid is not None:
            self.muPrime = np.full((mesh.NumberOfElts,), fluid.muPrime, dtype=np.float64)

        if simulProp.saveReynNumb:
            self.ReynoldsNumber = np.full((4, mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.ReynoldsNumber = None

        # regime variable (goes from 0 for fully toughness dominated and one for fully viscosity dominated propagation)
        if simulProp.saveRegime:
            self.regime = np.full((mesh.NumberOfElts,), np.nan, dtype=np.float32)
        else:
            self.regime = None

        if simulProp.saveFluidFlux:
            self.fluidFlux = np.full((4, mesh.NumberOfElts), np.nan, dtype=np.float32)
            self.fluidFlux_components = np.full((8, mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidFlux = None
            self.fluidFlux_components = None

        if simulProp.saveFluidVel:
            self.fluidVelocity = np.full((4, mesh.NumberOfElts), np.nan, dtype=np.float32)
            self.fluidVelocity_components = np.full((8, mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidVelocity = None
            self.fluidVelocity_components = None

        if simulProp.saveFluidFluxAsVector:
            self.fluidFlux_components = np.full((8, mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidFlux_components = None

        if simulProp.saveFluidVelAsVector:
            self.fluidVelocity_components = np.full((8, mesh.NumberOfElts), np.nan, dtype=np.float32)
        else:
            self.fluidVelocity_components = None

        self.closed = np.array([], dtype=int)

        self.TarrvlZrVrtx = np.full((mesh.NumberOfElts,), np.nan, dtype=np.float64)
        if self.v is not None:
            self.TarrvlZrVrtx[self.EltTip] = self.time - self.l / self.v


#-----------------------------------------------------------------------------------------------------------------------

    def plot_fracture(self, variable='complete', mat_properties=None, projection='3D', elements=None,
                      backGround_param=None, plot_prop=None, fig=None, edge=4, contours_at=None, labels=None,
                      plot_non_zero=True):
        """
        This function plots the fracture.

        Args:
            variable (string):                  -- the variable to be plotted. See :py:data:`supported_variables` of the
                                                   :py:mod:`labels` module for a list of supported variables.
            mat_properties (MaterialProperties):-- the material properties. It is mainly used to colormap the mesh.
            projection (string):                -- a string specifying the projection. See :py:data:`supported_projections`
                                                   for the supported projections for each of the supported variable. If
                                                   not provided, the default will be used.
            elements (ndarray):                 -- the elements to be plotted.
            backGround_param (string):          -- the parameter according to which the the mesh will be color-mapped.\
                                                   Options are listed below:

                                                        - 'confining stress' or 'sigma0'
                                                        - 'fracture toughness' or 'K1c'
                                                        - 'leak off coefficient', 'Cl'
            plot_prop (PlotProperties):         -- the properties to be used for the plot.
            fig (Figure):                       -- the figure to superimpose on. New figure will be made if not
                                                   provided.
            edge (int):                         -- the edge of the cell that will be plotted. This is for variables that
                                                    are evaluated on the cell edges instead of cell center. It can have
                                                    a value from 0 to 4 (0->left, 1->right, 2->bottome, 3->top,
                                                    4->average).
            contours_at (list):                 -- the values at which the contours are to be plotted.
            labels (LabelProperties):           -- the labels to be used for the plot.
            plot_non_zero (bool):               -- if true, only non-zero values will be plotted.

        Returns:
            (Figure):                           -- A Figure object that can be used superimpose further plots.
        """

        if variable in unidimensional_variables:
            raise ValueError("The variable does not vary spatially!")

        if variable == 'complete':
            proj = '3D'
            if '2D' in projection:
                proj = '2D'
            fig = plot_fracture_list([self],
                                       variable='mesh',
                                       mat_properties=mat_properties,
                                       projection=proj,
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
                                     projection=proj,
                                     elements=elements,
                                     backGround_param=backGround_param,
                                     plot_prop=plot_prop,
                                     fig=fig,
                                     edge=edge,
                                     contours_at=contours_at,
                                     labels=labels)
            variable = 'width'

        if projection == '3D':
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
        """

        # list of points where fracture front is intersecting the grid lines. 
        intrsct1 = np.zeros((2, len(self.l)))
        intrsct2 = np.zeros((2, len(self.l)))

        # todo: commenting

        for i in range(0, len(self.l)):
            if self.alpha[i] != 0 and self.alpha[i] != math.pi / 2:  # for angles greater than zero and less than 90 deg
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

        self.Ffront = tmp


#-----------------------------------------------------------------------------------------------------------------------

    def plot_fracture_slice(self, variable='width', point1=None, point2=None, projection='2D', plot_prop=None,
                            fig=None, edge=4, labels=None, plot_cell_center=False, orientation='horizontal'):
        """
        This function plots the fracture on a given slice of the domain. Two points are to be given that will be
        joined to form the slice. The values on the slice are interpolated from the values available on the cell
        centers. Exact values on the cell centers can also be plotted.

        Args:
            variable (string):              -- the variable to be plotted. See :py:data:`supported_variables` of the
                                               :py:mod:`labels` module for a list of supported variables.
            point1 (list or ndarray):       -- the left point from which the slice should pass [x, y].
            point2 (list or ndarray):       -- the right point from which the slice should pass [x, y].
            projection (string):            -- a string specifying the projection. It can either '3D' or '2D'.
            plot_prop (PlotProperties):     -- the properties to be used for the plot.
            fig (Figure):                   -- the figure to superimpose on. New figure will be made if not provided.
            edge (int):                     -- the edge of the cell that will be plotted. This is for variables that
                                               are evaluated on the cell edges instead of cell center. It can have a
                                               value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
            labels (LabelProperties):       -- the labels to be used for the plot.
            plot_cell_center (bool):        -- if True, the discrete values at the cell centers will be plotted. In this
                                               case, the slice passing through the center of the cell containing
                                               point1 will be taken. The slice will be made according to the given
                                               orientation (see orientation). If False, the values will be interpolated
                                               on the line joining the given two points.
            orientation (string):           -- the orientation according to which the slice is made in the case the
                                               plotted values are not interpolated and are taken at the cell centers.
                                               Any of the four ('vertical', 'horizontal', 'ascending' and 'descending')
                                               orientation can be used.

        Returns:
            (Figure):                       -- A Figure object that can be used superimpose further plots.

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
        """ This function saves the fracture object to a file on hard dist using dill module"""

        with open(filename, 'wb') as output:
            dill.dump(self, output, -1)

# -----------------------------------------------------------------------------------------------------------------------

    def plot_front(self, fig=None, plot_prop=None):
        """
        This function plots the front lines in the tip cells of the fracture taken from the fFront variable.
        """

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
        """
        This function plots the front lines with 3D projection in the tip cells of the fracture taken from the fFront
        variable.
        """

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

    def remesh(self, factor, C, coarse_mesh, material_prop, fluid_prop, inj_prop, sim_prop):
        """
        This function compresses the fracture by the given factor once it has reached the end of the mesh. If the
        compression factor is two, each set of four cells in the fine mesh is replaced by a single cell. The volume of
        the fracture is conserved upto machine precision. The elasticity matrix and the properties objects are also
        re-adjusted according to the new mesh.

        Arguments:
            factor (float):                     -- the factor by which the domain is to be compressed. For example, a
                                                   factor of 2 will merge the adjacent four cells to a single cell.
            C (ndarray):                        -- the elasticity matrix to be re-evaluated for the new mesh.
            coarse_mesh (CartesianMesh):        -- the coarse Cartesian mesh.
            material_prop (MaterialProperties): -- the MaterialProperties object giving the material properties.
            fluid_prop(FluidProperties):        -- the FluidProperties class object giving the fluid properties to be
                                                   re-evaluated for the new mesh..
            inj_prop(InjectionProperties):      -- the InjectionProperties class object giving the injection properties
                                                   to be re-evaluated for the new mesh.
            sim_prop (SimulationParameters):    -- the SimulationParameters class object giving the numerical parameters
                                                   to be used in the simulation.

        Returns:
            Fr_coarse (Fracture):   -- the new fracture after re-meshing.
        """

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
            # finding the intersecting cells of the fine and course mesh
            intersecting = np.array([], dtype=int)
            #todo: a description is to be written, its not readable
            for i in range(-int(((self.mesh.ny - 1) / 2 + 1) / 2) + 1, int(((self.mesh.ny - 1) / 2 + 1) / 2)):
                center = self.mesh.CenterElts[0] + i * self.mesh.nx
                row_to_add = np.arange(center - int(((self.mesh.nx - 1) / 2 + 1) / 2) + 1,
                                       center + int(((self.mesh.nx - 1) / 2 + 1) / 2),
                                       dtype=int)
                intersecting = np.append(intersecting, row_to_add)

            # getting the corresponding cells of the coarse mesh in the fine mesh
            corresponding = []
            for i in intersecting:
                corresponding.append(self.mesh.locate_element(coarse_mesh.CenterCoor[i, 0],
                                                            coarse_mesh.CenterCoor[i, 1]))
            corresponding = np.asarray(corresponding, dtype=int)

            # weighted sum to conserve volume upto machine precision
            w_coarse = np.zeros((coarse_mesh.NumberOfElts, ), dtype=np.float64)
            w_coarse[intersecting] = (self.w[corresponding]
                                        + np.sum(self.w[enclosing[corresponding, :4]] / 2, axis=1) +
                                        np.sum(self.w[enclosing[corresponding, 4:8]] / 4, axis=1)) / 4

            LkOff = np.zeros((coarse_mesh.NumberOfElts,), dtype=np.float64)
            LkOff[intersecting] = (self.LkOff[corresponding]
                                        + np.sum(self.LkOff[enclosing[corresponding, :4]] / 2, axis=1) +
                                        np.sum(self.LkOff[enclosing[corresponding, 4:8]] / 4, axis=1))

            wHist_coarse = np.zeros((coarse_mesh.NumberOfElts,), dtype=np.float64)
            wHist_coarse[intersecting] = (self.wHist[corresponding]
                                      + np.sum(self.wHist[enclosing[corresponding, :4]] / 2, axis=1) +
                                      np.sum(self.wHist[enclosing[corresponding, 4:8]] / 4, axis=1)) / 4

        else:
            # In case the factor by which mesh is compressed is not 2
            w_coarse = griddata(self.mesh.CenterCoor[self.EltChannel],
                                self.w[self.EltChannel],
                                coarse_mesh.CenterCoor,
                                method='linear',
                                fill_value=0.)

            LkOff = 4 * griddata(self.mesh.CenterCoor[self.EltChannel],
                                self.LkOff[self.EltChannel],
                                coarse_mesh.CenterCoor,
                                method='linear',
                                fill_value=0.)

            wHist_coarse = griddata(self.mesh.CenterCoor[self.EltChannel],
                                self.wHist[self.EltChannel],
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

        Fr_Geometry = Geometry(shape='level set',
                               survey_cells=excluding_tip,
                               inner_cells=excluding_tip,
                               tip_distances=-sgndDist_coarse[excluding_tip])
        init_data = InitializationParameters(geometry=Fr_Geometry,
                                             regime='static',
                                             width=w_coarse,
                                             elasticity_matrix=C,
                                             tip_velocity=np.nan)

        Fr_coarse = Fracture(coarse_mesh,
                            init_data,
                            solid=material_prop,
                            fluid=fluid_prop,
                            injection=inj_prop,
                            simulProp=sim_prop)

        # evaluate current level set on the coarse mesh
        EltRibbon = np.delete(Fr_coarse.EltRibbon, np.where(sgndDist_copy[Fr_coarse.EltRibbon] >= 1e10)[0])
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
        Tarrival_nan = np.where(np.isnan(Fr_coarse.Tarrival[Fr_coarse.EltChannel]))[0]
        if Tarrival_nan.size > 0:
            for elt in Tarrival_nan:
                Fr_coarse.Tarrival[Fr_coarse.EltChannel[elt]] = np.nanmean(
                                            Fr_coarse.Tarrival[coarse_mesh.NeiElements[Fr_coarse.EltChannel[elt]]])

        Fr_coarse.TarrvlZrVrtx[Fr_coarse.EltChannel] = griddata(self.mesh.CenterCoor[self.EltChannel],
                                                            self.TarrvlZrVrtx[self.EltChannel],
                                                            coarse_mesh.CenterCoor[Fr_coarse.EltChannel],
                                                            method='linear')

        # The zero vertex arrival time for the tip elements is taken equal to the corresponding element in the
        # fine mesh. If not available, average is taken of the enclosing elements
        to_correct = []
        for indx, elt in enumerate(Fr_coarse.EltTip):
            corr_tip = self.mesh.locate_element(coarse_mesh.CenterCoor[elt, 0], coarse_mesh.CenterCoor[elt, 1])
            if np.isnan(self.TarrvlZrVrtx[corr_tip]):
                TarrvlZrVrtx = 0
                cnt = 0
                for j in range(8):
                    if not np.isnan(self.TarrvlZrVrtx[enclosing[corr_tip][j]]):
                        TarrvlZrVrtx += self.TarrvlZrVrtx[enclosing[corr_tip][j]]
                        cnt += 1
                if cnt > 0:
                    Fr_coarse.TarrvlZrVrtx[elt] = TarrvlZrVrtx / cnt
                else:
                    to_correct.append(indx)
                    Fr_coarse.TarrvlZrVrtx[elt] = np.nan
            else:
                Fr_coarse.TarrvlZrVrtx[elt] = self.TarrvlZrVrtx[corr_tip]
        if len(to_correct) > 0:
            for elt in to_correct:
                Fr_coarse.TarrvlZrVrtx[Fr_coarse.EltTip[elt]] = np.nanmean(Fr_coarse.TarrvlZrVrtx[
                                                Fr_coarse.mesh.NeiElements[Fr_coarse.EltTip[elt]]])

        Fr_coarse.LkOff = LkOff
        Fr_coarse.LkOffTotal = self.LkOffTotal
        Fr_coarse.injectedVol = self.injectedVol
        Fr_coarse.efficiency = (Fr_coarse.injectedVol - Fr_coarse.LkOffTotal) / Fr_coarse.injectedVol
        Fr_coarse.time = self.time
        Fr_coarse.closed = np.asarray([])
        Fr_coarse.wHist = wHist_coarse

        return Fr_coarse
