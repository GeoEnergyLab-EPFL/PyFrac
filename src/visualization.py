# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Friday, July 06, 2018.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021. All rights
reserved. See the LICENSE.TXT file for more details.
"""
import logging
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import copy
import pickle
import io

# local imports
from postprocess_fracture import *
from properties import PlotProperties, LabelProperties
from labels import supported_variables, supported_projections, \
                   unidimensional_variables, suitable_elements


def plot_fracture_list(fracture_list, variable='footprint', projection=None, elements=None, plot_prop=None, fig=None,
                       edge=4, contours_at=None, labels=None, mat_properties=None, backGround_param=None,
                       plot_non_zero=True, source_loc=np.asarray([0,0])):
    """
    This function plots the fracture evolution with time. The state of the fracture at different times is provided in
    the form of a list of Fracture objects.

    Args:
        fracture_list (list):               -- the list of Fracture objects giving the evolution of fracture with
                                                    time.
        variable (string):                  -- the variable to be plotted. See :py:data:`supported_variables` of the
                                                :py:mod:`labels` module for a list of supported variables.
        mat_properties (MaterialProperties):-- the material properties. It is mainly used to colormap the mesh.
        projection (string):                -- a string specifying the projection. See :py:data:`supported_projections`
                                                for the supported projections for each of the supported variable. If not
                                                provided, the default will be used.
        elements (ndarray):                 -- the elements to be plotted.
        backGround_param (string):          -- the parameter according to which the the mesh will be colormapped.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        edge (int):                         -- the edge of the cell that will be plotted. This is for variables that
                                                are evaluated on the cell edges instead of cell center. It can have a
                                                value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        contours_at (list):                 -- the values at which the contours are to be plotted.
        labels (LabelProperties):           -- the labels to be used for the plot.
        plot_non_zero (bool):               -- if true, only non-zero values will be plotted.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_fracture_list')
    log.info("Plotting " + variable + '...')

    if not isinstance(fracture_list, list):
        raise ValueError("The provided fracture_list is not list type object!")

    if len(fracture_list) == 0:
        raise ValueError("Provided fracture list is empty!")

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if projection is None:
        projection = supported_projections[variable][0]
    elif projection not in supported_projections[variable]:
        raise ValueError("The given projection is not supported for \'" + variable +
                         '\'. Select one of the following\n' + repr(supported_projections[variable]))

    if plot_prop is None:
        plot_prop = PlotProperties()

    if labels is None:
        labels = LabelProperties(variable, 'whole mesh', projection)

    max_Lx = 0.
    max_Ly = 0.
    for i in fracture_list:
        if i.mesh.Lx > max_Lx:
            largest_mesh = i.mesh
            max_Lx = i.mesh.Lx
        if i.mesh.Ly > max_Ly:
            largest_mesh = i.mesh
            max_Ly = i.mesh.Ly

    if variable == 'mesh':
        if backGround_param is not None and mat_properties is None:
            raise ValueError("Material properties are required to color code background")
        if projection == '2D':
            fig = largest_mesh.plot(fig=fig,
                                    material_prop=mat_properties,
                                    backGround_param=backGround_param,
                                    plot_prop=plot_prop)

        else:
            fig = largest_mesh.plot_3D(fig=fig,
                                 material_prop=mat_properties,
                                 backGround_param=backGround_param,
                                 plot_prop=plot_prop)

    elif variable == 'footprint':
        if projection == '2D':
            for i in fracture_list:
                fig = i.plot_front(fig=fig, plot_prop=plot_prop)
        else:
            for i in fracture_list:
                fig = i.plot_front_3D(fig=fig, plot_prop=plot_prop)

    elif variable in ['source elements', 'se']:
        for fr in fracture_list:
            fig = plot_injection_source(fr, fig=fig, plot_prop=plot_prop)

    else:
        if variable == 'chi':
            vel_list, time_list = get_fracture_variable(fracture_list,
                                                            'v',
                                                            edge=edge,
                                                            return_time=True)
            var_val_list = []
            for i in vel_list:
                actual_ki = 2 * mat_properties.Cprime * mat_properties.Eprime / \
                            (np.sqrt(np.asarray(i)) * mat_properties.Kprime)
                var_val_list.append(actual_ki.tolist())

        elif variable == 'regime':
            var_val_list, legend_coord, time_list = get_fracture_variable(fracture_list,
                                                            variable,
                                                            edge=edge,
                                                            return_time=True)

        else:
            var_val_list, time_list = get_fracture_variable(fracture_list,
                                                            variable,
                                                            edge=edge,
                                                            return_time=True)

        var_val_copy = np.copy(var_val_list)
        for i in range(len(var_val_copy)):
            var_val_copy[i] /= labels.unitConversion

        if projection != '2D_vectorfield':
            var_value_tmp = np.copy(var_val_copy)
            if elements is not None:
                var_value_tmp = var_value_tmp[:, elements]
            if plot_non_zero:
                var_value_tmp = var_value_tmp[var_value_tmp != 0]
            vmin, vmax = np.inf, -np.inf
            if len(np.shape(var_value_tmp)) > 1:
                var_value_tmp = list(var_value_tmp[0])
            for i in var_value_tmp:
                if plot_non_zero:
                    i = i[i != 0]
                i = np.delete(i, np.where(np.isinf(i))[0])
                i = np.delete(i, np.where(np.isnan(i))[0])
                if not (not isinstance(i, float) and len(i) == 0):
                    if variable in ('p', 'pressure'):
                        non_zero = np.where(abs(i) > 0)[0]
                        i_min, i_max = -0.2 * np.median(i[non_zero]), 1.5 * np.median(i[non_zero])
                    else:
                        i_min, i_max = np.min(i), np.max(i)
                    vmin, vmax = min(vmin, i_min), max(vmax, i_max)

    if variable == 'regime':
        for i in range(len(var_val_list)):
            fig = plot_regime(var_val_copy[i],
                              fracture_list[i].mesh,
                              elements=fracture_list[i].EltRibbon,
                              fig=fig)

    elif variable in unidimensional_variables:
        fig = plot_variable_vs_time(time_list,
                                    var_val_list,
                                    fig=fig,
                                    plot_prop=plot_prop,
                                    label=labels.legend)
    #todo: the following was elif variable not in ['mesh', 'footprint']:
    elif variable in bidimensional_variables:
        if projection != '2D_vectorfield':
            if plot_non_zero:
                for indx, value in enumerate(var_val_copy):
                    remove_zeros(value, fracture_list[indx].mesh)#i[np.where(abs(i) < 1e-16)[0]] = np.nan

        if variable == 'surface':
            plot_prop.colorMap = 'cool'
            for i in range(len(var_val_list)):
                fig = plot_fracture_surface(var_val_copy[i],
                                            fracture_list[i].mesh,
                                            fig=fig,
                                            plot_prop=plot_prop,
                                            plot_colorbar=False,
                                            elements=elements,
                                            vmin=vmin,
                                            vmax=vmax)

        elif projection == '2D_clrmap':
            for i in range(len(var_val_list)):
                fig = plot_fracture_variable_as_image(var_val_copy[i],
                                                          fracture_list[i].mesh,
                                                          fig=fig,
                                                          plot_prop=plot_prop,
                                                          elements=elements,
                                                          plt_colorbar=False,
                                                          vmin=vmin,
                                                          vmax=vmax)
        elif projection == '2D_contours':
            for i in range(len(var_val_list)):
                labels.legend = 't= ' + to_precision(time_list[i], plot_prop.dispPrecision)
                plot_prop.lineColor = plot_prop.colorsList[i % len(plot_prop.colorsList)]
                fig = plot_fracture_variable_as_contours(var_val_copy[i],
                                                         fracture_list[i].mesh,
                                                         fig=fig,
                                                         plot_prop=plot_prop,
                                                         contours_at=contours_at,
                                                         plt_colorbar=False,
                                                         vmin=vmin,
                                                         vmax=vmax)
        elif projection == '3D':
            for i in range(len(var_val_list)):
                fig = plot_fracture_variable_as_surface(var_val_copy[i],
                                                        fracture_list[i].mesh,
                                                        fig=fig,
                                                        plot_prop=plot_prop,
                                                        plot_colorbar=False,
                                                        elements=elements,
                                                        vmin=vmin,
                                                        vmax=vmax)
        elif projection == '2D_vectorfield' and not np.isnan(var_val_copy[i]).any():
            # fracture_list[i].EltCrack => ribbon+tip+other in crack
            # fracture_list[i].EltChannel => ribbon+other in crack

            # multiple options:
            # elements_where_to_plot = fracture_list[i].EltCrack
            elements_where_to_plot = fracture_list[i].EltChannel
            # elements_where_to_plot = np.setdiff1d(fracture_list[i].EltChannel,fracture_list[i].EltRibbon)
            # elements_where_to_plot = np.setdiff1d(elements_where_to_plot, np.unique(np.ndarray.flatten(fracture_list[i].mesh.NeiElements[fracture_list[i].EltRibbon])))
            fig = plot_fracture_variable_as_vector(var_val_copy[i],
                                                      fracture_list[i].mesh,
                                                      elements_where_to_plot,
                                                      fig=fig)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)
    ax.set_title(labels.figLabel)
    if projection == '3D' and variable not in ['mesh', 'footprint', 'se', 'source elements']:
        ax.set_zlabel(labels.zLabel)
        sm = plt.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                   norm=plt.Normalize(vmin=vmin,
                                                      vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm, alpha=plot_prop.alpha)
        cb.set_label(labels.colorbarLabel)

    elif projection in ('2D_clrmap', '2D_contours') and variable != 'regime':
        im = ax.images
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
        cb.set_label(labels.colorbarLabel)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_list_slice(fracture_list, variable='width', point1=None, point2=None, projection='2D', plot_prop=None,
                             fig=None, edge=4, labels=None, plot_cell_center=False, orientation='horizontal',
                             extreme_points=None, export2Json=False, export2Json_assuming_no_remeshing=True):
    """
    This function plots the fracture evolution on a given slice of the domain. Two points are to be given that will be
    joined to form the slice. The values on the slice are either interpolated from the values available on the cell
    centers. Exact values on the cell centers can also be plotted.

    Args:
        fracture_list (list):               -- the list of Fracture objects giving the evolution of fracture with
                                                time.
        variable (string):                  -- the variable to be plotted. See :py:data:`supported_variables` of the
                                                :py:mod:`labels` module for a list of supported variables.
        point1 (list or ndarray):           -- the left point from which the slice should pass [x, y].
        point2 (list or ndarray):           -- the right point from which the slice should pass [x, y].
        projection (string):                -- a string specifying the projection. It can either '3D' or '2D'.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        edge (int):                         -- the edge of the cell that will be plotted. This is for variables that
                                                are evaluated on the cell edges instead of cell center. It can have a
                                                value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        labels (LabelProperties):           -- the labels to be used for the plot.
        plot_cell_center (bool):            -- if True, the discrete values at the cell centers will be plotted. In this
                                                case, the slice passing through the center of the cell containing
                                                point1 will be taken. The slice will be made according to the given
                                                orientation (see orientation). If False, the values will be interpolated
                                                on the line joining the given two points.
        orientation (string):               -- the orientation according to which the slice is made in the case the
                                               plotted values are not interpolated and are taken at the cell centers.
                                               Any of the four ('vertical', 'horizontal', 'ascending' and 'descending')
                                               orientation can be used.
        extreme_points (ndarray)            -- An empty array of shape (2, 2). It will be used to return the extreme
                                               points of the plotted slice. These points can be used to plot analytical
                                               solution.
        export2Json (bool)                  -- If you set it to True the function will return a dictionary with the
                                               data of the corresponding plot
    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if variable in unidimensional_variables:
        raise ValueError("The given variable does not vary spatially.")

    if plot_prop is None:
        plot_prop = PlotProperties()
        if plot_cell_center:
            plot_prop.lineStyle = '.'

    if labels is None:
        labels = LabelProperties(variable, 'slice', projection)

    mesh_list = get_fracture_variable(fracture_list,
                                                'mesh',
                                                edge=edge,
                                                return_time=False)
    var_val_list, time_list = get_fracture_variable(fracture_list,
                                                        variable,
                                                       edge=edge,
                                                    return_time=True)

    var_val_copy = np.copy(var_val_list)
    for i in range(len(var_val_copy)):
        var_val_copy[i] /= labels.unitConversion

    # find maximum and minimum to set the viewing limits on axis
    var_value_tmp = np.copy(var_val_copy)
    vmin, vmax = np.inf, -np.inf
    for i in var_value_tmp:
        i = np.delete(i, np.where(np.isinf(i))[0])
        i = np.delete(i, np.where(np.isnan(i))[0])
        if not (not isinstance(i, float) and len(i) == 0):
            if variable in ('p', 'pressure'):
                non_zero = np.where(abs(i) > 0)[0]
                i_min, i_max = -0.2 * np.median(i[non_zero]), 1.5 * np.median(i[non_zero])
            else:
                if len(i) > 0:
                    i_min, i_max = np.min(i), np.max(i)
                else:
                    i_min, i_max = np.inf, -np.inf
            vmin, vmax = min(vmin, i_min), max(vmax, i_max)

    label = labels.legend

    if export2Json:
        to_write = {
            'size_of_data': len(time_list),
            'time_list': time_list}

    for i in range(len(var_val_list)):
        labels.legend = label + ' t= ' + to_precision(time_list[i],
                                                      plot_prop.dispPrecision)
        plot_prop.lineColor = plot_prop.colorsList[i % len(plot_prop.colorsList)]
        if '2D' in projection:
            if plot_cell_center:
                fig ,sampling_line_out, var_value_selected, sampling_cells = plot_fracture_slice_cell_center(var_val_copy[i],
                                                                  mesh_list[i],
                                                                  point=point1,
                                                                  orientation=orientation,
                                                                  fig=fig,
                                                                  plot_prop=plot_prop,
                                                                  vmin=vmin,
                                                                  vmax=vmax,
                                                                  plot_colorbar=False,
                                                                  labels=labels,
                                                                  extreme_points=extreme_points,
                                                                  export2Json = export2Json)
                if i == 0 and export2Json and export2Json_assuming_no_remeshing: #write ones the sampling line, assuming no remeshing
                    to_write[variable+'_sampling_coords_'] = sampling_line_out.tolist()
                    to_write[variable+'_sampling_cells']  = sampling_cells.tolist()
                if export2Json and not export2Json_assuming_no_remeshing:
                    to_write[variable+'_sampling_coords_'+str(i)] = sampling_line_out.tolist()
                    to_write[variable+'_sampling_cells_'+str(i)] = sampling_cells.tolist()
                    to_write[variable+'_'+str(i)] = var_value_selected.tolist()
                if export2Json and export2Json_assuming_no_remeshing:
                    to_write[str(i)] = var_value_selected.tolist()
            else:
                fig = plot_fracture_slice_interpolated(var_val_copy[i],
                                                                mesh_list[i],
                                                                point1=point1,
                                                                point2=point2,
                                                                fig=fig,
                                                                plot_prop=plot_prop,
                                                                vmin=vmin,
                                                                vmax=vmax,
                                                                plot_colorbar=False,
                                                                labels=labels,
                                                                export2Json = export2Json)
            if not export2Json:
                ax_tv = fig.get_axes()[0]
                ax_tv.set_xlabel('meter')
                ax_tv.set_ylabel('meter')
                ax_tv.set_title('Top View')

                # making colorbar
                im = ax_tv.images
                divider = make_axes_locatable(ax_tv)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
                cb.set_label(labels.colorbarLabel)

                ax_slice = fig.get_axes()[1]
                ax_slice.set_ylabel(labels.colorbarLabel)
                ax_slice.set_xlabel('(x,y) ' + labels.xLabel)

        elif projection == '3D' and not export2Json:
            fig = plot_slice_3D(var_val_copy[i],
                                mesh_list[i],
                                point1=point1,
                                point2=point2,
                                fig=fig,
                                plot_prop=plot_prop,
                                vmin=vmin,
                                vmax=vmax,
                                label=labels.legend)
            ax_slice = fig.get_axes()[0]
            ax_slice.set_xlabel('meter')
            ax_slice.set_ylabel('meter')
            ax_slice.set_zlabel(labels.zLabel)
            ax_slice.title(labels.figLabel)
        else:
            raise ValueError("Given Projection is not correct!")

    if plot_prop.plotLegend and not export2Json:
        ax_slice.legend()

    if export2Json:
        return to_write
    else:
        return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_list_at_point(fracture_list, variable='width', point=None, plot_prop=None, fig=None,
                             edge=4, labels=None):
    """
    This function plots the fracture evolution on a given point.

    Args:
        fracture_list (list):               -- the list of Fracture objects giving the evolution of fracture with
                                                time.
        variable (string):                  -- the variable to be plotted. See :py:data:`supported_variables` of the
                                                :py:mod:`labels` module for a list of supported variables.
        point (list or ndarray):            -- the point at which the given variable is plotted against time [x, y].
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        edge (int):                         -- the edge of the cell that will be plotted. This is for variables that
                                                are evaluated on the cell edges instead of cell center. It can have a
                                                value from 0 to 4 (0->left, 1->right, 2->bottome, 3->top, 4->average).
        labels (LabelProperties):           -- the labels to be used for the plot.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if variable in unidimensional_variables:
        raise ValueError("The given variable does not vary spatially.")

    if plot_prop is None:
        plot_prop = PlotProperties()

    if labels is None:
        labels = LabelProperties(variable, 'point', '2D')

    if point is None:
        point = [0., 0.]

    point_values, time_list = get_fracture_variable_at_point(fracture_list,
                                                            variable,
                                                            point=point,
                                                            edge=edge)

    point_values = np.asarray(point_values) / labels.unitConversion

    fig = plot_variable_vs_time(time_list,
                                point_values,
                                fig=fig,
                                plot_prop=plot_prop,
                                label=labels.legend)

    ax = fig.get_axes()[0]
    ax.set_xlabel('time ($s$)')
    ax.set_ylabel(labels.colorbarLabel)
    ax.set_title(labels.figLabel)
    if plot_prop.plotLegend:
        ax.legend()

    plot_prop_fp = PlotProperties(line_color='k')
    labels_fp = LabelProperties('footprint', 'whole mesh', '2D')
    labels_fp.figLabel = ''
    fig_image = plot_fracture_list([fracture_list[-1]],
                                   variable='footprint',
                                   projection='2D',
                                   plot_prop=plot_prop_fp,
                                   labels=labels_fp)

    labels_2D = LabelProperties(variable, 'whole mesh', '2D_clrmap')
    labels_2D.figLabel = 'Sampling Point'
    fig_image = plot_fracture_list([fracture_list[-1]],
                       variable=variable,
                       projection='2D_clrmap',
                       fig=fig_image,
                       plot_prop=plot_prop,
                       edge=edge,
                       labels=labels_2D)

    ax_image = fig_image.get_axes()[0]
    ax_image.plot([point[0]], [point[1]], 'ko')

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_fracture_variable_as_vector(var_value, mesh, Elements_to_plot, fig=None):
    """
    This function plots a given 2D vector field.

    Args:
        var_value:                      -- an array with each column having the following information:
                                            [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge,
                                             fy bottom edge, fx top edge, fy top edge]
                                            note that "fx left edge" is the component along the x direction of the
                                            vector at the left edge of the cell. The name of the cell is coincident with
                                            the column position.
        mesh (CartesianMesh):           -- a CartesianMesh object giving the descritization of the domain.
        Elements_to_plot:               -- list of cell names on whose edges plot the vectors.
        fig (Figure):                   -- the figure to superimpose on. New figure will be made if not provided.

    Returns:
        (Figure):                       -- A Figure object that can be used superimpose further plots.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    U = np.vstack((var_value[0,Elements_to_plot], var_value[2,Elements_to_plot]))
    U = np.vstack((U, var_value[4,Elements_to_plot]))
    U = np.vstack((U, var_value[6,Elements_to_plot]))
    U = np.ndarray.flatten(U)

    V = np.vstack((var_value[1,Elements_to_plot], var_value[3,Elements_to_plot]))
    V = np.vstack((V, var_value[5,Elements_to_plot]))
    V = np.vstack((V, var_value[7,Elements_to_plot]))
    V = np.ndarray.flatten(V)

    X = np.vstack((mesh.CenterCoor[Elements_to_plot,0]-mesh.hx*0.5, mesh.CenterCoor[Elements_to_plot,0]+mesh.hx*0.5))
    X = np.vstack((X, mesh.CenterCoor[Elements_to_plot,0]))
    X = np.vstack((X, mesh.CenterCoor[Elements_to_plot,0]))
    X = np.ndarray.flatten(X)

    Y = np.vstack((mesh.CenterCoor[Elements_to_plot,1], mesh.CenterCoor[Elements_to_plot,1]))
    Y = np.vstack((Y, mesh.CenterCoor[Elements_to_plot,1]-mesh.hy*0.5))
    Y = np.vstack((Y, mesh.CenterCoor[Elements_to_plot,1]+mesh.hy*0.5))
    Y = np.ndarray.flatten(Y)

    M = np.hypot(U, V)

    ax.quiver(X,Y,U,V,M,pivot='mid')

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_variable_vs_time(time_list, value_list, fig=None, plot_prop=None, label=None):
    """
    This function plots a given list of values against time.

    Args:
        time_list (list or array):      -- the list of times.
        value_list (list or array):     -- the list of values.
        fig (Figure):                   -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):     -- the properties to be used for the plot.
        label (string):                 -- the label given to the plot line.

    Returns:
        (Figure):                       -- A Figure object that can be used superimpose further plots.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    if plot_prop.plotLegend and label is not None:
        label_copy = label
    else:
        label_copy = None

    if plot_prop.graphScaling == 'linear':
        ax.plot(time_list,
                value_list,
                plot_prop.lineStyle,
                color=plot_prop.lineColor,
                label=label_copy)

    elif plot_prop.graphScaling == 'loglog':
        ax.loglog(time_list,
                  value_list,
                  plot_prop.lineStyle,
                  color=plot_prop.lineColor,
                  label=label_copy)

    elif plot_prop.graphScaling == 'semilogx':
        ax.semilogx(time_list,
                    value_list,
                    plot_prop.lineStyle,
                    color=plot_prop.lineColor,
                    label=label_copy)

    elif plot_prop.graphScaling == 'semilogy':
        ax.semilogy(time_list,
                    value_list,
                    plot_prop.lineStyle,
                    color=plot_prop.lineColor,
                    label=label_copy)
    else:
        raise ValueError("Graph scaling type not supported")

    if plot_prop.plotLegend:
        ax.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_variable_as_image(var_value, mesh, fig=None, plot_prop=None, elements=None, vmin=None,
                                    vmax=None, plt_colorbar=True):
    """
    This function plots the 2D fracture variable in the form of a colormap.

    Args:
        var_value (ndarray):                -- a ndarray of the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        elements (ndarray):                 -- the elements to be plotted.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        plt_colorbar (bool):                -- if True, colorbar will be plotted.
    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """

    if elements is not None:
        var_value_fullMesh = np.full((mesh.NumberOfElts, ), np.nan)
        var_value_fullMesh[elements] = var_value[elements]
        var_value = var_value_fullMesh

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    x = mesh.CenterCoor[:, 0].reshape((mesh.ny, mesh.nx))
    y = mesh.CenterCoor[:, 1].reshape((mesh.ny, mesh.nx))

    var_value_2D = var_value.reshape((mesh.ny, mesh.nx))

    dx = (x[0, 1] - x[0, 0]) / 2.
    dy = (y[1, 0] - y[0, 0]) / 2.
    extent = [x[0, 0] - dx, x[-1, -1] + dx, y[0, 0] - dy, y[-1, -1] + dy]

    if plot_prop is None:
        plot_prop = PlotProperties()

    if vmin is None and vmax is None:
        var_value = np.delete(var_value, np.where(np.isinf(var_value))[0])
        var_value = np.delete(var_value, np.where(np.isnan(var_value))[0])
        vmin, vmax = np.min(var_value), np.max(var_value)

    cax = ax.imshow(var_value_2D,
              cmap=plot_prop.colorMap,
              interpolation=plot_prop.interpolation,
              extent=extent,
              alpha=0.8,
              vmin=vmin,
              vmax=vmax,
              origin='lower')

    if plt_colorbar:
        fig.colorbar(cax)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_variable_as_surface(var_value, mesh, fig=None, plot_prop=None, plot_colorbar=True, elements=None,
                                      vmin=None, vmax=None):
    """
    This function plots the 2D fracture variable in the form of a surface.

    Args:
        var_value (ndarray):                -- a ndarray of the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        elements (ndarray):                 -- the elements to be plotted.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        plot_colorbar (bool):               -- if True, colorbar will be plotted.
    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        scale = 1.1
        zoom_factory(ax, base_scale=scale)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    if elements is None:
        elements = np.arange(0, mesh.NumberOfElts)

    if vmin is None and vmax is None:
        var_value = np.delete(var_value, np.where(np.isinf(var_value))[0])
        var_value = np.delete(var_value, np.where(np.isnan(var_value))[0])
        vmin, vmax = np.min(var_value), np.max(var_value)

    ax.plot_trisurf(mesh.CenterCoor[elements, 0],
                          mesh.CenterCoor[elements, 1],
                          var_value[elements],
                          cmap=plot_prop.colorMap,
                          linewidth=plot_prop.lineWidth,
                          alpha=plot_prop.alpha,
                          vmin=vmin,
                          vmax=vmax)

    if plot_colorbar:
        sm = plt.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        plt.colorbar(sm, alpha=plot_prop.alpha)

    ax.set_zlim(vmin, vmax)

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_fracture_surface(width, mesh, fig=None, plot_prop=None, plot_colorbar=True, elements=None,
                                      vmin=None, vmax=None):
    """
    This function plots the 2D fracture variable in the form of a surface.

    Args:
        width (ndarray):                    -- the fracture width.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        elements (ndarray):                 -- the elements to be plotted.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        plt_colorbar (bool):                -- if True, colorbar will be plotted.
    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        scale = 1.1
        zoom_factory(ax, base_scale=scale)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    if elements is None:
        elements = np.arange(0, mesh.NumberOfElts)

    if vmin is None and vmax is None:
        width = np.delete(width, np.where(np.isinf(width))[0])
        width = np.delete(width, np.where(np.isnan(width))[0])
        vmin, vmax = np.min(width), np.max(width)

    ax.plot_trisurf(mesh.CenterCoor[elements, 0],
                  mesh.CenterCoor[elements, 1],
                  width[elements] / 2,
                  cmap=plot_prop.colorMap,
                  linewidth=plot_prop.lineWidth,
                  alpha=plot_prop.alpha,
                  vmin=vmin,
                  vmax=vmax)

    ax.plot_trisurf(mesh.CenterCoor[elements, 0],
                    mesh.CenterCoor[elements, 1],
                    -width[elements] / 2,
                    cmap=plot_prop.colorMap,
                    linewidth=plot_prop.lineWidth,
                    alpha=plot_prop.alpha,
                    vmin=vmin,
                    vmax=vmax)

    if plot_colorbar:
        sm = plt.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        plt.colorbar(sm, alpha=plot_prop.alpha)

    ax.set_zlim(vmin, vmax)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_variable_as_contours(var_value, mesh, fig=None, plot_prop=None, plt_backGround=True,
                                       plt_colorbar=True, contours_at=None, vmin=None, vmax=None):
    """
    This function plots the contours of the 2D fracture variable.

    Args:
        var_value (ndarray):                -- a ndarray of the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        plt_backGround(bool):               -- if True, the colormap of the variable will also be plotted.
        plt_colorbar (bool):                -- if True, colorbar will be plotted.
        contours_at (list or ndarray):      -- the values at which the countours are to be plotted.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    x = mesh.CenterCoor[:, 0].reshape((mesh.ny, mesh.nx))
    y = mesh.CenterCoor[:, 1].reshape((mesh.ny, mesh.nx))

    var_value_2D = var_value.reshape((mesh.ny, mesh.nx))

    dx = (x[0,1] - x[0,0]) / 2.
    dy = (y[1,0] - y[0,0]) / 2.
    extent = [x[0,0] - dx, x[-1, -1] + dx, y[0,0] - dy, y[-1, -1] + dy]

    if vmin is None and vmax is None:
        var_value = np.delete(var_value, np.where(np.isinf(var_value))[0])
        var_value = np.delete(var_value, np.where(np.isnan(var_value))[0])
        vmin, vmax = np.min(var_value), np.max(var_value)

    if plt_backGround:
        cax = ax.imshow(var_value_2D,
                  cmap=plot_prop.colorMap,
                  interpolation=plot_prop.interpolation,
                  extent=extent,
                  vmin=vmin,
                  vmax=vmax,
                  origin='lower')

        if plt_colorbar:
            cbar = fig.colorbar(cax)

    if contours_at is None:
        contours_at = vmin + (vmax-vmin) * np.asarray([0.01, 0.3, 0.5, 0.7, 0.9])

    CS = ax.contour(x,
                    y,
                    var_value_2D,
                    contours_at,
                    colors=plot_prop.lineColor,
                    label='fsd'
                    )

    plt.clabel(CS)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_slice_interpolated(var_value, mesh, point1=None, point2=None, fig=None, plot_prop=None, vmin=None,
                                     vmax=None, plot_colorbar=True, labels=None, plt_2D_image=True, export2Json = False):
    """
    This function plots the fracture on a given slice of the domain. Two points are to be given that will be
    joined to form the slice. The values on the slice are interpolated from the values available on the cell
    centers.

    Args:
        var_value (ndarray):                -- a ndarray with the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        point1 (list or ndarray):           -- the left point from which the slice should pass [x, y].
        point2 (list or ndarray):           -- the right point from which the slice should pass [x, y].
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        plot_colorbar (bool):               -- if True, colorbar will be plotted.
        labels (LabelProperties):           -- the labels to be used for the plot.
        plt_2D_image (bool):                -- if True, a subplot showing the colormap and the slice will also be
                                                plotted.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_fracture_slice_interpolated')
    if not export2Json:
        log.info("Plotting slice...")
    if plt_2D_image:
        if fig is None:
            fig = plt.figure()
            ax_2D = fig.add_subplot(211)
            ax_slice = fig.add_subplot(212)
        else:
            ax_2D = fig.get_axes()[0]
            ax_slice = fig.get_axes()[1]
    else:
        if fig is None:
            fig = plt.figure()
            ax_slice = fig.add_subplot(111)
        else:
            if len(fig.get_axes()) > 1:
                ax_slice = fig.get_axes()[1]
            else:
                ax_slice = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()


    if plt_2D_image:
        x = mesh.CenterCoor[:, 0].reshape((mesh.ny, mesh.nx))
        y = mesh.CenterCoor[:, 1].reshape((mesh.ny, mesh.nx))

        var_value_2D = var_value.reshape((mesh.ny, mesh.nx))

        dx = (x[0,1] - x[0,0]) / 2.
        dy = (y[1,0] - y[0,0]) / 2.
        extent = [x[0,0] - dx, x[-1, -1] + dx, y[0,0] - dy, y[-1, -1] + dy]

        im_2D = ax_2D.imshow(var_value_2D,
                            cmap=plot_prop.colorMap,
                            interpolation=plot_prop.interpolation,
                            extent=extent,
                            vmin=vmin,
                            vmax=vmax,
                            origin='lower')

        if plot_colorbar:
            divider = make_axes_locatable(ax_2D)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_2D, cax=cax, orientation='vertical')


    if point1 is None:
        point1 = np.array([-mesh.Lx, 0.])
    if point2 is None:
        point2 = np.array([mesh.Lx, 0.])

    # the code below find the extreme points of the line joining the two given points with the current mesh
    if point2[0] == point1[0]:
        point1[1] = -mesh.Ly
        point2[1] = mesh.Ly
    elif point2[1] == point1[1]:
        point1[0] = -mesh.Lx
        point2[0] = mesh.Lx
    else:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        y_intrcpt_lft = slope * (-mesh.Lx - point1[0]) + point1[1]
        y_intrcpt_rgt = slope * (mesh.Lx - point1[0]) + point1[1]
        x_intrcpt_btm = (-mesh.Ly - point1[1]) / slope + point1[0]
        x_intrcpt_top = (mesh.Ly - point1[1]) / slope + point1[0]

        if abs(y_intrcpt_lft) < mesh.Ly:
            point1[0] = -mesh.Lx
            point1[1] = y_intrcpt_lft
        if y_intrcpt_lft > mesh.Ly:
            point1[0] = x_intrcpt_top
            point1[1] = mesh.Ly
        if y_intrcpt_lft < -mesh.Ly:
            point1[0] = x_intrcpt_btm
            point1[1] = -mesh.Ly

        if abs(y_intrcpt_rgt) < mesh.Ly:
            point2[0] = mesh.Lx
            point2[1] = y_intrcpt_rgt
        if y_intrcpt_rgt > mesh.Ly:
            point2[0] = x_intrcpt_top
            point2[1] = mesh.Ly
        if y_intrcpt_rgt < -mesh.Ly:
            point2[0] = x_intrcpt_btm
            point2[1] = -mesh.Ly


    if plt_2D_image:
        ax_2D.plot(np.array([point1[0], point2[0]]),
               np.array([point1[1], point2[1]]),
               plot_prop.lineStyle,
               color=plot_prop.lineColor)


    sampling_points = np.hstack((np.linspace(point1[0], point2[0], 105).reshape((105, 1)),
                                 np.linspace(point1[1], point2[1], 105).reshape((105, 1))))

    value_samp_points = griddata(mesh.CenterCoor,
                                    var_value,
                                    sampling_points,
                                    method='linear',
                                    fill_value=np.nan)

    sampling_line_lft = ((sampling_points[:52, 0] - sampling_points[52, 0]) ** 2 +
                     (sampling_points[:52, 1] - sampling_points[52, 1]) ** 2) ** 0.5
    sampling_line_rgt = ((sampling_points[52:, 0] - sampling_points[52, 0]) ** 2 +
                         (sampling_points[52:, 1] - sampling_points[52, 1]) ** 2) ** 0.5
    sampling_line = np.concatenate((-sampling_line_lft, sampling_line_rgt))

    if labels is None:
        legend = None
    else:
        legend = labels.legend

    ax_slice.plot(sampling_line,
                  value_samp_points,
                  plot_prop.lineStyle,
                  color=plot_prop.lineColor,
                  label=legend)

    #ax_slice.set_xticks(np.hstack((sampling_line[[0, 20, 41, 62, 83, 104]], sampling_line[104])))
    ax_slice.set_xticks(np.hstack((sampling_line[[0, 20, 41, 52, 62, 83, 104]])))

    xtick_labels = []
    for i in [0, 20, 41, 52, 62, 83, 104]:
        xtick_labels.append('(' + to_precision(sampling_points[i, 0],
                                                plot_prop.dispPrecision) + ', ' +
                                  to_precision(sampling_points[i, 1],
                                                plot_prop.dispPrecision) + ')')

    ax_slice.set_xticklabels(xtick_labels)
    if vmin is not None and vmax is not None:
        ax_slice.set_ylim((vmin - 0.1*vmin, vmax + 0.1*vmax))

    return fig


#-----------------------------------------------------------------------------------------------------------------------

def plot_fracture_slice_cell_center(var_value, mesh, point=None, orientation='horizontal', fig=None, plot_prop=None,
                                vmin=None, vmax=None, plot_colorbar=True, labels=None, plt_2D_image=True,
                                extreme_points=None,export2Json=False):
    """
    This function plots the fracture on a given slice of the domain. A points along with the direction of the slice is
    given to form the slice. The slice is made from the center of the cell containing the given point along the given
    orientation.

    Args:
        var_value (ndarray):                -- a ndarray with the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        point (list or ndarray):            -- the point from which the slice should pass [x, y].
        orientation (string):               -- the orientation according to which the slice is made. Any of the four
                                               ('vertical', 'horizontal', 'ascending' and 'descending') orientations
                                               can be used.
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        plot_colorbar (bool):               -- if True, colorbar will be plotted.
        labels (LabelProperties):           -- the labels to be used for the plot.
        plt_2D_image (bool):                -- if True, a subplot showing the colormap and the slice will also be
                                                plotted.
        extreme_points (ndarray)            -- An empty array of shape (2, 2). It will be used to return the extreme
                                                points of the plotted slice. These points can be used to plot analytical
                                                solution.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_fracture_slice_cell_center')
    if not export2Json:
        log.info("Plotting slice...")
        if plt_2D_image:
            if fig is None:
                fig = plt.figure()
                ax_2D = fig.add_subplot(211)
                ax_slice = fig.add_subplot(212)
            else:
                ax_2D = fig.get_axes()[0]
                ax_slice = fig.get_axes()[1]
        else:
            if fig is None:
                fig = plt.figure()
                ax_slice = fig.add_subplot(111)
            else:
                ax_slice = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()
        plot_prop.lineStyle = '.'


    if plt_2D_image and not export2Json:
        x = mesh.CenterCoor[:, 0].reshape((mesh.ny, mesh.nx))
        y = mesh.CenterCoor[:, 1].reshape((mesh.ny, mesh.nx))

        var_value_2D = var_value.reshape((mesh.ny, mesh.nx))

        dx = (x[0, 1] - x[0, 0]) / 2.
        dy = (y[1, 0] - y[0, 0]) / 2.
        extent = [x[0, 0] - dx, x[-1, -1] + dx, y[0, 0] - dy, y[-1, -1] + dy]

        im_2D = ax_2D.imshow(var_value_2D,
                            cmap=plot_prop.colorMap,
                            interpolation=plot_prop.interpolation,
                            extent=extent,
                            vmin=vmin,
                            vmax=vmax,
                            origin='lower')

        if plt_2D_image and plot_colorbar:
            divider = make_axes_locatable(ax_2D)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_2D, cax=cax, orientation='vertical')


    if point is None:
        point = np.array([0., 0.])
    if orientation not in ('horizontal', 'vertical', 'increasing', 'decreasing'):
        raise ValueError("Given orientation is not supported. Possible options:\n 'horizontal', 'vertical',"
                         " 'increasing', 'decreasing'")

    zero_cell = mesh.locate_element(point[0], point[1])[0]
    if np.isnan(zero_cell).any():
        raise ValueError("The given point does not lie in the grid!")

    if orientation == 'vertical':
        sampling_cells = np.hstack((np.arange(zero_cell, 0, -mesh.nx)[::-1],
                                    np.arange(zero_cell, mesh.NumberOfElts, mesh.nx)))
        x_plot_coord = mesh.CenterCoor[sampling_cells, 1]
    elif orientation == 'horizontal':
        sampling_cells = np.arange(zero_cell // mesh.nx * mesh.nx, (zero_cell // mesh.nx + 1) * mesh.nx)
        x_plot_coord = mesh.CenterCoor[sampling_cells, 0]
    elif orientation == 'increasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx - 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx + 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] <
                                                mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))

        x_plot_coord = np.hstack((- np.sqrt([sum(tup) for tup in (mesh.CenterCoor[bottom_half] -
                                                               mesh.CenterCoor[zero_cell]) ** 2]),
                                 np.sqrt([sum(tup) for tup in (mesh.CenterCoor[top_half] -
                                                               mesh.CenterCoor[zero_cell]) ** 2])))

    elif orientation == 'decreasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx + 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] <
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx - 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))


        x_plot_coord = np.hstack((- np.sqrt([sum(tup) for tup in (mesh.CenterCoor[bottom_half] -
                                                               mesh.CenterCoor[zero_cell]) ** 2]),
                                 np.sqrt([sum(tup) for tup in (mesh.CenterCoor[top_half] -
                                                               mesh.CenterCoor[zero_cell]) ** 2])))


    if plt_2D_image and not export2Json:
        ax_2D.plot(mesh.CenterCoor[sampling_cells, 0],
                   mesh.CenterCoor[sampling_cells, 1],
                   'k.',
                   linewidth=plot_prop.lineWidth,
                   alpha=plot_prop.alpha,
                   markersize='1')

    # sampling_len = ((mesh.CenterCoor[sampling_cells[0], 0] - mesh.CenterCoor[sampling_cells[-1], 0]) ** 2 + \
    #                (mesh.CenterCoor[sampling_cells[0], 1] - mesh.CenterCoor[sampling_cells[-1], 1]) ** 2) ** 0.5
    #
    # # making x-axis centered at zero for the 1D slice. Necessary to have same reference with different meshes and
    # # analytical solution plots.
    # sampling_line = np.linspace(0, sampling_len, len(sampling_cells)) - sampling_len / 2
    if not export2Json:
        # ax_slice.plot(sampling_line,
        #               var_value[sampling_cells],
        #               plot_prop.lineStyle,
        #               color=plot_prop.lineColor,
        #               label=labels.legend)
        ax_slice.plot(x_plot_coord,
                      var_value[sampling_cells],
                      plot_prop.lineStyle,
                      color=plot_prop.lineColor,
                      label=labels.legend)

    if len(sampling_cells) > 7:
        mid = len(sampling_cells) // 2
        half_1st = np.arange(0, mid, mid // 3)
        half_2nd = np.arange(mid + mid // 3, len(sampling_cells), mid // 3)
        if len(half_2nd) < 3:
            half_2nd = np.append(half_2nd, len(sampling_cells) - 1)
        x_ticks = np.hstack((half_1st[:3], np.array([mid], dtype=int)))
        x_ticks = np.hstack((x_ticks, half_2nd))
    else:
        x_ticks = len(sampling_cells)

    # if not export2Json: ax_slice.set_xticks(sampling_line[x_ticks])

    if not export2Json: ax_slice.set_xticks(x_plot_coord[x_ticks])

    xtick_labels = []
    for i in x_ticks:
        xtick_labels.append('(' + to_precision(np.round(mesh.CenterCoor[sampling_cells[i], 0], 3),
                                                plot_prop.dispPrecision) + ', ' +
                                  to_precision(np.round(mesh.CenterCoor[sampling_cells[i], 1], 3),
                                                plot_prop.dispPrecision) + ')')

    if not export2Json:
        ax_slice.set_xticklabels(xtick_labels)
        if vmin is not None and vmax is not None:
            ax_slice.set_ylim((vmin - 0.1*vmin, vmax + 0.1*vmax))

    if extreme_points is not None:
        extreme_points[0] = mesh.CenterCoor[sampling_cells[0]]
        extreme_points[1] = mesh.CenterCoor[sampling_cells[-1]]

    if export2Json: fig = None
    # return fig, sampling_line, var_value[sampling_cells], sampling_cells
    return fig, x_plot_coord, var_value[sampling_cells], sampling_cells

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution_slice(regime, variable, mat_prop, inj_prop, mesh=None, fluid_prop=None, fig=None,
                             point1=None, point2=None, time_srs=None, length_srs=None, h=None, samp_cell=None,
                             plot_prop=None, labels=None, gamma=None, plt_top_view=False):
    """
    This function plots slice of the given analytical solution. It can be used to compare simulation results by
    superimposing on the figure obtained from the slice plot function.

    Args:
        regime (string):                        -- the string specifying the limiting case solution to be plotted. The
                                                   available options are.

            ========    ============================
            option      limiting solution
            ========    ============================
            'M'         viscosity storage
            'Mp'        finite pulse viscosity storage
            'Mt'        viscosity leak-off
            'K'         toughness storage
            'Kt'        toughness leak-off
            'PKN'       PKN
            'KGD_K'     KGD toughness
            'MDR'       MDR turbulent viscosity
            'E_K'       anisotropic toughness
            'E_E'       anisotropic elasticity
            ========    ============================

        variable (string):                      -- the variable to be plotted. Possible options are 'w', 'width' or 'p',
                                                   'pressure'.
        mat_prop (MaterialProperties):          -- the MaterialProperties object giving the material properties.
        inj_prop (InjectionProperties):         -- the InjectionProperties object giving the injection properties.
        mesh (CartesianMesh):                   -- a CartesianMesh class object describing the grid.
        fluid_prop( FluidProperties):           -- the FluidProperties object giving the fluid properties.
        fig (figure):                           -- figure object to superimpose the image.
        point1 (list or ndarray):               -- the left point from which the slice should pass [x, y].
        point2 (list or ndarray):               -- the right point from which the slice should pass [x, y].
        time_srs (list or ndarray):             -- the times at which the analytical solution is to be plotted.
        length_srs (list or ndarray):           -- the length at which the analytical solution is to be plotted. It will
                                                    be the radius of the fracture in the case of a radial fractures,
                                                    length of the fracture in case of height contained fractures and the
                                                    length of the minor axis in case of elliptical fractures.
        h (float):                              -- the height of fracture in case of height contained hydraulic
                                                   fractures
        samp_cell (int):                        -- the cell from where the values of the parameter to be taken. If not
                                                   given, values from the cell containing the injection point is taken
        plot_prop (PlotProperties):             -- the properties to be used for the plot.
        labels (LabelProperties):               -- the labels to be used for the plot.
        gamma (float):                          -- the aspect ratio, used in the case of elliptical fracture.
        plt_top_view (bool):                    -- if True, top view will be plotted also

    Returns:
        (Figure):                               -- A Figure object that can be used superimpose further plots.

    """

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean'):
        raise ValueError("The given variable does not vary spatially.")

    if plot_prop is None:
        plot_prop = PlotProperties()
    plot_prop_cp = copy.copy(plot_prop)

    if labels is None:
        labels = LabelProperties(variable, 'slice', '2D')

    analytical_list, mesh_list = get_HF_analytical_solution(regime,
                                                      variable,
                                                      mat_prop,
                                                      inj_prop,
                                                      mesh=mesh,
                                                      fluid_prop=fluid_prop,
                                                      time_srs=time_srs,
                                                      length_srs=length_srs,
                                                      h=h,
                                                      samp_cell=samp_cell,
                                                      gamma=gamma)
    for i in range(len(analytical_list)):
        analytical_list[i] /= labels.unitConversion
        if variable in ('pn', 'pressure'):
            analytical_list[i][(analytical_list[i] < 0)] = 0.

    # finding maximum and minimum values in complete list
    analytical_value = np.copy(analytical_list)
    vmin, vmax = np.inf, -np.inf
    for i in analytical_value:
        i = np.delete(i, np.where(np.isinf(i))[0])
        i = np.delete(i, np.where(np.isneginf(i))[0])
        i = np.delete(i, np.where(np.isnan(i))[0])
        if variable in ('p', 'pressure'):
            non_zero = np.where(abs(i) > 0)[0]
            i_min, i_max = -0.2 * np.median(i[non_zero]), 1.5 * np.median(i[non_zero])
        else:
            i_min, i_max = np.min(i), np.max(i)
        vmin, vmax = min(vmin, i_min), max(vmax, i_max)

    plot_prop_cp.colorMap = plot_prop.colorMaps[1]
    plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
    plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
    for i in range(len(analytical_list)):
        labels.legend = 'analytical (' + regime + ') t= ' + to_precision(time_srs[i],
                                                             plot_prop.dispPrecision)
        plot_prop_cp.lineColor = plot_prop_cp.colorsList[i % len(plot_prop.colorsList)]

        fig = plot_fracture_slice_interpolated(analytical_list[i],
                                                mesh_list[i],
                                                point1=point1,
                                                point2=point2,
                                                fig=fig,
                                                plot_prop=plot_prop_cp,
                                                vmin=vmin,
                                                vmax=vmax,
                                                plot_colorbar=False,
                                                labels=labels,
                                                plt_2D_image=plt_top_view)
    if plt_top_view:
        ax_tv = fig.get_axes()[0]
        ax_tv.set_xlabel('meter')
        ax_tv.set_ylabel('meter')
        ax_tv.set_title('Top View')

        # making colorbar
        im = ax_tv.images
        divider = make_axes_locatable(ax_tv)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
        cb.set_label(labels.colorbarLabel)

        ax_slice = fig.get_axes()[1]
    else:
        ax_slice = fig.get_axes()[0]
    ax_slice.set_ylabel(labels.colorbarLabel)
    ax_slice.set_xlabel('(x,y) ' + labels.xLabel)

    # if plot_prop.plotLegend:
    #     ax_slice.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution_at_point(regime, variable, mat_prop, inj_prop, fluid_prop=None, fig=None, point=None,
                                      time_srs=None, length_srs=None, h=None, samp_cell=None, plot_prop=None,
                                      labels=None, gamma=None):
    """
    This function plots the given analytical solution at a given point. It can be used to compare simulation results by
    superimposing on the figure obtained from the plot at point function.

    Args:
        regime (string):                        -- the string specifying the limiting case solution to be plotted. The
                                                   available options are.

            ========    ============================
            option      limiting solution
            ========    ============================
            'M'         viscosity storage
            'Mt'        viscosity leak-off
            'K'         toughness storage
            'Kt'        toughness leak-off
            'PKN'       PKN
            'KGD_K'     KGD toughness
            'MDR'       MDR turbulent viscosity
            'E_K'       anisotropic toughness
            'E_E'       anisotropic elasticity
            ========    ============================

        variable (string):                      -- the variable to be plotted. Possible options are 'w', 'width' or 'p',
                                                   'pressure'.
        mat_prop (MaterialProperties):          -- the MaterialProperties object giving the material properties.
        inj_prop (InjectionProperties):         -- the InjectionProperties object giving the injection properties.
        fluid_prop( FluidProperties):           -- the FluidProperties object giving the fluid properties.
        fig (figure):                           -- figure object to superimpose the image.
        point (list or ndarray):                -- the point at which the solution to be plotted [x, y].
        time_srs (list or ndarray):             -- the times at which the analytical solution is to be plotted.
        length_srs (list or ndarray):           -- the length at which the analytical solution is to be plotted. It will
                                                    be the radius of the fracture in the case of a radial fractures,
                                                    length of the fracture in case of height contained fractures and the
                                                    length of the minor axis in case of elliptical fractures.
        h (float):                              -- the height of fracture in case of height contained hydraulic
                                                   fractures
        samp_cell (int):                        -- the cell from where the values of the parameter to be taken. If not
                                                   given, values from the cell containing the injection point is taken
        plot_prop (PlotProperties):             -- the properties to be used for the plot.
        labels (LabelProperties):               -- the labels to be used for the plot.
        gamma (float):                          -- the aspect ratio, used in the case of elliptical fracture.

    Returns:
        (Figure):                               -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_analytical_solution_at_point')
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if time_srs is None and length_srs is None:
        raise ValueError("Either time series or length series is to be provided!")

    if plot_prop is None:
        plot_prop = PlotProperties()
    plot_prop_cp = copy.copy(plot_prop)

    if labels is None:
        labels_given = False
        labels = LabelProperties(variable, 'point', '2D')
    else:
        labels_given = True

    if point is None:
        point = [0., 0.]
    analytical_list = get_HF_analytical_solution_at_point(regime,
                                                        variable,
                                                        point,
                                                        mat_prop,
                                                        inj_prop,
                                                        fluid_prop=fluid_prop,
                                                        length_srs=length_srs,
                                                        time_srs=time_srs,
                                                        h=h,
                                                        samp_cell=samp_cell,
                                                        gamma=gamma)
    if time_srs is None:
        time_srs = get_HF_analytical_solution_at_point(regime,
                                                        't',
                                                        point,
                                                        mat_prop,
                                                        inj_prop,
                                                        fluid_prop=fluid_prop,
                                                        length_srs=length_srs,
                                                        time_srs=time_srs,
                                                        h=h,
                                                        samp_cell=samp_cell,
                                                        gamma=gamma)

    for i in range(len(analytical_list)):
        analytical_list[i] /= labels.unitConversion

    if variable in ['time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean']:
        log.warning("The given variable does not vary spatially.")

    plot_prop_cp.lineColor = plot_prop.lineColorAnal
    plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
    plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
    if not labels_given:
        labels.legend = labels.legend + ' analytical'
    labels.xLabel = 'time ($s$)'

    fig = plot_variable_vs_time(time_srs,
                                analytical_list,
                                fig=fig,
                                plot_prop=plot_prop_cp,
                                label=labels.legend)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.colorbarLabel)
    ax.set_title(labels.figLabel)
    if plot_prop.plotLegend:
        ax.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_scale_3D(fracture, fig=None, plot_prop=None):
    """ This function plots lines with dimensions on the 3D fracture plot."""
    log = logging.getLogger('PyFrac.plot_scale_3D')
    log.info('Plotting scale...')
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    I = fracture.Ffront[:, 0:2]

    max_x = max(I[:, 0])
    max_y = max(I[:, 1])
    min_x = min(I[:, 0])
    min_y = min(I[:, 1])
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, [min_x, min_y - 2 * fracture.mesh.hy]),
        (Path.LINETO, [max_x, min_y - 2 * fracture.mesh.hy]),
        (Path.MOVETO, [min_x, min_y - 2.5 * fracture.mesh.hy]),
        (Path.LINETO, [min_x, min_y - 1.5 * fracture.mesh.hy]),
        (Path.MOVETO, [max_x, min_y - 2.5 * fracture.mesh.hy]),
        (Path.LINETO, [max_x, min_y - 1.5 * fracture.mesh.hy]),
        (Path.MOVETO, [min_x - 2.5 * fracture.mesh.hx, min_y - fracture.mesh.hy]),
        (Path.LINETO, [min_x - 2.5 * fracture.mesh.hx, max_y]),
        (Path.MOVETO, [min_x - 3. * fracture.mesh.hx, min_y - fracture.mesh.hy]),
        (Path.LINETO, [min_x - 2. * fracture.mesh.hx, min_y - fracture.mesh.hy]),
        (Path.MOVETO, [min_x - 3. * fracture.mesh.hx, max_y]),
        (Path.LINETO, [min_x - 2. * fracture.mesh.hx, max_y]),
    ]

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, lw=1, facecolor='none')
    ax.add_patch(patch)
    art3d.pathpatch_2d_to_3d(patch)

    if plot_prop.textSize is None:
        plot_prop.textSize = max(fracture.mesh.hx, fracture.mesh.hx)

    y_len = to_precision(max_y - min_y + fracture.mesh.hy, plot_prop.dispPrecision)
    text3d(ax,
           (min_x - 2.5 * fracture.mesh.hx - 5 * plot_prop.textSize, (max_y + min_y) / 2, 0),
           y_len + "$m$",
           zdir="z",
           size=plot_prop.textSize,
           usetex=True,
           ec="none",
           fc="k")
    x_len = to_precision(max_x - min_x + fracture.mesh.hy, plot_prop.dispPrecision)
    text3d(ax,
           ((max_x + min_x) / 2, min_y - 2 * fracture.mesh.hy - 2 * plot_prop.textSize, 0),
           x_len + "$m$",
           zdir="z",
           size=plot_prop.textSize,
           usetex=True,
           ec="none",
           fc="k")

    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_axis_off()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_slice_3D(var_value, mesh, point1=None, point2=None, fig=None, plot_prop=None, vmin=None, vmax=None,
                  label=None):
    """
    This function plots the fracture on a given slice of the domain in 3D. Two points are to be given that will be
    joined to form the slice. The values on the slice are interpolated from the values available on the cell
    centers.

    Args:
        var_value (ndarray):                -- a ndarray with the length of the number of cells in the mesh.
        mesh (CartesianMesh):               -- a CartesianMesh object giving the descritization of the domain.
        point1 (list or ndarray):           -- the left point from which the slice should pass [x, y].
        point2 (list or ndarray):           -- the right point from which the slice should pass [x, y].
        fig (Figure):                       -- the figure to superimpose on. New figure will be made if not provided.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        vmin (float):                       -- the minimum value to be used to colormap and make the colorbar.
        vmax (float):                       -- the maximum value to be used to colormap and make the colorbar.
        label (LabelProperties):            -- the label of plotted line to be used for legend.

    Returns:
        (Figure):                           -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_slice_3D')
    log.info('Plotting slice in 3D...')

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()
        plot_prop.lineStyle = 'k--'

    if point1 is None:
        point1 = np.array([-mesh.Lx, 0.])
    if point2 is None:
        point2 = np.array([mesh.Lx, 0.])
    sampling_points = np.hstack((np.linspace(point1[0], point2[0], 100).reshape((100, 1)),
                                 np.linspace(point1[1], point2[1], 100).reshape((100, 1))))

    value_samp_points = griddata(mesh.CenterCoor,
                                 var_value,
                                 sampling_points,
                                 method='linear',
                                 fill_value=np.nan)

    ax.plot(sampling_points[:,0],
            sampling_points[:,1],
            value_samp_points,
            plot_prop.lineStyle,
            color=plot_prop.lineColor,
            label=label)
    if vmin is None and vmax is None:
        vmin, vmax = np.min(var_value), np.max(var_value)
    ax.set_zlim(vmin, vmax)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_footprint_analytical(regime, mat_prop, inj_prop, fluid_prop=None, time_srs=None, h=None, samp_cell=None,
                              fig=None, plot_prop=None, gamma=None, inj_point=None):
    """
    This function plots footprint of the analytical solution fracture. It can be used to compare simulation results by
    superimposing on the figure obtained from the footprint plot.

    Args:
        regime (string):                        -- the string specifying the limiting case solution to be plotted. The
                                                   available options are.

            ========    ============================
            option      limiting solution
            ========    ============================
            'M'         viscosity storage
            'Mt'        viscosity leak-off
            'K'         toughness storage
            'Kt'        toughness leak-off
            'PKN'       PKN
            'KGD_K'     KGD toughness
            'MDR'       MDR turbulent viscosity
            'E_K'       anisotropic toughness
            'E_E'       anisotropic elasticity
            ========    ============================

        mat_prop (MaterialProperties):          -- the MaterialProperties object giving the material properties.
        inj_prop (InjectionProperties):         -- the InjectionProperties object giving the injection properties.
        fluid_prop( FluidProperties):           -- the FluidProperties object giving the fluid properties.
        time_srs (list or ndarray):             -- the times at which the analytical solution is to be plotted.
        h (float):                              -- the height of fracture in case of height contained hydraulic
                                                   fractures
        samp_cell (int):                        -- the cell from where the values of the parameter to be taken. If not
                                                   given, values from the cell containing the injection point is taken.
        fig (figure):                           -- figure object to superimpose the image.
        plot_prop (PlotProperties):             -- the properties to be used for the plot.
        gamma (float):                          -- the aspect ratio, used in the case of elliptical fracture.
        inj_point (list):                       -- a list of size 2, giving the x and y coordinate of the injection
                                                   point.

    Returns:
        (Figure):                               -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_footprint_analytical')
    log.info("Plotting analytical footprint...")

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    footprint_patches = get_HF_analytical_solution_footprint(regime,
                                         mat_prop,
                                         inj_prop,
                                         plot_prop,
                                         fluid_prop=fluid_prop,
                                         time_srs=time_srs,
                                         h=h,
                                         samp_cell=samp_cell,
                                         gamma=gamma,
                                         inj_point=inj_point)

    for i in footprint_patches:
        ax.add_patch(i)
        if hasattr(ax, 'get_zlim'):
            art3d.pathpatch_2d_to_3d(i)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=None, fluid_prop=None, fig=None,
                             projection='2D', time_srs=None, length_srs=None, h=None, samp_cell=None, plot_prop=None,
                             labels=None, contours_at=None, gamma=None):
    """
    This function plots the analytical solution according to the given regime. It can be used to compare simulation
    results by superimposing on the figure obtained from the plot function.

    Args:
        regime (string):                        -- the string specifying the limiting case solution to be plotted. The
                                                   available options are.

            ========    ============================
            option      limiting solution
            ========    ============================
            'M'         viscosity storage
            'Mt'        viscosity leak-off
            'K'         toughness storage
            'Kt'        toughness leak-off
            'PKN'       PKN
            'KGD_K'     KGD toughness
            'MDR'       MDR turbulent viscosity
            'E_K'       anisotropic toughness
            'E_E'       anisotropic elasticity
            ========    ============================

        variable (string):                      -- the variable to be plotted. Possible options are 'w', 'width' or 'p',
                                                   'pressure'.
        mat_prop (MaterialProperties):          -- the MaterialProperties object giving the material properties.
        inj_prop (InjectionProperties):         -- the InjectionProperties object giving the injection properties.
        mesh (CartesianMesh):                   -- a CartesianMesh class object describing the grid.
        fluid_prop( FluidProperties):           -- the FluidProperties object giving the fluid properties.
        fig (figure):                           -- figure object to superimpose the image.
        projection (string):                    -- a string specifying the projection.
        time_srs (list or ndarray):             -- the times at which the analytical solution is to be plotted.
        length_srs (list or ndarray):           -- the length at which the analytical solution is to be plotted. It will
                                                    be the radius of the fracture in the case of a radial fractures,
                                                    length of the fracture in case of height contained fractures and the
                                                    length of the minor axis in case of elliptical fractures.
        h (float):                              -- the height of fracture in case of height contained hydraulic
                                                   fractures
        samp_cell (int):                        -- the cell from where the values of the parameter to be taken. If not
                                                   given, values from the cell containing the injection point is taken
        plot_prop (PlotProperties):             -- the properties to be used for the plot.
        labels (LabelProperties):               -- the labels to be used for the plot.
        contours_at (list):                     -- the values at which the contours are to be plotted.
        gamma (float):                          -- the aspect ratio, used in the case of elliptical fracture.


    Returns:
        (Figure):                               -- A Figure object that can be used superimpose further plots.

    """
    log = logging.getLogger('PyFrac.plot_analytical_solution')
    log.info("Plotting analytical " + variable + " " + regime + " solution...")
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if labels is None:
        labels_given = False
        labels = LabelProperties(variable, 'whole mesh', projection)
    else:
        labels_given = True

    if variable == 'footprint':
        fig = plot_footprint_analytical(regime,
                                        mat_prop,
                                        inj_prop,
                                        fluid_prop=fluid_prop,
                                        time_srs=time_srs,
                                        h=h,
                                        samp_cell=samp_cell,
                                        fig=fig,
                                        plot_prop=plot_prop,
                                        gamma=gamma,
                                        inj_point=inj_prop.sourceCoordinates)
    else:

        if plot_prop is None:
            plot_prop = PlotProperties()
        plot_prop_cp = copy.copy(plot_prop)

        analytical_list, mesh_list = get_HF_analytical_solution(regime,
                                          variable,
                                          mat_prop,
                                          inj_prop,
                                          mesh=mesh,
                                          fluid_prop=fluid_prop,
                                          time_srs=time_srs,
                                          length_srs=length_srs,
                                          h=h,
                                          samp_cell=samp_cell,
                                          gamma=gamma)

        for i in range(len(analytical_list)):
            analytical_list[i] /= labels.unitConversion

        analytical_value = np.copy(analytical_list)
        vmin, vmax = np.inf, -np.inf
        for i in analytical_value:
            i = np.delete(i, np.where(np.isinf(i))[0])
            i = np.delete(i, np.where(np.isnan(i))[0])
            i_min, i_max = np.min(i), np.max(i)
            vmin, vmax = min(vmin, i_min), max(vmax, i_max)

        if variable in unidimensional_variables:
            plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
            plot_prop_cp.lineColor = plot_prop.lineColorAnal
            plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
            if not labels_given:
                labels.legend = labels.legend + ' analytical'
            labels.xLabel = 'time ($s$)'
            fig = plot_variable_vs_time(time_srs,
                                        analytical_list,
                                        fig=fig,
                                        plot_prop=plot_prop_cp,
                                        label=labels.legend)
            projection = '2D'
        else:
            if projection == '2D_clrmap':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_image(analytical_list[i],
                                                          mesh_list[i],
                                                          fig=fig,
                                                          plot_prop=plot_prop_cp,
                                                          vmin=vmin,
                                                          vmax=vmax)
            elif projection == '2D_contours':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_contours(analytical_list[i],
                                                             mesh_list[i],
                                                             fig=fig,
                                                             plot_prop=plot_prop_cp,
                                                             contours_at=contours_at,
                                                             vmin=vmin,
                                                             vmax=vmax)
            elif projection == '3D':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_surface(analytical_list[i],
                                                            mesh_list[i],
                                                            fig=fig,
                                                            plot_prop=plot_prop_cp,
                                                            plot_colorbar=False,
                                                            vmin=vmin,
                                                            vmax=vmax)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)
    ax.set_title(labels.figLabel)
    if variable not in ['footprint']:
        if projection == '3D':
            ax.set_zlabel(labels.zLabel)
            sm = plt.cm.ScalarMappable(cmap=plot_prop_cp.colorMap,
                                       norm=plt.Normalize(vmin=vmin,
                                                          vmax=vmax))
            sm._A = []
            cb = plt.colorbar(sm, alpha=plot_prop_cp.alpha)
            cb.set_label(labels.colorbarLabel + ' analytical')
        elif projection in ('2D_clrmap', '2D_contours'):
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label(labels.colorbarLabel + ' analytical')
        elif projection == '2D':
            ax.set_title(labels.figLabel)
            if plot_prop_cp.plotLegend:
                ax.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def get_HF_analytical_solution_footprint(regime, mat_prop, inj_prop, plot_prop, fluid_prop=None, time_srs=None,
                                         h=None, samp_cell=None, gamma=None, inj_point=None):
    """ This function returns footprint of the analytical solution in the form of patches"""

    if time_srs is None:
        raise ValueError("Time series is to be provided.")

    if regime == 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime == "MDR":
        density = fluid_prop.density
    else:
        density = None

    if samp_cell is None:
        samp_cell = int(len(mat_prop.Kprime) / 2)

    if regime in ['K', 'M']:
        Cprime = None
    else:
        Cprime = mat_prop.Cprime[samp_cell]

    if regime == 'K':
        muPrime = None
    else:
        muPrime = fluid_prop.muPrime

    if regime == 'M':
        Kprime = None
    else:
        Kprime = mat_prop.Kprime[samp_cell]

    if regime == 'PKN' and h is None:
        raise ValueError("Fracture height is required to plot PKN fracture!")

    if len(inj_prop.injectionRate[0]) > 1:
        V0 = inj_prop.injectionRate[0, 1] * inj_prop.injectionRate[1, 0]
    else:
        V0=None

    return_patches = []
    for i in time_srs:
        if len(inj_prop.injectionRate[0]) > 1:
            if i > inj_prop.injectionRate[0, 1]:
                Q0 = 0.0
            else:
                Q0 = inj_prop.injectionRate[1, 0]
        else:
            Q0 = inj_prop.injectionRate[1,0]

        x_len, y_len = get_fracture_dimensions_analytical(regime,
                                                          i,
                                                          mat_prop.Eprime,
                                                          Q0,
                                                          muPrime=muPrime,
                                                          Kprime=Kprime,
                                                          Cprime=Cprime,
                                                          Kc_1=Kc_1,
                                                          h=h,
                                                          density=density,
                                                          gamma=gamma,
                                                          Vinj=V0)

        if inj_point is None:
            inj_point = [0., 0.]

        if regime in ('M', 'Mt', 'K', 'Kt', 'E', 'MDR'):
            return_patches.append(mpatches.Circle((inj_point[0], inj_point[1]),
                                   x_len,
                                   edgecolor=plot_prop.lineColorAnal,
                                   facecolor='none'))
        elif regime in ('PKN', 'KGD_K'):
            return_patches.append(mpatches.Rectangle(xy=(-x_len + inj_point[0], -y_len + inj_point[1]),
                                      width=2 * x_len,
                                      height=2 * y_len,
                                      edgecolor=plot_prop.lineColorAnal,
                                      facecolor='none'))
        elif regime in ('E_K', 'E_E'):
            return_patches.append(mpatches.Ellipse(xy=(inj_point[0], inj_point[1]),
                                   width=2 * x_len,
                                   height=2 * y_len,
                                   edgecolor=plot_prop.lineColorAnal,
                                   facecolor='none'))
        else:
            raise ValueError("Regime not supported.")

    return return_patches

#-----------------------------------------------------------------------------------------------------------------------

def plot_injection_source(frac, fig=None, plot_prop=None):
    """
    This function plots the location of the source.
    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()

    ax.plot(frac.mesh.CenterCoor[frac.source, 0],
            frac.mesh.CenterCoor[frac.source, 1],
            '.',
            color=plot_prop.lineColor)

    ax.plot(frac.mesh.CenterCoor[frac.sink, 0],
            frac.mesh.CenterCoor[frac.sink, 1],
            '.',
            color='w')

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def animate_simulation_results(fracture_list, variable='footprint', projection=None, elements=None,
                                 plot_prop=None, edge=4, contours_at=None, labels=None, mat_properties=None,
                                 backGround_param=None, block_figure=False, plot_non_zero=True, pause_time=0.2,
                                 save_images=False, images_address='.'):
    """
    This function plots the fracture evolution with time. The state of the fracture at different times is provided in
    the form of a list of Fracture objects.

    Args:
        fracture_list (list):               -- the list of Fracture objects giving the evolution of fracture with
                                                    time.
        variable (string):                  -- the variable to be plotted. See :py:data:`supported_variables` of the
                                               :py:mod:`labels` module for a list of supported variables. It can also
                                               be a list of variables which will be plotted in separate widows.
        projection (string):                -- a string specifying the projection. See :py:data:`supported_projections`
                                                for the supported projections for each of the supprted variable. If not
                                                provided, the default will be used.
        elements (ndarray):                 -- the elements to be plotted.
        plot_prop (PlotProperties):         -- the properties to be used for the plot.
        edge (int):                         -- the edge of the cell that will be plotted. This is for variables that
                                                are evaluated on the cell edges instead of cell center. It can have a
                                                value from 0 to 4 (0->left, 1->right, 2->bottome, 3->top, 4->average).
        labels (LabelProperties):           -- the labels to be used for the plot.
        mat_properties (MaterialProperties):-- the material properties. It is mainly used to colormap the mesh.
        backGround_param (string):          -- the parameter according to which the the mesh will be colormapped.
        block_figure (bool):                -- if True, a key would be needed to be pressed to proceed to the next
                                               frame.
        contours_at (list):                 -- the values at which the contours are to be plotted.
        plot_non_zero (bool):               -- if true, only non-zero values will be plotted.
        pause_time (float):                 -- time (in seconds) between two successive updates of frames.

    """
    log = logging.getLogger('PyFrac.animate_simulation_results')
    if not isinstance(variable, list):
        variable = [variable]
    figures = [None for i in range(len(variable))]

    setFigPos = True
    for Fr_i, fracture in enumerate(fracture_list):
        for indx, plt_var in enumerate(variable):
            log.info("Plotting solution at " + repr(fracture.time) + "...")
            if plot_prop is None:
                plot_prop = PlotProperties()


            if figures[indx]:
                ax = figures[indx].get_axes()[0]  # save axes from last figure
                plt.figure(figures[indx].number)
                plt.clf()  # clear figure
                figures[indx].add_axes(ax)  # add axis to the figure

            if plt_var == 'footprint':
                figures[indx] = fracture.plot_fracture(variable='mesh',
                                                       mat_properties=mat_properties,
                                                       projection=projection,
                                                       backGround_param=backGround_param,
                                                       fig=figures[indx],
                                                       plot_prop=plot_prop)

                plot_prop.lineColor = 'k'
                figures[indx] = fracture.plot_fracture(variable='footprint',
                                                       projection=projection,
                                                       fig=figures[indx],
                                                       plot_prop=plot_prop,
                                                       labels=labels)

            else:
                fp_projection = '2D'
                if projection is not None:
                    if '2D' in projection:
                        fp_projection = '2D'
                    else:
                        fp_projection = '3D'
                fig_labels = LabelProperties(plt_var, 'whole mesh', fp_projection)
                fig_labels.figLabel = ''

                figures[indx] = fracture.plot_fracture(variable='footprint',
                                                       projection=fp_projection,
                                                       fig=figures[indx],
                                                       labels=fig_labels)
                
                if elements is None:
                    elems = get_elements(suitable_elements[plt_var], fracture)
                else:
                    elems = elements
                figures[indx] = fracture.plot_fracture(variable=plt_var,
                                                       projection=projection,
                                                       elements=elems,
                                                       mat_properties=mat_properties,
                                                       fig=figures[indx],
                                                       plot_prop=plot_prop,
                                                       edge=edge,
                                                       contours_at=contours_at,
                                                       labels=labels,
                                                       plot_non_zero=plot_non_zero)

            # plotting source elements
            plot_injection_source(fracture, fig=figures[indx])

            # plotting closed cells
            if len(fracture.closed) > 0:
                plot_prop.lineColor = 'orangered'
                figures[indx] = fracture.mesh.identify_elements(fracture.closed,
                                                                        fig=figures[indx],
                                                                        plot_prop=plot_prop,
                                                                        plot_mesh=False,
                                                                        print_number=False)
            # plot the figure
            plt.ion()
            plt.pause(pause_time)

            if save_images:
                image_name = plt_var + repr(Fr_i)
                plt.savefig(images_address + image_name + '.png')
                
        # set figure position
        if setFigPos:
            for i in range(len(variable)):
                plt.figure(i + 1)
                mngr = plt.get_current_fig_manager()
                x_offset = 650 * i
                y_ofset = 50
                if i >= 3:
                    x_offset = (i - 3) * 650
                    y_ofset = 500
                try:
                    mngr.window.setGeometry(x_offset, y_ofset, 640, 545)
                except AttributeError:
                    pass
            setFigPos = False

        
        if block_figure:
            plt.pause(0.5)
            plt.ion()
            plt.show()
            plt.waitforbuttonpress()
            # input("Press any key to continue.")
    plt.show(block=True)

#-----------------------------------------------------------------------------------------------------------------------

def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    """
    Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
    and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
    as the third dimension.  usetex is a boolean indicating whether the string
    should be interpreted as latex or not.  Any additional keyword arguments
    are passed on to transform_path.

    Note: zdir affects the interpretation of xyz.
    """

    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "y":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = mpatches.PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            # printevent.button
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

#-----------------------------------------------------------------------------------------------------------------------


def to_precision(x, p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

#-----------------------------------------------------------------------------------------------------------------------


def save_images_to_video(image_folder, video_name='movie'):
    """ This function makes a video from the images in the given folder."""
    import cv2
    import os
    log = logging.getLogger('PyFrac.save_images_to_video')
    if ".avi" not in video_name:
        video_name = video_name + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    img_no = 0
    for image in images:
        log.info("adding image no " + repr(img_no))
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.waitKey(1)
        img_no += 1

    cv2.destroyAllWindows()
    video.release()

#-----------------------------------------------------------------------------------------------------------------------

def remove_zeros(var_value, mesh, plot_boundary=False):

    if plot_boundary:
        zero = np.full(mesh.NumberOfElts, False, dtype=bool)
        zero[abs(var_value) < 3 * np.finfo(float).eps] = True
        for i in range(mesh.NumberOfElts-1):
            not_left = zero[i] and zero[i + 1]
            not_right = zero[i] and zero[i - 1]
            not_bottom = zero[i] and zero[i - mesh.nx]
            not_top = zero[i] and zero[(i + mesh.nx) % mesh.NumberOfElts]
            if not_left and not_right and not_bottom and not_top:
                var_value[i] = np.nan
        var_value[mesh.NumberOfElts - 1] = np.nan
    else:
        var_value[abs(var_value) < 3 * np.finfo(float).eps] = np.nan


#-----------------------------------------------------------------------------------------------------------------------

def get_elements(specifier, fr):
    if specifier == 'crack':
        return fr.EltCrack
    elif specifier == 'channel':
        return fr.EltChannel
    elif specifier == 'tip':
        return fr.EltTip

#-----------------------------------------------------------------------------------------------------------------------


def plot_regime(var_value, mesh, fig=None, elements=None):
    """
       This function plots the fracture regime with the color code defined by Dontsov. Plotting is done at the ribbon
       cells. The colorbar is replaced by the colorcoded triangle.

       Args:
           var_value (list):                   -- List containing the color code at the tip.
           mesh (object):                      -- mesh of the current timestep
           fig (figure):                       -- Figure of the current footprint
           elements (ndarray):                 -- the elements to be plotted.

        Return:
           fig (figure):                       -- Adapted figure

       """

    # getting the extent of the figure
    x = mesh.CenterCoor[:, 0].reshape((mesh.ny, mesh.nx))
    y = mesh.CenterCoor[:, 1].reshape((mesh.ny, mesh.nx))

    dx = (x[0, 1] - x[0, 0]) / 2.
    dy = (y[1, 0] - y[0, 0]) / 2.

    extent = [x[0, 0] - dx, x[-1, -1] + dx, y[0, 0] - dy, y[-1, -1] + dy]

    # selecting only the relevant elements
    if elements is not None:
        var_value_fullMesh = np.full((mesh.NumberOfElts, 3), 1.)
        var_value_fullMesh[elements, ::] = var_value[elements, ::]
        var_value = var_value_fullMesh

    # re-arrange the solution for plotting
    var_value_2D = var_value.reshape((mesh.ny, mesh.nx, 3))

    # decide where we are not stagnant
    non_stagnant = np.where(np.prod(var_value[elements, ::] == [1., 1., 1.], axis=1) != 1.)[0]

    # use footprint if provided
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]
        l = list(ax.get_lines())

        fig.clf()
        fig.add_subplot(121)
        for line in l:
            plt.plot(line.get_data()[0],line.get_data()[1],'k')

    # plotting the colored cells
    ax = fig.get_axes()[0]
    ax.imshow(var_value_2D,
              extent=extent,
              origin='lower')

    # plotting the triangle with the location of the tip cells
    leg = fig.add_subplot(122)
    leg = mkmtTriangle(leg)
    leg = fill_mkmtTriangle(leg)
    plot_points_to_mkmtTriangle(leg, var_value[elements[non_stagnant], ::])

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def mkmtTriangle(fig):
    """
       This function draws the Maxwell triangle used to higlight the regime dominant in the ribbon cell.

       Args:
           fig (figure):               -- The figure to place the Maxwell triangle in.

       """

    # Plot the triangle
    a = 1.0 / math.sqrt(3)
    fig.plot([0., 1., 0.5, 0.], [0., 0., 0.5/a, 0], 'k-')
    fig.axis([-0.25, 1.2, -0.2, 1.05])
    # Remove axes
    fig.axis('off')
    #Label the corners of the triangle
    fig.text(1.0, 0, r"$k$", fontsize=18, verticalalignment='top')
    fig.text(-0.1, 0, r"$m$", fontsize=18, verticalalignment='top')
    fig.text(0.45, 0.575/a, r"$\tilde{m}$", fontsize=18, verticalalignment='top')

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def fill_mkmtTriangle(fig):
    """
       This function colors the Maxwell triangle used to highlight the regime dominant in the ribbon cell.

       Args:
           fig (figure):               -- The figure with the Maxwell triangle to color.

       """

    # Generate an image with 300x300 pixels
    Nlignes = 300
    Ncol = 300
    img = np.zeros((Nlignes, Ncol, 4))
    dx = 2.0 / (Ncol - 1)
    dy = 1.0 / (Nlignes - 1)

    # choose color of pixels.
    for i in range(Ncol - 1):
        for j in range(Nlignes - 1):
            x = -1.0 + i * dx
            y = j * dy
            v = y
            r = (x + 1 - v) / 2.0
            b = 1.0 - v - r
            if (r >= 0) and (r <= 1.0) and (v >= 0) and (v <= 1.0) and (b >= 0) and (b <= 1.0):
                img[j][i] = np.array([r, v, b, 1.0])
            else:
                img[j][i] = np.array([1.0, 1.0, 1.0, 0.0])
    a = 1.0 / math.sqrt(3)
    fig.imshow(img, origin='lower', extent=[0.0, 1, 0.0, 0.5 / a])

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_points_to_mkmtTriangle(fig, rgbpoints):
    """
       This function plots a set of points in the m-k-mtilde triangle

       Args:
           fig (figure):               -- The figure with the Maxwell triangle to place the points.
           rgbpoints (ndarraz):        -- Color code in RGB of the points to plot.

       """

    
    nOFpoits = rgbpoints.shape[0]
    x = np.zeros(nOFpoits)
    y = np.zeros(nOFpoits)
    
    # Transform color into coordinates
    a = 1.0 / math.sqrt(3)
    for k in range(nOFpoits):
        rgb = rgbpoints[k,:]
        somme = rgb[0] + rgb[1] + rgb[2]
        x[k] = ( (rgb[0] - rgb[2]) / math.sqrt(3) / somme) /(2*a) + 0.5
        y[k] = 0.5/a * rgb[1] / somme
    
    # Plot the points
    fig.plot(x, y, "k.", markersize=9)