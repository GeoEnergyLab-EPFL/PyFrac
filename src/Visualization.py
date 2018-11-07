# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Friday, July 06, 2018.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.
"""


from src.Properties import PlotProperties
from src.HFAnalyticalSolutions import get_fracture_dimensions_analytical
from src.Labels import *
from src.PostProcessFracture import *

import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation
import copy


def plot_fracture_list(fracture_list, variable='footprint', mat_properties=None, projection='2D', elements=None,
                       backGround_param=None, plot_prop=None, fig=None, edge=4, contours_at=None, labels=None,
                       plot_non_zero=True):

    print("Plotting " + variable + '...')

    if len(fracture_list) == 0:
        raise ValueError("Provided fracture list is empty!")
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)
    if projection not in supported_projections:
        raise ValueError(err_msg_projection)

    if plot_prop is None:
        plot_prop = PlotProperties()

    if labels is None:
        labels = get_labels(variable, 'wole_mesh', projection)

    max_Lx = 0.
    for i in fracture_list:
        if i.mesh.Lx > max_Lx:
            largest_mesh = i.mesh
            max_Lx = i.mesh.Lx

    if variable is 'mesh':
        if backGround_param is not None and mat_properties is None:
            raise ValueError("Material properties are required to color code background")
        if projection is '3D':
            fig = largest_mesh.plot_3D(fig=fig,
                                    material_prop=mat_properties,
                                    backGround_param=backGround_param,
                                    plot_prop=plot_prop)

        else:
            fig = largest_mesh.plot(fig=fig,
                                 material_prop=mat_properties,
                                 backGround_param=backGround_param,
                                 plot_prop=plot_prop)

    elif variable is 'footprint':
        if '2D' in projection:
            for i in fracture_list:
                fig = i.plot_front(fig=fig, plot_prop=plot_prop)
        else:
            for i in fracture_list:
                fig = i.plot_front_3D(fig=fig, plot_prop=plot_prop)

    else:
        var_val_list, time_list = get_fracture_variable(fracture_list,
                                                                    variable,
                                                                    edge=edge)

        var_val_copy = np.copy(var_val_list)
        for i in range(len(var_val_copy)):
            var_val_copy[i] /= labels.unitConversion


        var_value_tmp = np.copy(var_val_copy)
        vmin, vmax = np.inf, -np.inf
        for i in var_value_tmp:
            i = np.delete(i, np.where(np.isinf(i))[0])
            i = np.delete(i, np.where(np.isnan(i))[0])
            if variable in ('p', 'pressure'):
                non_zero = np.where(abs(i) > 0)[0]
                i_min, i_max = -0.2 * np.median(i[non_zero]), 1.5 * np.median(i[non_zero])
            else:
                i_min, i_max = np.min(i), np.max(i)
            vmin, vmax = min(vmin, i_min), max(vmax, i_max)


    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max', 'V', 'volume'
                    'front_dist_mean', 'd_mean', 'efficiency', 'ef', 'aspect ratio', 'ar'):
        labels.xLabel = 'time'
        fig = plot_variable_vs_time(time_list, var_val_list, fig=fig, plot_prop=plot_prop, label=labels.legend)
        projection = '2D'
    elif variable not in ('mesh', 'footprint'):

        if plot_non_zero:
            for i in var_val_copy:
                i[np.where(abs(i) < 1e-16)[0]] = np.nan

        if projection is '2D_image':
            for i in range(len(var_val_list)):
                fig = plot_fracture_variable_as_image(var_val_copy[i],
                                                          fracture_list[i].mesh,
                                                          fig=fig,
                                                          plot_prop=plot_prop,
                                                          elements=elements,
                                                          plt_colorbar=False,
                                                          vmin=vmin,
                                                          vmax=vmax)
        elif projection is '2D_contours':
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
        elif projection is '3D':
            for i in range(len(var_val_list)):
                fig = plot_fracture_variable_as_surface(var_val_copy[i],
                                                        fracture_list[i].mesh,
                                                        fig=fig,
                                                        plot_prop=plot_prop,
                                                        plot_colorbar=False,
                                                        elements=elements,
                                                        vmin=vmin,
                                                        vmax=vmax)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)
    plt.title(labels.figLabel)
    if projection is '3D' and variable not in ('mesh', 'footprint'):
        ax.set_zlabel(labels.zLabel)
        sm = plt.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                   norm=plt.Normalize(vmin=vmin,
                                                      vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm, alpha=plot_prop.alpha)
        cb.set_label(labels.colorbarLabel)
    elif projection in ('2D_image', '2D_contours'):
        im = ax.images
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
        cb.set_label(labels.colorbarLabel)
    elif projection is '2D':
        plt.title(labels.figLabel)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_list_slice(fracture_list, variable='width', point1=None, point2=None, projection='2D', plot_prop=None,
                             fig=None, edge=4, labels=None, plt_2D_image=True, plot_cell_center=False,
                             orientation='horizontal'):

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if plot_prop is None:
        plot_prop = PlotProperties()

    if labels is None:
        labels = get_labels(variable, 'wole_mesh', projection)

    mesh_list = get_fracture_variable(fracture_list,
                                                'mesh',
                                                edge=edge,
                                                return_time=False)
    var_val_list, time_list = get_fracture_variable(fracture_list,
                                                                variable,
                                                               edge=edge)

    var_val_copy = np.copy(var_val_list)
    for i in range(len(var_val_copy)):
        var_val_copy[i] /= labels.unitConversion

    var_value_tmp = np.copy(var_val_copy)
    vmin, vmax = np.inf, -np.inf
    for i in var_value_tmp:
        i = np.delete(i, np.where(np.isinf(i))[0])
        i = np.delete(i, np.where(np.isnan(i))[0])
        if variable in ('p', 'pressure'):
            non_zero = np.where(abs(i) > 0)[0]
            i_min, i_max = -0.2 * np.median(i[non_zero]), 1.5 * np.median(i[non_zero])
        else:
            if len(i) >0:
                i_min, i_max = np.min(i), np.max(i)
            else:
                i_min, i_max = np.inf, -np.inf
        vmin, vmax = min(vmin, i_min), max(vmax, i_max)

    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean'):
        raise ValueError("The given variable does not vary spatially.")

    else:
        label = labels.legend
        for i in range(len(var_val_list)):
            labels.legend = label + ' t= ' + to_precision(time_list[i],
                                                          plot_prop.dispPrecision)
            plot_prop.lineColor = plot_prop.colorsList[i % len(plot_prop.colorsList)]
            if '2D' in projection:
                if plot_cell_center:
                    plot_prop.lineStyle = '.'
                    fig, return_pnt1, return_pnt2= plot_fracture_slice_cell_center(var_val_copy[i],
                                                                  mesh_list[i],
                                                                  point=point1,
                                                                  orientation=orientation,
                                                                  fig=fig,
                                                                  plot_prop=plot_prop,
                                                                  vmin=vmin,
                                                                  vmax=vmax,
                                                                  plot_colorbar=False,
                                                                  labels=labels,
                                                                  plt_2D_image=plt_2D_image,
                                                                  return_points=True)
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
                                plt_2D_image=plt_2D_image)
            else:
                fig = plot_slice_3D(var_val_copy[i],
                                    mesh_list[i],
                                    point1=point1,
                                    point2=point2,
                                    fig=fig,
                                    plot_prop=plot_prop,
                                    vmin=vmin,
                                    vmax=vmax,
                                    label=labels.legend)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)
    if '2D' in projection and plt_2D_image:
        plt.subplot(211)
        plt.title('Top View')
        im = ax.images
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
        cb.set_label(labels.colorbarLabel)
    elif projection == '3D':
        ax.set_zlabel(labels.zLabel)
        plt.title(labels.figLabel)

    if plt_2D_image:
        ax = fig.get_axes()[1]
    else:
        ax = fig.get_axes()[0]
    ax.set_ylabel(labels.colorbarLabel)
    ax.set_xlabel('(x,y) ' + labels.xLabel )

    if plot_prop.plotLegend:
        ax.legend()

    if plot_cell_center:
        return fig, return_pnt1, return_pnt2

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_list_at_point(fracture_list, variable='width', point=None, plot_prop=None, fig=None,
                             edge=4, labels=None):

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean'):
        raise ValueError("The given variable does not vary spatially.")

    if plot_prop is None:
        plot_prop = PlotProperties()

    if labels is None:
        labels = get_labels(variable, 'wm', '2D')

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
    plt.title(labels.figLabel)
    if plot_prop.plotLegend:
        ax.legend()

    labels.figLabel = 'Sampling Point'
    fig_image = plot_fracture_list([fracture_list[-1]],
                       variable,
                       projection='2D_image',
                       plot_prop=plot_prop,
                       edge=edge,
                       labels=labels)
    plot_prop.lineColor = to_rgb('black')
    plot_prop.colorsList = ['black']
    plot_prop.lineStyle = '-'
    labels.figLabel=''
    fig_image = plot_fracture_list([fracture_list[-1]],
                                   fig=fig_image,
                                   projection='2D',
                                   variable='footprint',
                                   plot_prop=plot_prop,
                                   labels=labels)

    ax_image = fig_image.get_axes()[0]
    ax_image.plot([point[0]], [point[1]], 'ko')

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_variable_vs_time(time_list, value_list, fig=None, plot_prop=None, label=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop.graphScaling is 'linear':
        ax.plot(time_list,
                value_list,
                plot_prop.lineStyle,
                color=plot_prop.lineColor,
                label=label)

    elif plot_prop.graphScaling is 'loglog':
        ax.loglog(time_list,
                  value_list,
                  plot_prop.lineStyle,
                  color=plot_prop.lineColor,
                  label=label)

    elif plot_prop.graphScaling is 'semilogx':
        ax.semilogx(time_list,
                    value_list,
                    plot_prop.lineStyle,
                    color=plot_prop.lineColor,
                    label=label)

    elif plot_prop.graphScaling is 'semilogy':
        ax.semilogy(time_list,
                    value_list,
                    plot_prop.lineStyle,
                    color=plot_prop.lineColor,
                    label=label)
    else:
        raise ValueError("Graph scaling type not supported")

    return fig

def plot_fracture_variable_as_image(var_value, mesh, fig=None, plot_prop=None, elements=None, vmin=None,
                                    vmax=None, plt_colorbar=True):
    """
    This function plots the contours of the fracture width in millimeters.

    Arguments:
        fracture (Fracture):        -- the fracture object for which the fracture width is to be plotted.
        fig (figure)                -- figure to superimpose. A new figure will be created if not provided.
        bck_colMap (Colormaps)      -- colormap for the fracture width shown in the background.
        line_color (color)          -- the color of the contour line (e.g. 'r' will plot in red).
        contours_at (ndarray)       -- a list of fracture widths to plot contours at.

    Returns:
         Fig
    """

    if elements is not None:
        if len(var_value) == len(elements):
            var_value_fullMesh = np.full((mesh.NumberOfElts, ), np.nan)
            var_value_fullMesh[elements] = var_value
            var_value = var_value_fullMesh
        else:
            raise ValueError("The var_value and elements arguments should have same lengths.")

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
              vmax=vmax)

    if plt_colorbar:
        fig.colorbar(cax)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_fracture_variable_as_surface(var_value, mesh, fig=None, plot_prop=None, plot_colorbar=True, elements=None,
                                      vmin=None, vmax=None):
    """
    This function plots the contours of the fracture width in millimeters.

    Arguments:
        fracture (Fracture):        -- the fracture object for which the fracture width is to be plotted.
        fig (figure)                -- figure to superimpose. A new figure will be created if not provided.
        bck_colMap (Colormaps)      -- colormap for the fracture width shown in the background.
        line_color (color)          -- the color of the contour line (e.g. 'r' will plot in red).
        contours_at (ndarray)       -- a list of fracture widths to plot contours at.

    Returns:
         Fig
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

    if vmin is None and vmax is None:
        var_value = np.delete(var_value, np.where(np.isinf(var_value))[0])
        var_value = np.delete(var_value, np.where(np.isnan(var_value))[0])
        vmin, vmax = np.min(var_value), np.max(var_value)

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
    This function plots the contours of the fracture width in millimeters.

    Arguments:
        fracture (Fracture):        -- the fracture object for which the fracture width is to be plotted.
        fig (figure)                -- figure to superimpose. A new figure will be created if not provided.
        bck_colMap (Colormaps)      -- colormap for the fracture width shown in the background.
        line_color (color)          -- the color of the contour line (e.g. 'r' will plot in red).
        contours_at (ndarray)       -- a list of fracture widths to plot contours at.

    Returns:
         Fig
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
                  vmax=vmax)
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
                                     vmax=None, plot_colorbar=True, labels=None, plt_2D_image=True):

    print("Plotting slice...")
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

    if labels is None:
        labels = LabelProperties()

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
                            vmax=vmax)

        if plt_2D_image and plot_colorbar:
            divider = make_axes_locatable(ax_2D)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_2D, cax=cax, orientation='vertical')


    if point1 is None:
        point1 = np.array([-mesh.Lx, 0.])
    if point2 is None:
        point2 = np.array([mesh.Lx, 0.])

    # the code below find the extreme points of the line joining the two given points with the current mesh
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    if slope == 0.:
        point1[0] = -mesh.Lx
        point2[0] = mesh.Lx
    else:
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
    ax_slice.plot(sampling_line,
                  value_samp_points,
                  plot_prop.lineStyle,
                  color=plot_prop.lineColor,
                  label=labels.legend)

    ax_slice.set_xticks(np.hstack((sampling_line[[0, 20, 41, 62, 83, 104]], sampling_line[104])))

    xtick_labels = []
    for i in [0, 20, 41, 62, 83, 104]:
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
                                return_points=False):

    print("Plotting slice...")
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

    if labels is None:
        labels = LabelProperties()

    if plt_2D_image:
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
                            vmax=vmax)

        if plt_2D_image and plot_colorbar:
            divider = make_axes_locatable(ax_2D)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_2D, cax=cax, orientation='vertical')


    if point is None:
        point = np.array([0., 0.])
    if orientation not in ('horizontal', 'vertical', 'increasing', 'decreasing'):
        raise ValueError("Given orientation is not supported. Possible options:\n 'horizontal', 'vertical',"
                         " 'increasing', 'decreasing'")

    zero_cell = mesh.locate_element(point[0], point[1])
    if len(zero_cell) < 1:
        raise ValueError("The given point does not lie in the grid!")

    if orientation is 'vertical':
        sampling_cells = np.hstack((np.arange(zero_cell, 0, -mesh.nx)[::-1],
                                    np.arange(zero_cell, mesh.NumberOfElts, mesh.nx)))
    elif orientation is 'horizontal':
        sampling_cells = np.arange(zero_cell // mesh.nx * mesh.nx, (zero_cell // mesh.nx + 1) * mesh.nx)

    elif orientation is 'increasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx - 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx + 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] <
                                                mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))

    elif orientation is 'decreasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx + 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] <
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx - 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))


    if plt_2D_image:
        ax_2D.plot(mesh.CenterCoor[sampling_cells, 0],
                   mesh.CenterCoor[sampling_cells, 1],
                   'k.')

    sampling_len = ((mesh.CenterCoor[sampling_cells[0], 0] - mesh.CenterCoor[sampling_cells[-1], 0]) ** 2 + \
                   (mesh.CenterCoor[sampling_cells[0], 1] - mesh.CenterCoor[sampling_cells[-1], 1]) ** 2) ** 0.5

    # making x-axis centered at zero for the 1D slice. Neccessary to have same reference with different meshes and
    # analytical solution plots.
    sampling_line = np.linspace(0, sampling_len, len(sampling_cells)) - sampling_len / 2

    ax_slice.plot(sampling_line,
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

    ax_slice.set_xticks(x_ticks)

    xtick_labels = []
    for i in x_ticks:
        xtick_labels.append('(' + to_precision(np.round(mesh.CenterCoor[sampling_cells[i], 0], 3),
                                                plot_prop.dispPrecision) + ', ' +
                                  to_precision(np.round(mesh.CenterCoor[sampling_cells[i], 1], 3),
                                                plot_prop.dispPrecision) + ')')

    ax_slice.set_xticklabels(xtick_labels)
    if vmin is not None and vmax is not None:
        ax_slice.set_ylim((vmin - 0.1*vmin, vmax + 0.1*vmax))

    if return_points:
        return fig, mesh.CenterCoor[sampling_cells[0]], mesh.CenterCoor[sampling_cells[-1]]

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution_slice(regime, variable, mat_prop, inj_prop, mesh=None, fluid_prop=None, fig=None,
                             point1=None, point2=None, time_srs=None, length_srs=None, h=None, samp_cell=None,
                             plot_prop=None, labels=None, plt_2D_image=True, gamma=None):

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean'):
        raise ValueError("The given variable does not vary spatially.")

    if plot_prop is None:
        plot_prop = PlotProperties()
    plot_prop_cp = copy.copy(plot_prop)

    if labels is None:
        labels = get_labels(variable, 'wm', '2D')

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

    # finding maximum and minimum values in complete list
    analytical_value = np.copy(analytical_list)
    vmin, vmax = np.inf, -np.inf
    for i in analytical_value:
        i = np.delete(i, np.where(np.isinf(i))[0])
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
                            plt_2D_image=plt_2D_image)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)

    if plt_2D_image:
        plt.subplot(211)
        plt.title('Top View')
        im = ax.images
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im[-1], cax=cax, orientation='vertical')
        cb.set_label(labels.colorbarLabel + ' analytical')

        ax = fig.get_axes()[1]
        ax.set_ylabel(labels.colorbarLabel)

    if plot_prop.plotLegend:
        ax.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution_at_point(regime, variable, mat_prop, inj_prop, fluid_prop=None, fig=None,
                                point=None, time_srs=None, h=None, samp_cell=None, plot_prop=None, labels=None):

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if plot_prop is None:
        plot_prop = PlotProperties()
    plot_prop_cp = copy.copy(plot_prop)

    if labels is None:
        labels_given = False
        labels = get_labels(variable, 'wole_mesh', '2D')
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
                                                        time_srs=time_srs,
                                                        h=h,
                                                        samp_cell=samp_cell)

    for i in range(len(analytical_list)):
        analytical_list[i] /= labels.unitConversion

    if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                    'front_dist_mean', 'd_mean'):
        print("The given variable does not vary spatially.")

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

    print('Plotting scale...')
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

    print('Plotting slice in 3D...')

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
                              fig=None, plot_prop=None, color='b', gamma=None):

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if plot_prop is None:
        plot_prop = PlotProperties()
        plot_prop.lineColorAnal = color

    footprint_patches = get_HF_analytical_solution_footprint(regime,
                                         mat_prop,
                                         inj_prop,
                                         plot_prop,
                                         fluid_prop=fluid_prop,
                                         time_srs=time_srs,
                                         h=h,
                                         samp_cell=samp_cell,
                                         gamma=gamma)

    for i in footprint_patches:
        ax.add_patch(i)
        if hasattr(ax, 'get_zlim'):
            art3d.pathpatch_2d_to_3d(i)

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=None, fluid_prop=None, fig=None,
                             projection='2D', time_srs=None, length_srs=None, h=None, samp_cell=None, plot_prop=None,
                             labels=None, contours_at=None, gamma=None):

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    if labels is None:
        labels_given = False
        labels = get_labels(variable, 'wole_mesh', projection)
    else:
        labels_given = True

    if variable is 'footprint':
        fig = plot_footprint_analytical(regime,
                                        mat_prop,
                                        inj_prop,
                                        fluid_prop=fluid_prop,
                                        time_srs=time_srs,
                                        h=h,
                                        samp_cell=samp_cell,
                                        fig=fig,
                                        plot_prop=plot_prop,
                                        gamma=gamma)
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

        if variable in ('time', 't', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
                        'front_dist_mean', 'd_mean'):
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
            plot_prop_cp.colorMap = plot_prop.colorMapAnal
            if projection is '2D_image':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_image(analytical_list[i],
                                                          mesh[i],
                                                          fig=fig,
                                                          plot_prop=plot_prop_cp,
                                                          vmin=vmin,
                                                          vmax=vmax)
            elif projection is '2D_contours':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_contours(analytical_list[i],
                                                             mesh[i],
                                                             fig=fig,
                                                             plot_prop=plot_prop_cp,
                                                             contours_at=contours_at,
                                                             vmin=vmin,
                                                             vmax=vmax)
            elif projection is '3D':
                for i in range(len(analytical_list)):
                    fig = plot_fracture_variable_as_surface(analytical_list[i],
                                                            mesh[i],
                                                            fig=fig,
                                                            plot_prop=plot_prop_cp,
                                                            plot_colorbar=False,
                                                            vmin=vmin,
                                                            vmax=vmax)

    ax = fig.get_axes()[0]
    ax.set_xlabel(labels.xLabel)
    ax.set_ylabel(labels.yLabel)
    plt.title(labels.figLabel)
    if projection is '3D':
        ax.set_zlabel(labels.zLabel)
        sm = plt.cm.ScalarMappable(cmap=plot_prop_cp.colorMap,
                                   norm=plt.Normalize(vmin=vmin,
                                                      vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm, alpha=plot_prop_cp.alpha)
        cb.set_label(labels.colorbarLabel + ' analytical')
    elif projection in ('2D_image', '2D_contours'):
        im = ax.images
        cb = im[-1].colorbar
        cb.set_label(labels.colorbarLabel + ' analytical')
    elif projection is '2D':
        plt.title(labels.figLabel)
        if variable not in ('footprint'):
            ax.legend()

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def get_HF_analytical_solution_footprint(regime, mat_prop, inj_prop, plot_prop, fluid_prop=None, time_srs=None,
                                         h=None, samp_cell=None, gamma=None):

    if time_srs is None:
        raise ValueError("Time series is to be provided.")

    if regime is 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime is "MDR":
        density = fluid_prop.density
    else:
        density = None

    if samp_cell is None:
        samp_cell = int(len(mat_prop.Kprime) / 2)

    if regime is 'K':
        muPrime = None
        Cprime = None
    else:
        muPrime = fluid_prop.muPrime
        Cprime = mat_prop.Cprime[samp_cell]

    if regime is 'M':
        Kprime = None
        Cprime = None
    else:
        Kprime = mat_prop.Kprime[samp_cell]
        Cprime = mat_prop.Cprime[samp_cell]


    return_pathces = []
    for i in time_srs:
        x_len, y_len = get_fracture_dimensions_analytical(regime,
                                                          i,
                                                          mat_prop.Eprime,
                                                          inj_prop.injectionRate[1, 0],
                                                          muPrime=muPrime,
                                                          Kprime=Kprime,
                                                          Cprime=Cprime,
                                                          Kc_1=Kc_1,
                                                          h=h,
                                                          density=density,
                                                          gamma=gamma)


        if regime in ('M', 'Mt', 'K', 'Kt', 'E', 'MDR'):
            return_pathces.append(mpatches.Circle((0., 0.),
                                   x_len,
                                   edgecolor=plot_prop.lineColorAnal,
                                   facecolor='none'))
        elif regime in ('PKN', 'KGD_K'):
            return_pathces.append(mpatches.Rectangle(xy=(-x_len, -y_len),
                                      width=2 * x_len,
                                      height=2 * y_len,
                                      edgecolor=plot_prop.lineColorAnal,
                                      facecolor='none'))
        elif regime in ('E_K', 'E_E'):
            return_pathces.append(mpatches.Ellipse(xy=(0., 0.),
                                   width=2 * x_len,
                                   height=2 * y_len,
                                   edgecolor=plot_prop.lineColorAnal,
                                   facecolor='none'))
        else:
            raise ValueError("Regime not supported.")

    return return_pathces

#-----------------------------------------------------------------------------------------------------------------------


def animate_simulation_results(fracture_list, variable='width', mat_properties=None, projection='3D', elements=None,
                                backGround_param=None, plot_prop=None, edge=4, contours_at=None, labels=None,
                                plot_non_zero=True,  Interval=400, Repeat=None, save=False, address=None):
    """
    This function plays an animation of the evolution of fracture with time. See the arguments list for options

    Arguments:
        address (string):               the folder containing the fracture files
        time_period (float):            the output time period after which the next available fracture will be plotted.
                                        This is the minimum time between two ploted fractures and can be used to avoid
                                        clutter.
        colormap (matplotlib colormap): the color map used to plot
        edge_color (matplotlib colors): the color used to plot the grid lines
        Interval (float):               time in milliseconds between the frames of animation
        Repeat (boolean):               True will play the animation in a loop
        maxFiles (int):                 the maximum no. of files to be loaded

    """

    if plot_prop is None:
        plot_prop = PlotProperties()

    fig = plot_fracture_list(fracture_list,
                                variable='mesh',
                                projection=projection,
                                elements=elements,
                                plot_prop=plot_prop,
                                backGround_param=backGround_param,
                                mat_properties=mat_properties,
                                labels=labels)

    args = (fracture_list, variable, mat_properties, projection, elements,
            backGround_param, plot_prop, edge, contours_at, labels, fig, plot_non_zero)
    # animate fracture
    movie = animation.FuncAnimation(fig,
                              update,
                              fargs=args,
                              frames=len(fracture_list),
                              interval=Interval,
                              repeat=Repeat,
                              repeat_delay=1000)
    if save:
        if address is None:
            raise ValueError("Please provide the address to save the video file")
        Writer = animation.writers['ffmpeg']
        writer = Writer(metadata={'copyright':'EPFL - GeoEnergy Lab'})
        movie.save(address + 'Footprint-evol.mp4', writer=writer)
    else:
        plt.show()

#-----------------------------------------------------------------------------------------------------------------------

def update(frame, *args):
    """
    This function update the frames to be used in the animation.

    """

    (fracture_list, variable, mat_properties, projection, elements,
     backGround_param, plot_prop, edge, contours_at, labels, fig, plot_non_zero) = args

    ffi = fracture_list[frame]
    labels = LabelProperties()
    labels.figLabel = 't = ' + to_precision(ffi.time, plot_prop.dispPrecision) + "($s$)"
    ffi.plot_fracture(variable=variable,
                      mat_properties=mat_properties,
                      projection=projection,
                      elements=elements,
                      backGround_param=backGround_param,
                      plot_prop=plot_prop,
                      fig=fig,
                      edge=edge,
                      contours_at=contours_at,
                      labels=labels)

#-----------------------------------------------------------------------------------------------------------------------


def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    '''
    Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
    and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
    as the third dimension.  usetex is a boolean indicating whether the string
    should be interpreted as latex or not.  Any additional keyword arguments
    are passed on to transform_path.

    Note: zdir affects the interpretation of xyz.
    '''
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


def to_precision(x,p):
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

    import cv2
    import os

    if ".avi" not in video_name:
        video_name = video_name + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    img_no = 0
    for image in images:
        print("adding image no " + repr(img_no))
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.waitKey(1)
        img_no += 1

    cv2.destroyAllWindows()
    video.release()