#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 12.06.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
#
# Post-process scripts to plot results for a fracture

# local
import src.Fracture

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def plot_Reynolds_number(fracture,Rec=2200,fig=None, bck_colMap='cool', line_color = 'k', contours_at=None):
    """
    This function plots the average Reynolds number of the four edges of the cells in a fracture

    Arguments:
        fracture (Fracture):        -- the fracture object for which the Reynolds number is to be plotted.
        fig (figure)                -- figure to superimpose. A new figure will be created if not provided.
        bck_colMap (Colormaps)      -- colormap for the Reynold's number shown in the background.
        line_color (color)          -- the color of the contour line (e.g. 'r' will plot in red).
        contours_at (ndarray)       -- a list of Reynold's numbers to plot contours at.

    Returns:
         Fig
    """

    if fracture.ReynoldsNumber is None:
        print("Reynold's numbers not available for time = " + repr(fracture.time) + '\nProbably initial fracture')
        return

    ReyNum = np.mean(fracture.ReynoldsNumber, axis=0)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    x = fracture.mesh.CenterCoor[:, 0].reshape((fracture.mesh.ny, fracture.mesh.nx))
    y = fracture.mesh.CenterCoor[:, 1].reshape((fracture.mesh.ny, fracture.mesh.nx))
    ReyNum = ReyNum.reshape((fracture.mesh.ny, fracture.mesh.nx))

    dx = (x[0,1] - x[0,0]) / 2.
    dy = (y[1,0] - y[0,0]) / 2.
    extent = [x[0,0] - dx, x[-1, -1] + dx, y[0,0] - dy, y[-1, -1] + dy]

    cax = ax.imshow(ReyNum,
              cmap=bck_colMap,
              interpolation='spline16',
              extent=extent)
    cbar = fig.colorbar(cax)

    if contours_at is None:
        contours_at = np.max(ReyNum) * np.asarray([0.01, 0.07, 0.15, 0.5, 0.7, 0.9])

    CS = ax.contour(x,
                    y,
                    ReyNum,
                    contours_at,
                    colors=line_color)

    plt.clabel(CS, fmt='%1.0f')

    contours_at = np.asarray([Rec])
    CS = ax.contour(x,
                    y,
                    ReyNum,
                    contours_at,
                    colors='w',
                    linewidths=2)
    # fmt = {}
    # strs = ['transition']
    # for l, s in zip(CS.levels, strs):
    #     fmt[l] = s
    #
    # # Label every other level using strings
    # plt.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=10)


    custom_line = [Line2D([0], [0], color='w', lw=2)]
    ax.legend(custom_line, ['turbulent to laminar transition'])

    ax.set_ylabel('meters')
    ax.set_xlabel('meters')
    ax.set_title('Reynolds number')

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_width_contour(fracture, fig=None, bck_colMap='cool', line_color = 'k', contours_at=None):
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

    x = fracture.mesh.CenterCoor[:, 0].reshape((fracture.mesh.ny, fracture.mesh.nx))
    y = fracture.mesh.CenterCoor[:, 1].reshape((fracture.mesh.ny, fracture.mesh.nx))

    #
    width = fracture.w.reshape((fracture.mesh.ny, fracture.mesh.nx)) * 1e3

    dx = (x[0,1] - x[0,0]) / 2.
    dy = (y[1,0] - y[0,0]) / 2.
    extent = [x[0,0] - dx, x[-1, -1] + dx, y[0,0] - dy, y[-1, -1] + dy]

    cax = ax.imshow(width,
              cmap=bck_colMap,
              interpolation='spline16',
              extent=extent)
    cbar = fig.colorbar(cax)

    if contours_at is None:
        contours_at = np.max(width) * np.asarray([0.01, 0.15, 0.5, 0.7, 0.9])

    CS = ax.contour(x,
                    y,
                    width,
                    contours_at,
                    colors=line_color)

    plt.clabel(CS)

    ax.set_ylabel('meters')
    ax.set_xlabel('meters')
    ax.set_title('Fracture width (mm)')

    return fig


# -----------------------------------------------------------------------------------------------------------------------

def plot_pressure_contour(fracture, fig=None, bck_colMap='cool', line_color='k', contours_at=None):
    """
    This function plots the contours of the fracture pressure in mega pascals.

    Arguments:
        fracture (Fracture):        -- the fracture object for which the fracture pressure is to be plotted.
        fig (figure)                -- figure to superimpose. A new figure will be created if not provided.
        bck_colMap (Colormaps)      -- colormap for the fracture pressure shown in the background.
        line_color (color)          -- the color of the contour line (e.g. 'r' will plot in red).
        contours_at (ndarray)       -- a list of fracture pressures to plot contours at.

    Returns:
         Fig
    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    x = fracture.mesh.CenterCoor[:, 0].reshape((fracture.mesh.ny, fracture.mesh.nx))
    y = fracture.mesh.CenterCoor[:, 1].reshape((fracture.mesh.ny, fracture.mesh.nx))

    #
    pressure = fracture.p.reshape((fracture.mesh.ny, fracture.mesh.nx)) /1e6

    dx = (x[0, 1] - x[0, 0]) / 2.
    dy = (y[1, 0] - y[0, 0]) / 2.
    extent = [x[0, 0] - dx, x[-1, -1] + dx, y[0, 0] - dy, y[-1, -1] + dy]

    cax = ax.imshow(pressure,
                    cmap=bck_colMap,
                    interpolation='spline16',
                    extent=extent)
    cbar = fig.colorbar(cax)

    if contours_at is None:
        pressure_range = np.max(pressure) - np.min(pressure)
        contours_at = np.min(pressure) + pressure_range * np.asarray([0.00, 0.03, 0.25, 0.6, 0.8])

    CS = ax.contour(x,
                    y,
                    pressure,
                    contours_at,
                    colors=line_color)

    plt.clabel(CS)

    ax.set_ylabel('meters')
    ax.set_xlabel('meters')
    ax.set_title('Fracture pressure (MPa)')

    return fig
