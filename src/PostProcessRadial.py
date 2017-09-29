# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Apr 19 11:07:38 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from src.CartesianMesh import *
from src.Fracture import *
from src.VolIntegral import *
from src.Properties import *
from src.Utility import *


def plot_radial_data(address, plot_analytical=True, regime = "M", plot_w_cntr=True, fig_w_cntr=None, plot_r=True,
                     fig_r=None, plot_p_cntr=False, fig_p_cntr=None, maxFiles=150, loglog=True, plot_w_prfl=False,
                     plot_p_prfl=False):
    """
    This function reads the saved files in the given folder and plots the figures that are enabled. Analytical solutions
    according to the given regime can also be plotted along with the numerical solution read from the saved files.
    
    Argument:
        address (string):                           address of the saved data files (e.g ".\\Data\\simulation" for
                                                    windows system, "./Data/simulation" for linux or mac system).   
        plot_analytical (boolean, default True):    if True, analytical solution specified by the argument 'regime' will
                                                    be plotted along with the numerical solution
        regime (string, default "M"):               the regime of the analytical solution to be plotted.
        plot_w_cntr (boolean, default True):        if True, width at the center against time will be plotted.
        fig_w_cntr (matplotlib.figure object):      a figure object to superimpose width at the center plot(s). A new
                                                    figure will be made if not given.
        plot_r (boolean, default True):             if True, fracture radius against time will be plotted.
        fig_r: (matplotlib.figure object):          a figure object to superimpose the radius plot(s). A new figure will
                                                    be made if not given.
        plot_p_cntr (boolean, default False):       if True, pressure at center against time will be plotted.
        fig_p_cntr: (matplotlib.figure object):     a figure object to superimpose the pressure at center plot(s). A new
                                                    figure will be made if not given.
        maxFiles (int, default 150):                maximum number of files to be read.
        loglog (boolean, default True):             if True, plots will be made on loglog scale.
        plot_w_prfl (boolean, default False):       if True, width profile at the horizontal cross section (passing 
                                                    through the center) will be plotted. Six plots, equally spaced in
                                                    time, will be plotted along with the analytical solution at the
                                                    time. 
        plot_p_prfl (boolean, default False):       if True, pressure profile at the horizontal cross section (passing
                                                    through the center) will be plotted. Six plots, equally spaced in
                                                    time, will be plotted along with the analytical solution at the
                                                    time.
    
    Returns:
       matplotlib.figure object:                    a figure object of the width at center plot.  
       matplotlib.figure object:                    a figure object of the radius fracture plot.  
       matplotlib.figure object:                    a figure object of the pressure at center plot.  
    """

    # loading parameters

    import sys
    if "win32" in sys.platform or "win64" in sys.platform:
        slash = "\\"
    else:
        slash = "/"

    if not slash in address[-2:]:
        address = address + slash
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (solid, fluid, injection, simulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Properties file not found.")

    # loading the first fracture
    Fr = ReadFracture(simulProp.outFileAddress + "file_" + repr(0))

    # initializing arrays
    R_numrcl = np.asarray([], dtype=np.float64)
    w_numrcl_centr = np.asarray([], dtype=np.float64)
    p_numrcl_centr = np.asarray([], dtype=np.float64)
    vol_numrcl = np.asarray([], dtype=np.float64)
    time_series = np.asarray([], dtype=np.float64)

    # initializing arrays for width profile of the cross section if enabled
    if plot_w_prfl:
        if not plot_analytical:
            plot_analytical = True
        # elements in the horizontal cross section at the center
        hrzntl = np.where(abs(Fr.mesh.CenterCoor[:, 1]) < 1e-8)[0]
        x = Fr.mesh.CenterCoor[hrzntl, 0]
        w_hrzntl_anltcl = np.empty((hrzntl.size,), dtype=np.float64)
        w_hrzntl_numrcl = np.empty((hrzntl.size,), dtype=np.float64)

    # initializing arrays for pressure profile of the cross section if enabled
    if plot_p_prfl:
        if not plot_analytical:
            plot_analytical = True
        # elements in the horizontal cross section at the center
        hrzntl = np.where(abs(Fr.mesh.CenterCoor[:, 1]) < 1e-8)[0]
        x = Fr.mesh.CenterCoor[hrzntl, 0]
        p_hrzntl_anltcl = np.empty((hrzntl.size,), dtype=np.float64)
        p_hrzntl_numrcl = np.empty((hrzntl.size,), dtype=np.float64)

    # initializing arrays for analytical solution if enabled
    if plot_analytical:
        R_anltcl = np.asarray([], dtype=np.float64)
        w_anltcl_centr = np.asarray([], dtype=np.float64)
        p_anltcl_centr = np.asarray([], dtype=np.float64)
        vol_anltcl = np.asarray([], dtype=np.float64)

    fileNo = 0

    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            Fr = ReadFracture(simulProp.outFileAddress + "file_" + repr(fileNo))
        except FileNotFoundError:
            break

        time_series = np.append(time_series, Fr.time)
        # radius of the fracture is evaluated by adding the distance of the zero vertex from the center (injection
        # point) to the length of the perpendicular drawn on the fracture front in the tip cell
        tipVrtxCoord = Fr.mesh.VertexCoor[Fr.mesh.Connectivity[Fr.EltTip, Fr.ZeroVertex]]
        R_numrcl = np.append(R_numrcl, max((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + Fr.l))

        # width at the injection point
        w_numrcl_centr = np.append(w_numrcl_centr, Fr.w[injection.source_location])
        # pressure at the injection point
        p_numrcl_centr = np.append(p_numrcl_centr, Fr.p[injection.source_location])
        # total volume inside the fracture
        vol_numrcl = np.append(vol_numrcl, sum(Fr.w) * Fr.mesh.EltArea)

        if plot_analytical:
        # getting the analytical solution
            if regime == "M":
                (R_a, p_a, w_a, v_a) = M_vertex_solution_t_given(solid.Eprime,
                                                                 injection.injectionRate[1, 0],
                                                                 fluid.muPrime,
                                                                 Fr.mesh,
                                                                 Fr.time)
            elif regime == "K":
                (R_a, p_a, w_a, v_a) = K_vertex_solution_t_given(solid.Kprime,
                                                                 solid.Eprime,
                                                                 injection.injectionRate[1, 0],
                                                                 Fr.mesh,
                                                                 Fr.time)
            R_anltcl = np.append(R_anltcl, R_a)
            w_anltcl_centr = np.append(w_anltcl_centr, w_a[injection.source_location])
            p_anltcl_centr = np.append(p_anltcl_centr, p_a[injection.source_location])
            vol_anltcl = np.append(vol_anltcl, sum(w_a) * Fr.mesh.EltArea)

        # saving width of the cross section if plotting is enabled
        if plot_w_prfl:
            w_hrzntl_numrcl = np.vstack((w_hrzntl_numrcl, Fr.w[hrzntl]))
            w_hrzntl_anltcl = np.vstack((w_hrzntl_anltcl, w_a[hrzntl]))

        # saving pressure of the cross section if plotting is enabled
        if plot_p_prfl:
            p_a = p_a[hrzntl]
            # get maximum pressure. This value is used for representing singularities in order to get reasonable
            # axis values.
            maxP = max(p_a)
            # setting to nan where fracture is not propagated
            zero = np.where(p_a == 0)[0]
            p_a[zero] = np.nan

            # finding the location of singularities
            nonZero = np.where(np.logical_not(np.isnan(p_a)))[0]
            p_a[max(nonZero)] = - maxP
            p_a[min(nonZero)] = - maxP

            # saving the pressure at cross section
            p_hrzntl_numrcl = np.vstack((p_hrzntl_numrcl, Fr.p[hrzntl]))
            p_hrzntl_anltcl = np.vstack((p_hrzntl_anltcl, p_a))


        fileNo += 1

    # plotting width at center
    if plot_w_cntr:
        if fig_w_cntr is None:
            fig_w_cntr = plt.figure()
            ax_w = fig_w_cntr.add_subplot(111)
        ax_w = fig_w_cntr.add_subplot(111)
        if loglog:
            ax_w.loglog(time_series, w_numrcl_centr, 'o')
            if plot_analytical:
                ax_w.loglog(time_series, w_anltcl_centr)
        else:
            ax_w.plot(time_series, w_numrcl_centr, 'o')
            if plot_analytical:
                ax_w.plot(time_series, w_anltcl_centr)
        plt.ylabel('width at injection point (meters)')
        plt.xlabel('time (sec)')


    # plotting radius
    if plot_r:
        if fig_r is None:
            fig_r = plt.figure()
            ax_r = fig_r.add_subplot(111)
        ax_r = fig_r.add_subplot(111)
        if loglog:
            if plot_analytical:
                ax_r.loglog(time_series, R_anltcl)
            ax_r.loglog(time_series, R_numrcl, 'o')
        else:
            if plot_analytical:
                ax_r.plot(time_series, R_anltcl)
            ax_r.plot(time_series, R_numrcl, 'o')
        plt.ylabel('fracture radius')
        plt.xlabel('time')

    # plotting pressure at center
    if plot_p_cntr:
        if fig_p_cntr is None:
            fig_p_cntr = plt.figure()
            ax_p = fig_p_cntr.add_subplot(111)
        ax_p = fig_p_cntr.add_subplot(111)
        if loglog:
            if plot_analytical:
                ax_p.loglog(time_series, p_anltcl_centr)
            ax_p.loglog(time_series, p_numrcl_centr, 'o')
        else:
            if plot_analytical:
                ax_r.plot(time_series, p_anltcl_centr)
            ax_r.plot(time_series, p_numrcl_centr, 'o')
        plt.ylabel('pressure at injection point')
        plt.xlabel('time')

    #plotting width profile at the center
    if plot_w_prfl:
        fig_w_prfl = plt.figure()
        for i in range(1, 7):
            ax = fig_w_prfl.add_subplot(2, 3, i)
            ax.plot(x, w_hrzntl_numrcl[((fileNo - 1) // 6) * i],'o')
            ax.plot(x, w_hrzntl_anltcl[((fileNo - 1) // 6) * i])
            plt.ylabel("width (meters)")
            plt.xlabel("x (meters)")
            plt.title("time = " + str(round(time_series[((fileNo-1)//6) * i],5)) + " sec")
        plt.suptitle('Width profile at cross section passing through the injection point (center)')
    else:
        fig_w_prfl = None

    # plotting pressure profile at the center
    if plot_p_prfl:
        fig_p_prfl = plt.figure()
        for i in range(1, 7):
            ax = fig_p_prfl.add_subplot(2, 3, i)
            ax.plot(x, p_hrzntl_numrcl[((fileNo - 1) // 6) * i], 'o')
            ax.plot(x, p_hrzntl_anltcl[((fileNo - 1) // 6) * i])
            plt.ylabel("pressure (pascals)")
            plt.xlabel("x (meters)")
            plt.title("time = " + str(round(time_series[((fileNo - 1) // 6) * i],5)) + " sec")
        plt.suptitle('Pressure profile at cross section passing through the injection point (center)')
    else:
        fig_p_prfl = None

    plt.show()
    return fig_w_cntr, fig_r, fig_p_cntr
