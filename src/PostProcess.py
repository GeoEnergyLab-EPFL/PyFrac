#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 12.06.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#
#
# post-process script from reading files  in simulation folder.....

# adding src folder to the path
import sys
if "win32" in sys.platform or "win64" in sys.platform:
    slash = "\\"
else:
    slash = "/"

# imports
from src.Fracture import *
from src.Properties import *

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import dill
import os

def animate_simulation_results(address=None, time_period= 0.0, sol_time_series=None, Interval=400, Repeat=None,
                               maxFiles=1000, save=False):
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
    print("Animating the fracture evolution...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if address[-1] is not slash:
        address = address + slash

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    fileNo = 0
    fraclist = []
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_time_series, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_time_series[t_srs_indx]

    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            break
        prev_modified_at = stats[-2]

        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period
            fraclist.append(ff)

            if t_srs_given:
                if t_srs_indx < len(sol_time_series) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_time_series[t_srs_indx]
                if ff.time > max(sol_time_series):
                    break
            else:
                nxt_plt_t = ff.time + time_period


    fig = fraclist[-1].plot_fracture(parameter='mesh', mat_properties=Solid, sim_properties=SimulProp)

    args = (fraclist, fileNo, Solid, Fluid, Injection)
    # animate fracture
    movie = animation.FuncAnimation(fig,
                              update,
                              fargs=args,
                              frames=len(fraclist),
                              interval=Interval,
                              repeat=Repeat,
                              repeat_delay=1000)
    if save:
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

    # loading the fracture list
    (fraclist, noFractures, Solid, Fluid, Injection) = args

    ffi = fraclist[frame]

    I =ffi.Ffront[:,0:2]
    J= ffi.Ffront[:,2:4]

    for e in range(0,len(I)):
        plt.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), '-k')

    plt.title('Time ='+ "%.4f" % ffi.time+ ' sec.')
    plt.axis('equal')

#-----------------------------------------------------------------------------------------------------------------------


def plot_profile(address=None, fig_w_x=None, fig_w_y=None, fig_p_x=None, fig_p_y=None, plt_pressure=False,
                 time_period=0.0, plot_at_times=None, analytical_sol='n', plt_symbol='k.', anltcl_lnStyle='b'):
    """
    This function plots the width and pressure at the injection point of the fracture.

    Arguments:
        address (string)                -- the folder address containing the saved files
        fig_w_x (figure)                -- figure for fracture width at x-axis to superimpose. A new figure will be
                                           created if not provided.
        fig_w_y (figure)                -- figure for fracture width at y-axis to superimpose. A new figure will be
                                           created if not provided.
        fig_p_x (figure)                -- figure for fracture pressure at x-axis to superimpose. A new figure will be
                                           created if not provided.
        fig_p_y (figure)                -- figure for fracture pressure at y-axis to superimpose. A new figure will be
                                           created if not provided.
        plt_pressure (boolean)          -- if True, pressure will also be plotted.
        time_period (float)             -- time period between two successive fracture plots.
        plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
        analytical_sol (string)         -- the following options can be provided
                                                'M'     -- radial fracture in viscosity dominated regime
                                                'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                'K'     -- radial fracture in toughness dominated regime
                                                'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                'E'     -- elliptical fracture in toughness dominated regime
                                                'PKN'   -- PKN fracture
        plt_symbol (string)             -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                           continous line with data points marked with dots )
        anltcl_lnStyle (string)         -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                           continous line with data points marked with dots )

    Returns:
        fig_w_x (figure)                -- figure for fracture width at x-axis to superimpose.
        fig_w_y (figure)                -- figure for fracture width at y-axis to superimpose.
        fig_p_x (figure)                -- figure for fracture pressure at x-axis to superimpose.
        fig_p_y (figure)                -- figure for fracture pressure at y-axis to superimpose.
    """

    print("Plotting fracture profile...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(plot_at_times, np.ndarray)
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    if fig_w_x is None:
        fig_w_x = plt.figure()
        ax_w_x = fig_w_x.add_subplot(111)
    else:
        ax_w_x = fig_w_x.get_axes()[0]

    if fig_w_y is None:
        fig_w_y = plt.figure()
        ax_w_y = fig_w_y.add_subplot(111)
    else:
        ax_w_y = fig_w_y.get_axes()[0]

    if plt_pressure:
        if fig_p_x is None:
            fig_p_x = plt.figure()
            ax_p_x = fig_p_x.add_subplot(111)
        else:
            ax_p_x = fig_p_x.get_axes()[0]
        if fig_p_y is None:
            fig_p_y = plt.figure()
            ax_p_y = fig_p_y.add_subplot(111)
        else:
            ax_p_y = fig_p_y.get_axes()[0]

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            break
        prev_modified_at = stats[-2]

        fileNo += 1
        plotted_fractures = 0

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period
            if not analytical_sol is 'n':
                from src.CartesianMesh import CartesianMesh
                mesh_refined = CartesianMesh(ff.mesh.Lx, ff.mesh.Ly, 201, 201)
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                mesh_refined,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)
                    hrzntl_rfnd = np.where(abs(mesh_refined.CenterCoor[:, 1]) < 1e-10)[0]
                    x_refined = mesh_refined.CenterCoor[hrzntl_rfnd, 0]
                    vrtcl_rfnd = np.where(abs(mesh_refined.CenterCoor[:, 0]) < 1e-10)[0]
                    y_refined = mesh_refined.CenterCoor[vrtcl_rfnd, 1]
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")

            # cells on x and y axes
            hrzntl = np.where(abs(ff.mesh.CenterCoor[:, 1]) < 1e-10)[0]
            x = ff.mesh.CenterCoor[hrzntl, 0]
            vrtcl = np.where(abs(ff.mesh.CenterCoor[:, 0]) < 1e-10)[0]
            y = ff.mesh.CenterCoor[vrtcl, 1]

            line_wx_num, = ax_w_x.plot(x, ff.w[hrzntl],plt_symbol)
            line_wy_num, = ax_w_y.plot(y, ff.w[vrtcl], plt_symbol)
            if not analytical_sol is 'n':
                line_wx_anl, = ax_w_x.plot(x_refined, w[hrzntl_rfnd],anltcl_lnStyle)
                line_wy_anl, = ax_w_y.plot(y_refined, w[vrtcl_rfnd], anltcl_lnStyle)

            ax_w_x.set_ylabel('width')
            ax_w_x.set_xlabel('meters')
            ax_w_x.set_title('Width profile along x-axis')

            ax_w_y.set_ylabel('width')
            ax_w_y.set_xlabel('meters')
            ax_w_y.set_title('Width profile along y-axis')

            if plt_pressure:
                line_px_num, = ax_p_x.plot(x, ff.p[hrzntl], plt_symbol)
                line_py_num, = ax_p_y.plot(y, ff.p[vrtcl], plt_symbol)
                if not analytical_sol is 'n':
                    # np.delete(hrzntl, np.where(ff.p[hrzntl]!=0.)[0], 0)
                    line_px_anl, = ax_p_x.plot(x_refined, p[hrzntl_rfnd], anltcl_lnStyle)
                    line_py_anl, = ax_p_y.plot(y_refined, p[vrtcl_rfnd], anltcl_lnStyle)

                ax_p_x.set_ylabel('pressure')
                ax_p_x.set_xlabel('meters')
                ax_p_x.set_title('Pressure profile along x-axis')

                ax_p_y.set_ylabel('pressure')
                ax_p_y.set_xlabel('meters')
                ax_p_y.set_title('Pressure profile along y-axis')

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

            plotted_fractures += 1

    if not analytical_sol is 'n' and plotted_fractures > 0:
        ax_w_x.legend((line_wx_num, line_wx_anl),('numerical','analytical'))
        ax_w_y.legend((line_wy_num, line_wy_anl), ('numerical', 'analytical'))
        if plt_pressure:
            ax_p_x.legend((line_px_num, line_px_anl), ('numerical', 'analytical'))
            ax_p_y.legend((line_py_num, line_py_anl), ('numerical', 'analytical'))


    return fig_w_x, fig_w_y, fig_p_x, fig_p_y


#-----------------------------------------------------------------------------------------------------------------------
def plot_at_injection_point(address=None, fig_w=None, fig_p=None, plt_pressure=True, time_period=0.0, plot_at_times=None,
                analytical_sol='n', plt_symbol='r.', anltcl_lnStyle='b', loglog=True, plt_t_dimensionless=False,
                plt_error=True, add_labels=True):
    """
        This function plots the width and pressure at the injection point of the fracture.

        Arguments:
            address (string)                -- the folder address containing the saved files
            fig_w (figure)                  -- figure for fracture width to superimpose. A new figure will be created
                                               if not provided.
            fig_p (figure)                  -- figure for pressure to superimpose. A new figure will be created if not
                                               provided.
            plt_pressure (boolean)          -- if True, pressure will also be plotted.
            time_period (float)             -- time period between two successive fracture plots.
            plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
            analytical_sol (string)         -- the following options can be provided
                                                    'M'     -- radial fracture in viscosity dominated regime
                                                    'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                    'K'     -- radial fracture in toughness dominated regime
                                                    'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                    'E'     -- elliptical fracture in toughness dominated regime
                                                    'PKN'   -- PKN fracture
            plt_symbol (string)             -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            anltcl_lnStyle (string)         -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            loglog (True)                   -- if True, plots will be loglog.
            plt_dimensionless (True)        -- if True, time will be scaled with the viscosity to toughness transition
                                               time.

        Returns:
            fig_w (figure)                    -- width figure to superimpose.
            fig_p (figure)                    -- pressure figure to superimpose.
    """

    print("Ploting solution at the injection point...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    # add slash at the end if not present
    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(plot_at_times, np.ndarray)
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    if fig_w is None:
        fig_w = plt.figure()
        ax_w = fig_w.add_subplot(111)
    else:
        ax_w = fig_w.get_axes()[0]
    if plt_pressure:
        if fig_p is None:
            fig_p = plt.figure()
            ax_p = fig_p.add_subplot(111)
        else:
            ax_p = fig_p.get_axes()[0]

    if not analytical_sol is 'n':
        w_anltcl = np.array([], dtype=np.float64)
        time_srs = np.array([], dtype=np.float64)
        if plt_pressure:
            p_anltcl = np.array([], dtype=np.float64)

        if plt_error:
            fig_err = plt.figure()
            ax_err = fig_err.add_subplot(111)
            w_err = np.array([], dtype=np.float64)
            p_err = np.array([], dtype=np.float64)

    if plt_t_dimensionless:
        # viscosity to toughness transition time
        tmk = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
                (32 / math.pi) ** 0.5 * Solid.K1c[0]) ** 18) ** 0.5
        tmk2 = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
                (32 / math.pi) ** 0.5 * Solid.K1c[0]) ** 18) ** 0.5
    else:
        tmk = 1.

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            break
        prev_modified_at = stats[-2]

        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            if not analytical_sol is 'n':
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                ff.mesh,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")

            if loglog:
                # ax_w.semilogx(ff.time, ff.w[ff.mesh.CenterElts], plt_symbol)
                ax_w.loglog(ff.time/tmk, ff.w[ff.mesh.CenterElts], plt_symbol)
            else:
                ax_w.plot(ff.time/tmk, ff.w[ff.mesh.CenterElts],plt_symbol)
            if add_labels:
                ax_w.set_ylabel('width')
                ax_w.set_xlabel('time')
                ax_w.set_title('Width at injection point')

            if not analytical_sol is 'n':
                w_anltcl = np.append(w_anltcl, w[ff.mesh.CenterElts])
                time_srs = np.append(time_srs, ff.time)
                if plt_error:
                    w_err = np.append(w_err, 1. - w[ff.mesh.CenterElts]/ff.w[ff.mesh.CenterElts])

            if plt_pressure:
                if isinstance(ff.p, np.ndarray):
                    p_num = ff.p[ff.mesh.CenterElts]
                else:
                    p_num = ff.p
                if loglog:
                    # ax_p.semilogx(ff.time, p_num, plt_symbol)
                    ax_p.loglog(ff.time/tmk, p_num, plt_symbol)
                else:
                    ax_p.plot(ff.time/tmk, p_num, plt_symbol)


                if not analytical_sol is 'n':
                    if isinstance(p, np.ndarray):
                        p_aa = p[ff.mesh.CenterElts]
                    else:
                        p_aa = p
                    p_anltcl = np.append(p_anltcl, p_aa)
                    if plt_error:
                        p_err = np.append(p_err, (p_aa - p_num)/p_aa)

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if not analytical_sol is 'n':
        if loglog:
            ax_w.semilogx(time_srs/tmk, w_anltcl, anltcl_lnStyle, label='analytical')
        else:
            ax_w.plot(time_srs/tmk, w_anltcl, anltcl_lnStyle, label='analytical')
        if add_labels:
            ax_w.legend()

        if plt_pressure:
            if analytical_sol in ('M', 'Mt'):
                print("Singularity at injection point for pressure. Analytical solution not plotted")
            else:
                if loglog:
                    ax_p.semilogx(time_srs/tmk, p_anltcl, anltcl_lnStyle, label='analytical')
                else:
                    ax_p.plot(time_srs/tmk, p_anltcl, anltcl_lnStyle, label='analytical')
            if add_labels:
                ax_p.set_ylabel('pressure')
                ax_p.set_xlabel('time')
                ax_p.set_title('Pressure at injection point')
                ax_p.legend()

    # ax_w.plot(tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000 * tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000, 1e-4, 'k.')
    # print(repr(time_srs))
        if plt_error:
            ax_err.semilogx(time_srs, abs(w_err), 'bo-', label='error on width')
            if plt_pressure and not analytical_sol in ('M', 'Mt'):
                ax_err.semilogx(time_srs, abs(p_err), 'ro-', label='error on pressure')
            if add_labels:
                ax_err.set_ylabel('error')
                ax_err.set_xlabel('time')
                ax_err.set_title('Relative error at injection point')
                ax_err.legend()


    return fig_w, fig_p


def plot_footprint(address=None, fig=None, time_period=0.0, plot_at_times=None, analytical_sol='n',
                            plt_color='k', anltcl_lnStyle='b', plt_mesh=True, plt_regime=False, Sim_prop=None):
    """
    This function plots the footprints of the fractures saved in the given folder.

    Arguments:
        address (string)                -- the folder address containing the saved files
        fig (figure)                    -- figure to superimpose. A new figure will be created if not provided.
        time_period (float)             -- time period between two successive fracture plots.
        plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
        analytical_sol (string)         -- the following options can be provided
                                                'M'     -- radial fracture in viscosity dominated regime
                                                'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                'K'     -- radial fracture in toughness dominated regime
                                                'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                'E'     -- elliptical fracture in toughness dominated regime
                                                'PKN'   -- PKN fracture
        plt_symbol (string)             -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
        anltcl_lnStyle (string)         -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
        plt_mesh (boolean)              -- if true, mesh will also be plotted.
        plt_regime (boolean)            -- if true, regime evaluated at the ribbon cells will be ploted (see Zia and
                                           Lecampion 2018)
        Sim_prop (SimulationParameters) -- if provided, the simulation properties read for file will be overriden by
                                           these parameters.

    Returns:
        fig (figure)                    -- a figure to superimpose.

    """

    print("Ploting footprint of the fracture...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    if plot_at_times is None and time_period == 0.0 and Sim_prop is None and SimulProp.get_solTimeSeries() is not None:
        plot_at_times = SimulProp.get_solTimeSeries()
    elif plot_at_times is None and time_period == 0.0 and Sim_prop is not None:
        plot_at_times = Sim_prop.get_solTimeSeries()


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0

    t_srs_given = isinstance(plot_at_times, np.ndarray) #time series is given
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    # new figure if not provided
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]
    ff = None
    while fileNo < 5000:

        # saving last fracture in case the loaded file is to be discarded (possibly its from an old simulation)
        ff_last = copy.deepcopy(ff)

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            ff = ff_last
            break
        prev_modified_at = stats[-2]

        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period
            print("Plotting at " + repr(ff.time) + ' s')

            I = ff.Ffront[:, 0:2]
            J = ff.Ffront[:, 2:4]

            for e in range(0, len(I)):
                ax.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), color=plt_color)

            # plot analytical solution
            if not analytical_sol is 'n':
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                ff.mesh,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")

                if analytical_sol in ('M', 'Mt', 'K', 'Kt'):
                    circle = plt.Circle((0, 0), radius=R)
                    circle.set_ec(anltcl_lnStyle)
                    circle.set_fill(False)
                    ax.add_patch(circle)
                elif analytical_sol == 'E':
                    from matplotlib.patches import Ellipse
                    import matplotlib as mpl
                    a = (Solid.K1c[0] / Solid.K1c_perp) ** 2 * R
                    ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * R, angle=360.,
                                                  color=anltcl_lnStyle)
                    ellipse.set_fill(False)
                    ellipse.set_ec(anltcl_lnStyle)
                    ax.add_patch(ellipse)

            # plot regime if enabled
            if plt_regime:
                ribbon_elts = ff.regime[1, :].astype(int)
                patches = []
                for i in range(ribbon_elts.size):
                    polygon = Polygon(np.reshape(ff.mesh.VertexCoor[ff.mesh.Connectivity[ribbon_elts[i]], :], (4, 2)),
                                      True)
                    patches.append(polygon)

                p = PatchCollection(patches)

                # applying colors for regime
                regime = ff.regime[0, :]
                regime[np.where(regime > 1)[0]] = np.nan
                regime[np.where(regime < 0)[0]] = np.nan
                colors = regime
                p.set_array(np.array(colors))

                p.set_clim(0., 1.)
                ax.add_collection(p)

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if plt_regime:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable.
        sm._A = []
        clr_bar = plt.colorbar(sm)
        clr_bar.set_label("regime")
        plt.axis('equal')

    if fileNo >= 5000:
        raise SystemExit("too many files.")

    if plt_mesh:
        if Sim_prop is not None:
            SimulProp = Sim_prop
        ff.plot_fracture(parameter='mesh', mat_properties=Solid, sim_properties=SimulProp, fig=fig)

    ax.set_title("Fracture footprint")

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_radius(address=None, r_type='mean', fig_r=None, fig_err=None, plot_at_times=None, time_period=0., loglog=True,
                plt_symbol='.', analytical_sol='n', anltcl_lnStyle='k', plt_error=True, error_lnStyle='bo-',
                add_labels=True):
    """
        This function plots the footprints of the fractures saved in the given folder.

        Arguments:
            address (string)                -- the folder address containing the saved files
            r_type (string)                 -- specifies the radius to plot. Possible options:
                                                    'max'   -- plots the distance of the furthest point on the fracture
                                                               front from the injection point.
                                                    'mmean' -- plots the mean distance of the points on the fracture
                                                               front from the injection point.
                                                    'min'   -- plots the distance of the closest point on the fracture
                                                               front from the injection point.
            fig_r (figure)                  -- figure to superimpose. A new figure will be created if not provided.
            time_period (float)             -- time period between two successive fracture plots.
            plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
            analytical_sol (string)         -- the following options can be provided
                                                    'M'     -- radial fracture in viscosity dominated regime
                                                    'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                    'K'     -- radial fracture in toughness dominated regime
                                                    'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                    'E'     -- elliptical fracture in toughness dominated regime
                                                    'PKN'   -- PKN fracture
            plt_symbol (string)             -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            anltcl_lnStyle (string)         -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            plt_error (bool)                -- if True, error between numerical and analytical would be plotted

        Returns:
            fig_r (figure)                    -- a figure to superimpose.

    """

    print("Plotting radius of the fracture...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(plot_at_times, np.ndarray)
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    r_numrcl = np.asarray([])
    time_srs = np.asarray([])

    if plt_error:
        if fig_err is None:
            fig_err = plt.figure()
            ax_err = fig_err.add_subplot(111)
        else:
            ax_err = fig_err.get_axes()[0]
    else:
        fig_err = None

    if not analytical_sol is 'n':
        r_anltcl = np.asarray([])
        if plt_error:
            err = np.asarray([])
    elif analytical_sol is 'n':
        plt_error = False
    else:
        raise ValueError("Analytical solution type not supported")

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            break
        prev_modified_at = stats[-2]

        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            time_srs = np.append(time_srs, ff.time)
            tipVrtxCoord = ff.mesh.VertexCoor[ff.mesh.Connectivity[ff.EltTip, ff.ZeroVertex]]
            if r_type is 'mean':
                r_numrcl = np.append(r_numrcl, np.mean((tipVrtxCoord[:, 0]**2 + tipVrtxCoord[:, 1] ** 2)**0.5 + ff.l))
            elif r_type is 'max':
                r_numrcl = np.append(r_numrcl, max((tipVrtxCoord[:, 0]**2 + tipVrtxCoord[:, 1] ** 2)**0.5 + ff.l))
            elif r_type is 'min':
                r_numrcl = np.append(r_numrcl, min((tipVrtxCoord[:, 0]**2 + tipVrtxCoord[:, 1] ** 2)**0.5 + ff.l))
            else:
                raise ValueError("Radius type not supported!")

            if not analytical_sol is 'n':
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                ff.mesh,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")


                if r_type is 'max' and analytical_sol is 'E':
                    R = (Solid.K1c[ff.mesh.CenterElts] / Solid.K1c_perp)**2 * R
                    r_anltcl = np.append(r_anltcl, R)
                else:
                    r_anltcl = np.append(r_anltcl, R)

                if plt_error:
                    err = np.append(err, abs(R - r_numrcl[-1]) / R)

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if fig_r is None:
        fig_r = plt.figure()
        ax = fig_r.add_subplot(111)
    else:
        ax = fig_r.get_axes()[0]

    if loglog:
        if not analytical_sol is 'n':
            # ax.semilogx(time_srs, r_anltcl, anltcl_lnStyle)
            ax.loglog(time_srs, r_anltcl, anltcl_lnStyle, label='radius analytical')
        # ax.semilogx(time_srs, r_numrcl, plt_symbol)
        ax.loglog(time_srs, r_numrcl, plt_symbol, label='radius numerical')
    else:
        if not analytical_sol is 'n':
            ax.plot(time_srs, r_anltcl, anltcl_lnStyle, label='radius analytical')
        ax.plot(time_srs, r_numrcl, plt_symbol, label='radius numerical')
    if add_labels:
        ax.set_ylabel('radius')
        ax.set_xlabel('time')
        ax.set_title(r_type + ' distance from injection point')
        ax.legend()

    if plt_error:
        ax_err.semilogx(time_srs, err, error_lnStyle, label='error on radius')
        if add_labels:
            ax_err.set_ylabel('error')
            ax_err.set_xlabel('time')
            ax_err.set_title('Relative error on radius')
            ax_err.legend()


    return fig_r, fig_err

#-----------------------------------------------------------------------------------------------------------------------


def plot_leakOff(address=None, fig_lk=None, fig_eff=None, plot_at_times=None, time_period=0., loglog=True, plt_symbol='.',
                analytical_sol='n', anltcl_lnStyle='k', plt_efficiency=True, add_labels=True):
    """
        This function plots the footprints of the fractures saved in the given folder.

        Arguments:
            address (string)                -- the folder address containing the saved files
            fig_lk (figure)                 -- leaked off volume figure to superimpose. A new figure will be created if
                                               not provided.
            fig_eff (figure)                -- fracturing efficiency figure to superimpose. A new figure will be
                                               created if not provided.
            time_period (float)             -- time period between two successive fracture plots.
            plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
            analytical_sol (string)         -- the following options can be provided
                                                    'M'     -- radial fracture in viscosity dominated regime
                                                    'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                    'K'     -- radial fracture in toughness dominated regime
                                                    'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                    'E'     -- elliptical fracture in toughness dominated regime
                                                    'PKN'   -- PKN fracture
            plt_symbol (string)             -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            anltcl_lnStyle (string)         -- the line style of the analytical solution lines (e.g. '.k-' for a black
                                               continous line with data points marked with dots )
            plt_efficiency (bool)           -- if True, fracturing efficiency would be plotted

        Returns:
            fig_lk (figure)                    -- a figure to superimpose leaked off volume.
            fig_eff (figure)                   -- a figure to superimpose fracturing efficiency.
        """
    print("Plotting leak off from the fracture...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(plot_at_times, np.ndarray)
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    lk_numrcl = np.asarray([])
    efficiency = np.asarray([])
    time_srs = np.asarray([])

    if not analytical_sol is 'n':
        lk_anltcl = np.asarray([])

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    # loop to load fracture files
    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            break
        prev_modified_at = stats[-2]

        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            time_srs = np.append(time_srs, ff.time)

            if analytical_sol is not 'n':
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                ff.mesh,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)

                    leaked_off_anltcl = Injection.injectionRate[1, 0] * t - sum(w) * ff.mesh.EltArea
                    lk_anltcl = np.append(lk_anltcl, leaked_off_anltcl)
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")

            leaked_off = sum(ff.LkOff_vol[ff.EltCrack])
            lk_numrcl = np.append(lk_numrcl, leaked_off)
            efficiency = np.append(efficiency, ff.efficiency)

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if fig_lk is None:
        fig_lk = plt.figure()
        ax_lk = fig_lk.add_subplot(111)
    else:
        ax_lk = fig_lk.get_axes()[0]

    if loglog:
        if not analytical_sol is 'n':
            # ax_lk.semilogx(time_srs, r_anltcl, anltcl_lnStyle)
            ax_lk.loglog(time_srs, lk_anltcl, anltcl_lnStyle, label='leaked off analytical')
        # ax_lk.semilogx(time_srs, r_numrcl, plt_symbol)
        ax_lk.loglog(time_srs, lk_numrcl, plt_symbol, label='leaked off numerical')
    else:
        if not analytical_sol is 'n':
            ax_lk.plot(time_srs, lk_anltcl, anltcl_lnStyle, label='radius analytical')
        ax_lk.plot(time_srs, lk_numrcl, plt_symbol, label='radius numerical')
    if add_labels:
        ax_lk.set_ylabel('leaked off volume (m^3)')
        ax_lk.set_xlabel('time')
        ax_lk.set_title(' Leaked off volume')
        ax_lk.legend()

    if plt_efficiency:
        if fig_eff is None:
            fig_eff = plt.figure()
            ax_eff = fig_eff.add_subplot(111)
        else:
            ax_eff = fig_eff.get_axes()[0]

        ax_eff.semilogx(time_srs, efficiency, plt_symbol, label='hydraulic fracturing efficiency numerical')
        if add_labels:
            ax_eff.set_ylabel('efficiency')
            ax_eff.set_xlabel('time')
            ax_eff.set_title('Hydraulic fracturing efficiency')
            ax_eff.legend()


    return fig_lk, fig_eff

#-----------------------------------------------------------------------------------------------------------------------

def plot_simulation_results(address=None, plot_at_times=None, time_period=0., analytical_sol='n', footprint=True, inj_pnt=True,
                           radius=True,  profile=True):
    """
    This function plot the simulation results from the given folder
    Arguments:
            address (string)                -- the folder address containing the saved files
            plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
            time_period (float)             -- time period between two successive fracture plots.
            analytical_sol (string)         -- the following options can be provided
                                                    'M'     -- radial fracture in viscosity dominated regime
                                                    'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                    'K'     -- radial fracture in toughness dominated regime
                                                    'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                    'E'     -- elliptical fracture in toughness dominated regime
                                                    'PKN'   -- PKN fracture
            footprint (boolean)             -- if True, fracture footprints will be plotted.
            inj_pnt (boolean)               -- if True, fracture fracture width and pressure at the injection points
                                               will be plotted.
            radius (boolean)                -- if True, mean distance of the fracture front from the injection point
                                               will be plotted.
            profile (boolean)               -- if True, width profile at the x and y axes will be plotted.

        Returns:

    """

    if footprint:
        plot_footprint(address=address,
                                plot_at_times=plot_at_times,
                                time_period=time_period,
                                analytical_sol=analytical_sol)
    if inj_pnt:
        plot_at_injection_point(address=address,
                                plot_at_times=plot_at_times,
                                time_period=time_period,
                                analytical_sol=analytical_sol)
    if radius:
        plot_radius(address=address,
                                plot_at_times=plot_at_times,
                                time_period=time_period,
                                analytical_sol = analytical_sol)
    if profile:
        plot_profile(address=address,
                                plot_at_times=plot_at_times,
                                time_period=time_period,
                                analytical_sol=analytical_sol)


#-----------------------------------------------------------------------------------------------------------------------


def plot_footprint_3d(address=None, fig=None, time_period=0.0, plot_at_times=None, plt_time=True, txt_size=None,
                      plt_axis=False, plt_grid=False, plt_mesh=True, plt_bckColor=True, alternate=False,
                      plt_clr_bar=True, disp_precision=3):
    """
    This function plots the footprints of the fractures saved in the given folder.

    Arguments:
        address (string)                -- the folder address containing the saved files
        fig (figure)                    -- figure to superimpose. A new figure will be created if not provided.
        time_period (float)             -- time period between two successive fracture plots.
        plot_at_times (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
        plt_time (bool)                 -- if True, time will be displayed close to each of the front.
        txt_size (float)                -- the size of text to display time, lengths etc.
        plt_axis (bool)                 -- if True, axis will be plotted.
        plt_grid (bool)                 -- if True, grid will also be plotted.
        plt_mesh (bool)                 -- if True, mesh will be plotted on the plain containing the fracture.
        plt_bckColor (bool)             -- if True, the mesh will be color coded according to the parameter specified
                                           in the loaded parameter properties object from the given address.
        alternate (bool)                -- if True, the time will be displayed alternatively at the furthest and closest
                                           points of the front from the injection point.
        plt_clr_bar (bool)              -- if True, the color bar will generated.
        disp_precision (int)            -- the precision upto which the time, length etc. are displayed

    Returns:
        fig (figure)                    -- a figure to superimpose.

    """

    print("Plotting the fracture evolution...")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    if isinstance(plot_at_times, float) or isinstance(plot_at_times, int):
        plot_at_times = np.array([plot_at_times])

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0

    t_srs_given = isinstance(plot_at_times, np.ndarray) #time series is given
    if t_srs_given:
        nxt_plt_t = plot_at_times[t_srs_indx]

    # new figure if not provided
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax = fig.get_axes()[0]
    ax.set_frame_on(False)

    # time at wich the first fracture file was modified
    stats = os.stat(address + "fracture_0")
    prev_modified_at = stats[-2]

    printed_fronts = 0
    ff = None
    while fileNo < 5000:

        # saving last fracture in case the loaded file is to be discarded (possibly its from an old simulation)
        ff_last = copy.deepcopy(ff)

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "fracture_" + repr(fileNo))
        except FileNotFoundError:
            break

        stats = os.stat(address + "fracture_" + repr(fileNo))
        # if the next file was modified before the last one, it means it is from some older simulation
        if stats[-2] < prev_modified_at:
            ff = ff_last
            break
        prev_modified_at = stats[-2]

        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:

            # if the current fracture time has advanced the output time period
            I = ff.Ffront[:, 0:2]
            J = ff.Ffront[:, 2:4]

            # draw front lines
            for e in range(0, len(I)):
                Path = mpath.Path
                path_data = [
                    (Path.MOVETO, [I[e, 0], I[e, 1]]),
                    (Path.LINETO, [J[e, 0], J[e, 1]])]

                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(path, lw=1)
                ax.add_patch(patch)
                art3d.pathpatch_2d_to_3d(patch)

            printed_fronts += 1

            # plot width
            fig = ff.plot_fracture(parameter='width', elts='crack', fig=fig, alpha= 0.3)
            # ax.set_zlim([min(ff.w) - min(ff.w) * 0.25, max(ff.w) + max(ff.w) * 0.25])

            # print time close to the front edge
            if plt_time:
                tipVrtxCoord = ff.mesh.VertexCoor[ff.mesh.Connectivity[ff.EltTip, ff.ZeroVertex]]
                if printed_fronts % 2 == 0:
                    r_indx = np.argmax((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + ff.l)
                elif alternate:
                    r_indx = np.argmin((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + ff.l)
                else:
                    r_indx = np.argmax((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + ff.l)
                x_coor = ff.mesh.CenterCoor[ff.EltTip[r_indx], 0] + 0.75 * ff.mesh.hx
                y_coor = ff.mesh.CenterCoor[ff.EltTip[r_indx], 1] + 0.75 * ff.mesh.hy
                if txt_size is None:
                    txt_size = max(ff.mesh.hx, ff.mesh.hx)
                print("Plotting at time " + repr(ff.time) + "...")
                t = to_precision(ff.time, disp_precision)
                text3d(ax,
                       (x_coor, y_coor, 0),
                       t + "$sec$",
                       zdir="z",
                       size=txt_size,
                       usetex=True,
                       ec="none",
                       fc="k")

            if t_srs_given:
                if t_srs_indx < len(plot_at_times) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = plot_at_times[t_srs_indx]
                if ff.time > max(plot_at_times):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if not plt_grid:
        ax.grid(False)

    # print the grid. If grid is not printed, lines are drawn to show the length scale of the fracture
    if not plt_axis:
        ax.set_axis_off()

        max_x = max(I[:, 0])
        max_y = max(I[:, 1])
        min_x = min(I[:, 0])
        min_y = min(I[:, 1])
        Path = mpath.Path
        path_data = [
            (Path.MOVETO, [min_x, min_y - 2 * ff.mesh.hy]),
            (Path.LINETO, [max_x, min_y - 2 * ff.mesh.hy]),
            (Path.MOVETO, [min_x, min_y - 2.5 * ff.mesh.hy]),
            (Path.LINETO, [min_x, min_y - 1.5 * ff.mesh.hy]),
            (Path.MOVETO, [max_x, min_y - 2.5 * ff.mesh.hy]),
            (Path.LINETO, [max_x, min_y - 1.5 * ff.mesh.hy]),
            (Path.MOVETO, [min_x - 2.5 * ff.mesh.hx, min_y - ff.mesh.hy]),
            (Path.LINETO, [min_x - 2.5 * ff.mesh.hx, max_y]),
            (Path.MOVETO, [min_x - 3. * ff.mesh.hx, min_y - ff.mesh.hy]),
            (Path.LINETO, [min_x - 2. * ff.mesh.hx, min_y - ff.mesh.hy]),
            (Path.MOVETO, [min_x - 3. * ff.mesh.hx, max_y]),
            (Path.LINETO, [min_x - 2. * ff.mesh.hx, max_y]),
        ]

        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, lw=1, facecolor='none')
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch)
        if txt_size is None:
            txt_size = max(ff.mesh.hx, ff.mesh.hx)
        y_len = to_precision(max_y - min_y + ff.mesh.hy, disp_precision)
        text3d(ax,
               (min_x - 2.5 * ff.mesh.hx - 5 * txt_size, (max_y + min_y) / 2, 0),
               y_len + "$m$",
               zdir="z",
               size=txt_size,
               usetex=True,
               ec="none",
               fc="k")
        x_len = to_precision(max_x - min_x + ff.mesh.hy, disp_precision)
        text3d(ax,
               ((max_x + min_x) / 2, min_y - 2 * ff.mesh.hy - 2 * txt_size, 0),
               x_len + "$m$",
               zdir="z",
               size=txt_size,
               usetex=True,
               ec="none",
               fc="k")
    else:
        # ax.set_xbound(-ff.mesh.Lx*1.2,ff.mesh.Lx*1.2)
        ax.set_xticks(np.linspace(-ff.mesh.Lx*1.2,ff.mesh.Lx*1.2,5))
        ax.set_yticks(np.linspace(-ff.mesh.Ly, ff.mesh.Ly, 5))

    # plot mesh with the color superimposed according to the given parameter
    if plt_mesh:
        print("Plotting mesh...")
        if SimulProp is not None:
            if SimulProp.bckColor == 'sigma0':
                max_bck = max(Solid.SigmaO) / 1e6
                min_bck = min(Solid.SigmaO) / 1e6
                if max_bck - min_bck > 0:
                    colors = (Solid.SigmaO / 1e6 - min_bck) / (max_bck - min_bck)
                    parameter = "confining stress ($MPa$)"
            elif SimulProp.bckColor == 'K1c':
                max_bck = max(Solid.K1c)/ 1e6
                min_bck = min(Solid.K1c)/ 1e6
                if max_bck - min_bck > 0:
                    colors = (Solid.K1c/ 1e6 - min_bck) / (max_bck - min_bck)
                    parameter = "fracture toughness ($Mpa\sqrt{m})"
            elif SimulProp.bckColor == 'Cl':
                max_bck = max(Solid.Cl)
                min_bck = min(Solid.Cl)
                if max_bck - min_bck > 0:
                    colors = (Solid.Cl - min_bck) / (max_bck - min_bck)
                    parameter = "Carter's Leak off"
            elif not SimulProp.bckColor is None:
                raise ValueError("Back ground color identifier not supported!")
            else:
                colors = np.full((ff.mesh.NumberOfElts, ), 0.5)
                plt_mesh = False

        # add rectangle for each cell
        for i in range(ff.mesh.NumberOfElts):
            if plt_bckColor:
                face_color = (0, colors[i], 0, 0.2)
            else:
                face_color = (0, 0.0, 0, 0.2)
            cell = mpatches.Rectangle((ff.mesh.CenterCoor[i, 0],
                                      ff.mesh.CenterCoor[i, 1]),
                                      ff.mesh.hx,
                                      ff.mesh.hy,
                                      ec=(0, 0, 0, 0.05),
                                      fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)

    if plt_clr_bar and plt_bckColor and plt_mesh:
        print("Making colorbar...")
        clr_range = np.linspace(0, 1., 11)
        y = np.linspace(min_y, max_y, 11)
        dy = y[1] - y[0]
        for i in range(11):
            face_color = (0, clr_range[i], 0, 0.2)
            cell = mpatches.Rectangle((ff.mesh.Lx + 4 * ff.mesh.hx,
                                      y[i]),
                                      2 * dy,
                                      dy,
                                      ec=(0, 0, 0, 0.05),
                                      fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)

        text3d(ax,
               (ff.mesh.Lx + 4 * ff.mesh.hx, y[9] + 3 * dy, 0),
               parameter,
               zdir="z",
               size=txt_size,
               usetex=True,
               ec="none",
               fc="k")
        y = [y[0], y[5], y[10]]
        values = np.linspace(min_bck, max_bck, 11)
        values = [values[0], values[5], values[10]]
        for i in range(3):
            text3d(ax,
                   (ff.mesh.Lx + 4 * ff.mesh.hx + 2 * dy, y[i] + dy/2, 0),
                   "%.2f" % values[i],
                   zdir="z",
                   size=txt_size,
                   usetex=True,
                   ec="none",
                   fc="k")

    # mark maximum width for scale
    w_max = to_precision(max(ff.w) * 1e3, disp_precision)
    w_max_indx = np.argmax(ff.w)
    text3d(ax,
           (ff.mesh.CenterCoor[w_max_indx,0], ff.mesh.CenterCoor[w_max_indx,1], max(ff.w)),
           w_max + "$mm$",
           zdir="z",
           size=txt_size,
           # angle=np.pi / 2,
           usetex=True,
           ec="none",
           fc="k")

    ax.axis('equal')
    ax.set_frame_on(False)
    ax.set_zlim([min(ff.w)-min(ff.w)*0.25, max(ff.w)+max(ff.w)*0.25])
    ax.set_title("Fracture evolution")
    scale = 1.1
    zoom_factory(ax, base_scale=scale)

    return fig

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
