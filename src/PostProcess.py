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
import numpy as np
from src.CartesianMesh import *
from src.Fracture import *
from src.Properties import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.colors as mcolors




def animate_simulation_results(address, time_period= 0.0, sol_time_series=None, Interval=400, Repeat=None,
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

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        #todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)


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
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
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


def plot_profile(address, fig_w_x=None, fig_w_y=None, fig_p_x=None, fig_p_y=None, plt_pressure=False,
                 time_period=0.0, sol_t_srs=None, analytical_sol='n', plt_color='k.', analytical_color='b'):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        #todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    if fig_w_x is None:
        fig_w_x = plt.figure()
    ax_w_x = fig_w_x.add_subplot(111)

    if fig_w_y is None:
        fig_w_y = plt.figure()
    ax_w_y = fig_w_y.add_subplot(111)

    if plt_pressure:
        if fig_p_x is None:
            fig_p_x = plt.figure()
        ax_p_x = fig_p_x.add_subplot(111)
        if fig_p_y is None:
            fig_p_y = plt.figure()
        ax_p_y = fig_p_y.add_subplot(111)

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
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

            # cells on x and y axes
            hrzntl = np.where(abs(ff.mesh.CenterCoor[:, 1]) < 1e-10)[0]
            x = ff.mesh.CenterCoor[hrzntl, 0]
            vrtcl = np.where(abs(ff.mesh.CenterCoor[:, 0]) < 1e-10)[0]
            y = ff.mesh.CenterCoor[vrtcl, 1]

            line_wx_num, = ax_w_x.plot(x, ff.w[hrzntl],plt_color)
            line_wy_num, = ax_w_y.plot(y, ff.w[vrtcl], plt_color)
            if not analytical_sol is 'n':
                line_wx_anl, = ax_w_x.plot(x, w[hrzntl],analytical_color)
                line_wy_anl, = ax_w_y.plot(y, w[vrtcl], analytical_color)

            ax_w_x.set_ylabel('width')
            ax_w_x.set_xlabel('meters')
            ax_w_x.set_title('Width profile along x-axis')

            ax_w_y.set_ylabel('width')
            ax_w_y.set_xlabel('meters')
            ax_w_y.set_title('Width profile along y-axis')

            if plt_pressure:
                line_px_num, = ax_p_x.plot(x, ff.p[hrzntl], plt_color)
                line_py_num, = ax_p_y.plot(y, ff.p[vrtcl], plt_color)
                if not analytical_sol is 'n':
                    np.delete(hrzntl, np.where(ff.p[hrzntl]!=0.)[0], 0)
                    line_px_anl, = ax_p_x.plot(x, p[hrzntl], analytical_color)
                    line_py_anl, = ax_p_y.plot(y, p[vrtcl], analytical_color)

                ax_p_x.set_ylabel('pressure')
                ax_p_x.set_xlabel('meters')
                ax_p_x.set_title('Pressure profile along x-axis')

                ax_p_y.set_ylabel('pressure')
                ax_p_y.set_xlabel('meters')
                ax_p_y.set_title('Pressure profile along y-axis')

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if not analytical_sol is 'n':
        ax_w_x.legend((line_wx_num, line_wx_anl),('numerical','analytical'))
        ax_w_y.legend((line_wy_num, line_wy_anl), ('numerical', 'analytical'))
        if plt_pressure:
            ax_p_x.legend((line_px_num, line_px_anl), ('numerical', 'analytical'))
            ax_p_y.legend((line_py_num, line_py_anl), ('numerical', 'analytical'))

    return fig_w_x, fig_w_y, fig_p_x, fig_p_y


#-----------------------------------------------------------------------------------------------------------------------
def plot_at_injection_point(address, fig_w=None, fig_p=None, plt_pressure=False, time_period=0.0, sol_t_srs=None,
                analytical_sol='n', plt_color='r.', analytical_color='b', loglog=False, plt_dimensionless=False):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        #todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    if fig_w is None:
        fig_w = plt.figure()
    ax_w = fig_w.add_subplot(111)

    if plt_pressure:
        if fig_p is None:
            fig_p = plt.figure()
        ax_p = fig_p.add_subplot(111)

    if not analytical_sol is 'n':
        w_anltcl = np.array([], dtype=np.float64)
        time_srs = np.array([], dtype=np.float64)
        if plt_pressure:
            p_anltcl = np.array([], dtype=np.float64)

        fig_err = plt.figure()
        ax_err = fig_err.add_subplot(111)
        w_err = np.array([], dtype=np.float64)
        p_err = np.array([], dtype=np.float64)
    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
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

            if plt_dimensionless:
                tmk = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
                    (32 / math.pi) ** 0.5 * Solid.K1c[0]) ** 18) ** 0.5
                tmk2 = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
                (32 / math.pi) ** 0.5 * Solid.K1c[0]) ** 18) ** 0.5
            else:
                tmk = 1.

            if loglog:
                # ax_w.semilogx(ff.time, ff.w[ff.mesh.CenterElts], plt_color)
                ax_w.loglog(ff.time/tmk, ff.w[ff.mesh.CenterElts], plt_color)
            else:
                ax_w.plot(ff.time/tmk, ff.w[ff.mesh.CenterElts],plt_color)
            ax_w.set_ylabel('width')
            ax_w.set_xlabel('time')
            ax_w.set_title('Width at injection point')

            if not analytical_sol is 'n':
                w_anltcl = np.append(w_anltcl, w[ff.mesh.CenterElts])
                time_srs = np.append(time_srs, ff.time)
                w_err = np.append(w_err, 1. - w[ff.mesh.CenterElts]/ff.w[ff.mesh.CenterElts])


            if plt_pressure:
                if isinstance(ff.p, np.ndarray):
                    p_num = ff.p[ff.mesh.CenterElts]
                else:
                    p_num = ff.p
                if loglog:
                    # ax_p.semilogx(ff.time, p_num, plt_color)
                    ax_p.loglog(ff.time/tmk, p_num, plt_color)
                else:
                    ax_p.plot(ff.time/tmk, p_num, plt_color)


                if not analytical_sol is 'n':
                    if isinstance(p, np.ndarray):
                        p_aa = p[ff.mesh.CenterElts]
                    else:
                        p_aa = p
                    p_anltcl = np.append(p_anltcl, p_aa)
                    p_err = np.append(p_err, (p_aa - p_num)/p_aa)

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if not analytical_sol is 'n':
        if loglog:
            ax_w.semilogx(time_srs/tmk, w_anltcl, analytical_color, label='analytical')
        else:
            ax_w.plot(time_srs/tmk, w_anltcl, analytical_color, label='analytical')
        ax_w.legend()

        if plt_pressure:
            if loglog:
                ax_p.semilogx(time_srs/tmk, p_anltcl, analytical_color, label='analytical')
            else:
                ax_p.plot(time_srs/tmk, p_anltcl, analytical_color, label='analytical')
            ax_p.set_ylabel('pressure')
            ax_p.set_xlabel('time')
            ax_p.set_title('Pressure at injection point')
            ax_p.legend()
    # ax_w.plot(tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000 * tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000, 1e-4, 'k.')
    # print(repr(time_srs))

        ax_err.plot(time_srs, abs(w_err), 'b.', label='error on width')
        if plt_pressure:
            ax_err.plot(time_srs, abs(p_err), 'r.', label='error on pressure')
        ax_err.set_ylabel('error')
        ax_err.set_xlabel('time')
        ax_err.set_title('relative error')
        ax_err.legend()

    return fig_w, fig_p


def plot_footprint(address, fig=None, time_period=0.0, sol_t_srs=None, analytical_sol='n',
                            plt_color='k', analytical_color='b', plt_mesh=True, plt_regime=False, Sim_prop=None):
    """
    This function plots the footprints of the fractures saved in the given folder.

    Arguments:
        address (string)                -- the folder address containing the saved files
        fig (figure)                    -- figure to superimpose. A new figure will be created if not provided.
        time_period (float)             -- time period between two successive fracture plots.
        sol_t_srs (ndarray)             -- if provided, the fracture footprints will be plotted at the given times.
        analytical_sol (string)         -- the following options can be provided
                                                'M'     -- radial fracture in viscosity dominated regime
                                                'Mt'    -- radial fracture in viscosity dominated regime with leak-off
                                                'K'     -- radial fracture in toughness dominated regime
                                                'Kt'    -- radial fracture in toughness dominated regime with leak-off
                                                'E'     -- elliptical fracture in toughness dominated regime
                                                'PKN'   -- PKN fracture
        plt_color (string)              -- the color(matplotlib colors) of the fracture lines.
        analytical_color (string)       -- the color(matplotlib colors) of the analytical solution lines.
        plt_mesh (boolean)              -- if true, mesh will also be plotted.
        plt_regime (boolean)            -- if true, regime evaluated at the ribbon cells will be ploted (see Zia and
                                           Lecampion 2018)
        Sim_prop (SimulationParameters) -- if provided, the simulation properties file read will be overriden by these
                                           parameters.

    Returns:
        fig (figure)                    -- a figure to superimpose.

    """

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        #todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0

    t_srs_given = isinstance(sol_t_srs, np.ndarray) #time series is given
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    # new figure if not provided
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period
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
                    circle.set_ec(analytical_color)
                    circle.set_fill(False)
                    ax.add_patch(circle)
                elif analytical_sol == 'E':
                    from matplotlib.patches import Ellipse
                    import matplotlib as mpl
                    a = (Solid.K1c[0] / Solid.K1c_perp) ** 2 * R
                    ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * R, angle=360.,
                                                  color=analytical_color)
                    ellipse.set_fill(False)
                    ellipse.set_ec(analytical_color)
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
                regime[np.where(regime < 0)[0]] = 0
                colors = regime
                p.set_array(np.array(colors))

                p.set_clim(0., 1.)
                ax.add_collection(p)

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if fileNo >= 5000:
        raise SystemExit("too many files.")

    if plt_mesh:
        if not Sim_prop is None:
            SimulProp = Sim_prop
        ff.plot_fracture(parameter='mesh', mat_properties=Solid, sim_properties=SimulProp, fig=fig)

    return fig

#-----------------------------------------------------------------------------------------------------------------------

def plot_radius(address, r_type='mean', fig_r=None, sol_t_srs=None, time_period=0.,
                loglog=True, plt_symbol='o', analytical_sol='E', anltcl_clr='k'):
    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        # todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    r_numrcl = np.asarray([])
    r_anltcl = np.asarray([])
    time_srs = np.asarray([])

    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
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

                r_anltcl = np.append(r_anltcl, R)

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if fig_r is None:
        fig_r = plt.figure()
        ax = fig_r.add_subplot(111)
    ax = fig_r.add_subplot(111)
    if loglog:
        if not analytical_sol is 'n':
            # ax.semilogx(time_srs, r_anltcl, anltcl_clr)
            ax.loglog(time_srs, r_anltcl, anltcl_clr, label='radius analytical')
        # ax.semilogx(time_srs, r_numrcl, plt_symbol)
        ax.loglog(time_srs, r_numrcl, plt_symbol, label='radius numerical')
    else:
        if not analytical_sol is 'n':
            ax.plot(time_srs, r_anltcl, anltcl_clr, label='radius analytical')
        ax.plot(time_srs, r_numrcl, plt_symbol, label='radius numerical')

    plt.ylabel('radius')
    plt.xlabel('time')
    plt.title('Distance from injection point')
    plt.legend()

    return fig_r

#-----------------------------------------------------------------------------------------------------------------------


def plt_simulation_results(address, sol_t_srs=None, time_period=0., analytical_sol='n', footprint=True, inj_pnt=True,
                           radius=True,  profile=True):

    if footprint:
        plot_footprint(address=address,
                                sol_t_srs=sol_t_srs,
                                time_period=time_period,
                                analytical_sol=analytical_sol)
    if inj_pnt:
        plot_at_injection_point(address=address,
                                sol_t_srs=sol_t_srs,
                                time_period=time_period,
                                analytical_sol=analytical_sol)
    if radius:
        plot_radius(address=address,
                                sol_t_srs=sol_t_srs,
                                time_period=time_period,
                                analytical_sol = analytical_sol)
    if profile:
        plot_profile(address=address,
                                sol_t_srs=sol_t_srs,
                                time_period=time_period,
                                analytical_sol=analytical_sol)

