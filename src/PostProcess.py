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
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors




def animate_simulation_results(address, time_period= 0.0, sol_time_series=None, colormap=cm.jet, edge_color = '0.5', Interval=400,
                               Repeat=None, maxFiles=1000 ):
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


    fileNo = 0
    fraclist = [];
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

    #todo decide weather mesh should be kept seperate from fracture
    Mesh = ff.mesh

    fig, ax = plt.subplots()
    ax.set_xlim([-Mesh.Lx, Mesh.Lx])
    ax.set_ylim([-Mesh.Ly, Mesh.Ly])

    # make grid cells
    patches = []
    for i in range(Mesh.NumberOfElts):
        polygon = Polygon(np.reshape(Mesh.VertexCoor[Mesh.Connectivity[i], :], (4, 2)), True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=colormap, alpha=0.65, edgecolor=edge_color)

    # applying different colors for different types of elements
    # todo needs to be done properly
    colors = 100. * np.full(len(patches), 0.9)
    if np.max(Solid.SigmaO) > 0:
        colors += -100. * (Solid.SigmaO) / np.max(Solid.SigmaO)
    if np.max(Solid.Kprime) > 0:
        colors += -100. * (Solid.Kprime) / np.max(Solid.Kprime)

    p.set_array(np.array(colors))
    ax.add_collection(p)



    args = (fraclist, fileNo, Solid, Fluid, Injection)
    # animate fracture
    animation = FuncAnimation(fig,
                              update,
                              fargs=args,
                              frames=len(fraclist),
                              interval=Interval,
                              repeat=Repeat,
                              repeat_delay=1000)  # ,extra_args=['-vcodec', 'libxvid']

    # animation.save(address + 'Footprint-evol.mp4', metadata={'copyright':'EPFL - GeoEnergy Lab'})
    plt.show()

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

    # R, a, w, p = anisotropic_toughness_elliptical_solution(Solid.K1c,
    #                                                        Solid.K1c_perp,
    #                                                        Solid.Eprime,
    #                                                        Injection.injectionRate[1, 0],
    #                                                        ffi.mesh,
    #                                                        t=ffi.time)
    # from matplotlib.patches import Ellipse
    # import matplotlib as mpl
    # ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * R, angle=360., color='r')
    # # ellipse.set_clip_box(ax.bbox)
    # ellipse.set_fill(False)
    # ellipse.set_ec('r')
    # fig, ax = plt.subplots()
    # ax.add_patch(ellipse)
    plt.title('Time ='+ "%.4f" % ffi.time+ ' sec.')
    plt.axis('equal')



def plot_simulation_results(address, fig = None, time_period= 0.0, sol_time_series=None, maxFiles=2000,
        plot_analytical=False, analytical_sol='M', plt_color='b', analytical_color = 'k', plt_mesh=False,
        mesh_clr_map=cm.jet, mesh_edge_color='0.5', plt_regime=False, clr_bar=False, ln_style='-'):
    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    #todo: do not make a fracture list. Plot as you read to save memory
    fileNo = 0
    fraclist = [];
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
        # print(repr(fileNo)+' '+repr(ff.time))

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

    #todo mesh seperate from fracture
    Mesh = ff.mesh   # because Mesh is not stored in a separate file for now

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)


    for ffi in fraclist:
        tmk = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
        (32 / math.pi) ** 0.5 * Solid.K1c_perp) ** 18) ** 0.5
        print(repr(tmk))
        I = ffi.Ffront[:, 0:2]
        J = ffi.Ffront[:, 2:4]

        for e in range(0, len(I)):
            ax.plot(np.array([I[e, 0], J[e, 0]]), np.array([I[e, 1], J[e, 1]]), color=plt_color, ls=ln_style)

        if plot_analytical:
            if analytical_sol == 'E':
                R, a, w, p = anisotropic_toughness_elliptical_solution(Solid.K1c,
                                                                       Solid.K1c_perp,
                                                                       Solid.Eprime,
                                                                       Injection.injectionRate[1, 0],
                                                                       ffi.mesh,
                                                                       t=ffi.time)
                from matplotlib.patches import Ellipse
                import matplotlib as mpl
                ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=2 * a, height=2 * R, angle=360., color=analytical_color)
                # ellipse.set_clip_box(ax.bbox)
                ellipse.set_fill(False)
                ellipse.set_ec(analytical_color)
                ax.add_patch(ellipse)

            elif analytical_sol == 'M':
                R, p, w, v = M_vertex_solution_t_given(Solid.Eprime,
                                                       Injection.injectionRate[1, 0],
                                                       Fluid.muPrime,
                                                       ffi.mesh,
                                                       ffi.time)
                circle = plt.Circle((0, 0), radius=R)
                circle.set_ec(analytical_color)
                circle.set_fill(False)
                ax.add_patch(circle)

            elif analytical_sol == "K":
                R, p, w, v = K_vertex_solution_t_given(Solid.Kprime,
                                                       Solid.Eprime,
                                                       Injection.injectionRate[1, 0],
                                                       ffi.mesh,
                                                       ffi.time)
                circle = plt.Circle((0, 0), radius=R)
                circle.set_ec(analytical_color)
                circle.set_fill(False)
                ax.add_patch(circle)
    if plt_mesh:
        ax.set_xlim([-Mesh.Lx, Mesh.Lx])
        ax.set_ylim([-Mesh.Ly, Mesh.Ly])

        # make grid cells
        patches = []
        for i in range(Mesh.NumberOfElts):
            polygon = Polygon(np.reshape(Mesh.VertexCoor[Mesh.Connectivity[i], :], (4, 2)), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=mesh_clr_map, alpha=0.65, edgecolor=mesh_edge_color)

        # applying different colors for different types of elements
        # todo needs to be done properly
        colors = 100. * np.full(len(patches), 0.9)
        if np.max(Solid.SigmaO) > 0:
            colors += -100. * (Solid.SigmaO) / np.max(Solid.SigmaO)
        if np.max(Solid.Kprime) > 0:
            colors += -100. * (Solid.Kprime) / np.max(Solid.Kprime)

        p.set_array(np.array(colors))
        ax.add_collection(p)

    if plt_regime:

        for ffi in fraclist:
            ribbon_elts = ffi.regime[1,:].astype(int)
            patches = []
            for i in range(ribbon_elts.size):
                polygon = Polygon(np.reshape(ffi.mesh.VertexCoor[ffi.mesh.Connectivity[ribbon_elts[i]], :], (4, 2)), True)
                patches.append(polygon)

            p = PatchCollection(patches, cmap=mesh_clr_map)

            c = mcolors.ColorConverter().to_rgb
            rvb = make_colormap(
                [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])

            # s = ffi.sgndDist[ffi.EltRibbon]
            # eta = s/(Solid.)

            # applying colors for regime
            regime = ffi.regime[0,:]
            regime[np.where(regime<0)[0]]=0
            colors = regime
            p.set_array(np.array(colors))
            # p.set_cmap(rvb)
            p.set_clim(0., 1.)
            ax.add_collection(p)
        sm = plt.cm.ScalarMappable(cmap=mesh_clr_map, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable.
        sm._A = []
        if clr_bar:
            plt.colorbar(sm)

    # plt.axis('equal')
    return fig


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

#-----------------------------------------------------------------------------------------------------------------------

def plot_profile(address, fig_w_a=None, fig_w_b=None, fig_p_a=None, fig_p_b=None, plt_pressure=False,
                 time_period=0.0, sol_time_series=None, maxFiles=1000, plt_analytical=False, analytical_sol='M',
                 plt_color='k', analytical_color='b'):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_time_series, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_time_series[t_srs_indx]

    if fig_w_a is None:
        fig_w_a = plt.figure()
    ax_w_a = fig_w_a.add_subplot(111)

    if fig_w_b is None:
        fig_w_b = plt.figure()
    ax_w_b = fig_w_b.add_subplot(111)

    if plt_pressure:
        if fig_p_a is None:
            fig_p_a = plt.figure()
        ax_p_a = fig_p_a.add_subplot(111)
        if fig_p_b is None:
            fig_p_b = plt.figure()
        ax_p_b = fig_p_b.add_subplot(111)

    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period
            tmk = (Solid.Eprime ** 13 * Fluid.muPrime ** 5 * Injection.injectionRate[1, 0] ** 3 / (
            (32 / math.pi) ** 0.5 * Solid.K1c_perp) ** 18) ** 0.5
            print(repr(ff.time/tmk))
            if plt_analytical:
                if analytical_sol == 'E':
                    R, a, w, p = anisotropic_toughness_elliptical_solution(Solid.K1c,
                                                                           Solid.K1c_perp,
                                                                           Solid.Eprime,
                                                                           Injection.injectionRate[1, 0],
                                                                           ff.mesh,
                                                                           t=ff.time)

                elif analytical_sol == 'M':
                    R, p, w, v = M_vertex_solution_t_given(Solid.Eprime,
                                                           Injection.injectionRate[1, 0],
                                                           Fluid.muPrime,
                                                           ff.mesh,
                                                           ff.time)
                    # w[ff.mesh.CenterElts] = np.nan
                    # (minx, miny) = (min(abs(ff.mesh.CenterCoor[:, 0])), min(abs(ff.mesh.CenterCoor[:, 1])))
                    CenterElts = np.intersect1d(np.where(abs(ff.mesh.CenterCoor[:, 0]) < 1e-15)[0],
                                                np.where(abs(ff.mesh.CenterCoor[:, 1]) < 1e-15)[0])
                    # p[CenterElts] = np.nan
                    # p[np.where(p==0)[0]] = np.nan

                elif analytical_sol == "K":
                    R, p, w, v = K_vertex_solution_t_given(Solid.Kprime,
                                                           Solid.Eprime,
                                                           Injection.injectionRate[1, 0],
                                                           ff.mesh,
                                                           ff.time)

            hrzntl = np.where(abs(ff.mesh.CenterCoor[:, 1]) < 1e-8)[0]
            x = ff.mesh.CenterCoor[hrzntl, 0]
            vrtcl = np.where(abs(ff.mesh.CenterCoor[:, 0]) < 1e-8)[0]
            y = ff.mesh.CenterCoor[vrtcl, 1]

            ax_w_a.plot(x, ff.w[hrzntl],plt_color)
            ax_w_b.plot(y, ff.w[vrtcl], plt_color)
            if plt_analytical:
                ax_w_a.plot(x, w[hrzntl],analytical_color)
                ax_w_b.plot(y, w[vrtcl], analytical_color)

            if plt_pressure:
                # ff.p[np.where(ff.p == 0)[0]] = np.nan
                ax_p_a.plot(x, ff.p[hrzntl], plt_color)
                ax_p_b.plot(y, ff.p[vrtcl], plt_color)
                if plt_analytical:
                    np.delete(hrzntl, np.where(ff.p[hrzntl]!=0.)[0], 0)
                    ax_p_a.plot(x, p[hrzntl], analytical_color)
                    ax_p_b.plot(y, p[vrtcl], analytical_color)

            if t_srs_given:
                if t_srs_indx < len(sol_time_series) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_time_series[t_srs_indx]
                if ff.time > max(sol_time_series):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    return fig_w_a, fig_w_b, fig_p_a, fig_p_b


#-----------------------------------------------------------------------------------------------------------------------
def plot_at_injection_point(address, fig_w=None, fig_p=None, plt_pressure=False, time_period=0.0, sol_time_series=None,
                maxFiles=1000, plt_analytical=False, analytical_sol='M', plt_color='r.', analytical_color='b',
                loglog=False, plt_dimensionless=False):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_time_series, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_time_series[t_srs_indx]

    if fig_w is None:
        fig_w = plt.figure()
    ax_w = fig_w.add_subplot(111)

    if plt_pressure:
        if fig_p is None:
            fig_p = plt.figure()
        ax_p = fig_p.add_subplot(111)

    if plt_analytical:
        w_anltcl = np.array([], dtype=np.float64)
        time_srs = np.array([], dtype=np.float64)
        if plt_pressure:
            p_anltcl = np.array([], dtype=np.float64)

        fig_err = plt.figure()
        ax_err = fig_err.add_subplot(111)
        w_err = np.array([], dtype=np.float64)
        p_err = np.array([], dtype=np.float64)
    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo += 1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            if plt_analytical:
                if analytical_sol == 'E':
                    R, a, w, p = anisotropic_toughness_elliptical_solution(Solid.K1c,
                                                                           Solid.K1c_perp,
                                                                           Solid.Eprime,
                                                                           Injection.injectionRate[1, 0],
                                                                           ff.mesh,
                                                                           t=ff.time)

                elif analytical_sol == 'M':
                    R, p, w, v = M_vertex_solution_t_given(Solid.Eprime,
                                                           Injection.injectionRate[1, 0],
                                                           Fluid.muPrime,
                                                           ff.mesh,
                                                           ff.time)
                elif analytical_sol == "K":
                    R, p, w, v = K_vertex_solution_t_given(Solid.Kprime,
                                                                     Solid.Eprime,
                                                                     Injection.injectionRate[1, 0],
                                                                     ff.mesh,
                                                                     ff.time)
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
            if plt_analytical:
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
                if plt_analytical:
                    if isinstance(p, np.ndarray):
                        p_aa = p[ff.mesh.CenterElts]
                    else:
                        p_aa = p
                    p_anltcl = np.append(p_anltcl, p_aa)
                    p_err = np.append(p_err, (p_aa - p_num)/p_aa)

            if t_srs_given:
                if t_srs_indx < len(sol_time_series) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_time_series[t_srs_indx]
                if ff.time > max(sol_time_series):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    if plt_analytical:
        if loglog:
            ax_w.semilogx(time_srs/tmk, w_anltcl, analytical_color)
        else:
            ax_w.plot(time_srs/tmk, w_anltcl, analytical_color)
        if plt_pressure:
            if loglog:
                ax_p.semilogx(time_srs/tmk, p_anltcl, analytical_color)
            else:
                ax_p.plot(time_srs/tmk, p_anltcl, analytical_color)
    # ax_w.plot(tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000 * tmk2 / tmk, 1e-4, 'k.')
    # ax_w.plot(7000, 1e-4, 'k.')
    # print(repr(time_srs))

        ax_err.plot(time_srs, abs(w_err), 'b.')
        ax_err.plot(time_srs, abs(p_err), 'r.')

    return fig_w, fig_p