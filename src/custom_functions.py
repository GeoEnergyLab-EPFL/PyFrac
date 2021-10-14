import numpy as np
import matplotlib.pyplot as plt


def getH__(Ffront):
    return np.abs(np.max(np.hstack((Ffront[::, 1], Ffront[::, 3]))) - np.min(np.hstack((Ffront[::, 1], Ffront[::, 3]))))

def update_limits(x1, x2, xmax, xmin):
    #the following function updates x and y
    if x1 > xmax:
        xmax = x1
    if x1 < xmin:
        xmin = x1
    if x2 > xmax:
        xmax = x2
    if x2 < xmin:
        xmin = x2
    return xmax, xmin

def getFfrontBounds__(Ffront):
    xmax = 0.
    xmin = 0.
    ymax = 0.
    ymin = 0.

    for segment in Ffront:
        x1, y1, x2, y2 = segment
        xmax, xmin = update_limits(x1, x2, xmax, xmin)
        ymax, ymin = update_limits(y1, y2, ymax, ymin)
        return xmax, xmin, ymax, ymin

def getL__(Ffront):
    xmax, xmin, ymax, ymin = getFfrontBounds__(Ffront)
    Lhalf = (np.abs(xmax)+np.abs(xmin))/2.
    return Lhalf

def getwAtMovingCenter__(fr):
    xmax, xmin, ymax, ymin = getFfrontBounds__(fr.Ffront)
    xmean = 0.5 * (xmax + xmin)
    ymean = 0.5 * (ymax + ymin)
    elem_id = fr.mesh.locate_element(xmean, ymean)
    return fr.w[elem_id]


def getwmax__(w):
    return w.max()

def apply_custom_prop(sim_prop, fr):
    #sim_prop.LHyst__.append(getH__(fr.Ffront))
    #sim_prop.LHyst__.append(getL__(fr.Ffront))
    #sim_prop.LHyst__.append(getwmax__(fr.w))
    sim_prop.LHyst__.append(getwmax__(fr.pFluid))
    #sim_prop.LHyst__.append(getwAtMovingCenter__(fr))
    sim_prop.tHyst__.append(fr.time)

def custom_plot(sim_prop, fig = None):
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = fig.get_axes()[0]
    # plot L vs time
    xlabel = 'time [s]'
    #ylabel = 'H [m]'
    ylabel = 'p [m]'
    ax.scatter(sim_prop.tHyst__, sim_prop.LHyst__, color='k')
    # straight line
    sl= []
    sl_ana = []
    K_Ic = 0.5e6
    H=2*1.48
    p_limit = 1.47*K_Ic/np.sqrt(np.pi*H/2)
    p_ana = 2.*K_Ic/np.sqrt(np.pi*H)
    for i in range(len(sim_prop.tHyst__)):
        sl.append(p_limit)
        sl_ana.append(p_ana)
    ax.plot(sim_prop.tHyst__, sl, color='r')
    ax.plot(sim_prop.tHyst__, sl_ana, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig