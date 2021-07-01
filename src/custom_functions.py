import numpy as np
import matplotlib.pyplot as plt
def getL__(Ffront):
    xmax = 0.
    xmin = 0.

    for segment in Ffront:
        x1, y1, x2, y2 = segment
        if x1 > xmax:
            xmax = x1
        if x1 < xmin:
            xmin = x1
        if x2 > xmax:
            xmax = x2
        if x2 < xmin:
            xmin = x2
        Lhalf = (np.abs(xmax)+np.abs(xmin))/2.
    return Lhalf

def apply_custom_prop(sim_prop, fr):
    sim_prop.LHyst__.append(getL__(fr.Ffront))
    sim_prop.tHyst__.append(fr.time)

def custom_plot(sim_prop, fig = None):
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = fig.get_axes()[0]
    # plot L vs time
    xlabel = 'time [s]'
    ylabel = 'L [m]'
    ax.scatter(sim_prop.tHyst__, sim_prop.LHyst__, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig