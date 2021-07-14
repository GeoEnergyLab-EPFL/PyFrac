import numpy as np
import matplotlib.pyplot as plt
def getH__(Ffront):
    return np.abs(np.max(np.hstack((Ffront[::, 1], Ffront[::, 3]))) - np.min(np.hstack((Ffront[::, 1], Ffront[::, 3]))))

def apply_custom_prop(sim_prop, fr):
    sim_prop.LHyst__.append(getH__(fr.Ffront))
    sim_prop.tHyst__.append(fr.time)

def custom_plot(sim_prop, fig = None):
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = fig.get_axes()[0]
    # plot L vs time
    xlabel = 'time [s]'
    ylabel = 'H [m]'
    ax.scatter(sim_prop.tHyst__, sim_prop.LHyst__, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig