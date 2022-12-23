#external imports
import copy
import json
import os
import shutil
import numpy as np
from numba import jit, prange

# internal imports
from controller import Controller
from matplotlib import pyplot as plt
from properties import InjectionProperties
from solid.solid_prop import MaterialProperties
from utilities.postprocess_fracture import load_fractures, append_to_json_file
from utilities.utility import setup_logging_to_console


# setting up the verbosity level of the log at console
from src.properties import PlotProperties
from src.utilities.visualization import plot_fracture_list

setup_logging_to_console(verbosity_level='debug')

@jit
def smoothing(K1, K2, r, delta, x):
    # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
    #### LINEAR - DIRAC DELTA ####

    if  x < r-delta :
        return K1
    elif x >= r-delta and x<r :
        K12 = K1 + (K2-K1)*1.
        a = (K12 - K1) / (delta)
        b = K1 - a * (r - delta)
        return a * x + b
    elif x >= r:
        return K2
    else:
        print("ERROR")

@jit
def rot( x, y, psi):
     x_rot = np.cos(psi)*x-np.sin(psi)*y
     y_rot = np.sin(psi)*x-np.cos(psi)*y
     return [x_rot, y_rot]


@jit
def random_sampling(K,NoP,x_cV,y_cV,r_oV,K_Ic,KIc_ratio,delta,x,y):
    for i in prange(NoP):
        xc = x_cV[i]
        yc = y_cV[i]
        r_o = r_oV[i]
        psi = np.pi / 3.
        l = np.asarray([1,2])
        # l = rot(x, y, psi)
        # xp = l[0]
        # yp = l[1]

        xp = x
        yp = y
        # l = rot(xc, yc, psi)
        # xcp = l[0]
        # ycp = l[1]
        xcp = xc
        ycp = yc
        radius = np.sqrt((xp - xcp) ** 2 + (yp - ycp) ** 2)
        d = np.sqrt((xcp) ** 2 + (ycp) ** 2)
        K[i] = smoothing(KIc_ratio * K_Ic, K_Ic, r_o, delta, radius)
    return K

class K1c_func_factory:
     def __init__(self, r_o, K_Ic, KIc_ratio, hx, hy, delta = 0.001):
         self.K_Ic = K_Ic # fracture toughness
         self.NoP = 80
         self.r_o = 0.1 * r_o * np.random.uniform(0.9,1,self.NoP) # distance to the heterogeneity (regularization is before)
         self.x_c = 2*r_o * (2* np.random.uniform(0.,1,self.NoP) -1)
         self.y_c = 2*r_o * (2* np.random.uniform(0.,1,self.NoP) -1)
         self.KIc_ratio = KIc_ratio
         self.delta = delta
         # check
         if delta > hx/20. or delta > hy/20.:
             print('regularization param > 1/20 cell size')



     def __call__(self, x, y, alpha):
         K=np.zeros(self.NoP)
         random_sampling(K, self.NoP, self.x_c, self.y_c, self.r_o, self.K_Ic, self.KIc_ratio, self.delta, x, y)

         """ The function providing the toughness"""
         #return smoothing(self.K_Ic, self.KIc_ratio * self.K_Ic, self.r, self.delta, x)
         return np.max(K)


def sigmaO_func(x, y):
    return 0

# --------------------------------------------------------------
def get_fracture_sizes(Fr):
    # Now we are at a given time step.
    # This function returns the coordinates of the smallest rectangle containing the fracture footprint

    x_min_temp = 0.
    x_max_temp = 0.
    y_min_temp = 0.
    y_max_temp = 0.
    hx = Fr.mesh.hx; hy = Fr.mesh.hy
    # loop over the segments defining the fracture front
    for i in range(Fr.Ffront.shape[0]):
        segment = Fr.Ffront[i]

        # to find the x_max at this segment:
        if segment[0] > x_max_temp and np.abs(segment[1])<2.*hy:
            x_max_temp = segment[0]
        if segment[2] > x_max_temp and np.abs(segment[3])<2.*hy:
            x_max_temp = segment[2]

        # to find the n_min at this segment:
        if segment[0] < x_min_temp and np.abs(segment[1])<2.*hy:
            x_min_temp = segment[0]
        if segment[2] < x_min_temp and np.abs(segment[3])<2.*hy:
            x_min_temp = segment[2]

        # to find the y_max at this segment:
        if segment[1] > y_max_temp and np.abs(segment[0])<2.*hx:
            y_max_temp = segment[1]
        if segment[3] > y_max_temp and np.abs(segment[2])<2.*hx:
            y_max_temp = segment[3]

        # to find the y_min at this segment:
        if segment[1] < y_min_temp and np.abs(segment[0])<2.*hx:
            y_min_temp = segment[1]
        if segment[3] < y_min_temp and np.abs(segment[2])<2.*hx:
            y_min_temp = segment[3]

    return x_min_temp, x_max_temp, y_min_temp, y_max_temp

class adapive_time_ref_factory():
    def __init__(self ):
        print('')
    def __call__(self, Fr_current, Fr_new, timestep):
        return timestep, False

def default_terminating_criterion(fracture):
        # criterion based on the final time
        if fracture.time < 0.999 * 1.e100:
            return True
        else:
            return False


class custom_factory():
    def __init__(self, r_0, xlabel, ylabel):
        self.data = {'xlabel' : xlabel,
                     'ylabel': ylabel,
                     'xdata': [],
                     'ydata': [],
                     'H/2': r_0} # max value of x that can be reached during the simulation

    def custom_plot(self, sim_prop, fig=None):
        # this method is mandatory
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]

        ax.scatter(self.data['xdata'], self.data['ydata'], color='k')
        ax.set_xlabel(self.data['xlabel'])
        ax.set_ylabel(self.data['ylabel'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        return fig

    def postprocess_fracture(self, sim_prop, fr):
        # this method is mandatory
        x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(fr)
        self.data['xdata'].append(y_max_n / self.data['H/2'])
        self.data['ydata'].append(x_max_n / self.data['H/2'])
        fr.postprocess_info = self.data
        return fr


def run(r_0, Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid, simdir):
    # define the toughenss function
    K1c_func = K1c_func_factory(r_0, Solid_loaded.K1c[0], KIc_ratio, Fr.mesh.hx, Fr.mesh.hy, delta=delta)
    Solid = MaterialProperties(Fr.mesh,
                               Solid_loaded.Eprime,
                               K1c_func=K1c_func,
                               confining_stress_func=sigmaO_func,
                               confining_stress=0.,
                               minimum_width=0.)
    Injection = InjectionProperties(Injection.injectionRate[1, 0], Fr.mesh)
    simulProp.meshReductionPossible = False
    simulProp.meshExtensionAllDir = True
    simulProp.finalTime = 10. ** 30
    simulProp.maxFrontItrs = 45
    simulProp.tmStpPrefactor = 1
    simulProp.tolFractFront = 0.0001
    simulProp.frontAdvancing = 'explicit'
    simulProp.set_outputFolder(simdir)
    simulProp.custom = custom_factory(r_0, 'y/(0.5 H)', 'x/(0.5 H)')
    simulProp.simID = '1560'
    # define the adaptive time step function to get the simulation reaching ar = ar_desired +/- toll
    simulProp.adaptive_time_refinement = adapive_time_ref_factory()
    simulProp.terminating_criterion = default_terminating_criterion


    # create a Controller
    controller = Controller(Fr,
                            Solid,
                            Fluid,
                            Injection,
                            simulProp)

    # run the simulation
    last_Fr = controller.run()
    return last_Fr

# --------------------------------------------------------------
# --------------------------------------------------------------
print('STARTING SIMULATION:')
# educated  guess
TR = np.asarray([106.917185481925, 23.988447321363665, 7.999417382598588, 7.078244250253137, 6.674610843009765, 4.25540730566231, 1.7741874780639637, 1.4287380449729226, 1.4060753589175625, 1.34375, 110.2154412738598, 77.24778713486634, 32.87565342867487, 31.879539259028782, 49.53006308329436, 45.55276195149959, 117.87413570419311, 91.80141011018539, 56.847730402357456, 44.90514509436578, 27.647511224415616, 27.823979905656486, 64.6152790349907, 25.237905789552585, 21.474085895589518, 16.277029034014507, 13.629805293793265, 11.683741966638067, 10.099601610262912, 10.25443151013145, 9.340843671961121, 5.850190820510264, 5.697990573314791, 4.936385510414589, 5.003375991127431, 4.550624892690718, 3.848481392454521, 2.8963250655543007, 2.449034498944342, 2.1680369874482697, 1.8200747720671098, 1.6121301060492836, 1.5840184937241348, 1.5514235878663767, 1.3124865389282827, 1.2975685012603453, 1.210736583581057, 1.2061131990555705, 1.4152003420871764, 1.4039258535445787, 1.2109375, 1.2754254657920683, 1.2049531456395393, 1.0544406793105026, 1.1247531793105026, 1.3172580621230026, 1.1467716727514226, 1.049220173014882, 1.2777413159664945, 1.10205078125, 1.1152692157920683, 1.0796001517557432, 1.0830205056228568, 1.189253663087742, 58.10357100799947, 37.79264252139599, 1.075421473714736])
SIM_ID = np.asarray([10, 420, 720, 760, 770, 910, 1190, 1330, 1340, 1640, 20, 80, 320, 330, 220, 230, 0, 30, 170, 270, 370, 380, 130, 430, 470, 520, 570, 620, 660, 670, 710, 810, 820, 860, 870, 920, 960, 1010, 1060, 1110, 1180, 1250, 1260, 1270, 1400, 1410, 1480, 1490, 1560, 1570, 1630, 1720, 1890, 1960, 1970, 1980, 2050, 2060, 1420, 1650, 1710, 1730, 1800, 1810, 180, 280, 1880])
#num = '1560'
#num = '1650'
num = '1980'

globalpath = '/home/carlo/Desktop/PyFrac/17_heterog_fields'
date_ext = '2022-02-02__09_02_40'
basename = '/simulation__'+date_ext+'_file_'

# load the fracture obj
Fr_list, properties = load_fractures(address=globalpath + '/sim_1330', step_size=1 )#, sim_name='simulation_'+ num)
Solid_loaded, Fluid, Injection, simulProp = properties

# plot_prop = PlotProperties()
# Fig_R = plot_fracture_list(Fr_list,
#                            variable='footprint',
#                            plot_prop=plot_prop)
# Fig_R = plot_fracture_list(Fr_list,
#                            fig=Fig_R,
#                            variable='mesh',
#                            mat_properties=Solid_loaded,
#                            backGround_param='K1c',
#                            plot_prop=plot_prop)
# plt.show()

contunue_loop = True
it_count = 0



Fr = copy.deepcopy(Fr_list[-1])

# define the hard limit
x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + Fr.mesh.hx
delta = Fr.mesh.hx / 100.
x_lim = r_0

# tollerance aspect ratio
aspect_ratio_toll = 0.001
# target aspect ratio
aspect_ratio_max = 1.02
# aspect ratio when to stop the simulation
aspect_ratio_target = aspect_ratio_max

# tollerance xmax
toll_xmax = delta
xmax_lim = x_lim + toll_xmax

pos = np.where(SIM_ID==int(num))[0][0]
KIc_ratio = 120#TR[pos]

last_Fr = run(r_0, Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid, globalpath + '/sim_1330')
