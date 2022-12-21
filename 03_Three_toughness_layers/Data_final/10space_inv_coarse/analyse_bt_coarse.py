#external imports
import copy
import json
import os
import shutil
import numpy as np
import glob

# internal imports
from controller import Controller
from matplotlib import pyplot as plt
from properties import InjectionProperties
from solid.solid_prop import MaterialProperties
from utilities.postprocess_fracture import load_fractures, append_to_json_file
from utilities.utility import setup_logging_to_console


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

def check_make_folder(simdir):
    if os.path.isdir(simdir):
        print('  the folder exist')
    else:
        print('  the folder do not exist')
        os.mkdir(simdir)

def check_copy_file(dest_file, src_file):

    if os.path.isfile(dest_file):
        print('  the file exist')
    else:
        print('  the file does not exist: copying it')
        shutil.copy(src_file, dest_file,follow_symlinks=True)

def copy_dir(dest_folder, src_folder):

    if os.path.isdir(src_folder):
        print('  the folder exist: copying to the new location')
        if os.path.isdir(dest_folder):
            print('  the new location existed and it will be removed')
            shutil.rmtree(dest_folder)
        shutil.copytree(src_folder, dest_folder)
    else:
        print('  the folder does not exist: abort')
        raise SystemExit
# --------------------------------------------------------------
def smoothing(K1, K2, r, delta, x):
    # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
    #### LINEAR - DIRAC DELTA ####
    x = np.abs(x)
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

class K1c_func_factory:
     def __init__(self, r, K_Ic, KIc_ratio, hx, hy, delta = 0.001):
         self.K_Ic = K_Ic # fracture toughness
         self.r = r # 1/2 height of the layer
         self.KIc_ratio = KIc_ratio
         self.delta = delta
         # check
         if delta > hx/20. or delta > hy/20.:
             print('regularization param > 1/20 cell size')

     def __call__(self, x, y, alpha):
         """ The function providing the toughness"""
         return smoothing(self.K_Ic, self.KIc_ratio * self.K_Ic, self.r, self.delta, x)

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

# --------------------------------------------------------------
# define the terminating criterion function
class terminating_criterion_factory:
    def __init__(self, aspect_ratio_target, xmax_lim, aspect_ratio_toll):
        self.x_lim = x_lim  # max value of x that can be reached during the simulation
        self.aspect_ratio_target = aspect_ratio_target  # target aspect ratio that can be reached during the simulation
        self.aspect_ratio_toll = aspect_ratio_toll

    def __call__(self, fracture):
        """ The implementing the terminating_criterion"""
        x_min, x_max, y_min, y_max = get_fracture_sizes(fracture)
        larger_abs_x = np.maximum(np.abs(x_min),x_max)
        x_dimension = np.abs(x_min) + x_max
        y_dimension = np.abs(y_min) + y_max
        aspect_ratio = y_dimension / x_dimension
        if  aspect_ratio < (self.aspect_ratio_target) and larger_abs_x < self.x_lim :
            return True
        else:
            return False

# defining the return function in case the simulation ends according to the terminating criterion function
def return_function(fracture):
    return fracture

class adapive_time_ref_factory():
    def __init__(self, aspect_ratio_max, aspect_ratio_toll, xmax_lim):
        self.xmax_lim = xmax_lim  # max value of x that can be reached during the simulation
        self.aspect_ratio_max = aspect_ratio_max  # max aspect ratio that can be reached during the simulation
        self.aspect_ratio_toll = aspect_ratio_toll
        self.active_adaptive_time_ref=False
        self.upper_time_bound=None
        self.lower_time_bound=None

    def __call__(self, Fr_current, Fr_new, timestep):
        """ checking how far we are from the goal of getting an aspect ratio close to 1"""
        x_min_c, x_max_c, y_min_c, y_max_c = get_fracture_sizes(Fr_current)
        larger_abs_x_c = np.maximum(np.abs(x_min_c), x_max_c)
        x_dimension_c = np.abs(x_min_c) + x_max_c
        y_dimension_c = np.abs(y_min_c) + y_max_c
        aspect_ratio_c = y_dimension_c / x_dimension_c

        x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(Fr_new)
        larger_abs_x_n = np.maximum(np.abs(x_min_n),x_max_n)
        x_dimension_n = np.abs(x_min_n) + x_max_n
        y_dimension_n = np.abs(y_min_n) + y_max_n
        aspect_ratio_n = y_dimension_n / x_dimension_n

        aspect_ratio_bridging_target = (aspect_ratio_c < (self.aspect_ratio_max) and \
                aspect_ratio_n >= (self.aspect_ratio_max + self.aspect_ratio_toll))
        if aspect_ratio_bridging_target or self.active_adaptive_time_ref:
            if self.active_adaptive_time_ref == False:
                self.active_adaptive_time_ref = True
                self.lower_time_bound = Fr_current.time
                self.upper_time_bound = Fr_new.time
                # we should limit the time step
                timestep_new = np.abs(np.abs(self.upper_time_bound + self.lower_time_bound) * 0.5 - Fr_current.time)
                return  timestep_new, True
            elif self.active_adaptive_time_ref and aspect_ratio_n >= (self.aspect_ratio_max + self.aspect_ratio_toll):
                # the time step is still too large
                self.upper_time_bound = Fr_new.time
                # we should limit the time step and update the upper bound
                timestep_new = np.abs(np.abs(self.upper_time_bound + self.lower_time_bound) * 0.5 - Fr_current.time)
                return timestep_new, True
            elif self.active_adaptive_time_ref and aspect_ratio_n < (self.aspect_ratio_max):
                # the time step is too little
                self.lower_time_bound = Fr_new.time
                # we should increase the time step and update
                timestep_new = np.abs(np.abs(self.upper_time_bound + self.lower_time_bound) * 0.5 - Fr_current.time)
                return timestep_new, True
            elif self.active_adaptive_time_ref and (aspect_ratio_n >= self.aspect_ratio_max and aspect_ratio_n < (self.aspect_ratio_max + self.aspect_ratio_toll)):
                # accept time step!!! and restore active time ref
                self.active_adaptive_time_ref = False
                self.upper_time_bound = None
                self.lower_time_bound = None
                return timestep, False
            else:
                SystemExit("ERROR adapive_time_ref_factory: option not allowed")
        return timestep, False

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

def run(r_0, Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid):
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
    simulProp.maxFrontItrs = 95
    simulProp.tmStpPrefactor = 0.5
    simulProp.tolFractFront = 0.0001
    simulProp.gmres_Restart = 500
    simulProp.gmres_maxiter = 1000
    simulProp.set_outputFolder(simdir)
    simulProp.frontAdvancing = 'implicit'
    simulProp.plotFigure = False
    simulProp.custom = custom_factory(r_0, 'y/(0.5 H)', 'x/(0.5 H)')
    # define the adaptive time step function to get the simulation reaching ar = ar_desired +/- toll
    simulProp.adaptive_time_refinement = adapive_time_ref_factory(aspect_ratio_max, aspect_ratio_toll, xmax_lim)

    # define the terminating criterion function
    simulProp.terminating_criterion = terminating_criterion_factory(aspect_ratio_target, x_lim, aspect_ratio_toll)

    # defining the return function in case the simulation ends according to the terminating criterion function
    simulProp.return_function = return_function

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
"""
we fix an aspect ratio target and we loop on the toughness ratio from the moment we touch
"""
# educated  guess
# 430 is 25.15635786183006
TR = np.asarray([4.0, 4.4, 4.840000000000001, 5.324000000000002, 5.8564000000000025, 6.442040000000003, 7.086244000000004, 7.794868400000006, 8.574355240000006, 9.431790764000008, 10.37496984040001, 11.412466824440012, 12.553713506884014, 13.809084857572417, 15.18999334332966, 16.708992677662625, 18.379891945428888, 20.217881139971777, 22.239669253968955, 29.463636179365853, 32.40999979730244, 50.65099977703269, 55.716099754735964, 81.28770973020957, 89.41648070323053, 128.35812877355357, 151.19394165090895, 166.31333581599986])
SIM_ID = np.asarray([1968, 1959, 1878, 1800, 1794, 1791, 1719, 1713, 1641, 1635, 1563, 1557, 1482, 1101, 1056, 954, 909, 660, 615, 513, 468, 366, 321, 219, 174, 72, 30, 9])

file_name = "analyse_bt_res.json"
globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/10space_inv_coarse/contained_at_1p5/'
globalpath_json = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/10space_inv_coarse/'
date_ext = '2022-02-02__09_02_40'
basename = '/simulation__'+date_ext+'_file_'

todo = []
todo_n = []
#locallist = range(0, 2107, 3) # used with relative position interface to cell size between to 0.5 and 0.75
#locallist = range(1791, 1792, 1) # used with relative position interface to cell size between to 0.5 and 0.80
#remove some values
new_SIM_ID = SIM_ID.tolist()
to_remove = [72]
for elem in to_remove:
    new_SIM_ID.remove(elem)

locallist = np.sort(new_SIM_ID)
forced_recompute = []
for number in locallist: #range(0, 2107, 10):
    if number not in todo_n:
        todo.append(str(number))
todo_n = len(todo)

# copy the file for safety!
file_name_copy = "analyse_bt_res_copy.json"
if os.path.isfile(file_name):
    shutil.copyfile(globalpath_json+file_name, globalpath_json+file_name_copy)


# initialize some vars
results = {"toughness ratio" : [],
            "sim id" : [],
            "aspect ratio" : [],
            "ended" : [],
            "aspect_ratio_toll": [],
            "aspect_ratio_target": [],
            "x_max": [],
            "x_min": [],
            "x_lim": [],
            "xmax_lim": [],
            "delta": [],
            "halfH": [],
            }

KIc_ratio = None
KIc_ratio_upper = None
KIc_ratio_lower = None

# define the results
if not os.path.isfile(file_name):
    content = results
    action = 'dump_this_dictionary'
    append_to_json_file(file_name, [content], action, delete_existing_filename=False)
else:
    with open(file_name, "r+") as json_file:
        results = json.load(json_file)[0]  # get the data

# load the results with the min toughness ratio
if not os.path.isfile(globalpath_json + 'AR_high_K.json'):
    SystemExit("file not found")
else:
    with open(globalpath_json + 'AR_high_K.json', "r+") as json_file:
        results_AR_high_K = json.load(json_file)[0]  # get the data
    resARsimID = np.asarray(results_AR_high_K['sim id'])
    resARtoughRatio = np.asarray(results_AR_high_K['toughness ratio'])

for num_id, num in enumerate(todo):

    print(f'sim {num_id+1} of {todo_n}\n')

    # check error and eventually recompute
    if int(num) in results["sim id"]:
        pos = np.where(np.asarray(results["sim id"]) == int(num))[0][0]
        check_xbt = (results["x_max"][pos] >= results["x_lim"][pos]
                    and results["x_max"][pos] <= results["x_lim"][pos] + results["delta"][pos])
        # check_ar = results["aspect ratio"][pos] >= results["aspect_ratio_target"][pos] \
        #            and results["aspect ratio"][pos] <(results["aspect_ratio_target"][pos] + 0.001)
        check_ar = True
        check_xbt = True
        if not check_ar or not check_xbt or int(num) in forced_recompute:
            print(f'AR is in the proper range: {check_ar}, AR: {results["aspect ratio"][pos]}')
            print(f'xbt is in the proper range {check_xbt}, 100(xbt - x_lim)/delta {100*(results["x_max"][pos]-results["x_lim"][pos])/results["delta"][pos]}')
            results["toughness ratio"].pop(pos)
            results["sim id"].pop(pos)
            results["aspect ratio"].pop(pos)
            results["ended"].pop(pos)
            results["aspect_ratio_toll"].pop(pos)
            results["aspect_ratio_target"].pop(pos)
            results["x_max"].pop(pos)
            results["x_min"].pop(pos)
            results["x_lim"].pop(pos)
            results["xmax_lim"].pop(pos)
            results["delta"].pop(pos)
            results["halfH"].pop(pos)
            # remove the folder
            if os.path.isdir(globalpath_json + 'bt/simulation_' + num + '__' + date_ext):
                shutil.rmtree(globalpath_json + 'bt/simulation_' + num + '__' + date_ext)
            if os.path.isdir(globalpath_json + 'bt/simulation_' + num + '__' + date_ext + '_copy'):
                shutil.rmtree(globalpath_json + 'bt/simulation_' + num + '__' + date_ext + '_copy')

    if int(num) not in results["sim id"]:
        simdir = globalpath_json + 'bt/simulation_'+num+'__' + date_ext

        # make the folder if it does not exist
        print(' -check if the folder existed')
        check_make_folder(simdir)

        # copy properties if they do not exist
        print('\n -check if the properties existed')
        dest_file = simdir + '/properties'
        src_file = globalpath + 'simulation_'+num+'__'+date_ext+'/simulation__'+date_ext+'/properties'
        check_copy_file(dest_file, src_file)

        # check if the timestep exist in the source dir and copy
        print('\n -check if the initial file existed')
        dest_file = simdir + '/simulation_'+num+'__' + date_ext +'_file_0'
        base = globalpath + 'simulation_'+num+'__'+date_ext+'/simulation__'+date_ext+ '/simulation__' + date_ext + '_file_*'
        filelist = glob.glob(base)
        filelist_n = []
        for name in filelist:
            filelist_n.append(int(name[len(base[:-1]):]))
        filelist_n = np.asarray(filelist_n)
        np.ndarray.sort(filelist_n)
        src_file = globalpath + 'simulation_'+num+'__'+date_ext+'/simulation__'+date_ext+ '/simulation__' + date_ext + '_file_' + str(filelist_n[-3])
        check_copy_file(dest_file, src_file)

        # make a copy of the input folder
        print('\n -check if the newly created folder exist and make a copy of it')
        dest_folder = simdir + '_copy'
        src_folder = simdir
        copy_dir(dest_folder, src_folder)

        # load the fracture obj
        Fr_list, properties = load_fractures(address=globalpath_json + 'bt', step_size=100, sim_name='simulation_' + num)
        Solid_loaded, Fluid, Injection, simulProp = properties
        contunue_loop = True
        it_count = 0

        # check the location of the barrier
        Fr = copy.deepcopy(Fr_list[-1])

        # define the hard limit
        x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
        r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + Fr.mesh.hx
        delta = Fr.mesh.hx / 100.
        x_lim = r_0

        relative_pos_xlim = ((r_0 - 0.5 * Fr.mesh.hx) % Fr.mesh.hx) / Fr.mesh.hx

        print(f'\n -number of elts {len(Fr_list[-1].EltCrack)} \n sim {num_id + 1}\n and rel pos x_lim {relative_pos_xlim}')
        if not len(Fr_list[-1].EltCrack) > 28000 and relative_pos_xlim > .5 and relative_pos_xlim < .75:
            while contunue_loop:

                Fr = copy.deepcopy(Fr_list[-1])

                # define the hard limit
                x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
                delta = Fr.mesh.hx / 100.
                r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + delta
                x_lim = r_0

                # tollerance aspect ratio
                aspect_ratio_toll = 0.001
                # target aspect ratio
                aspect_ratio_max = 1.6
                # aspect ratio when to stop the simulation
                aspect_ratio_target = aspect_ratio_max

                # tollerance xmax
                toll_xmax = delta
                xmax_lim = x_lim + toll_xmax

                # current state variables
                skip = False
                if KIc_ratio is None or (it_count ==0):
                    if int(num) in resARsimID:
                        pos = np.where(resARsimID==int(num))[0][0]
                        KIc_ratio = resARtoughRatio[pos]
                        KIc_ratio_upper = KIc_ratio + 0.2 * resARtoughRatio[pos]
                        KIc_ratio_lower = resARtoughRatio[pos]
                        KIc_ratio = (KIc_ratio_upper + KIc_ratio_lower) * .5
                        if KIc_ratio_lower < 1.:
                            KIc_ratio_lower = 1.
                        skip = True
                    else:
                        KIc_ratio = 2.

                if not skip:
                    if KIc_ratio_upper is None or (num_id == 0 and it_count ==0):
                        KIc_ratio_upper = 3.
                    elif KIc_ratio_upper is not None and it_count ==0:
                        KIc_ratio_upper = KIc_ratio

                    if KIc_ratio_lower is None or (it_count ==0):
                        KIc_ratio_lower = 1.

                print(f'\n iterations on tough. ratio: {it_count} of 200, ID: {num}')
                print(f' toughness ratio: {KIc_ratio}')
                print(f' tough. min: {KIc_ratio_lower}')
                print(f' tough. max: {KIc_ratio_upper}')
                print(f' rel diff limits: {100 * np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower} %')

                last_Fr = run(r_0, Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid)

                # check if xmax < xlim
                x_min_c, x_max_c, y_min_c, y_max_c = get_fracture_sizes(last_Fr)
                larger_abs_x_c = np.maximum(np.abs(x_min_c), x_max_c)
                smaller_abs_x_c = np.minimum(np.abs(x_min_c), x_max_c)
                x_dimension_c = np.abs(x_min_c) + x_max_c
                y_dimension_c = np.abs(y_min_c) + y_max_c
                aspect_ratio_c = y_dimension_c / x_dimension_c

                # checks:
                print("checks:")

                target_reduction = (np.abs(KIc_ratio_lower - KIc_ratio_upper) / KIc_ratio_lower > 0.001)
                if target_reduction:
                    print(f'np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower = {np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower} > 0.001')
                else:
                    print(f' |KIc_ratio_lower-KIc_ratio_upper|/KIc_ratio_lower = {np.abs(KIc_ratio_lower - KIc_ratio_upper) / KIc_ratio_lower} < 0.001')

                ar_GE_target = aspect_ratio_c >= aspect_ratio_target
                target_reduction = target_reduction or not ar_GE_target
                print(f"aspect ratio {aspect_ratio_c} vs {aspect_ratio_target}")
                if ar_GE_target:
                    print(" aspect ratio >= target ")
                else:
                    print(" aspect ratio < target ")

                x_GE_xmax_lim = larger_abs_x_c >= xmax_lim
                if x_GE_xmax_lim:
                    print(" x >= x max lim ")
                else:
                    print(" x < x max lim ")

                x_G_x_lim = larger_abs_x_c > x_lim
                if x_G_x_lim:
                    print(" x > x_lim ")
                else:
                    print(" x <= x_lim ")

                # update the counter:
                it_count = it_count + 1
                if it_count < 300:
                    if ((ar_GE_target and x_GE_xmax_lim) or (not ar_GE_target and larger_abs_x_c > x_lim)) \
                        and target_reduction:
                        print(' increasing toughness ratio')
                        print(f' x/xlim: {larger_abs_x_c / x_lim}')
                        # increase toughness in the bounding layers
                        if (aspect_ratio_c <= aspect_ratio_target and larger_abs_x_c > x_lim and KIc_ratio >= KIc_ratio_upper):
                            KIc_ratio_upper = 1. + KIc_ratio_upper
                            KIc_ratio_new = KIc_ratio_upper
                        else:
                            KIc_ratio_new = (KIc_ratio + KIc_ratio_upper) * 0.5
                        KIc_ratio_lower = KIc_ratio
                        KIc_ratio = KIc_ratio_new
                        # delete the folder and get a new one
                        src_folder = simdir + '_copy'
                        dest_folder = simdir
                        copy_dir(dest_folder, src_folder)
                    elif ar_GE_target and not x_G_x_lim \
                          and target_reduction:
                        print(' decreasing toughness ratio')
                        print(f' x/xlim: {larger_abs_x_c/x_lim}')
                        # decrease toughness in the bounding layers
                        if KIc_ratio <= KIc_ratio_lower :
                            if KIc_ratio_lower - 1. > 1.:
                                KIc_ratio_lower = KIc_ratio_lower - 1.
                            else:
                                KIc_ratio_lower = 1.
                            KIc_ratio_new = KIc_ratio_lower
                        else:
                            KIc_ratio_new = (KIc_ratio + KIc_ratio_lower) * 0.5

                        KIc_ratio_upper = KIc_ratio
                        KIc_ratio = KIc_ratio_new
                        # KIc_ratio_new = smoothing(Solid_loaded.K1c[0], Solid_loaded.K1c[0] * KIc_ratio, r_0, delta, smaller_abs_x_c)/Solid_loaded.K1c[0]
                        # delete the folder and get a new one
                        src_folder = simdir + '_copy'
                        dest_folder = simdir
                        copy_dir(dest_folder, src_folder)
                    elif (not ar_GE_target and not x_G_x_lim):
                        print("\n aspect_ratio_c < aspect_ratio_target and larger_abs_x_c < x_lim")
                        b = input("    -->press a button to kill the program")
                        SystemExit()
                    else:
                        # accept solution
                        print('-solution achieved')
                        print(f' x/xlim: {larger_abs_x_c / x_lim}')
                        contunue_loop = False
                        results["toughness ratio"].append(KIc_ratio)
                        results["sim id"].append(int(num))
                        results["aspect ratio"].append(aspect_ratio_c)
                        results["aspect_ratio_toll"].append(aspect_ratio_toll)
                        results["aspect_ratio_target"].append(aspect_ratio_target)
                        results["x_max"].append(x_max_c)
                        results["x_min"].append(x_min_c)
                        results["x_lim"].append(x_lim)
                        results["xmax_lim"].append(xmax_lim)
                        results["delta"].append(delta)
                        results["halfH"].append(r_0)
                        results["ended"].append(True)
                        print(" Saving to file")
                        content = results
                        action = 'dump_this_dictionary'
                        append_to_json_file(file_name, [content], action, delete_existing_filename=True)

                        # make a copy of the input folder
                        print(' delete the copy of the starting folder')
                        dest_folder = simdir + '_copy'
                        shutil.rmtree(dest_folder)

                        # copy the file for safety!
                        file_name_copy = "analyse_bt_res_copy.json"
                        shutil.copyfile(globalpath_json + file_name, globalpath + file_name_copy)
                        print('-----------------------------')
                else:
                    print('-convergence on the toughness ratio not achieved!')
                    print(f'simulaton ID: '+num)
                    print('-----------------------------')
                    SystemExit()
        else:
            # remove the copies and go next
            dest_folder = simdir + '_copy'
            shutil.rmtree(dest_folder)
            dest_folder = simdir
            shutil.rmtree(dest_folder)

            # delete some variables
            del Fr_list, properties, Solid_loaded, Fluid, Injection, simulProp