#external imports
import copy
import json
import os
import shutil
import numpy as np

# internal imports
from controller import Controller
from properties import InjectionProperties
from solid.solid_prop import MaterialProperties
from utilities.postprocess_fracture import load_fractures, append_to_json_file
from utilities.utility import setup_logging_to_console


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

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

    # loop over the segments defining the fracture front
    for i in range(Fr.Ffront.shape[0]):
        segment = Fr.Ffront[i]

        # to find the x_max at this segment:
        if segment[0] > x_max_temp:
            x_max_temp = segment[0]
        if segment[2] > x_max_temp:
            x_max_temp = segment[2]

        # to find the n_min at this segment:
        if segment[0] < x_min_temp:
            x_min_temp = segment[0]
        if segment[2] < x_min_temp:
            x_min_temp = segment[2]

        # to find the y_max at this segment:
        if segment[1] > y_max_temp:
            y_max_temp = segment[1]
        if segment[3] > y_max_temp:
            y_max_temp = segment[3]

        # to find the y_min at this segment:
        if segment[1] < y_min_temp:
            y_min_temp = segment[1]
        if segment[3] < y_min_temp:
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

print('STARTING SIMULATION:')
file_name = "analyse_bt_res.json"
globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/10space_inv_coarse/'
date_ext = '2022-02-11__09_41_49'
basename = '/simulation__'+date_ext+'_file_'

todo = []
todo_n = []
for number in range(1100, 1210, 10):
    if number not in todo_n:
        todo.append(str(number))
todo_n = len(todo)

# copy the file for safety!
file_name_copy = "analyse_bt_res_copy.json"
if os.path.isfile(file_name):
    shutil.copyfile(globalpath+file_name, globalpath+file_name_copy)


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

for num_id, num in enumerate(todo):

    print(f'sim {num_id+1} of {todo_n}\n')


    # check error and eventually recompute
    if int(num) in results["sim id"]:
        pos = np.where(np.asarray(results["sim id"]) == int(num))[0][0]
        err_xbt = 100 * np.abs(results["x_max"][pos] - results["x_lim"][pos]) / results["delta"][pos]
        err_ar = 100 * np.abs(results["aspect ratio"][pos] - results["aspect_ratio_target"][pos]) / results["aspect_ratio_target"][pos]
        if err_ar > 30. or err_xbt > 30.:
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
            shutil.rmtree(globalpath + '/bt/simulation_'+num+'__' + date_ext)

    if int(num) not in results["sim id"]:
        simdir = globalpath + 'bt/simulation_'+num+'__' + date_ext

        # make the folder if it does not exist
        print(' -check if the folder existed')
        check_make_folder(simdir)

        # copy properties if they do not exist
        print('\n -check if the properties existed')
        dest_file = simdir + '/properties'
        src_file = globalpath + 'simulation__' + date_ext + '/properties'
        check_copy_file(dest_file, src_file)

        # check if the timestep exist in the source dir and copy
        print('\n -check if the initial file existed')
        dest_file = simdir + '/simulation_'+num+'__' + date_ext +'_file_0'
        src_file = globalpath + 'simulation__' + date_ext + '/simulation__' + date_ext + '_file_' + num
        check_copy_file(dest_file, src_file)

        # make a copy of the input folder
        print('\n -check if the newly created folder exist and make a copy of it')
        dest_folder = simdir + '_copy'
        src_folder = simdir
        copy_dir(dest_folder, src_folder)

        # load the fracture obj
        Fr_list, properties = load_fractures(address=globalpath + 'bt', step_size=100, sim_name='simulation_' + num)
        Solid_loaded, Fluid, Injection, simulProp = properties
        contunue_loop = True
        it_count = 0

        if not len(Fr_list[-1].EltCrack) > 3000:
            while contunue_loop:

                Fr = copy.deepcopy(Fr_list[-1])

                # define the hard limit
                x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
                r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + Fr.mesh.hx
                delta = Fr.mesh.hx / 100.
                x_lim = r_0

                # tollerance aspect ratio
                aspect_ratio_toll = 0.0001
                # target aspect ratio
                aspect_ratio_max = 1.55
                # aspect ratio when to stop the simulation
                aspect_ratio_target = aspect_ratio_max

                # tollerance xmax
                toll_xmax = delta
                xmax_lim = x_lim + toll_xmax

                # current state variables
                if KIc_ratio is None or (num_id == 0 and it_count ==0): KIc_ratio = 34.0625

                if KIc_ratio_upper is None or (num_id == 0 and it_count ==0): KIc_ratio_upper = 40.
                elif KIc_ratio_upper is not None and it_count ==0: KIc_ratio_upper = KIc_ratio

                if KIc_ratio_lower is None or (it_count ==0): KIc_ratio_lower = 1.

                print(f'\n iterations on tough. ratio: {it_count} of 200, ID: {num}')
                print(f' toughness ratio: {KIc_ratio}')
                print(f' tough. min: {KIc_ratio_lower}')
                print(f' tough. max: {KIc_ratio_upper}')
                print(f' rel diff limits: {100 * np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower} %')

                # define the toughenss function
                K1c_func = K1c_func_factory(r_0, Solid_loaded.K1c[0], KIc_ratio, Fr.mesh.hx, Fr.mesh.hy, delta = delta)
                Solid = MaterialProperties(Fr.mesh,
                                           Solid_loaded.Eprime,
                                           K1c_func=K1c_func,
                                           confining_stress_func=sigmaO_func,
                                           confining_stress=0.,
                                           minimum_width=0.)
                Injection = InjectionProperties(Injection.injectionRate[1,0], Fr.mesh)
                simulProp.meshReductionPossible = False
                simulProp.meshExtensionAllDir = True
                simulProp.finalTime = 10. ** 30
                simulProp.maxFrontItrs = 45
                simulProp.set_outputFolder(simdir)
                simulProp.LHyst__ = []
                simulProp.tHyst__ = []
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

                # check if xmax < xlim
                x_min_c, x_max_c, y_min_c, y_max_c = get_fracture_sizes(last_Fr)
                larger_abs_x_c = np.maximum(np.abs(x_min_c), x_max_c)
                smaller_abs_x_c = np.minimum(np.abs(x_min_c), x_max_c)
                x_dimension_c = np.abs(x_min_c) + x_max_c
                y_dimension_c = np.abs(y_min_c) + y_max_c
                aspect_ratio_c = y_dimension_c / x_dimension_c

                # update the counter:
                it_count = it_count + 1
                if it_count < 200:
                    target_reduction = (np.abs(KIc_ratio_lower - KIc_ratio_upper) / KIc_ratio_lower > 0.005 or
                                        100 * np.abs(larger_abs_x_c - xmax_lim) / delta > 30.)
                    if ((aspect_ratio_c >= aspect_ratio_target and larger_abs_x_c >= xmax_lim) or
                        (aspect_ratio_c <= aspect_ratio_target and larger_abs_x_c > x_lim)) \
                            and target_reduction:
                        print(' increasing toughness ratio')
                        print(f' x/xlim: {larger_abs_x_c / x_lim}')
                        # increase toughness in the bounding layers
                        if KIc_ratio >= KIc_ratio_upper or (aspect_ratio_c <= aspect_ratio_target and larger_abs_x_c > x_lim) :
                            KIc_ratio_upper = 10.+KIc_ratio_upper
                        KIc_ratio_new = (KIc_ratio + KIc_ratio_upper) * 0.5
                        KIc_ratio_lower = KIc_ratio
                        KIc_ratio = KIc_ratio_new
                        # delete the folder and get a new one
                        src_folder = simdir + '_copy'
                        dest_folder = simdir
                        copy_dir(dest_folder, src_folder)
                    elif (aspect_ratio_c >= aspect_ratio_target and larger_abs_x_c <= x_lim) \
                             and target_reduction:
                        print(' decreasing toughness ratio')
                        print(f' x/xlim: {larger_abs_x_c/x_lim}')
                        # decrease toughness in the bounding layers
                        KIc_ratio_upper = KIc_ratio
                        KIc_ratio_new = (KIc_ratio + KIc_ratio_lower) * 0.5
                        KIc_ratio = KIc_ratio_new
                        #KIc_ratio_new = smoothing(Solid_loaded.K1c[0], Solid_loaded.K1c[0] * KIc_ratio, r_0, delta, smaller_abs_x_c)/Solid_loaded.K1c[0]
                        # delete the folder and get a new one
                        src_folder = simdir + '_copy'
                        dest_folder = simdir
                        copy_dir(dest_folder, src_folder)
                    elif (aspect_ratio_c < aspect_ratio_target and larger_abs_x_c < x_lim):
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
                        shutil.copyfile(globalpath + file_name, globalpath + file_name_copy)
                        print('-----------------------------')
                else:
                    print('-convergence on the toughness ratio not achieved!')
                    print(f'simulaton ID: '+num)
                    print('-----------------------------')
        else:
            # remove the copies and go next
            dest_folder = simdir + '_copy'
            shutil.rmtree(dest_folder)
            dest_folder = simdir
            shutil.rmtree(dest_folder)

            # delete some variables
            del Fr_list, properties, Solid_loaded, Fluid, Injection, simulProp