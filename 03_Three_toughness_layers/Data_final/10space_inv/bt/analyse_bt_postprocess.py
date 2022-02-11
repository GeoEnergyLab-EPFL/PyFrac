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
    def __init__(self, aspect_ratio_target, xmax_lim):
        self.xmax_lim = xmax_lim  # max value of x that can be reached during the simulation
        self.aspect_ratio_target = aspect_ratio_target  # target aspect ratio that can be reached during the simulation

    def __call__(self, fracture):
        """ The implementing the terminating_criterion"""
        x_min, x_max, y_min, y_max = get_fracture_sizes(fracture)
        larger_abs_x = np.maximum(np.abs(x_min),x_max)
        x_dimension = np.abs(x_min) + x_max
        y_dimension = np.abs(y_min) + y_max
        aspect_ratio = y_dimension / x_dimension
        if larger_abs_x < self.xmax_lim and aspect_ratio < self.aspect_ratio_target:
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

        if aspect_ratio_c < (self.aspect_ratio_max - self.aspect_ratio_toll) and \
                aspect_ratio_n > (self.aspect_ratio_max + self.aspect_ratio_toll):
            # we should limit the time step
            timestep_new = Fr_current.time + (Fr_new.time + Fr_current.time) * 0.5
            return  timestep_new, True
        return timestep, False
#-----------------------------------------------------------------------
print('STARTING SIMULATION:')
# load the file with the results
file_name = "analyse_bt_res_copy.json"
baseloc = "/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv/"
with open(baseloc+file_name, "r+") as json_file:
    results = json.load(json_file)[0]  # get the data

# get the list with simulations names
simlist = results["sim id"]

t_touch_lst = []
error_on_xtouch_lst = []
average_R_lst = []
dimlessK_lst = []
elts_in_crack_lst = []
time_bt_lst = []
velocity_ttouch_lst = []
max_velocity_ttouch_lst = []
min_velocity_ttouch_lst = []
variance_velocity_ttouch_lst = []

Fr_list = []
properties = []
for num_id, num in enumerate(simlist):
    print(f'analyzing {num_id+1} out of {len(simlist)}')
    # get H/2
    xlim = results["x_lim"][num_id]
    # get results path
    globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv/bt/simulation_'+str(num)+'__2022-02-02__09_02_40/'

    # load the fracture obj
    del Fr_list, properties
    Fr_list, properties = load_fractures(address=globalpath, step_size=1)
    Solid_loaded, Fluid, Injection, simulProp = properties


    for II in range(len(Fr_list)):
        x_min, x_max, y_min, y_max = get_fracture_sizes(Fr_list[II])
        if x_max > xlim or np.abs(x_min) > xlim:
            break
        else :
            target_fr = II
            e1 = np.abs(x_max - xlim)/xlim
            e2 = np.abs(np.abs(x_min)  - xlim)/xlim
            error_on_xtouch = np.maximum(e1,e2)

    # append the error on xtouch
    error_on_xtouch_lst.append(error_on_xtouch)

    # get the time of touch
    time_touch = Fr_list[target_fr].time
    t_touch_lst.append(time_touch)

    # get the average radius at the time of touch
    sumR = 0.
    for seg in Fr_list[target_fr].Ffront:
        x1, y1, x2, y2 = seg
        sumR = sumR + np.sqrt(x1*x1 + y1*y1)
    averageR = sumR / len(Fr_list[target_fr].Ffront)
    average_R_lst.append(averageR)

    # get the number of elements in the crack
    elts_in_crack_lst.append(len(Fr_list[target_fr].EltCrack))

    # get the dimensionless toughness
    Q_o = Injection.injectionRate[1][0]
    Eprime = Solid_loaded.Eprime
    K1c1 = np.min(Solid_loaded.K1c)
    mu = Fluid.muPrime/12.

    dimlessK = ( ((K1c1**18)*(time_touch**2)) / ((mu**5)*(Q_o**3)*(Eprime**13)) )**(1./18.)
    dimlessK_lst.append(dimlessK)

    # get the average velocity at t_touch
    # get the min and max velocity at t_touch
    velocity_ttouch_lst.append(np.mean(Fr_list[target_fr].v))
    max_velocity_ttouch_lst.append(np.max(Fr_list[target_fr].v))
    min_velocity_ttouch_lst.append(np.min(Fr_list[target_fr].v))
    variance_velocity_ttouch_lst.append(np.var(Fr_list[target_fr].v))

# save the results
results["t_touch_lst"] = t_touch_lst
results["error_on_xtouch_lst"] = error_on_xtouch_lst
results["average_R_lst"] = average_R_lst
results["dimlessK_lst"] = dimlessK_lst
results["elts_in_crack_lst"] = elts_in_crack_lst
results["time_bt_lst"] = time_bt_lst
results["velocity_ttouch_lst"] = velocity_ttouch_lst
results["max_velocity_ttouch_lst"] = max_velocity_ttouch_lst
results["min_velocity_ttouch_lst"] = min_velocity_ttouch_lst
results["variance_velocity_ttouch_lst"] = variance_velocity_ttouch_lst
results["mu"]=mu
results["Eprime"]=Eprime
results["K1c"]=K1c1
results["Qo"]=Q_o

print(" Saving to file")
content = results
action = 'dump_this_dictionary'
file_name_post = "analyse_bt_res_copy_post.json"
append_to_json_file(baseloc+file_name_post, [content], action, delete_existing_filename=True)

