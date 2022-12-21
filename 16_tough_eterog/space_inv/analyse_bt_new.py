#external imports
import copy
import json
import os
import shutil
import numpy as np
import random

# internal imports
from controller import Controller
from matplotlib import pyplot as plt
from properties import InjectionProperties
from solid.solid_prop import MaterialProperties
from utilities.postprocess_fracture import load_fractures, append_to_json_file
from utilities.utility import setup_logging_to_console
from level_set.anisotropy import get_toughness_from_cellCenter_iter

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

def rot(x, y):
     psi = 0.#np.pi / 4.
     [x_rot, y_rot] = np.dot([[np.cos(psi), -np.sin(psi)],
                              [np.sin(psi), np.cos(psi)]], [x, y])
     return x_rot, y_rot

class K1c_func_factory:
     def __init__(self, r_o, K_Ic, KIc_ratio, hx, hy, delta = 0.001):
         self.K_Ic = K_Ic # fracture toughness
         self.r_o = r_o # distance to the heterogeneity (regularization is before)
         print(f"radius: {r_o}")
         self.KIc_ratio = KIc_ratio
         self.delta = delta
         # check
         if delta > hx/20. or delta > hy/20.:
             print('regularization param > 1/20 cell size')



     def __call__(self, x, y, alpha):
         mult = 0.25# if it is 1 the heterog will have the same radius
         xc = (1+mult)*self.r_o #* np.sqrt(2)
         yc = 0

         xp, yp = rot(x,y)
         radius = np.sqrt((xp-xc)**2 + (yp-yc)**2)

         """ The function providing the toughness"""
         #return smoothing(self.K_Ic, self.KIc_ratio * self.K_Ic, self.r, self.delta, x)
         return smoothing(self.KIc_ratio * self.K_Ic, self.K_Ic, mult*self.r_o-self.delta, self.delta, radius)


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
    def __init__(self, r_0, delta, xmax_lim, v_toll, mat_properties, maxKIc):
        self.xmax_lim = xmax_lim  # max value of x that can be reached during the simulation
        self.delta = delta
        self.target_vel_toll = v_toll
        self.mat_properties = mat_properties
        self.maxKIc = maxKIc

    def __call__(self, fracture):
        """ The implementing the terminating_criterion"""
        #x_min, x_max, y_min, y_max = get_fracture_sizes(fracture)
        vel = get_front_velocity(fracture)
        maxKIc = get_current_max_KIc(fracture,self.mat_properties)

        if maxKIc < self.maxKIc and vel > self.target_vel_toll :
            print(" KI < KIc_max and vel > target ")
            return True # keep going and see if it stops
        elif maxKIc >= self.maxKIc and vel > self.target_vel_toll  :
            print(" KI >= KIc_max and vel > target ")
            return False #stop because the chance of having v to zero are gone
        elif maxKIc < self.maxKIc and vel <= self.target_vel_toll :
            print(" KI < KIc_max and vel <= target")
            return False #toughness ratio too high
        elif maxKIc >= self.maxKIc and vel <= self.target_vel_toll :
            print(" KI >= KIc_max and vel <= target")
            return False #keep going
        else:
            print("to be investigated --> END")
            return False

# defining the return function in case the simulation ends according to the terminating criterion function
def return_function(fracture):
    return fracture

def get_front_velocity(Fr):
    # get the length of the fracture circumference
    l_collection = np.zeros(len(Fr.Ffront))
    center_collenction_x =  np.zeros(len(Fr.Ffront))
    center_collenction_y =  np.zeros(len(Fr.Ffront))
    frontIDlist =   np.zeros(len(Fr.Ffront),dtype=int)
    for point_ID, point in enumerate(Fr.Ffront):
        [x1, y1] = [point[0],point[1]]
        [x2, y2] = [point[2], point[3]]
        l = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        l_collection[point_ID] = l
        center_collenction_x[point_ID] = 0.5 * (x1+x2)
        center_collenction_y[point_ID] = 0.5 * (y1+y2)
        frontIDlist[point_ID]  = Fr.mesh.locate_element(center_collenction_x[point_ID],center_collenction_y[point_ID])[0]
    front_length = np.sum(l_collection)

    # check if there are duplicates
    frontIDlist_new = np.unique(frontIDlist)
    if len(frontIDlist) != len(frontIDlist_new):
        print("ERROR locating the front cell from the front edge coordinates")
        SystemExit()
    # check if the number of elements is the same to the tip elements
    if len(Fr.EltTip) != len(frontIDlist_new):
        print("ERROR the number of tip elements is not the same as the one derived by the front elements")
        SystemExit()
    #check if the elements are the same:
    if np.sum(np.sort(Fr.EltTip) - np.sort(frontIDlist_new)) !=0:
        print("ERROR Fr.EltTip does not contain the same elements as in 'frontIDlist_new' ")
        SystemExit()
    #check if the elements are the same:
    if np.sum(Fr.EltTip - frontIDlist_new) !=0:                               # get the sorting vectors
        # strategy:
        #   v1 = [1 3 5 2]      -> [1 2 3 5]  and via [1 4 2 3]
        #   v2 = [5 2 3 1]      -> [4 2 3 1]  and via [4 2 3 1]
        #      so element in pos 1 correspond to elem in pos 4
        #      so element in pos 4 correspond to elem in pos 2
        #      so element in pos 2 correspond to elem in pos 3
        #      so element in pos 3 correspond to elem in pos 1

        indSort1=np.argsort(Fr.EltTip)
        indSort2=np.argsort(frontIDlist_new)
        sorted_l = np.zeros(len(Fr.EltTip))
        check_  = np.zeros(len(Fr.EltTip))
        # sort  frontIDlist_new and length as Fr.EltTip
        for ii in range(len(indSort1)):
             sorted_l[indSort1[ii]] = l_collection[indSort2[ii]]
             check_[indSort1[ii]] = frontIDlist_new[indSort2[ii]]
        if  np.sum(check_ -  Fr.EltTip) !=0:
             print("ERROR sorting the arrays has a bug ")
             SystemExit()
        else:
             l_collection = sorted_l


    # get the coordinates of the element with the portion of the front "y=0, x>0"
    tipcoor = Fr.mesh.CenterCoor[Fr.EltTip]
    elID1 = None
    for point_ID, point in enumerate(tipcoor):
        if np.abs(point[1]) < Fr.mesh.hy and point[0] > 0:
            elID1 = Fr.EltTip[point_ID]
            break
    xc = Fr.mesh.CenterCoor[elID1][0]
    yc = Fr.mesh.CenterCoor[elID1][1]

    #get distances of teh tip elements from that point
    # distances are ordered as EltTip
    distances=np.zeros(len(Fr.EltTip))
    for i in range(len(Fr.EltTip)):
        i_ID = Fr.EltTip[i]
        [x1, y1] = [Fr.mesh.CenterCoor[i_ID][0], Fr.mesh.CenterCoor[i_ID][1]]
        distances[i] =  np.sqrt((x1-xc)*(x1-xc)+(y1-yc)*(y1-yc))

    # find the array that sorts the array of distances
    sorting_array = np.argsort(distances)

    # get the radius where to find the velocity
    front_portion =  np.maximum(0.02 * front_length,0.8 * Fr.mesh.hy  )

    l = 0.
    index = 0
    tip_cells_IDs = []
    while l <= front_portion:
        l = l + l_collection[sorting_array[index]]
        tip_cells_IDs.append(sorting_array[index])
        index = index + 1

    # compute the velocity along the front
    v = -(Fr.sgndDist[Fr.EltTip] - Fr.sgndDist_last[Fr.EltTip]) / Fr.timeStep_last

    # compute the average velocity
    print(f" number of cells involved {len(tip_cells_IDs)}\n")
    print(f" max v =  {np.max(v[tip_cells_IDs])}\n")
    vel = np.max(v[tip_cells_IDs])
    if vel<0.: vel = 0.
    return vel

def get_current_max_KIc(fracture, matProp):
    Ffront = fracture.Ffront
    KIc_i = np.zeros(len(Ffront))
    for i , segment in enumerate(Ffront):
        x1,y1,x2,y2 = segment
        xc = 0.5 * (x1+x2)
        yc = 0.5 * (y1+y2)
        KIc_i[i] = matProp.K1cFunc(xc, yc, 0.)
    return KIc_i.max()

class adapive_time_ref_factory():
    def __init__(self, r_0, delta, xmax_lim):
        self.r_0 = r_0
        self.delta = delta
        self.xmax_lim = xmax_lim


    def __call__(self, Fr_current, Fr_new, timestep):
        x_min_c, x_max_c, y_min_c, y_max_c = get_fracture_sizes(Fr_current)
        x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(Fr_new)
        if x_max_c <= self.r_0 and x_max_n > self.xmax_lim:
            time_step_new = (Fr_new.time + Fr_current.time)*0.5
            return time_step_new, True
        else:
            return timestep, False
        # x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(Fr_new)
        # vel_c = get_front_velocity(Fr_current)
        # vel_n = get_front_velocity(Fr_new)
        # if vel_c >= self.target_V and x_max_n > self.r_0:
        #     if np.abs(vel_n-self.target_V)>self.target_vel_toll:
        #         return timestep, False
        #     else:
        #         time_step_new = (Fr_new.time + Fr_current.time)*0.5
        #         return time_step_new, True
        # else:
        #     return timestep, False


class custom_factory():
    def __init__(self, r_0, delta,t_zero,xlabel, ylabel):
        self.data = {'xlabel' : xlabel,
                     'ylabel': ylabel,
                     'xdata': [],
                     'ydata': [],
                     'time_data':[t_zero],
                     'time_touch': None,
                     'x_plot_data': [],
                     'y_plot_data': [],
                     'H/2': r_0,
                     'delta':delta} # max value of x that can be reached during the simulation

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
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig

    def postprocess_fracture(self, sim_prop, fr):
        # this method is mandatory
        x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(fr)
        if x_max_n > self.data['H/2']- self.data['delta'] and self.data['time_touch'] == None:
            if len(self.data['time_data'])>0:
                self.data['time_touch'] = 0.5 * ((self.data['time_data'])[-1] + fr.time)
                for i in range(len(self.data['x_plot_data'])):
                    self.data['x_plot_data'][i]= self.data['x_plot_data'][i] / self.data['time_touch']
                self.data['xlabel'] = "t/t_touch"
            else:
                print('NO TIME STEP BEFORE BREAKTHROUGH ')


        self.data['xdata'].append(y_max_n / self.data['H/2'])
        self.data['ydata'].append(x_max_n / self.data['H/2'])
        self.data['time_data'].append(fr.time)
        if self.data['time_touch'] == None:
            self.data['x_plot_data'].append(fr.time)
        else:
            self.data['x_plot_data'].append(fr.time/self.data['time_touch'])
        self.data['y_plot_data'].append(100*(x_max_n / self.data['H/2']-1.))
        fr.postprocess_info = self.data
        return fr

def run(r_0, v_toll, xmax_lim, Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid):
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
    simulProp.set_outputFolder(simdir)
    simulProp.frontAdvancing = 'implicit'
    simulProp.plotVar = ['footprint','custom']
    #simulProp.EHL_iter_lin_solve=True
    simulProp.plotFigure = True
    simulProp.custom = custom_factory(r_0, delta,Fr.time,'time','100*(x_max_n /(.5H)-1.')

    simulProp.adaptive_time_refinement = adapive_time_ref_factory(r_0, delta, xmax_lim)
    Fr.postprocess_info={'xlabel' : None,
                     'ylabel': None,
                     'xdata': [],
                     'ydata': [],
                     'time_data':[],
                     'time_touch': None,
                     'x_plot_data': [],
                     'y_plot_data': [],
                     'H/2': r_0,
                     'delta':delta}
    # define the terminating criterion function
    simulProp.terminating_criterion = terminating_criterion_factory(r_0, delta, xmax_lim,v_toll, Solid, Solid_loaded.K1c[0]*KIc_ratio )

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
    return last_Fr, Solid

# upper bound function
def upper_bound_Kratio(muPrime, Q_o, Eprime, K1c1, xlim):
    dimlessK = ( ((K1c1**4)*(xlim*2)) / ((muPrime)*(Q_o)*(Eprime**3)) )**(1./4.)
    upper_bound_Kr = np.maximum((1+0.31881409564007984/(dimlessK**3))**(1/3),2.)
    return upper_bound_Kr

# generate list of simulations to do in the proper order
def get_todo_list():
    aSIM_ID = np.asarray(
        [30, 9, 72, 81, 174, 183, 219, 276, 321, 330, 366, 423, 468, 477, 513, 570, 615, 624, 660, 717, 819, 864, 909,
         918, 954, 966, 1011, 1056, 1065, 1101, 1113, 1482, 1488, 1494, 1557, 1563, 1635, 1713, 1719, 1734, 1794, 1800,
         1815, 1192, 1194, 1254, 1256, 1258, 1260, 1262, 1264, 1266, 1268, 1270, 1404, 1406, 1408, 1410, 1412, 1414,
         1416, 1418, 1420, 1422, 766, 770, 1326, 1328, 1330, 130, 134, 1959, 1893, 1878, 1791])
    aTR = np.asarray([93.61391011018539, 106.61874640274064, 80.7970187583269, 77.00965850403033, 56.55396796389728,
                     54.34482859030754, 48.40086296324266, 38.94756941573432, 33.470567466646685, 32.44203925902878,
                     28.893691215072508, 23.13752616831978, 19.883811550899814, 19.262442439934194, 17.155612798066393,
                     13.723555293793265, 11.793680330603586, 11.425127820272223, 10.193351610262912, 8.282098183338617,
                     5.661590555016632, 4.843313795111884, 4.162222792674275, 4.064670695970972, 3.6200973385991473,
                     3.3938412549367003, 2.916582328461227, 2.5292237379624707, 2.4699450566039753, 2.219091261792634,
                     2.089066383171972, 1.207741502771296, 1.207741502771296, 1.201249580809693, 1.1484375, 1.1484375,
                     1.106109619140625, 1.0728979110717773, 1.0768041610717773, 1.0672036409378052, 1.0544101931154728,
                     1.0596709884164284, 1.0596709884164284, 1.782560198980692, 1.773382602299877, 1.6098982231623729,
                     1.6098982231623729, 1.5924362300972816, 1.5957372437241348, 1.5870641419922453, 1.5784481949216573,
                     1.569890511563989, 1.561392588765474, 1.5631423378663767, 1.2991693669915128, 1.2991693669915128,
                     1.2991693669915128, 1.2929189934281524, 1.2929189934281524, 1.2883421341558376, 1.2860894612327451,
                     1.2860894612327451, 1.2777413159664945, 1.2777413159664945, 6.892419811827882, 6.762501468009765,
                     1.4375, 1.4375, 1.4345974199729223, 65.3652790349907, 64.0674938341572, 1.0417187499999998,
                     1.04828125, 1.0406640624999999, 1.069853515625])
    SIM_ID=np.asarray([30, 183, 330, 477, 624, 770, 918, 1011, 1101, 1192, 1270, 1326, 1404, 1482, 1557, 1635, 1713, 1791, 1815, 1893, 1959])
    pos=[]
    for j in range(len(SIM_ID)):
        mysim=SIM_ID[j]
        for i in range(len(aSIM_ID))  :
            SI=aSIM_ID[i]
            if SI == mysim:
                pos.append(i)
                break
    TR = aTR[pos]

    # list of simulations
    todo = []
    todo_n = []
    locallist = np.sort(SIM_ID)

    for number in locallist:  # range(0, 2107, 10):
        if number not in todo_n:
            todo.append(str(number))

    todo_n = len(todo)
    return SIM_ID, TR, todo, todo_n

def make_copy_of_the_res_file():
    # copy the file with the results - for safety!
    file_name_copy = "analyse_bt_res_copy.json"
    if os.path.isfile(file_name):
        shutil.copyfile(baseloc + file_name, baseloc + file_name_copy)

def create_or_reload_the_res_file(file_name, results):
    # create or reload the results file
    if not os.path.isfile(file_name):
        content = results
        action = 'dump_this_dictionary'
        append_to_json_file(file_name, [content], action, delete_existing_filename=False)
    else:
        with open(file_name, "r+") as json_file:
            results = json.load(json_file)[0]  # get the data
    return results

def manage_files_and_folders(num, date_ext, globalpath):
    simdir = globalpath + '/bt/simulation_' + num + '__' + date_ext

    # make the folder if it does not exist
    print(' -check if the folder existed')
    check_make_folder(simdir)

    # copy properties if they do not exist
    print('\n -check if the properties existed')
    dest_file = simdir + '/properties'
    src_file = globalpath + '/simulation__' + date_ext + '/properties'
    check_copy_file(dest_file, src_file)

    # check if the timestep exist in the source dir and copy
    print('\n -check if the initial file existed')
    dest_file = simdir + '/simulation_' + num + '__' + date_ext + '_file_0'
    src_file = globalpath + '/simulation__' + date_ext + '/simulation__' + date_ext + '_file_' + num
    check_copy_file(dest_file, src_file)

    # make a copy of the input folder
    print('\n -check if the newly created folder exist and make a copy of it')
    dest_folder = simdir + '_copy'
    src_folder = simdir
    copy_dir(dest_folder, src_folder)

    return simdir

def check_if_needed_to_be_recomputed(num, results, globalpath, date_ext):
    # todo: check this
    # check error and eventually recompute
    if int(num) in results["sim id"]:
        pos = np.where(np.asarray(results["sim id"]) == int(num))[0][0]
        check_xbt = (results["x_max"][pos] >= results["x_lim"][pos]
                     and results["x_max"][pos] <= results["x_lim"][pos] + results["delta"][pos])
        check_xbt = True
        if not check_xbt or int(num) in forced_recompute:
            print(
                f'xbt is in the proper range {check_xbt}, 100(xbt - x_lim)/delta {100 * (results["x_max"][pos] - results["x_lim"][pos]) / results["delta"][pos]}')
            results["toughness ratio"].pop(pos)
            results["sim id"].pop(pos)
            results["aspect ratio"].pop(pos)
            results["ended"].pop(pos)
            results["aspect_ratio_toll"].pop(pos)
            results["penetration_target"].pop(pos)
            results["x_max"].pop(pos)
            results["x_min"].pop(pos)
            results["x_lim"].pop(pos)
            results["xmax_lim"].pop(pos)
            results["delta"].pop(pos)
            results["halfH"].pop(pos)
            # remove the folder
            if os.path.isdir(globalpath + '/bt/simulation_' + num + '__' + date_ext):
                shutil.rmtree(globalpath + '/bt/simulation_' + num + '__' + date_ext)
            if os.path.isdir(globalpath + '/bt/simulation_' + num + '__' + date_ext + '_copy'):
                shutil.rmtree(globalpath + '/bt/simulation_' + num + '__' + date_ext + '_copy')
    return results

# --------------------------------------------------------------
# --------------------------------------------------------------
print('STARTING SIMULATION:')
"""
we fix a penetration target and we loop on the toughness ratio from the moment we touch
"""

# user parameters
file_name = "analyse_bt_res.json"
globalpath = '/home/carlo/Desktop/PyFrac/16_tough_eterog/space_inv'
baseloc = "/home/carlo/Desktop/PyFrac/16_tough_eterog/space_inv/"
date_ext = '2022-02-02__09_02_40'
forced_recompute = []


SIM_ID, TR, todo, todo_n = get_todo_list()

make_copy_of_the_res_file()

# initialize some result vars
results = {"toughness ratio" : [],
            "sim id" : [],
            "time" : [],
            "target_v_toll" : [] ,
            "x_max": [],
            "x_min": [],
            "y_max": [],
            "y_min": [],
            "x_lim": [],
            "xmax_lim": [],
            "delta": [],
            "halfH": [],
            "ended": [],
            }

results = create_or_reload_the_res_file(file_name, results)


# initialize the loop variables
KIc_ratio = None
KIc_ratio_upper = None
KIc_ratio_lower = None

# LOOP ON THE SIMULATIONS
for num_id, num in enumerate(todo):
    print(f'sim {num_id+1} of {todo_n}\n')

    #results = check_if_needed_to_be_recomputed(num, results, globalpath, date_ext)

    if int(num) not in results["sim id"]:

        simdir = manage_files_and_folders(num, date_ext, globalpath)

        # load the fracture obj
        Fr_list, properties = load_fractures(address=globalpath + '/bt', step_size=100, sim_name='simulation_' + num)
        Solid_loaded, Fluid, Injection, simulProp = properties

        contunue_loop = True
        it_count = 0

        # check the location of the barrier
        Fr = copy.deepcopy(Fr_list[-1])

        # get the average velocity of the front
        v = -(Fr.sgndDist[Fr.EltTip] - Fr.sgndDist_last[Fr.EltTip]) / Fr.timeStep_last
        average_v_touch = np.mean(v)

        # define the hard limits
        x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
        delta = Fr.mesh.hx / 100.  # <-- a tolerance
        r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + delta # <-- the jump position
        x_lim = r_0

        # this is the relative position to the cartesian grid
        relative_pos_xlim = ((r_0 - 0.5 * Fr.mesh.hx) % Fr.mesh.hx) / Fr.mesh.hx

        print(f'\n -number of elts {len(Fr_list[-1].EltCrack)} \n sim {num_id + 1}\n and rel pos x_lim {relative_pos_xlim}')

        if not len(Fr_list[-1].EltCrack) > 8000 and relative_pos_xlim > .5 and relative_pos_xlim < .98:

            # inner loop on the  min toughness ratio
            while contunue_loop:

                Fr = copy.deepcopy(Fr_list[-1])

                # define the hard limit (again)
                x_min, x_max, y_min, y_max = get_fracture_sizes(Fr)
                r_0 = np.maximum(np.abs(x_min), np.abs(x_max)) + delta # <-- the jump position
                delta = Fr.mesh.hx / 100.  # <-- a tolerance
                x_lim = r_0 # <-- ?     # todo: check this

                # tollerance xmax
                toll_xmax = 10*delta    # the front can advance at most of this quantity
                xmax_lim = x_lim + toll_xmax    # the front can advance at most of this quantity

                v_toll =0.001 * average_v_touch

                # current state variables
                if int(num) == 1000000000 and it_count ==0 :
                    KIc_ratio = 50
                    KIc_ratio_upper = 100
                    KIc_ratio_lower = 1
                else:
                    if KIc_ratio is None or (it_count == 0):
                        if int(num) in SIM_ID:
                            pos = np.where(SIM_ID==int(num))[0][0]
                            KIc_ratio = TR[pos]
                            KIc_ratio_upper = KIc_ratio + 8. * KIc_ratio
                            KIc_ratio_lower = .3 * KIc_ratio
                            #KIc_ratio =  KIc_ratio_upper# (KIc_ratio_lower + KIc_ratio_upper)
                            if KIc_ratio_lower < 1.:
                                KIc_ratio_lower = 1.


                print(f'\n iterations on tough. ratio: {it_count} of 200, ID: {num}')
                print(f' toughness ratio: {KIc_ratio}')
                print(f' tough. min: {KIc_ratio_lower}')
                print(f' tough. max: {KIc_ratio_upper}')
                print(f' rel diff limits: {100 * np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower} %')

                last_Fr, Solid = run(r_0, v_toll, xmax_lim ,Solid_loaded, Injection, Fr, KIc_ratio, delta, simulProp, Fluid)

                # decide what to do with the aspect ratio
                print("checks:")

                # 1) check if the upper and the lower KIc ratio are close to each other
                target_reduction = (np.abs(KIc_ratio_lower - KIc_ratio_upper) / KIc_ratio_lower > 0.001)
                if target_reduction:
                    print(f'np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower = {np.abs(KIc_ratio_lower-KIc_ratio_upper)/KIc_ratio_lower} > 0.001')
                else:
                    print(f' |KIc_ratio_lower-KIc_ratio_upper|/KIc_ratio_lower = {np.abs(KIc_ratio_lower - KIc_ratio_upper) / KIc_ratio_lower} < 0.001')

                # 2) did the fracture fully penetrate the new medium?
                maxKIc = get_current_max_KIc(last_Fr, Solid)
                KI_GE_target = maxKIc >= KIc_ratio * Solid_loaded.K1c[0]

                print(f"KI/target_KIc {maxKIc/(KIc_ratio * Solid_loaded.K1c[0])} ")
                if KI_GE_target :
                    print(" KI >= target ")
                else:
                    print(" KI < target ")

                # 3) did we reduce the velocity to zero?
                vel_last_Fr = get_front_velocity(last_Fr)
                v_GE_vmin = vel_last_Fr >= v_toll
                if v_GE_vmin:
                    print(" v >= v min ")
                else:
                    print(" v < v min ")


                # 4) check if xmax < xlim i.e.: we penetrated too much
                x_min_c, x_max_c, y_min_c, y_max_c = get_fracture_sizes(last_Fr)
                x_GE_xmax_lim = x_max_c >= xmax_lim

                if x_GE_xmax_lim:
                    print(" x >= x max lim ")
                else:
                    print(" x < x max lim ")

                # x_G_x_lim = x_max_c > x_lim
                # if x_G_x_lim:
                #     print(" x > x_lim ")
                # else:
                #     print(" x <= x_lim ")

                # here we have got to decide what to do
                # update the counter of iterations
                it_count = it_count + 1
                if it_count < 300:
                    if ((KI_GE_target and v_GE_vmin) or (not v_GE_vmin and x_GE_xmax_lim)) and target_reduction:
                        print(' increasing toughness ratio')
                        # increase toughness in the bounding layer
                        if (v_GE_vmin and x_GE_xmax_lim and KIc_ratio >= KIc_ratio_upper):
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
                    elif (not KI_GE_target) and target_reduction:
                        print(' decreasing toughness ratio')
                        print(f' x/xlim: {x_max_c/x_lim}')
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
                        #KIc_ratio_new = smoothing(Solid_loaded.K1c[0], Solid_loaded.K1c[0] * KIc_ratio, r_0, delta, smaller_abs_x_c)/Solid_loaded.K1c[0]
                        # delete the folder and get a new one
                        src_folder = simdir + '_copy'
                        dest_folder = simdir
                        copy_dir(dest_folder, src_folder)

                    elif not target_reduction or (KI_GE_target and not v_GE_vmin and not x_GE_xmax_lim ):
                        # accept solution
                        print('-solution achieved')
                        print(f' x/xlim: {x_max_c / x_lim}')
                        contunue_loop = False
                        results["toughness ratio"].append(KIc_ratio)
                        results["sim id"].append(int(num))
                        results["time"].append(last_Fr.time)
                        results["target_v_toll"].append(v_toll)
                        results["x_max"].append(x_max_c)
                        results["x_min"].append(x_min_c)
                        results["y_max"].append(y_max_c)
                        results["y_min"].append(y_min_c)
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
                        shutil.copyfile(baseloc + file_name, baseloc + file_name_copy)
                        print('-----------------------------')
                    else :
                        print("\n to be investigated")
                        b = input("    -->press a button to kill the program")
                        SystemExit()
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