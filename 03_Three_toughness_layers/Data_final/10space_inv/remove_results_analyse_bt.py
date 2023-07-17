#external imports
import copy
import json
import os
import shutil
import numpy as np

# internal imports
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
print('STARTING SIMULATION:')
"""
remove info from one file 
"""

file_name = "analyse_bt_res.json"
globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv'

# copy the file for safety!
baseloc = "/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv/"
file_name_copy = "analyse_bt_res_copy.json"
if os.path.isfile(file_name):
    shutil.copyfile(baseloc+file_name, baseloc+file_name_copy)

with open(file_name, "r+") as json_file:
    results = json.load(json_file)[0]  # get the data

todo = [1791]
todo_n = len(todo)

for num_id, num in enumerate(todo):

    print(f'sim {num_id+1} of {todo_n}\n')

    # check error and eventually recompute
    if int(num) in results["sim id"]:
        pos = np.where(np.asarray(results["sim id"]) == int(num))[0][0]
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
        # if os.path.isdir(globalpath + '/bt/simulation_' + num + '__' + date_ext):
        #     shutil.rmtree(globalpath + '/bt/simulation_' + num + '__' + date_ext)
        # if os.path.isdir(globalpath + '/bt/simulation_' + num + '__' + date_ext + '_copy'):
        #     shutil.rmtree(globalpath + '/bt/simulation_' + num + '__' + date_ext + '_copy')

        print(" Saving to file")
        content = results
        action = 'dump_this_dictionary'
        append_to_json_file(file_name, [content], action, delete_existing_filename=True)

        # copy the file for safety!
        file_name_copy = "analyse_bt_res_copy.json"
        shutil.copyfile(baseloc + file_name, baseloc + file_name_copy)
        print('-----------------------------')

