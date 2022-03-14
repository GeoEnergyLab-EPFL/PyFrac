#external imports
import copy
import json
import os
import shutil
import numpy as np

from src.utilities.postprocess_fracture import append_to_json_file, load_fractures
# --------------------------------------------------------------
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


print('POST SIMULATION:')
globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv'
date_ext = '2022-02-02__09_02_40'
basename = '/simulation__'+date_ext+'_file_'

# load the fracture obj
Fr_list, properties = load_fractures(address=globalpath + '/test_bt_1p2/', step_size=1)
Solid_loaded, Fluid, Injection, simulProp = properties

Fr = copy.deepcopy(Fr_list[-1])
content = Fr.postprocess_info

content['w_center'] = []
content['x_av'] = []
content['t'] = []
content['xneg_min'] = []
content['xpos_max'] = []
for fr_i in Fr_list:
    x_min_temp, x_max_temp, y_min_temp, y_max_temp = get_fracture_sizes(fr_i)
    content['xneg_min'].append(x_min_temp)
    content['xpos_max'].append(x_max_temp)

    # get the points of Ffront that are ahead
    tn = 0.
    ti = 0.
    for seg in fr_i.Ffront:
        x1, y1, x2, y2 = seg
        if x1 > content["H/2"]:
            ti = ti + x1/content["H/2"]
            tn = tn + 1
    if tn > 0:
       content['x_av'].append(ti/tn)
    else:
        content['x_av'].append(1.)

    center_ID = fr_i.mesh.locate_element(0., 0.)
    content['w_center'].append(fr_i.w[center_ID][0].tolist())
    content['t'].append(fr_i.time)

content['viscosity'] = Fluid.viscosity
content['injrate'] = Injection.injectionRate[1][0]
content['K1c2'] = Solid_loaded.K1c.max()
content['K1c1'] = Solid_loaded.K1c.min()
content['Ep'] = Solid_loaded.Eprime
content['hx'] = Fr.mesh.hx
content['hy'] = Fr.mesh.hy

append_to_json_file(globalpath + '/test_bt_1p2/res06', content, 'dump_this_dictionary', delete_existing_filename=True)

